"""Where each map starts and ends, derived from the video rather than assumed.

Three independent mechanisms name a boundary: all eight scoreboard slots
resetting to 0/0 at once, the header's pips advancing, and the header's map
score resetting to 0-0. `derive(strict=True)` raises if a boundary is short a
witness, which is what a layout change should do.

The mode comes from the scoreboard's sixth column label (TIME/PLNT/OVLD), so
a map gets a mode even without a box score.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

# A run this short is a single mid-fade frame or a cut, not a state.
MIN_RUN_SECONDS = 0.5

# Two witnesses are the same boundary if they land this close.
WITNESS_TOLERANCE = 2.0

# The series is best of 9. A header frame saying otherwise is a recap clip of
# some other match.
SERIES_BEST_OF = 9

STAT_LABEL_MODES = {"TIME": "hardpoint", "PLNT": "search_and_destroy",
                    "OVLD": "overload"}


@dataclass(frozen=True)
class MapWindow:
    """One map's LIVE span, in seconds of the source VOD."""

    number: int
    start: float
    end: float
    mode: str | None = None
    pips: tuple[int, int] | None = None
    witnesses: tuple[str, ...] = ()

    def contains(self, t: float) -> bool:
        return self.start <= t <= self.end


@dataclass
class Boundary:
    """A candidate map boundary and which mechanisms saw it."""

    t: float
    witnesses: dict[str, float] = field(default_factory=dict)

    @property
    def confirmed(self) -> bool:
        return len(self.witnesses) >= 3


# -- the three witnesses ---------------------------------------------------


def scoreboard_resets(runs: list[dict], quorum: int = 8) -> dict[float, int]:
    """Times at which a slot's kills and deaths both fall back to 0.

    Returned as time -> how many of the eight slots reset there. Times are
    rounded to the second since run edges for different slots can differ by
    a frame.
    """
    votes: Counter[float] = Counter()
    for slot_runs in _by_slot(runs).values():
        previous = None
        for run in slot_runs:
            value = run["value"]
            if value["kills"] is None:
                continue
            if (value["kills"] == 0 and value["deaths"] == 0 and previous
                    and (previous["kills"] or previous["deaths"])):
                votes[round(run["start"])] += 1
            previous = value
    return {t: n for t, n in sorted(votes.items()) if n >= quorum}


def pip_advances(header_runs: list[dict]) -> dict[float, tuple[int, int]]:
    """Times at which the series score gains a map win, -> the new pips.

    Two filters: runs whose `best_of` is not 9 are dropped as recaps of some
    other match, and readings below the running maximum are dropped as
    replays of an earlier point in this series (the series score never goes
    down).
    """
    series = _pip_series(header_runs)
    out: dict[float, tuple[int, int]] = {}
    for (_, _, before), (start, _, after) in zip(series, series[1:]):
        if sum(after) == sum(before) + 1:
            out[round(start)] = after
    return out


def score_resets(header_runs: list[dict]) -> list[float]:
    """Times at which the map score goes back to 0-0.

    Tested on the pair, not on each side separately: a side that ends a map
    already at 0 never steps, so it never witnesses on its own.
    """
    series = pair_series(header_runs, "score_left", "score_right")
    return [
        round(start)
        for (_, _, before), (start, _, after) in zip(series, series[1:])
        if after == (0, 0) and before != (0, 0)
    ]


# -- putting them together -------------------------------------------------


def live_series_spans(header_runs: list[dict]) -> list[tuple[float, float]]:
    """Spans the header says are the present state of this series.

    Filters recaps of another match (`best_of` != 9) and recaps of an earlier
    map of this series (pips behind the running maximum). A span is further
    truncated where the map score decreases, since a recap can hold a valid
    best-of-9 header and un-advanced pips while showing an old map score.
    """
    return [
        _until_score_falls(header_runs, t0, t1)
        for t0, t1, _ in _pip_series(header_runs)
    ]


def _until_score_falls(header_runs: list[dict], start: float,
                       end: float) -> tuple[float, float]:
    """Truncate a span at the first frame where the map score goes backwards."""
    inside = [
        (t0, pair) for t0, _, pair
        in pair_series(header_runs, "score_left", "score_right")
        if start <= t0 <= end
    ]
    best = (0, 0)
    for t0, pair in inside:
        if pair[0] < best[0] or pair[1] < best[1]:
            return (start, t0 - MIN_RUN_SECONDS)
        best = (max(best[0], pair[0]), max(best[1], pair[1]))
    return (start, end)


def filter_to_series(runs: list[dict], header_runs: list[dict]):
    """Partition scoreboard runs into (the live series, replays of it).

    A cross-mechanism filter: the verdict comes from the header strip, the
    runs from the two panel strips. Catches recaps of an earlier map of this
    series, which `scoreboard_runs.split_foreign` cannot (same roster).
    """
    spans = live_series_spans(header_runs)
    ours, replays = [], []
    for run in runs:
        (ours if _within(spans, run["start"]) else replays).append(run)
    return ours, replays


def boundaries(runs: list[dict], header_runs: list[dict]) -> list[Boundary]:
    """Every candidate boundary, annotated with the witnesses that saw it.

    A header witness with no scoreboard reset beside it still gets its own
    Boundary, so a missing reset shows up as unconfirmed rather than nothing.
    """
    seen: dict[str, dict[float, object]] = {
        "scoreboard_reset": dict(scoreboard_resets(runs)),
        "pip_advance": dict(pip_advances(header_runs)),
        "score_reset": {t: 0 for t in score_resets(header_runs)},
    }
    out: list[Boundary] = []
    for name in ("scoreboard_reset", "pip_advance", "score_reset"):
        for t in sorted(seen[name]):
            match = next(
                (b for b in out if abs(b.t - t) <= WITNESS_TOLERANCE), None
            )
            if match is None:
                match = Boundary(t=float(t))
                out.append(match)
            match.witnesses[name] = float(t)
    out.sort(key=lambda b: b.t)
    return out


def derive(runs: list[dict], header_runs: list[dict],
           strict: bool = False) -> list[MapWindow]:
    """Per-map windows from the scoreboard and header timelines.

    `runs` must already have had foreign runs removed
    (`scoreboard_runs.split_foreign`), since a recap of another series carries
    its own scoreboard whose slots reset on that match's boundaries.
    """
    found = boundaries(runs, header_runs)
    unconfirmed = [b for b in found if not b.confirmed]
    if unconfirmed and strict:
        raise ValueError(
            "map boundaries disagree between mechanisms: "
            + ", ".join(f"t={b.t:.0f} seen only by {sorted(b.witnesses)}"
                        for b in unconfirmed)
        )
    starts = [b for b in found if b.confirmed]

    live = sorted(
        (r["start"], r["end"]) for r in runs if r["value"]["kills"] is not None
    )
    if not live:
        return []
    edges = [live[0][0]] + [b.t for b in starts]
    windows: list[MapWindow] = []
    for number, start in enumerate(edges, start=1):
        limit = edges[number] if number < len(edges) else float("inf")
        inside = [(s, e) for s, e in live if start <= s < limit]
        if not inside:
            continue
        end = max(e for _, e in inside)
        witnesses = (("first_reading",) if number == 1
                     else tuple(sorted(starts[number - 2].witnesses)))
        windows.append(MapWindow(
            number=number,
            start=start,
            end=end,
            mode=_mode_of(runs, start, end),
            pips=_pips_at(header_runs, start, end),
            witnesses=witnesses,
        ))
    return windows


def assign_boxscores(windows: list[MapWindow],
                     boxscores: list[dict]) -> dict[int, dict]:
    """Map number -> the box score drawn after it, where one exists.

    By time, not by the box score's own `map` field, which is just a count
    of box scores found and does not always match the map number.
    """
    out: dict[int, dict] = {}
    for box in sorted(boxscores, key=lambda b: b["t"]):
        before = [w for w in windows if w.start <= box["t"]]
        if before:
            out[max(before, key=lambda w: w.start).number] = box
    return out


# -- internals -------------------------------------------------------------


def _by_slot(runs: list[dict]) -> dict[tuple[str, int], list[dict]]:
    out: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for run in runs:
        out[(run["side"], run["slot"])].append(run)
    for slot_runs in out.values():
        slot_runs.sort(key=lambda r: r["start"])
    return out


def _field_runs(header_runs: list[dict], name: str) -> list[dict]:
    return sorted(
        (r for r in header_runs
         if r["field"] == name and r["end"] - r["start"] >= MIN_RUN_SECONDS),
        key=lambda r: r["start"],
    )


def pair_series(header_runs: list[dict], left_field: str,
                 right_field: str) -> list[tuple[float, float, tuple[int, int]]]:
    """(start, end, (left, right)) for a two-sided header field.

    Merged on the union of both sides' run edges, since either side can step
    on its own; restricted to spans the header itself calls best-of-9.
    """
    ours = _series_spans(header_runs)
    left = _field_runs(header_runs, left_field)
    right = _field_runs(header_runs, right_field)
    out: list[tuple[float, float, tuple[int, int]]] = []
    for t in sorted({r["start"] for r in left} | {r["start"] for r in right}):
        if not _within(ours, t):
            continue
        a, b = _run_at(left, t), _run_at(right, t)
        if a is None or b is None or a["value"] is None or b["value"] is None:
            continue
        pair = (a["value"], b["value"])
        # Ends at whichever side's run ends first; the pair only holds while
        # both do.
        end = min(a["end"], b["end"])
        if out and out[-1][2] == pair:
            out[-1] = (out[-1][0], max(out[-1][1], end), pair)
        else:
            out.append((t, end, pair))
    return out


def _run_at(runs: list[dict], t: float):
    return next((r for r in runs if r["start"] <= t <= r["end"]), None)


def _pip_series(header_runs: list[dict]) -> list[tuple[float, float, tuple[int, int]]]:
    """The series score over time, as a staircase that only ever goes up."""
    out: list[tuple[float, float, tuple[int, int]]] = []
    best = (0, 0)
    for start, end, pips in pair_series(header_runs, "maps_left", "maps_right"):
        if pips[0] < best[0] or pips[1] < best[1]:
            continue
        best = pips
        if out and out[-1][2] == pips:
            out[-1] = (out[-1][0], end, pips)
        else:
            out.append((start, end, pips))
    return out


def _series_spans(header_runs: list[dict]) -> list[tuple[float, float]]:
    return [
        (r["start"], r["end"]) for r in _field_runs(header_runs, "best_of")
        if r["value"] == SERIES_BEST_OF
    ]


def _within(spans: list[tuple[float, float]], t: float) -> bool:
    return any(s <= t <= e for s, e in spans)


def _mode_of(runs: list[dict], start: float, end: float) -> str | None:
    votes: Counter[str] = Counter()
    for run in runs:
        if start <= run["start"] <= end:
            label = run["value"].get("stat_label")
            if label in STAT_LABEL_MODES:
                votes[STAT_LABEL_MODES[label]] += run["samples"]
    return votes.most_common(1)[0][0] if votes else None


def _pips_at(header_runs: list[dict], start: float,
             end: float) -> tuple[int, int] | None:
    votes: Counter[tuple[int, int]] = Counter()
    for t0, t1, pips in _pip_series(header_runs):
        if start <= t0 <= end:
            votes[pips] += 1
    return votes.most_common(1)[0][0] if votes else None
