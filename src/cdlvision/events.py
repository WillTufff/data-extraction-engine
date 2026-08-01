"""Killfeed rows to discrete kill events.

Two distinct steps: reading turns one row in one frame into names, per frame
and stateless. Deduplicating collapses the several frames one entry is on
screen for into a single event with a single timestamp, using the overlay's
behaviour over time (bottom-anchored stack, entries' slot only decreases).

A genuine repeat kill (same attacker/victim twice in quick succession) must
stay two events; that is separated from a persisting entry by two-entries-
coexisting or slot-going-back-down.

Entry frames are pooled into one bitmap probe per name before matching,
which reads better than any single frame does alone.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from .names import NameMatch, NameMatcher, average_masks

# How far apart two observations of one entry may sit before the gap is read
# as a new entry rather than a persisting one.
MAX_GAP = 6.0

# How much a row's token widths may move between frames and still be
# considered the same entry.
WIDTH_SLACK = 3

# A chain seen in only one frame has no corroboration, so it must be read
# more confidently than a pooled one to be trusted.
LONE_MAX_DISTANCE = 0.45


@dataclass(frozen=True)
class ChainParams:
    """Chaining settings for one sampling rate.

    Not universal constants: `max_gap` and `min_frames` are thresholds in
    seconds/frames, so their meaning moves with the rate the rows were
    scanned at.
    """

    max_gap: float
    width_slack: int
    min_frames: int
    features: object

    def kwargs(self) -> dict:
        return {
            "max_gap": self.max_gap,
            "width_slack": self.width_slack,
            "min_frames": self.min_frames,
            "features": self.features,
        }


@dataclass(frozen=True)
class RowReading:
    """One killfeed row, in one frame, with its names read."""

    t: float
    slot: int
    attacker: NameMatch
    victim: NameMatch
    medal: bool
    signature: tuple[int, int, int]
    attacker_mask: np.ndarray | None = None
    victim_mask: np.ndarray | None = None

    @property
    def resolved(self) -> bool:
        return self.attacker.resolved and self.victim.resolved

    @property
    def pair(self) -> tuple[str | None, str | None]:
        return (self.attacker.label, self.victim.label)


@dataclass
class KillEvent:
    """One kill, collapsed from every frame that showed it."""

    t: float
    attacker: str | None
    victim: str | None
    medal: bool
    frames: int
    slots: list[int] = field(default_factory=list)
    same_team: bool = False

    @property
    def resolved(self) -> bool:
        return self.attacker is not None and self.victim is not None


def read_rows(matcher: NameMatcher, records: list[dict], decode) -> list[RowReading]:
    """Read cached scan rows into per-frame readings.

    `records` are what `tools/scan_killfeed.py` wrote and `decode` turns one
    stored bitmap back into a mask -- passed in so this module never has to
    know how the cache is encoded.
    """
    out: list[RowReading] = []
    for r in records:
        ma = decode(r["attacker_bmp"])
        mv = decode(r["victim_bmp"])
        a, v = matcher.match_row(ma, mv)
        out.append(
            RowReading(
                t=float(r["t"]),
                slot=int(r["slot"]),
                attacker=a,
                victim=v,
                medal=bool(r.get("medal_bmp")),
                signature=tuple(r["signature"]),
                attacker_mask=ma,
                victim_mask=mv,
            )
        )
    return out


def _key(reading: RowReading) -> tuple:
    """What identifies an entry across frames: the pair of names, nothing else.

    Unresolved rows all share the `(None, None)` key; the chain logic (not
    this key) is what separates distinct entries with the same identity.
    """
    return reading.pair


def dedupe(readings: list[RowReading], max_gap: float = MAX_GAP) -> list[KillEvent]:
    """Collapse persisting rows into one event each, keeping real repeats.

    Readings are grouped by identity and then followed frame by frame as
    chains, one chain per entry actually on screen; several entries with the
    same two names can be open at once. Within a frame, rows attach
    oldest-first (highest slot belongs to the oldest open chain), since
    entries enter at the bottom and only move up.
    """
    groups: dict[tuple, list[RowReading]] = defaultdict(list)
    for r in readings:
        groups[_key(r)].append(r)

    events: list[KillEvent] = []
    for group in groups.values():
        events.extend(_event(run) for run in _chains(group, max_gap))

    events.sort(key=lambda e: e.t)
    return events


def widths_of(row: dict) -> tuple[int, ...]:
    """Attacker, weapon and victim token widths."""
    return tuple(row["signature"])


def anchors_of(row: dict) -> tuple[int, ...]:
    """Where the weapon icon and victim name start.

    Steadier than token widths: the attacker token's left edge is fragile
    (bright gameplay bleed-in, threshold loss on dim glyphs), but downstream
    x-offsets are stable since they follow from the attacker name's rendered
    width.
    """
    return (row["x"][1], row["x"][2])


# Settings for a 1 Hz row cache.
SPARSE_1HZ = ChainParams(
    max_gap=MAX_GAP, width_slack=WIDTH_SLACK, min_frames=1, features=widths_of
)

# Settings for a 12 Hz row cache.
DENSE_12HZ = ChainParams(
    max_gap=1.25, width_slack=2, min_frames=5, features=anchors_of
)


def chain_rows(
    rows: list[dict],
    max_gap: float = MAX_GAP,
    width_slack: int = WIDTH_SLACK,
    features=widths_of,
) -> list[list[dict]]:
    """Group raw scan rows into one chain per on-screen entry, using geometry.

    Runs before any name is read, so a single misread frame cannot split an
    entry into two events. Two rows join the same chain when the slot has not
    moved back down, the gap is shorter than an entry lives, and the token
    widths still agree.
    """

    def compatible(prev: dict, cur: dict) -> bool:
        return all(
            abs(a - b) <= width_slack
            for a, b in zip(features(prev), features(cur))
        )

    frames: dict[float, list[dict]] = defaultdict(list)
    for r in rows:
        frames[float(r["t"])].append(r)

    done: list[list[dict]] = []
    open_runs: list[list[dict]] = []
    for t in sorted(frames):
        still_open = []
        for run in open_runs:
            (still_open if t - float(run[-1]["t"]) <= max_gap else done).append(run)
        open_runs = still_open

        used: set[int] = set()
        for r in sorted(frames[t], key=lambda r: r["slot"]):
            candidates = [
                (i, run)
                for i, run in enumerate(open_runs)
                if i not in used
                and run[-1]["slot"] >= r["slot"]
                and float(run[-1]["t"]) < t
                and compatible(run[-1], r)
            ]
            if candidates:
                i, run = min(candidates, key=lambda c: c[1][-1]["slot"])
                run.append(r)
                used.add(i)
            else:
                open_runs.append([r])
                used.add(len(open_runs) - 1)

    done.extend(open_runs)
    return done


def build_events(
    matcher: NameMatcher,
    records: list[dict],
    decode,
    lone_max_distance: float = LONE_MAX_DISTANCE,
    max_gap: float = MAX_GAP,
    width_slack: int = WIDTH_SLACK,
    min_frames: int = 1,
    features=widths_of,
) -> list[KillEvent]:
    """Scan rows to events: chain on geometry, then read each entry once.

    The pooled probe is a pixelwise majority over every frame the entry was
    on screen, so per-frame compression noise cancels instead of accumulating.
    Chaining parameters are arguments rather than constants because the right
    values depend on the sampling rate the rows were scanned at.
    """
    events: list[KillEvent] = []
    for chain in chain_rows(
        records, max_gap=max_gap, width_slack=width_slack, features=features
    ):
        if len(chain) < min_frames:
            continue
        a = average_masks([decode(r["attacker_bmp"]) for r in chain])
        v = average_masks([decode(r["victim_bmp"]) for r in chain])
        ma, mv = matcher.match_row(a, v)
        if len(chain) == 1 and max(ma.distance, mv.distance) > lone_max_distance:
            ma = mv = NameMatch(None, ma.distance, ma.runner_up, ma.runner_up_distance)
        events.append(
            KillEvent(
                t=float(chain[0]["t"]),
                attacker=ma.label,
                victim=mv.label,
                medal=any(r.get("medal_bmp") for r in chain),
                frames=len(chain),
                slots=[r["slot"] for r in chain],
                same_team=matcher.same_team(ma, mv),
            )
        )
    events.sort(key=lambda e: e.t)
    return events


def _chains(group: list[RowReading], max_gap: float) -> list[list[RowReading]]:
    """Follow one identity's rows into one run per entry."""
    frames: dict[float, list[RowReading]] = defaultdict(list)
    for r in group:
        frames[r.t].append(r)

    done: list[list[RowReading]] = []
    open_runs: list[list[RowReading]] = []
    for t in sorted(frames):
        # An entry not seen for longer than it lives is closed.
        still_open = []
        for run in open_runs:
            (still_open if t - run[-1].t <= max_gap else done).append(run)
        open_runs = still_open

        used: set[int] = set()
        for r in sorted(frames[t], key=lambda r: r.slot):
            candidates = [
                (i, run)
                for i, run in enumerate(open_runs)
                if i not in used and run[-1].slot >= r.slot and run[-1].t < t
            ]
            if candidates:
                i, run = min(candidates, key=lambda c: c[1][-1].slot)
                run.append(r)
                used.add(i)
            else:
                open_runs.append([r])
                used.add(len(open_runs) - 1)

    done.extend(open_runs)
    return done


def _event(run: list[RowReading]) -> KillEvent:
    first = run[0]
    return KillEvent(
        t=first.t,
        attacker=first.attacker.label,
        victim=first.victim.label,
        # A medal fades with the row, so trust any frame that saw one.
        medal=any(r.medal for r in run),
        frames=len(run),
        slots=[r.slot for r in run],
    )


def pooled_masks(run: list[RowReading]) -> tuple[np.ndarray, np.ndarray]:
    """Average one entry's frames into a single probe per name.

    The point of keeping the bitmaps: a row that no single frame reads
    confidently is often unambiguous once its frames are pooled.
    """
    return (
        average_masks([r.attacker_mask for r in run if r.attacker_mask is not None]),
        average_masks([r.victim_mask for r in run if r.victim_mask is not None]),
    )
