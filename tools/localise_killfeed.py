"""Name the seconds where the killfeed lost a row, using the scoreboard.

Diffs two records of the same events read from different pixels:

- every increment in a scoreboard kill or death count, bracketed to the
  interval between the last frame showing the old value and the first
  showing the new one;
- every killfeed event, with its attacker and victim.

An increment with no matching killfeed event is a lost row, timestamped.
A killfeed event with no matching increment means the killfeed invented a
kill. Both sides are gated on the map windows from `cdlvision.maps`, since
recap footage between maps draws both a scoreboard and a killfeed and would
otherwise manufacture increments and events that pair with nothing real.

The gutter skull, reported alongside a death increment, confirms the death
was real but does not identify suicides: it appears for every death however
caused, so a death with no killfeed row is still either a dropped row or a
suicide.

The matching window default (1.5s) is set from the offset distribution of
matched pairs rather than assumed.

    uv run python tools/localise_killfeed.py
    uv run python tools/localise_killfeed.py --window 3.0
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cdlvision import maps, scoreboard_runs  # noqa: E402
from cdlvision.events import build_events  # noqa: E402
from cdlvision.names import NameMatcher  # noqa: E402
from reconcile_killfeed import choose_params, decode  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Increment:
    """One step in a player's kill or death count, bracketed in time."""

    player: str
    field: str
    before: int
    after: int
    earliest: float   # last frame still showing the old value
    latest: float     # first frame showing the new value
    skull: bool       # gutter showed a skull across the step

    @property
    def t(self) -> float:
        return self.latest


def increments(runs: list[dict]) -> list[Increment]:
    """Every increase in a kill or death count, in time order. Decreases
    (misreads or map boundaries) are handled by `validate_scoreboard.py`."""
    by_slot: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for run in runs:
        by_slot[(run["side"], run["slot"])].append(run)

    out: list[Increment] = []
    for slot_runs in by_slot.values():
        slot_runs.sort(key=lambda r: r["start"])
        for previous, current in zip(slot_runs, slot_runs[1:]):
            a, b = previous["value"], current["value"]
            if a["name"] is None or a["name"] != b["name"]:
                continue
            if current["start"] - previous["end"] > 1.0:
                continue  # a gap: the two runs are not consecutive observations
            for field in ("kills", "deaths"):
                if a[field] is None or b[field] is None:
                    continue
                if b[field] == a[field] + 1:
                    out.append(Increment(
                        player=a["name"], field=field,
                        before=a[field], after=b[field],
                        earliest=previous["end"], latest=current["start"],
                        skull=b.get("gutter") == "skull"
                                or a.get("gutter") == "skull",
                    ))
    return sorted(out, key=lambda i: i.t)


def match(incs: list[Increment], events, window: float):
    """Greedily pair increments with killfeed events, nearest offset first;
    each event used once."""
    candidates = []
    for i, inc in enumerate(incs):
        for j, event in enumerate(events):
            who = event.attacker if inc.field == "kills" else event.victim
            if who != inc.player:
                continue
            offset = event.t - inc.t
            if abs(offset) <= window:
                candidates.append((abs(offset), offset, i, j))
    candidates.sort()

    used_inc: set[int] = set()
    used_event: set[int] = set()
    pairs = []
    for _, offset, i, j in candidates:
        if i in used_inc or j in used_event:
            continue
        used_inc.add(i)
        used_event.add(j)
        pairs.append((i, j, offset))
    return pairs, used_inc, used_event


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="data/scoreboard_runs.jsonl")
    ap.add_argument("--header", default="data/header_runs.jsonl")
    ap.add_argument("--rows", default="data/killfeed_rows_dense.jsonl")
    ap.add_argument("--names", default="src/cdlvision/config/killfeed_names.json")
    ap.add_argument("--window", type=float, default=1.5)
    ap.add_argument("--out", default="data/killfeed_losses.json")
    ap.add_argument("--show", type=int, default=25)
    args = ap.parse_args()

    all_runs = scoreboard_runs.load(ROOT / args.runs)
    roster = scoreboard_runs.series_roster(all_runs)
    runs, alien = scoreboard_runs.split_foreign(all_runs, roster)
    header_runs = scoreboard_runs.load(ROOT / args.header)
    runs, replays = maps.filter_to_series(runs, header_runs)
    print(f"{len(all_runs)} scoreboard runs, {len(alien)} dropped as another "
          f"match's recap footage, {len(replays)} as a replay of an earlier "
          f"map of this one")
    windows = maps.derive(runs, header_runs)
    print(f"{len(windows)} maps derived, "
          f"{sum(w.end - w.start for w in windows):.0f}s of map time")
    incs = increments(runs)
    print(f"{len(runs)} runs kept -> {len(incs)} increments "
          f"({sum(i.field == 'kills' for i in incs)} kills, "
          f"{sum(i.field == 'deaths' for i in incs)} deaths)")

    matcher = NameMatcher.load(ROOT / args.names)
    records = [
        json.loads(line)
        for line in (ROOT / args.rows).read_text().splitlines()
        if line.strip()
    ]
    params = choose_params(None, records)
    events = [e for e in build_events(matcher, records, decode, **params.kwargs())
              if e.resolved]
    # Gate on the same map windows as the increments; recap footage between
    # maps draws its own killfeed and would look like invented events.
    inside = [e for e in events if any(w.contains(e.t) for w in windows)]
    print(f"{len(records)} killfeed rows -> {len(events)} events with both "
          f"names, {len(events) - len(inside)} of them between maps\n")
    events = inside

    kill_incs = [i for i in incs if i.field == "kills"]
    death_incs = [i for i in incs if i.field == "deaths"]

    report = {}
    for label, subset in (("kills", kill_incs), ("deaths", death_incs)):
        pairs, used_inc, used_event = match(subset, events, args.window)
        offsets = np.array([p[2] for p in pairs])
        print(f"=== {label} ===")
        print(f"  {len(pairs)}/{len(subset)} increments matched a killfeed event "
              f"({100 * len(pairs) / max(len(subset), 1):.1f}%)")
        if len(offsets):
            print(f"  event minus scoreboard, seconds: "
                  f"median {np.median(offsets):+.2f}, "
                  f"5th {np.percentile(offsets, 5):+.2f}, "
                  f"95th {np.percentile(offsets, 95):+.2f}")

        missing = [i for n, i in enumerate(subset) if n not in used_inc]
        print(f"  {len(missing)} increments with no killfeed row")
        if label == "deaths":
            confirmed = [i for i in missing if i.skull]
            print(f"      {len(confirmed)} of them are confirmed by a gutter "
                  f"skull, so the death is real and the killfeed is at fault "
                  f"(this does not separate suicides from dropped rows)")
        by_player = Counter(i.player for i in missing)
        print(f"  by player: {dict(by_player.most_common())}")
        for i in missing[: args.show]:
            print(f"      t={i.earliest:8.1f}-{i.latest:8.1f}  {i.player:9s} "
                  f"{i.field[:-1]} #{i.after}"
                  f"{'  [skull]' if i.skull else ''}")
        if len(missing) > args.show:
            print(f"      ... and {len(missing) - args.show} more")

        report[label] = {
            "increments": len(subset),
            "matched": len(pairs),
            "missing": [
                {"player": i.player, "field": i.field, "n": i.after,
                 "earliest": i.earliest, "latest": i.latest, "skull": i.skull}
                for i in missing
            ],
            "unmatched_events": [
                {"t": e.t, "attacker": e.attacker, "victim": e.victim}
                for n, e in enumerate(events) if n not in used_event
            ],
        }
        print(f"  {len(report[label]['unmatched_events'])} killfeed events "
              f"with no matching {label[:-1]} increment "
              f"-- these are the invented ones\n")

    (ROOT / args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
