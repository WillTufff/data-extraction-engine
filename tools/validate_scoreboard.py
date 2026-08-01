"""Check the scanned scoreboard timeline against the box scores and itself.

Map windows come from `cdlvision.maps`, derived from the scoreboard's own
resets and the header's pips and score, rather than from gaps between box
scores (there are seven maps but only six box scores).

Three checks: the scoreboard's last reading before each box score must equal
it exactly; within a map, a player's kills and deaths only ever go up; and
one team's kills equal the other team's deaths minus suicides (also checked
in `reconcile_killfeed.py`, off different pixels).

    uv run python tools/validate_scoreboard.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cdlvision import maps, scoreboard_runs  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="data/scoreboard_runs.jsonl")
    ap.add_argument("--header", default="data/header_runs.jsonl")
    ap.add_argument("--boxscores", default="data/boxscore_kd.json")
    ap.add_argument("--strict", action="store_true",
                    help="fail if the three boundary mechanisms disagree")
    args = ap.parse_args()

    all_runs = scoreboard_runs.load(ROOT / args.runs)
    roster = scoreboard_runs.series_roster(all_runs)
    runs, alien = scoreboard_runs.split_foreign(all_runs, roster)
    header_runs = scoreboard_runs.load(ROOT / args.header)
    runs, replays = maps.filter_to_series(runs, header_runs)
    runs.sort(key=lambda r: (r["side"], r["slot"], r["start"]))
    boxscores = json.loads((ROOT / args.boxscores).read_text())

    print(f"series roster: {dict(sorted(roster.items()))}")
    print(f"{len(all_runs)} runs, {len(alien)} from another match (the roster "
          f"contradicts itself), {len(replays)} a replay of an earlier map "
          f"(the header's pips or score are behind), {len(runs)} kept\n")

    # -- 0. where the maps are --------------------------------------------
    print("=== map boundaries, derived ===")
    for boundary in maps.boundaries(runs, header_runs):
        mark = "ok" if boundary.confirmed else "UNCONFIRMED"
        print(f"  t={boundary.t:7.0f}  {mark:12} {', '.join(sorted(boundary.witnesses))}")
    windows = maps.derive(runs, header_runs, strict=args.strict)
    assigned = maps.assign_boxscores(windows, boxscores)
    print()
    for w in windows:
        box = assigned.get(w.number)
        anchor = "no box score" if box is None else f"box score at t={box['t']:.0f}"
        print(f"  map {w.number}: {w.start:7.1f}-{w.end:7.1f} "
              f"({w.end - w.start:4.0f}s) {w.mode!s:19} pips {w.pips}, {anchor}")
    print(f"\n  {len(windows)} maps from {len(boxscores)} box scores\n")

    # -- 1. against the box scores ---------------------------------------
    print("=== final scoreboard reading vs box score ===")
    agree = total = 0
    for w in windows:
        box = assigned.get(w.number)
        if box is None:
            print(f"  map {w.number} ({w.mode}): no box score to check against")
            continue
        if box["mode"] != w.mode:
            print(f"  map {w.number}: mode from the stat column is {w.mode}, "
                  f"box score says {box['mode']}")
        final = _final_by_name(runs, w.start, w.end)
        mismatches = []
        for name, expected in box["kd"].items():
            total += 2
            got = final.get(name)
            if got is None:
                mismatches.append(f"{name}: never read")
                continue
            for field in ("kills", "deaths"):
                if got[field] == expected[field]:
                    agree += 1
                else:
                    mismatches.append(
                        f"{name} {field}: scoreboard {got[field]}, "
                        f"box score {expected[field]}"
                    )
        status = "ok" if not mismatches else f"{len(mismatches)} disagree"
        print(f"  map {w.number} ({w.mode}, {w.start:.0f}-{w.end:.0f}s): {status}")
        for m in mismatches:
            print(f"      {m}")
    print(f"\n  {agree}/{total} totals agree ({100 * agree / total:.1f}%)\n")

    # -- 2. monotonicity --------------------------------------------------
    print("=== monotonicity within each map ===")
    drops = 0
    for number, mapno, field, a, b, ts in _drops(runs, windows):
        drops += 1
        if drops <= 20:
            print(f"  map {mapno} player {number} {field}: {a} -> {b} at t={ts:.1f}")
    print(f"  {drops} decreasing transitions\n")

    # -- 3. the team-sum invariant ---------------------------------------
    print("=== team kills vs opposing deaths (per map) ===")
    for w in windows:
        final = _final_by_number(runs, w.start, w.end)
        left = [final[n] for n in (1, 2, 3, 4) if n in final]
        right = [final[n] for n in (5, 6, 7, 8) if n in final]
        if len(left) != 4 or len(right) != 4:
            print(f"  map {w.number}: only {len(left)}+{len(right)} slots read")
            continue
        lk, ld = sum(r["kills"] for r in left), sum(r["deaths"] for r in left)
        rk, rd = sum(r["kills"] for r in right), sum(r["deaths"] for r in right)
        print(f"  map {w.number}: left {lk} kills vs right {rd} deaths "
              f"(diff {lk - rd:+d}); right {rk} kills vs left {ld} deaths "
              f"(diff {rk - ld:+d})")
    print("\n  A negative difference is suicides or friendly fire; a positive "
          "one is a misread, since a kill cannot exist without a death.")
    return 0


def _runs_in(runs, start, end):
    return [r for r in runs if start <= r["start"] <= end]


def _final_by_name(runs, start, end) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for run in sorted(_runs_in(runs, start, end), key=lambda r: r["start"]):
        v = run["value"]
        if v["name"] and v["kills"] is not None:
            out[v["name"]] = v
    return out


def _final_by_number(runs, start, end) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for run in sorted(_runs_in(runs, start, end), key=lambda r: r["start"]):
        v = run["value"]
        if v["number"] and v["kills"] is not None:
            out[v["number"]] = v
    return out


def _drops(runs, windows):
    by_slot: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for run in runs:
        by_slot[(run["side"], run["slot"])].append(run)
    for slot_runs in by_slot.values():
        slot_runs.sort(key=lambda r: r["start"])
        for w in windows:
            mapno = w.number
            window = [
                r for r in slot_runs
                if w.contains(r["start"]) and r["value"]["kills"] is not None
            ]
            for previous, current in zip(window, window[1:]):
                if previous["value"]["number"] != current["value"]["number"]:
                    continue
                for field in ("kills", "deaths"):
                    if current["value"][field] < previous["value"][field]:
                        yield (current["value"]["number"], mapno, field,
                               previous["value"][field], current["value"][field],
                               current["start"])


if __name__ == "__main__":
    raise SystemExit(main())
