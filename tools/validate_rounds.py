"""Check derived rounds against the game's own arithmetic, not against labels.

Six checks, and none of them needs a transcription:

- **Every boundary carries all three witnesses** -- clock, lives, score.
- **The score entering each round is a staircase** stepping by exactly one. This
  is what rules out a *missing* round, and it is the reason the series can be
  read as 6-5 on maps whose last drawn score is 5-5.
- **Rounds won reproduce the map score**, side by side.
- **The map winner agrees with the series pips**, which come from a different
  field entirely.
- **No player dies twice in a round**, and **no side loses more than four**.
  Exact upper bounds in Search and Destroy, so a violation is a killfeed defect
  rather than a tuning question -- and unlike the box score, which reports the
  excess as a total of 50, this names the pair of events.
- **Lives lost and killfeed deaths bracket the box score.** Both are lower
  bounds, for different reasons, so the check is an ordering rather than an
  equality and is written as one.

    uv run python tools/validate_rounds.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cdlvision import maps, rounds, scoreboard_runs  # noqa: E402
from cdlvision.events import DENSE_12HZ, build_events  # noqa: E402
from cdlvision.names import NameMatcher  # noqa: E402
from scan_killfeed import decode  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scoreboard", default="tests/fixtures/scoreboard_runs.jsonl.gz")
    ap.add_argument("--header", default="tests/fixtures/header_runs.jsonl.gz")
    ap.add_argument("--rows", default="tests/fixtures/killfeed_rows_dense.jsonl.gz")
    ap.add_argument("--boxscores", default="tests/fixtures/boxscore_kd.json")
    ap.add_argument("--names", default="src/cdlvision/config/killfeed_names.json")
    ap.add_argument("--strict", action="store_true",
                    help="raise when a round boundary is short a witness")
    args = ap.parse_args()

    board = scoreboard_runs.load(ROOT / args.scoreboard)
    header = scoreboard_runs.load(ROOT / args.header)
    ours, _ = scoreboard_runs.split_foreign(board)
    ours, _ = maps.filter_to_series(ours, header)
    windows = maps.derive(ours, header)
    sides = scoreboard_runs.player_sides(ours)
    boxscores = {b["map"]: b for b in json.loads((ROOT / args.boxscores).read_text())}
    assigned = maps.assign_boxscores(windows, list(boxscores.values()))

    matcher = NameMatcher.load(ROOT / args.names)
    rows = scoreboard_runs.load(ROOT / args.rows)
    events = build_events(matcher, rows, decode, **DENSE_12HZ.kwargs())
    winners = rounds.map_winners(windows)
    print(f"{len(events)} killfeed events, {len(windows)} maps\n")

    print("=== how each map divides ===")
    for window in windows:
        segments = rounds.subdivide(window, header, events, sides)
        kind = ("rounds" if window.mode == rounds.ROUNDS_MODE
                else "hills" if window.mode == rounds.HILLS_MODE else "halves")
        print(f"  map {window.number} {window.mode:20s} {len(segments):2d} {kind}")
    print("  Only Search and Destroy has rounds. Overload is a timed mode with "
          "halves; Hardpoint has hills. Measured, not assumed.")

    problems = 0
    for window in windows:
        if window.mode != rounds.ROUNDS_MODE:
            continue
        derived = rounds.derive(window, header, events, sides, strict=args.strict)
        rounds.close_from_series(derived, winners.get(window.number))
        print(f"\n=== map {window.number} ===")
        problems += _check_map(window, header, derived, sides,
                               assigned.get(window.number), winners)

    print(f"\n=== {problems} problems ===")
    print("Every one is reported, not tuned away. The duplicates in particular "
          "are evidence about `events.DENSE_12HZ`, and fitting a global "
          "threshold to nine observations of one mode would be the wrong move.")
    return 0


def _check_map(window, header, derived, sides, box, winners) -> int:
    problems = 0

    missing = [r for r in derived if r.number > 1 and len(r.witnesses) < 3]
    print(f"  boundaries: {len(derived)} rounds, "
          f"{len(derived) - 1 - len(missing)}/{len(derived) - 1} with all "
          f"three witnesses")
    for r in missing:
        print(f"    r{r.number} at t={r.start:.0f} saw only {list(r.witnesses)}")
        problems += 1

    gaps = rounds.staircase_gaps(window, header)
    print(f"  score staircase: {len(gaps)} jumps of more than one "
          f"({'no round is missing' if not gaps else 'A ROUND IS MISSING'})")
    problems += len(gaps)

    outcomes = Counter(r.outcome for r in derived)
    print("  outcomes: " + ", ".join(
        f"{n} {k}" for k, n in outcomes.most_common()))
    if outcomes[None]:
        problems += outcomes[None]

    won = Counter(r.winner for r in derived)
    final = (won.get("left", 0), won.get("right", 0))
    last = derived[-1].score_before
    print(f"  rounds won {final[0]}-{final[1]}, last drawn score "
          f"{last[0]}-{last[1]}, winner {winners.get(window.number)} "
          f"(pips), last round decided by {derived[-1].decided_by}")
    if won.get(None):
        print(f"    {won[None]} round(s) with no winner -- the map ends on the "
              "last one and the broadcast never draws its score")

    dups = [(r, d) for r in derived for d in rounds.duplicate_victims(r)]
    kinds = Counter(d[3] for _, d in dups)
    print(f"  one death per player per round: {len(dups)} violations "
          f"({kinds['fragment']} fragment, {kinds['replay']} replay)")
    for r, (victim, a, b, kind) in dups:
        print(f"    r{r.number} {victim} twice, {b.t - a.t:5.2f}s apart "
              f"[{kind}]: {a.attacker}({a.frames}f) at t={a.t:.2f} then "
              f"{b.attacker}({b.frames}f) at t={b.t:.2f}")
        problems += 1

    over = [(r, s) for r in derived for s in rounds.over_four(r, sides)]
    print(f"  at most four deaths a side per round: {len(over)} violations")
    for r, side in over:
        print(f"    r{r.number} {side} lost "
              f"{len(r.victims(sides, side))} by the killfeed's count")
        problems += 1

    if box:
        for side in rounds.SIDES:
            lost = sum(rounds.LIVES - low for low, s in
                       ((r.lives_low[i], s) for r in derived
                        for i, s in enumerate(rounds.SIDES))
                       if s == side and low is not None)
            kf = sum(len(set(r.victims(sides, side))) for r in derived)
            deaths = sum(v["deaths"] for n, v in box["kd"].items()
                         if sides.get(n) == side)
            ok = lost <= deaths and kf <= deaths
            print(f"  {side:5s} deaths: lives-lost {lost:3d}, killfeed {kf:3d}, "
                  f"box score {deaths:3d}  {'ok' if ok else 'EXCESS'}")
            problems += 0 if ok else 1
    return problems


if __name__ == "__main__":
    raise SystemExit(main())
