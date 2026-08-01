"""Check the spectated panel against the scoreboard, with no ground truth.

The panel and the live scoreboard show the same player's numbers at the same
instant, drawn 900px apart and read by different code off different strips,
so agreement between them is evidence.

Four checks: the name (scoreboard marks the inverted row, panel draws that
player's name); K/D, streak and the sixth stat; the team from the bar's
colour against the roster; and HP, which has no second witness anywhere and
is only reported as a distribution.

The offset between the two strips is measured by sweeping it and reporting
the agreement curve, rather than assumed.

    uv run python tools/validate_spectated.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cdlvision import maps, scoreboard_runs  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# Fields the two regions both carry, panel name -> scoreboard row attribute.
SHARED = {"name": "name", "kills": "kills", "deaths": "deaths",
          "streak": "streak", "stat": "stat"}

# Two strip frames; disagreements this close to a player switch are the two
# regions redrawing a frame apart, not misreads.
TRANSITION_WINDOW = 2 / 6


def load_runs(path: Path) -> dict[str, list[dict]]:
    by_field: dict[str, list[dict]] = defaultdict(list)
    for line in path.read_text().splitlines():
        if line.strip():
            run = json.loads(line)
            by_field[run["field"]].append(run)
    for runs in by_field.values():
        runs.sort(key=lambda r: r["start"])
    return by_field


def value_at(runs: list[dict], t: float):
    """The value of a run-length field at a time, or None if no run covers it."""
    lo, hi = 0, len(runs) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        run = runs[mid]
        if t < run["start"]:
            hi = mid - 1
        elif t > run["end"]:
            lo = mid + 1
        else:
            return run["value"]
    return None


def spectated_timeline(runs: list[dict]) -> list[dict]:
    """Every scoreboard run whose row is the inverted one."""
    out = [r for r in runs if r["value"].get("spectated")]
    out.sort(key=lambda r: r["start"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="data/spectated_runs.jsonl")
    ap.add_argument("--scoreboard", default="data/scoreboard_runs.jsonl")
    ap.add_argument("--header", default="data/header_runs.jsonl")
    ap.add_argument("--step", type=float, default=1 / 6)
    args = ap.parse_args()

    panel = load_runs(ROOT / args.panel)
    board = scoreboard_runs.load(ROOT / args.scoreboard)
    roster = scoreboard_runs.series_roster(board)
    ours, _ = scoreboard_runs.split_foreign(board, roster)
    header = scoreboard_runs.load(ROOT / args.header)
    ours, _ = maps.filter_to_series(ours, header)
    windows = maps.derive(ours, header)
    rows = spectated_timeline(ours)
    print(f"{sum(len(v) for v in panel.values())} panel runs, "
          f"{len(rows)} scoreboard runs on an inverted row, "
          f"{len(windows)} maps\n")

    # -- the offset between the two strips, measured ----------------------
    print("=== agreement on the name, by strip offset ===")
    best = None
    for shift in (-4, -3, -2, -1, 0, 1, 2, 3, 4):
        offset = shift * args.step
        agree, total = _score(panel, rows, windows, args.step, offset, ("name",))
        print(f"  {offset:+.3f}s: {agree}/{total} "
              f"({100 * agree / max(total, 1):.1f}%)")
        if best is None or agree > best[0]:
            best = (agree, offset)
    offset = best[1]
    print(f"\n  best at {offset:+.3f}s, which is what the rest uses\n")

    # -- per field ---------------------------------------------------------
    print("=== panel vs scoreboard, per field ===")
    disagreements: dict[str, Counter] = defaultdict(Counter)
    totals: Counter = Counter()
    agreed: Counter = Counter()
    for t, row in _pairs(panel, rows, windows, args.step, offset):
        for field, attribute in SHARED.items():
            want = row["value"].get(attribute)
            got = value_at(panel[field], t)
            if want is None or got is None:
                continue
            totals[field] += 1
            if str(want) == str(got):
                agreed[field] += 1
            else:
                disagreements[field][(str(want), str(got))] += 1
    for field in SHARED:
        n = totals[field]
        print(f"  {field:7} {agreed[field]}/{n} agree "
              f"({100 * agreed[field] / max(n, 1):.2f}%)")
        for (want, got), count in disagreements[field].most_common(3):
            print(f"      scoreboard {want!r}, panel {got!r}: {count}")

    # -- are the residuals transitions rather than misreads? ---------------
    print("\n=== the residual, against the moments the spectated player "
          "changes ===")
    changes = [run["start"] for run in panel["name"]]
    near = far = 0
    for t, row in _pairs(panel, rows, windows, args.step, offset):
        want, got = row["value"].get("name"), value_at(panel["name"], t)
        if want is None or got is None or str(want) == str(got):
            continue
        if any(abs(t - c) <= TRANSITION_WINDOW for c in changes):
            near += 1
        else:
            far += 1
    print(f"  {near} of {near + far} disagreements land within "
          f"{TRANSITION_WINDOW:.2f}s of the panel switching player")
    print("  A disagreement there is the two regions updating a frame apart, "
          "not a misread; one away from a switch would be the real thing.")

    print("\n=== how many rows the scoreboard calls spectated ===")
    counts = _spectated_counts(ours, windows, args.step)
    total = sum(counts.values())
    for n in sorted(counts):
        print(f"  {n} row(s): {counts[n]} frames "
              f"({100 * counts[n] / max(total, 1):.2f}%)")
    print("  Zero is a free camera, which the panel also declines to draw. "
          "More than one is the scoreboard's own polarity test firing on a "
          "row that is merely lighter, and the panel is the tie-breaker: it "
          "draws exactly one player and always the highlighted one.")

    # -- the bar's colour against the roster -------------------------------
    print("\n=== team from the bar's colour vs the roster ===")
    sides = {name: side for side, name in
             ((r["side"], r["value"]["name"]) for r in rows) if name}
    ok = bad = 0
    for t, row in _pairs(panel, rows, windows, args.step, offset):
        team = value_at(panel["team"], t)
        name = value_at(panel["name"], t)
        if team is None or name is None or name not in sides:
            continue
        if team == sides[name]:
            ok += 1
        else:
            bad += 1
    print(f"  {ok}/{ok + bad} frames put the panel's colour and the panel's "
          f"name on the same side ({100 * ok / max(ok + bad, 1):.2f}%)")

    # -- hp, which nothing else witnesses ----------------------------------
    print("\n=== HP, reported rather than checked ===")
    hp = Counter()
    for run in panel.get("hp", []):
        if run["value"] is not None:
            hp[run["value"]] += run["samples"]
    total = sum(hp.values())
    print(f"  {total} samples, {len(hp)} distinct values, "
          f"range {min(hp)}..{max(hp)}")
    print(f"  100: {100 * hp[100] / max(total, 1):.1f}%, "
          f"0: {100 * hp[0] / max(total, 1):.1f}%, "
          f"other: {100 * (total - hp[100] - hp[0]) / max(total, 1):.1f}%")
    out_of_range = sum(n for v, n in hp.items() if not 0 <= v <= 100)
    print(f"  outside 0..100: {out_of_range}")

    # -- the weapon, reported ----------------------------------------------
    print("\n=== weapon name ===")
    weapons = Counter()
    for run in panel.get("weapon", []):
        weapons[run["value"]] += run["samples"]
    total = sum(weapons.values())
    for name, count in weapons.most_common():
        print(f"  {str(name):12} {count:6d} ({100 * count / max(total, 1):5.1f}%)")
    return 0


def _spectated_counts(runs, windows, step) -> Counter:
    """How many rows the scoreboard marks inverted, per sampled instant."""
    by_slot = scoreboard_runs.by_slot(runs)
    counts: Counter = Counter()
    for window in windows:
        t = window.start
        while t <= window.end:
            drawn = spectated = 0
            for slot_runs in by_slot.values():
                value = value_at(slot_runs, t)
                if value is not None:
                    drawn += 1
                    spectated += bool(value.get("spectated"))
            if drawn:
                counts[spectated] += 1
            t += step
    return counts


def _pairs(panel, rows, windows, step, offset):
    """(panel time, scoreboard run) for each sampled instant inside a map."""
    for row in rows:
        t = row["start"]
        while t <= row["end"]:
            shifted = t + offset
            if any(w.contains(t) for w in windows):
                yield shifted, row
            t += step


def _score(panel, rows, windows, step, offset, fields):
    agree = total = 0
    for t, row in _pairs(panel, rows, windows, step, offset):
        for field in fields:
            want = row["value"].get(SHARED[field])
            got = value_at(panel[field], t)
            if want is None or got is None:
                continue
            total += 1
            agree += str(want) == str(got)
    return agree, total


if __name__ == "__main__":
    raise SystemExit(main())
