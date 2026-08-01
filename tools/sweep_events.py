"""Sweep the killfeed chaining parameters against box-score reconciliation.

Scores each setting the same way `reconcile_killfeed.py` does: total
absolute error, excess (kills/deaths beyond the box score, meaning an
invented event), and recovery (fraction of box-score kills accounted for).
Bitmaps are decoded once and re-chained per setting to keep the sweep cheap.

    uv run python tools/sweep_events.py --rows data/killfeed_rows_dense.jsonl
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cdlvision.events import anchors_of, build_events, widths_of  # noqa: E402
from cdlvision.names import NameMatcher  # noqa: E402
from cdlvision.progress import Progress  # noqa: E402
from reconcile_killfeed import boxscore_segments, map_of  # noqa: E402
from scan_killfeed import decode  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def score(events, maps, segments) -> dict:
    """Reconcile one setting's events against the box scores."""
    kills: dict[int, Counter] = {}
    deaths: dict[int, Counter] = {}
    for e in events:
        if not e.resolved:
            continue
        m = map_of(e.t, segments)
        if m is None:
            continue
        kills.setdefault(m, Counter())[e.attacker] += 1
        deaths.setdefault(m, Counter())[e.victim] += 1

    err = excess = total_k = total_bk = total_d = total_bd = 0
    for entry in maps:
        m = entry["map"]
        for name, box in entry["kd"].items():
            k = kills.get(m, Counter())[name]
            d = deaths.get(m, Counter())[name]
            err += abs(k - box["kills"]) + abs(d - box["deaths"])
            excess += max(0, k - box["kills"]) + max(0, d - box["deaths"])
            total_k += k
            total_d += d
            total_bk += box["kills"]
            total_bd += box["deaths"]

    return {
        "error": err,
        "excess": excess,
        "kills": total_k,
        "box_kills": total_bk,
        "deaths": total_d,
        "box_deaths": total_bd,
        "recovery": total_k / max(total_bk, 1),
        "events": len(events),
        "resolved": sum(1 for e in events if e.resolved),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="data/killfeed_rows_dense.jsonl")
    ap.add_argument("--cache", default="data/boxscore_kd.json")
    ap.add_argument("--timeline", default="data/state_timeline.json")
    ap.add_argument("--names", default="src/cdlvision/config/killfeed_names.json")
    ap.add_argument("--max-gap", type=float, nargs="+",
                    default=[0.25, 0.5, 1.0, 2.0, 6.0])
    ap.add_argument("--width-slack", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--min-frames", type=int, nargs="+", default=[1, 2, 3, 5])
    ap.add_argument("--features", nargs="+", choices=["widths", "anchors"],
                    default=["widths", "anchors"])
    args = ap.parse_args()

    feature_sets = {"widths": widths_of, "anchors": anchors_of}

    segments = boxscore_segments(ROOT / args.timeline)
    maps = json.loads((ROOT / args.cache).read_text())
    matcher = NameMatcher.load(ROOT / args.names)
    records = [
        json.loads(line)
        for line in (ROOT / args.rows).read_text().splitlines()
        if line.strip()
    ]
    # Decode every bitmap once rather than per setting; `medal_bmp` stays
    # encoded since it is only tested for truthiness.
    for r in records:
        r["attacker_bmp"] = decode(r["attacker_bmp"])
        r["victim_bmp"] = decode(r["victim_bmp"])
    identity = lambda payload: payload  # noqa: E731

    print(f"{len(records)} rows, {len(maps)} box scores\n")

    grid = list(itertools.product(
        args.features, args.max_gap, args.width_slack, args.min_frames
    ))
    print(f"{'feats':>7s} {'gap':>5s} {'slack':>5s} {'minf':>4s} "
          f"{'events':>7s} {'kills':>11s} {'deaths':>11s} "
          f"{'error':>6s} {'excess':>6s}")

    results = []
    with Progress("sweep_events", total=len(grid), label="event sweep") as bar:
        for feats, gap, slack, minf in grid:
            events = build_events(
                matcher, records, identity,
                max_gap=gap, width_slack=slack, min_frames=minf,
                features=feature_sets[feats],
            )
            s = score(events, maps, segments)
            s.update(features=feats, max_gap=gap, width_slack=slack, min_frames=minf)
            results.append(s)
            print(f"{feats:>7s} {gap:5.2f} {slack:5d} {minf:4d} "
                  f"{s['events']:7d} "
                  f"{s['kills']:5d}/{s['box_kills']:<5d} "
                  f"{s['deaths']:5d}/{s['box_deaths']:<5d} "
                  f"{s['error']:6d} {s['excess']:6d}", flush=True)
            best_so_far = min(results, key=lambda r: (r["error"], r["excess"]))
            bar.step(note=f"best error {best_so_far['error']} "
                          f"(gap={best_so_far['max_gap']} "
                          f"slack={best_so_far['width_slack']} "
                          f"minf={best_so_far['min_frames']})")

    best = min(results, key=lambda r: (r["error"], r["excess"]))
    print(f"\nlowest total error: features={best['features']} "
          f"gap={best['max_gap']} slack={best['width_slack']} "
          f"min_frames={best['min_frames']}")
    print(f"  error {best['error']}, excess {best['excess']}, "
          f"kills {best['kills']}/{best['box_kills']} "
          f"({best['recovery']:.1%}), deaths {best['deaths']}/{best['box_deaths']}")

    out = ROOT / "data" / "sweep_events.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
