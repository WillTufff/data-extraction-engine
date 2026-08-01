"""Measure the live scoreboard's geometry from the VOD rather than assuming it.

Run this before changing any constant in `scoreboard.py`. Samples LIVE frames
from the state timeline and reports the measured row grid, column extents,
panel mirroring, stat-column label widths, and spectated-row (inverted)
counts against what the module claims.

Usage:
    uv run python tools/probe_scoreboard.py <video> [--samples 400]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cdlvision import scoreboard, video  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# Bands generous enough that a row outside the claimed grid would be found
# rather than cropped away.
PROBE_Y = (60, 240)
LABEL_Y = (68, 100)


def _row_bands(frame: np.ndarray, side: scoreboard.Side) -> list[tuple[int, int]]:
    """Text row bands inside a panel, found by ink rather than by the grid."""
    x0 = scoreboard.COLUMNS["number"][0] + side.offset
    x1 = scoreboard.COLUMNS["name"][1] + side.offset
    y0, y1 = PROBE_Y
    mask = scoreboard.ink(frame[y0:y1, x0 : x1 + 1])
    out = []
    for a, b in scoreboard._spans(mask.sum(axis=1) > 3):
        out.append((y0 + a, y0 + b))
    return out


def _inverted(frame: np.ndarray, side: scoreboard.Side, row: int) -> bool:
    """Is this row drawn as dark text on a bright bar?"""
    cell = side.cell(frame, row, "name")
    return bool(scoreboard.ink(cell).mean() > 0.8)


def _column_spans(frame: np.ndarray, side: scoreboard.Side, row: int, key: str):
    mask = scoreboard.ink(side.cell(frame, row, key))
    x0 = side.column(key)[0]
    return [(x0 + a, x0 + b) for a, b in scoreboard.token_spans(mask)]


_SOURCE: str | None = None
_INFO: video.VideoInfo | None = None


def _init(source: str) -> None:
    global _SOURCE, _INFO
    _SOURCE, _INFO = source, video.probe(source)


def _one(ts: float):
    try:
        frame = video.frame_at(_SOURCE, ts, _INFO)
    except Exception:
        return None

    try:
        scoreboard.check_mirror(frame)
        mirror = "ok"
    except scoreboard.LayoutError as exc:
        mirror = str(exc)

    out = {
        "ts": ts,
        "mirror": mirror,
        "bands": {s.name: _row_bands(frame, s) for s in scoreboard.SIDES},
        "inverted": [
            (s.name, r)
            for s in scoreboard.SIDES
            for r in range(scoreboard.ROW_COUNT)
            if _inverted(frame, s, r)
        ],
        "labels": {
            s.name: {
                key: _label_span(frame, s, key)
                for key in ("kd", "strk", "stat")
            }
            for s in scoreboard.SIDES
        },
        "cols": {
            key: [
                span
                for s in scoreboard.SIDES
                for r in range(scoreboard.ROW_COUNT)
                if not _inverted(frame, s, r)
                for span in _column_spans(frame, s, r, key)
            ]
            for key in ("number", "name", "kd", "strk", "stat")
        },
    }
    return out


def _label_span(frame, side, key):
    spans = scoreboard.token_spans(scoreboard.ink(side.label_cell(frame, key)))
    if not spans:
        return None
    x0 = side.column(key)[0]
    return (x0 + spans[0][0], x0 + spans[-1][1])


def live_timestamps(timeline: Path) -> list[float]:
    stamps: list[float] = []
    for s in json.loads(timeline.read_text()):
        if s["state"] != "live":
            continue
        t = s["start"] + 1
        while t < s["end"] - 1:
            stamps.append(t)
            t += 1
    return stamps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--timeline", default="data/state_timeline.json")
    ap.add_argument("--samples", type=int, default=400)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    stamps = live_timestamps(ROOT / args.timeline)
    print(f"{len(stamps)} LIVE seconds available")
    random.Random(args.seed).shuffle(stamps)
    stamps = sorted(stamps[: args.samples])

    results = []
    with Pool(args.workers, initializer=_init, initargs=(args.source,)) as pool:
        for r in pool.imap(_one, stamps, chunksize=4):
            if r is not None:
                results.append(r)
    print(f"{len(results)} frames read\n")

    # -- the mirror invariant -------------------------------------------
    mirrors = Counter(r["mirror"] for r in results)
    ok = mirrors.get("ok", 0)
    print(f"mirror check: {ok}/{len(results)} pass ({100 * ok / len(results):.1f}%)")
    for msg, n in mirrors.most_common():
        if msg != "ok":
            print(f"  {n:4d}  {msg}")

    offsets = Counter()
    for r in results:
        for key in ("kd", "strk", "stat"):
            left, right = r["labels"]["left"][key], r["labels"]["right"][key]
            if left and right:
                offsets[right[0] - left[0]] += 1
    print(f"\nleft->right label offset: {sorted(offsets.items())}")

    # -- the row grid ----------------------------------------------------
    tops = Counter()
    heights = Counter()
    for r in results:
        for bands in r["bands"].values():
            for a, b in bands:
                tops[a] += 1
                heights[b - a + 1] += 1
    print(f"\nrow band tops (grid claims {list(scoreboard.ROWS)}, label at "
          f"{scoreboard.LABEL_TOP})")
    for y, n in sorted(tops.items()):
        on = "grid" if y in scoreboard.ROWS else (
            "label" if y == scoreboard.LABEL_TOP else "OFF-GRID")
        print(f"  y={y:4d}  {n:5d}  {on}")
    print(f"row band heights: {sorted(heights.items())}")

    # -- columns ---------------------------------------------------------
    print("\ncolumn extents over every non-inverted row "
          "(window from scoreboard.COLUMNS, left panel coordinates)")
    for key in ("number", "name", "kd", "strk", "stat"):
        seen = [
            (a - (scoreboard.MIRROR if a > 1000 else 0),
             b - (scoreboard.MIRROR if b > 1000 else 0))
            for r in results for a, b in r["cols"][key]
        ]
        if not seen:
            continue
        lo = min(a for a, _ in seen)
        hi = max(b for _, b in seen)
        widths = Counter(b - a + 1 for a, b in seen)
        w0, w1 = scoreboard.COLUMNS[key]
        slack = f"{lo - w0}px left, {w1 - hi}px right"
        print(f"  {key:7s} x {lo:4d}..{hi:4d}  window {w0}..{w1} ({slack})  "
              f"widths {min(widths)}..{max(widths)}  n={len(seen)}")

    # -- the stat schema -------------------------------------------------
    print("\nstat-column label width (TIME/PLNT/OVLD differ)")
    stat = Counter()
    for r in results:
        span = r["labels"]["left"]["stat"]
        if span:
            stat[span[1] - span[0] + 1] += 1
    for w, n in sorted(stat.items()):
        print(f"  width {w:3d}  {n:5d} frames")

    # -- the inverted row ------------------------------------------------
    inv = Counter(len(r["inverted"]) for r in results)
    print(f"\ninverted (spectated) rows per frame: {sorted(inv.items())}")
    where = Counter(tuple(x) for r in results for x in r["inverted"])
    print("  by slot: " + ", ".join(
        f"{s}[{i}]={n}" for (s, i), n in sorted(where.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
