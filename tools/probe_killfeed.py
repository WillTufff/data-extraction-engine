"""Measure the killfeed's geometry from the VOD rather than assuming it.

Samples LIVE frames from the state timeline and reports row positions, row
count per frame, row height and pitch, slide-in animation frequency, and row
text width. Run before changing any constant in `killfeed.py`.

Usage:
    uv run python tools/probe_killfeed.py <video> [--samples 200]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cdlvision import video  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# Generous search band, wider than the settled grid, to catch extra rows.
BAND = (400, 620)
XRANGE = (20, 640)

ROW_LEFT = 60          # every row's text starts here
LEFT_SLACK = 4
ROW_HEIGHT = (11, 17)
GRID_BOTTOM = 569      # top edge of the newest row, when settled
GRID_PITCH = 30
GRID_SLACK = 1         # anti-aliasing can shift a measured top by a pixel


def on_grid(top: int) -> bool:
    if top > GRID_BOTTOM + GRID_SLACK:
        return False
    offset = (GRID_BOTTOM - top) % GRID_PITCH
    return min(offset, GRID_PITCH - offset) <= GRID_SLACK


def ink(bgr: np.ndarray) -> np.ndarray:
    """Flat white or flat magenta killfeed text. Prefilter only, pair with the
    left-edge constraint since it also matches gameplay content."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    white = (v > 165) & (s < 70)
    magenta = (h >= 140) & (h <= 175) & (s > 90) & (v > 110)
    return white | magenta


def rows_in(bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Candidate killfeed rows as (top, bottom, x_first, x_last)."""
    y0, y1 = BAND
    x0, x1 = XRANGE
    mask = ink(bgr[y0:y1, x0:x1])
    proj = mask.sum(axis=1)

    bands: list[tuple[int, int]] = []
    run: tuple[int, int] | None = None
    for i, count in enumerate(proj):
        if count >= 10:
            run = (run[0], i) if run else (i, i)
        elif run:
            bands.append(run)
            run = None
    if run:
        bands.append(run)

    lo, hi = ROW_HEIGHT
    out = []
    for a, b in bands:
        if not lo <= b - a + 1 <= hi:
            continue
        cols = np.flatnonzero(mask[a : b + 1].sum(axis=0) > 0)
        if not len(cols):
            continue
        left = x0 + int(cols[0])
        if abs(left - ROW_LEFT) > LEFT_SLACK:
            continue  # gameplay content, not a row
        out.append((y0 + a, y0 + b, left, x0 + int(cols[-1])))
    return out


_SOURCE: str | None = None
_INFO: video.VideoInfo | None = None


def _init(source: str) -> None:
    global _SOURCE, _INFO
    _SOURCE = source
    _INFO = video.probe(source)


def _one(ts: float):
    try:
        bgr = video.frame_at(_SOURCE, ts, _INFO)
    except Exception:
        return None
    return ts, rows_in(bgr)


def live_timestamps(timeline: Path) -> list[float]:
    segments = json.loads(timeline.read_text())
    stamps: list[float] = []
    for s in segments:
        if s["state"] != "live":
            continue
        # Skip the first and last second: those straddle a transition.
        t = s["start"] + 1
        while t < s["end"] - 1:
            stamps.append(t)
            t += 1
    return stamps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--timeline", default="data/state_timeline.json")
    ap.add_argument("--samples", type=int, default=200)
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

    counts = Counter(len(rows) for _, rows in results)
    print(f"\nrows per sampled LIVE frame ({len(results)} frames)")
    for n in sorted(counts):
        print(f"  {n} rows  {counts[n]:4d}  {100 * counts[n] / len(results):5.1f}%")

    tops = Counter(top for _, rows in results for top, _, _, _ in rows)
    print("\nrow top positions (settled grid at 569/539/509/479/...)")
    for y in sorted(tops):
        print(f"  y={y:4d}  {tops[y]:4d}  {'on-grid' if on_grid(y) else 'OFF-GRID'}")

    heights = Counter(bottom - top + 1 for _, rows in results for top, bottom, _, _ in rows)
    print(f"\nrow heights: {sorted(heights.items())}")

    pitches: Counter[int] = Counter()
    off_grid_frames = 0
    for _, rows in results:
        ys = sorted(top for top, _, _, _ in rows)
        pitches.update(b - a for a, b in zip(ys, ys[1:]))
        if any(not on_grid(y) for y in ys):
            off_grid_frames += 1
    print(f"pitches: {sorted(pitches.items())}")
    print(f"frames with any off-grid row (mid-slide): {off_grid_frames} "
          f"({100 * off_grid_frames / len(results):.1f}%)")

    widths = [last - first for _, rows in results for _, _, first, last in rows]
    if widths:
        widths.sort()
        print(f"\nrow text width: min {widths[0]}, median "
              f"{widths[len(widths) // 2]}, max {widths[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
