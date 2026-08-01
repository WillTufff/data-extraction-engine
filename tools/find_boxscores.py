"""Locate post-map box-score screens in a VOD.

Grid detection doubles as a state detector: a frame either presents a
well-formed 8-row table or it does not. Column count identifies the mode.
Scans by parallel seeking rather than a linear decode pass, since box scores
are sparse.

Usage:
    uv run python tools/find_boxscores.py <video> [--step 10] [--workers 8]
"""

from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cdlvision import video  # noqa: E402
from cdlvision.boxscore import GridDetectionError, detect_grid  # noqa: E402

_SOURCE: str | None = None
_INFO: video.VideoInfo | None = None


def _init(source: str) -> None:
    global _SOURCE, _INFO
    _SOURCE = source
    _INFO = video.probe(source)


def _probe_one(ts: float) -> tuple[float, int, int] | None:
    try:
        frame = video.frame_at(_SOURCE, ts, _INFO)
    except Exception:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    try:
        grid = detect_grid(gray)
    except GridDetectionError:
        return None
    return ts, grid.n_columns, grid.row_height


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--step", type=float, default=10.0, help="probe spacing in seconds")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=None)
    args = ap.parse_args()

    info = video.probe(args.source)
    end = args.end if args.end is not None else info.duration
    stamps = [args.start + i * args.step
              for i in range(int((end - args.start) / args.step))]

    print(f"{info.width}x{info.height} @ {info.fps:.2f}fps, {info.duration:.0f}s")
    print(f"probing {len(stamps)} timestamps every {args.step}s "
          f"across {args.workers} workers", flush=True)

    hits = []
    with Pool(args.workers, initializer=_init, initargs=(args.source,)) as pool:
        for i, result in enumerate(pool.imap(_probe_one, stamps, chunksize=4)):
            if i % 100 == 0:
                print(f"  ... {i}/{len(stamps)}", flush=True)
            if result is not None:
                hits.append(result)
                ts, ncols, rh = result
                print(f"  HIT t={ts:8.1f}s  8 rows x {ncols} cols "
                      f"(row height {rh}px)", flush=True)

    print(f"\n{len(hits)} box-score frames found")
    by_cols: dict[int, list[float]] = {}
    for ts, ncols, _ in hits:
        by_cols.setdefault(ncols, []).append(ts)
    for ncols in sorted(by_cols):
        times = by_cols[ncols]
        groups, run = [], [times[0]]
        for a, b in zip(times, times[1:]):
            if b - a <= args.step * 1.5:
                run.append(b)
            else:
                groups.append(run); run = [b]
        groups.append(run)
        print(f"  {ncols} columns: {len(groups)} appearances -> "
              f"{[f'{g[0]:.0f}-{g[-1]:.0f}s' for g in groups]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
