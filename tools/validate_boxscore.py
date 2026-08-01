"""Parse box scores at given timestamps and report per-glyph confidence.

Held-out validation: the atlas is built from one frame, so results here are
on frames (and modes) it was not built from.

Usage:
    uv run python tools/validate_boxscore.py <video> 4920 7380 ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cdlvision import video  # noqa: E402
from cdlvision.boxscore import (  # noqa: E402
    MODE_SCHEMAS, NON_NUMERIC, GridDetectionError, detect_grid, detect_mode, parse,
)
from cdlvision.glyphs import GlyphAtlas, TemplateSet  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("timestamps", nargs="+", type=float)
    ap.add_argument("--mode", default=None,
                    help="override mode detection (default: read from header)")
    ap.add_argument("--atlas", default=ROOT / "src/cdlvision/config/atlas_boxscore.json")
    ap.add_argument("--modes", default=ROOT / "src/cdlvision/config/modes.json")
    ap.add_argument("--margin-floor", type=int, default=8,
                    help="flag any glyph whose best-vs-runner-up gap is below this")
    args = ap.parse_args()

    atlas = GlyphAtlas.load(args.atlas)
    modes = TemplateSet.load(args.modes)
    info = video.probe(args.source)
    all_margins: list[int] = []
    all_distances: list[int] = []

    for ts in args.timestamps:
        gray = cv2.cvtColor(video.frame_at(args.source, ts, info), cv2.COLOR_BGR2GRAY)
        try:
            grid = detect_grid(gray)
        except GridDetectionError as exc:
            print(f"t={ts:.0f}s  no box-score grid: {exc}")
            continue

        if args.mode:
            mode, mode_note = args.mode, "(forced)"
        else:
            m = detect_mode(gray, modes)
            mode, mode_note = m.char, f"(title distance {m.distance})"

        schema = MODE_SCHEMAS[mode]
        print(f"\nt={ts:.0f}s  {grid.n_rows}x{grid.n_columns} grid  mode={mode} {mode_note}")
        if grid.n_columns != len(schema):
            print(f"  column count {grid.n_columns} != schema {mode!r} ({len(schema)})")
            continue

        result = parse(gray, atlas, mode=mode, grid=grid)
        cols = [c for c in schema if c not in NON_NUMERIC]
        print("  " + "  ".join(f"{c[:9]:>9s}" for c in cols))
        for player in result.players:
            print("  " + "  ".join(f"{player.text(c):>9s}" for c in cols))

        margins = [m.margin for p in result.players
                   for f in p.fields.values() for m in f.matches]
        distances = [m.distance for p in result.players
                     for f in p.fields.values() for m in f.matches]
        all_margins += margins
        all_distances += distances

        weak = [
            (p_i, f.name, m.char, m.margin)
            for p_i, p in enumerate(result.players)
            for f in p.fields.values()
            for m in f.matches
            if m.margin < args.margin_floor
        ]
        print(f"  glyphs {len(margins)}  distance max {max(distances)}  "
              f"margin min {min(margins)} median {int(np.median(margins))}")
        if weak:
            print(f"  {len(weak)} low-confidence glyphs:")
            for row, fname, ch, margin in weak[:10]:
                print(f"    row {row} {fname}: {ch!r} margin {margin}")

    if all_margins:
        print(f"\n=== overall: {len(all_margins)} glyphs ===")
        print(f"  match distance    max {max(all_distances)}  "
              f"median {int(np.median(all_distances))}")
        print(f"  confidence margin min {min(all_margins)}  "
              f"p1 {int(np.percentile(all_margins, 1))}  "
              f"median {int(np.median(all_margins))}")
        below = sum(1 for m in all_margins if m < args.margin_floor)
        print(f"  below margin floor ({args.margin_floor}): {below} "
              f"({100 * below / len(all_margins):.2f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
