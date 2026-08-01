"""Build a glyph atlas from a ground-truth-labelled box-score frame.

Segments each cell into character spans and pairs them positionally with the
known string. A length mismatch is reported as a segmentation bug rather than
silently absorbed.

Usage:
    uv run python tools/build_atlas.py <video> [--fixture ...] [--out ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cdlvision import video  # noqa: E402
from cdlvision.boxscore import MODE_SCHEMAS, NON_NUMERIC, detect_grid  # noqa: E402
from cdlvision.glyphs import GlyphAtlas, binarize, segment_glyphs, tight_crop  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="video file")
    ap.add_argument(
        "--fixture", action="append", default=None,
        help="labelled frame spec; repeatable (defaults to all fixtures)",
    )
    ap.add_argument("--out", default=ROOT / "src/cdlvision/config/atlas_boxscore.json")
    ap.add_argument("--font", default="boxscore_stat")
    args = ap.parse_args()

    fixtures = args.fixture or sorted(
        str(p) for p in (ROOT / "tests/fixtures").glob("boxscore_t*.json")
    )

    samples: dict[str, list[np.ndarray]] = defaultdict(list)
    problems: list[str] = []

    for fixture in fixtures:
        spec = json.loads(Path(fixture).read_text())
        schema = MODE_SCHEMAS[spec["mode"]]

        frame = video.frame_at(args.source, spec["timestamp"])
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        grid = detect_grid(gray)
        print(f"{Path(fixture).name}: t={spec['timestamp']}s {spec['mode']!r} "
              f"-> {grid.n_rows}x{grid.n_columns} grid, row height {grid.row_height}px")
        if grid.n_columns != len(schema):
            print(f"  ! schema for {spec['mode']!r} expects {len(schema)} columns")
            return 1

        for row_spec, y0 in zip(spec["rows"], grid.rows):
            for name, (x0, x1) in zip(schema, grid.columns):
                if name in NON_NUMERIC:
                    continue
                truth = row_spec[name]
                cell = gray[y0 : y0 + grid.row_height, x0:x1]
                mask = binarize(cell)
                spans = segment_glyphs(mask)
                if len(spans) != len(truth):
                    problems.append(
                        f"{Path(fixture).stem}/{row_spec['name']}.{name}: expected "
                        f"{len(truth)} glyphs for {truth!r}, segmented {len(spans)}"
                    )
                    continue
                for ch, (sx0, sx1) in zip(truth, spans):
                    samples[ch].append(tight_crop(mask[:, sx0:sx1]))

    if problems:
        print(f"\n{len(problems)} segmentation mismatches:")
        for p in problems[:20]:
            print("  ", p)

    atlas = GlyphAtlas(args.font)
    print(f"\nglyph samples ({len(samples)} distinct characters):")
    for ch in sorted(samples):
        crops = samples[ch]
        shapes = {c.shape for c in crops}
        canon = max(crops, key=lambda c: int(c.sum()))
        spread = 0
        for c in crops:
            if c.shape == canon.shape:
                spread = max(spread, int(np.count_nonzero(c ^ canon)))
        flag = "" if len(shapes) == 1 else f"  <- {len(shapes)} distinct sizes {sorted(shapes)}"
        print(f"  {ch!r}: {len(crops):3d} samples, max intra-class diff {spread:2d}px{flag}")
        atlas.add(ch, canon)

    out = Path(args.out)
    atlas.save(out)
    print(f"\nwrote {len(atlas)} glyphs -> {out.relative_to(ROOT)}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
