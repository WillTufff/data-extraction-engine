"""Build whole-bitmap templates for the box-score mode title.

All three modes render 13 columns, so the mode has to be read rather than
inferred from table shape. The title is one of a closed set of fixed strings,
so it is matched as a single bitmap instead of character by character.

Usage:
    uv run python tools/build_mode_templates.py <video>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cdlvision import video  # noqa: E402
from cdlvision.boxscore import header_lines  # noqa: E402
from cdlvision.glyphs import TemplateSet, binarize  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# Timestamps of a known box score for each mode, confirmed against the series
# overview screen's map list.
LABELLED: list[tuple[float, str]] = [
    (760.0, "hardpoint"),
    (2800.0, "search_and_destroy"),
    (3940.0, "overload"),
]

X_WINDOW = (215, 640)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--out", default=ROOT / "src/cdlvision/config/modes.json")
    args = ap.parse_args()

    info = video.probe(args.source)
    templates = TemplateSet("boxscore_mode")

    for ts, mode in LABELLED:
        gray = cv2.cvtColor(video.frame_at(args.source, ts, info), cv2.COLOR_BGR2GRAY)
        lines = header_lines(gray, x_window=X_WINDOW)
        if not lines:
            print(f"  ! t={ts}: no header lines found")
            return 1
        y0, y1 = lines[0]
        mask = binarize(gray[y0 : y1 + 1, X_WINDOW[0] : X_WINDOW[1]])
        templates.add(mode, mask)
        print(f"t={ts:.0f}s {mode:20s} title band y={y0}-{y1} "
              f"bitmap {templates.templates[mode].shape}")

    # Cross-check: every template must be nearer to itself than to any other.
    print("\npairwise distances:")
    labels = sorted(templates.templates)
    for a in labels:
        row = []
        for b in labels:
            ta, tb = templates.templates[a], templates.templates[b]
            if ta.shape != tb.shape:
                row.append("  shape")
            else:
                row.append(f"{int(np.count_nonzero(ta ^ tb)):6d}")
        print(f"  {a:20s} " + " ".join(row))

    templates.save(args.out)
    print(f"\nwrote {len(templates)} templates -> {Path(args.out).relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
