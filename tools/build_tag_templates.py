"""Build whole-bitmap templates for the broadcast's replay/highlight tags.

Identifies the tag by its text rather than colour, since sponsor bugs and
warm gameplay content share the same colour band.

Usage:
    uv run python tools/build_tag_templates.py <video>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cdlvision import video  # noqa: E402
from cdlvision.glyphs import TemplateSet  # noqa: E402
from cdlvision.state import ROI_REPLAY_TAG, tag_text_masks  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# Frames confirmed by eye to carry each tag.
LABELLED: list[tuple[float, str]] = [
    (7320.0, "replay"),
    (840.0, "highlights"),
]

# Frames confirmed to carry no tag: sponsor branding and warm-toned gameplay.
NEGATIVES = [2520.0, 8100.0, 8700.0, 8940.0, 9120.0, 9540.0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--out", default=ROOT / "src/cdlvision/config/tags.json")
    args = ap.parse_args()

    info = video.probe(args.source)
    templates = TemplateSet("broadcast_tag")

    for ts, label in LABELLED:
        bgr = video.frame_at(args.source, ts, info)
        masks = tag_text_masks(bgr, ROI_REPLAY_TAG)
        if not masks:
            print(f"  ! t={ts}: no tag text extracted")
            return 1
        templates.add(label, max(masks, key=lambda m: m.size))
        print(f"t={ts:.0f}s {label:12s} bitmap {templates.templates[label].shape}")

    print("\nnegative controls:")
    for ts in NEGATIVES:
        bgr = video.frame_at(args.source, ts, info)
        masks = tag_text_masks(bgr, ROI_REPLAY_TAG)
        if not masks:
            print(f"  t={ts:6.0f}  no tag block                    -> rejected")
            continue
        verdicts = []
        for mk in masks:
            try:
                m = templates.match(mk)
                verdicts.append(f"{m.char}@{m.distance}")
            except LookupError:
                verdicts.append("shape-incompatible")
        print(f"  t={ts:6.0f}  {len(masks)} block(s): {', '.join(verdicts)}")

    templates.save(args.out)
    print(f"\nwrote {len(templates)} templates -> {Path(args.out).relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
