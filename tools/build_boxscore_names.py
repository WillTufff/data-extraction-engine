"""Read player names from the box-score name column.

Names are a closed set of eight, matched as whole-region templates (binarise
at 150, match the region) rather than per-character.

Templates are built from one box score fixture and checked against the
other, held out rather than memorised.

    uv run python tools/build_boxscore_names.py <video>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cdlvision import video  # noqa: E402
from cdlvision.boxscore import MODE_SCHEMAS, detect_grid, detect_mode  # noqa: E402
from cdlvision.glyphs import TemplateSet, binarize  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

BUILD_FIXTURE = "tests/fixtures/boxscore_t3940.json"
CHECK_FIXTURE = "tests/fixtures/boxscore_t4920.json"


def name_cells(source: str, ts: float, info) -> list:
    """The binarised name cell for each player row, top to bottom."""
    gray = cv2.cvtColor(video.frame_at(source, ts, info), cv2.COLOR_BGR2GRAY)
    grid = detect_grid(gray)
    mode = detect_mode(gray, TemplateSet.load(ROOT / "src/cdlvision/config/modes.json"))
    schema = MODE_SCHEMAS[mode.char]
    x0, x1 = grid.columns[schema.index("name")]
    return [
        binarize(gray[y : y + grid.row_height, x0:x1])
        for y in grid.rows
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--out", default="src/cdlvision/config/boxscore_names.json")
    args = ap.parse_args()

    info = video.probe(args.source)
    build = json.loads((ROOT / BUILD_FIXTURE).read_text())
    check = json.loads((ROOT / CHECK_FIXTURE).read_text())

    cells = name_cells(args.source, build["timestamp"], info)
    truth = [r["name"] for r in build["rows"]]
    if len(cells) != len(truth):
        print(f"{len(cells)} rows detected, fixture has {len(truth)}",
              file=sys.stderr)
        return 1

    templates = TemplateSet("boxscore_names")
    for name, cell in zip(truth, cells):
        templates.add(name, cell)
    print(f"built {len(templates)} templates from t={build['timestamp']}")

    print(f"\nheld out on t={check['timestamp']}:")
    cells = name_cells(args.source, check["timestamp"], info)
    expected = [r["name"] for r in check["rows"]]
    wrong = 0
    for want, cell in zip(expected, cells):
        m = templates.match(cell)
        flag = "ok " if m.char == want else "BAD"
        wrong += m.char != want
        print(f"  {flag} {want:10s} read {m.char:10s} "
              f"distance {m.distance:3d} margin {m.margin:4d}")
    if wrong:
        print(f"\n{wrong} mismatches; not writing", file=sys.stderr)
        return 1

    out = ROOT / args.out
    templates.save(out)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
