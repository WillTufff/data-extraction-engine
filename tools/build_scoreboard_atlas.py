"""Build the live scoreboard's glyph atlas, name templates and stat labels.

Three artefacts come out of one pass over the hand-transcribed fixtures:

- `atlas_scoreboard.json` -- per-character bitmaps for the numeric columns,
  a smaller font cut than the box score's (12px vs 23px), so not reusable.
- `scoreboard_names.json` -- whole-region templates for the closed set of
  eight players, one template per name.
- `scoreboard_labels.json` -- templates for the column headers. The sixth
  column reads TIME in Hardpoint, PLNT in Search and Destroy, OVLD in
  Control at identical geometry, so the label is what says which mode.

Segmentation and labelling are zipped positionally: a length disagreement is
a segmentation bug and is reported rather than absorbed.

Two fixtures build and two are held out, so agreement is measured rather than
memorised.

    uv run python tools/build_scoreboard_atlas.py <video>
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cdlvision import scoreboard as sb  # noqa: E402
from cdlvision import video  # noqa: E402
from cdlvision.glyphs import GlyphAtlas, TemplateSet, tight_crop  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "src/cdlvision/config"

BUILD = [
    "tests/fixtures/scoreboard_t300.json",   # Hardpoint, TIME
    "tests/fixtures/scoreboard_t1700.json",  # Search and Destroy, PLNT
    "tests/fixtures/scoreboard_t7600.json",  # Overload, OVLD
]
HELD_OUT = ["tests/fixtures/scoreboard_t3300.json", "tests/fixtures/scoreboard_t4500.json"]

# Columns read character by character against the atlas. The name column is a
# whole-region match instead, and the gutter is icons.
NUMERIC = ("number", "kd", "strk", "stat")


def _side(name: str) -> sb.Side:
    return sb.LEFT if name == "left" else sb.RIGHT


def _cells(source: str, spec: dict, info):
    """Yield (row_spec, key, mask) for every numeric cell, plus names/labels."""
    frame = video.frame_at(source, spec["timestamp"], info)
    for row in spec["rows"]:
        side, slot = _side(row["side"]), row["slot"]
        rule = sb.row_ink(frame, side, slot)
        if rule.inverted != row["spectated"]:
            raise SystemExit(
                f"t={spec['timestamp']} {row['side']}[{slot}]: polarity says "
                f"inverted={rule.inverted}, fixture says spectated={row['spectated']}"
            )
        for key in NUMERIC:
            yield row, key, sb.cell_mask(frame, side, slot, key, rule)
        yield row, "name", sb.cell_mask(frame, side, slot, "name", rule)
    for side in sb.SIDES:
        for key in ("kd", "strk", "stat"):
            mask = sb.ink(side.label_cell(frame, key))
            spans = sb.token_spans(mask)
            label = {"kd": "K/D", "strk": "STRK", "stat": spec["stat_label"]}[key]
            yield None, ("label", label), mask[:, spans[0][0] : spans[-1][1] + 1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--font", default="scoreboard_stat")
    args = ap.parse_args()

    info = video.probe(args.source)
    glyphs: dict[str, list[np.ndarray]] = defaultdict(list)
    names: dict[str, list[np.ndarray]] = defaultdict(list)
    labels: dict[str, list[np.ndarray]] = defaultdict(list)
    problems: list[str] = []

    for fixture in BUILD:
        spec = json.loads((ROOT / fixture).read_text())
        print(f"{Path(fixture).name}: t={spec['timestamp']}s, "
              f"stat column {spec['stat_label']!r}")
        for row, key, mask in _cells(args.source, spec, info):
            if isinstance(key, tuple):
                labels[key[1]].append(mask)
                continue
            if key == "name":
                names[row["name"]].append(mask)
                continue
            expected = row[key]
            spans = sb.segment_cell(mask)
            if len(spans) != len(expected):
                problems.append(
                    f"  t={spec['timestamp']} {row['side']}[{row['slot']}] {key}: "
                    f"{len(spans)} glyphs segmented for {expected!r} "
                    f"({len(expected)} expected)"
                )
                continue
            for (x0, x1), char in zip(spans, expected):
                glyphs[char].append(mask[:, x0:x1])

    if problems:
        print("\nsegmentation disagreed with the labelling:")
        print("\n".join(problems))
        return 1

    print(f"\n{len(glyphs)} distinct characters, "
          f"{sum(len(v) for v in glyphs.values())} instances")
    atlas = GlyphAtlas(args.font)
    for char, instances in sorted(glyphs.items()):
        canvases = [tight_crop(m) for m in instances]
        shapes = {c.shape for c in canvases}
        biggest = max(shapes, key=lambda s: s[0] * s[1])
        stack = np.zeros((len(canvases), *biggest), dtype=bool)
        for i, c in enumerate(canvases):
            stack[i, : c.shape[0], : c.shape[1]] = c
        disagree = int((stack.any(axis=0) & ~stack.all(axis=0)).sum())
        atlas.add(char, stack.mean(axis=0) > 0.5)
        print(f"  {char!r:4s} n={len(instances):3d} shapes={sorted(shapes)} "
              f"disagreeing pixels={disagree}")

    _write(atlas, CONFIG / "atlas_scoreboard.json")

    name_set = TemplateSet("scoreboard_names")
    for name, instances in sorted(names.items()):
        name_set.templates[name] = _majority(instances)
        print(f"  name {name:9s} n={len(instances)} "
              f"widths={sorted({m.shape[1] for m in instances})}")
    _write(name_set, CONFIG / "scoreboard_names.json")

    label_set = TemplateSet("scoreboard_labels")
    for label, instances in sorted(labels.items()):
        label_set.templates[label] = _majority(instances)
        print(f"  label {label:6s} n={len(instances)} "
              f"widths={sorted({m.shape[1] for m in instances})}")
    _write(label_set, CONFIG / "scoreboard_labels.json")

    return _score(args.source, info, atlas, name_set, label_set)


def _majority(instances: list[np.ndarray]) -> np.ndarray:
    """Pixelwise majority over every observation, at the modal shape.

    Instances of another shape are dropped rather than padded.
    """
    from collections import Counter

    crops = [tight_crop(m) for m in instances]
    modal = Counter(c.shape for c in crops).most_common(1)[0][0]
    stack = np.stack([c for c in crops if c.shape == modal])
    return stack.mean(axis=0) > 0.5


def _write(obj, path: Path) -> None:
    obj.save(path)
    print(f"wrote {path.relative_to(ROOT)}")


def _score(source, info, atlas, name_set, label_set) -> int:
    """Read the held-out fixtures and report every disagreement."""
    print("\n=== held out ===")
    wrong = 0
    total = 0
    # margin = distance to runner-up minus distance to winner; worst margin
    # over the set says whether it's separable, not just its accuracy.
    margins: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    for fixture in HELD_OUT:
        spec = json.loads((ROOT / fixture).read_text())
        frame = video.frame_at(source, spec["timestamp"], info)
        for row in spec["rows"]:
            side, slot = _side(row["side"]), row["slot"]
            rule = sb.row_ink(frame, side, slot)
            got = {}
            for key in NUMERIC:
                mask = sb.cell_mask(frame, side, slot, key, rule)
                hits = [atlas.match(mask[:, a:b]) for a, b in sb.segment_cell(mask)]
                got[key] = "".join(h.char for h in hits)
                margins["glyph"] += [_score_of(h) for h in hits]
            hit = name_set.match(sb.cell_mask(frame, side, slot, "name", rule))
            got["name"] = hit.char
            margins["name"].append(_score_of(hit))
            for key in (*NUMERIC, "name"):
                total += 1
                if got[key] != row[key]:
                    wrong += 1
                    print(f"  t={spec['timestamp']} {row['side']}[{slot}] {key}: "
                          f"read {got[key]!r}, expected {row[key]!r}")
        # the schema label
        mask = sb.ink(sb.LEFT.label_cell(frame, "stat"))
        spans = sb.token_spans(mask)
        hit = label_set.match(mask[:, spans[0][0] : spans[-1][1] + 1])
        total += 1
        margins["label"].append(_score_of(hit))
        if hit.char != spec["stat_label"]:
            wrong += 1
            print(f"  t={spec['timestamp']} stat label: read {hit.char!r}, "
                  f"expected {spec['stat_label']!r}")

    print(f"\n{total - wrong}/{total} held-out cells correct "
          f"({100 * (total - wrong) / total:.1f}%)")
    for kind, hits in sorted(margins.items()):
        contested = [h for h in hits if h[0] is not None]
        sole = len(hits) - len(contested)
        line = (f"  {kind:6s} n={len(hits):4d}  "
                f"max distance {max(h[1] for h in hits):3d}")
        if contested:
            worst = min(contested)
            line += (f"  worst margin {worst[0]:3d} "
                     f"(distance {worst[1]} on {worst[2]!r})")
        if sole:
            line += f"  [{sole} with no shape-compatible rival]"
        print(line)
    return 1 if wrong else 0


def _score_of(hit):
    """(margin, distance, label), with margin None when nothing else competed."""
    return (None if hit.runner_up is None else hit.margin, hit.distance, hit.char)


if __name__ == "__main__":
    raise SystemExit(main())
