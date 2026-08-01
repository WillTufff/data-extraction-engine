"""Build the spectated panel's atlases, labelled by the scoreboard.

Nothing here is transcribed. The panel and the live scoreboard show the same
player's numbers at the same instant, and the scoreboard already marks which
row is spectated -- so it labels the panel: the name bar, the sixth column's
label, and the K/D/streak/stat cells (segmented into glyphs and labelled
positionally from the scoreboard's already-read string). That last part is
where `/` comes from.

Every glyph and template is a pixelwise majority over all its observations,
not one frame's rendering.

Frames where the scoreboard and the panel disagree about how many glyphs a
cell holds are dropped, and the count is reported.

    uv run python tools/build_spectated_atlas.py <video> [--samples 600]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cdlvision import scoreboard as sb  # noqa: E402
from cdlvision import spectated as sp  # noqa: E402
from cdlvision import video  # noqa: E402
from cdlvision.glyphs import GlyphAtlas, TemplateSet, segment_glyphs, tight_crop  # noqa: E402
from cdlvision.names import NameMatcher, average_masks, normalised_distance  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# Name glyphs are 28px, value glyphs 18px; canvases add margin for outliers.
NAME_CANVAS = (34, 30)
VALUE_CANVAS = (22, 19)

# Ink-normalised distance below which two weapon-name bitmaps are one weapon.
WEAPON_CLUSTER_RADIUS = 0.30

_SOURCE: str | None = None
_INFO: video.VideoInfo | None = None
_READER: sb.Reader | None = None


def _init(source: str) -> None:
    global _SOURCE, _INFO, _READER
    _SOURCE, _INFO, _READER = source, video.probe(source), sb.Reader.load()


def _observe(ts: float) -> dict | None:
    """One frame: the panel's masks, labelled by the scoreboard beside them."""
    try:
        frame = video.frame_at(_SOURCE, ts, _INFO)
    except Exception:
        return None
    if not sp.present(frame):
        return None
    try:
        sp.check_layout(frame)
        reading = _READER.read(frame)
    except Exception:
        return None

    spectated = [r for r in reading.rows if r.spectated]
    if len(spectated) != 1:
        return None          # nobody, or two -- oracle is mute either way
    row = spectated[0]

    out: dict = {"ts": ts, "name": row.name, "side": row.side,
                 "label": reading.stat_label}
    out["name_mask"] = _encode(sp.name_mask(frame))
    out["label_mask"] = _encode(sp.cell_mask(frame, "stat", sp.LABEL_Y))
    cells: dict[str, list] = {}
    for column, truth in (("kd", _kd_text(row)), ("strk", _text(row.streak)),
                          ("stat", row.stat)):
        mask = sp.cell_mask(frame, column, sp.VALUE_Y)
        spans = segment_glyphs(mask)
        if truth is None or len(spans) != len(truth):
            continue
        cells[column] = [
            (char, _encode(mask[:, x0:x1])) for char, (x0, x1) in zip(truth, spans)
        ]
    out["cells"] = cells
    out["weapon_mask"] = _encode(sp.weapon_mask(frame))
    return out


def _kd_text(row) -> str | None:
    if row.kills is None or row.deaths is None:
        return None
    return f"{row.kills}/{row.deaths}"


def _text(value) -> str | None:
    return None if value is None else str(value)


def _encode(mask: np.ndarray) -> list[str]:
    return ["".join("#" if v else "." for v in r) for r in mask]


def _decode(rows: list[str]) -> np.ndarray:
    return np.array([[c == "#" for c in r] for r in rows], dtype=bool)


def _majority(masks: list[np.ndarray]) -> np.ndarray | None:
    """Pixelwise majority over every observation at the same tight size.

    Grouped by shape first, since a mask one pixel wider is a different
    rendering (usually a fade) and averaging it in blurs the edges.
    """
    by_shape: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
    for mask in masks:
        cropped = tight_crop(mask)
        if cropped.size:
            by_shape[cropped.shape].append(cropped)
    if not by_shape:
        return None
    best = max(by_shape.values(), key=len)
    return np.mean(best, axis=0) >= 0.5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?")
    ap.add_argument("--samples", type=int, default=600)
    ap.add_argument("--timeline", default="data/state_timeline.json")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default="src/cdlvision/config")
    ap.add_argument("--cache", default="data/spectated_observations.jsonl",
                    help="labelled masks; reused if present so the build is "
                         "tunable without decoding again")
    ap.add_argument("--rescan", action="store_true")
    ap.add_argument("--sheet", default="data/spectated_weapons.png")
    args = ap.parse_args()

    cache = ROOT / args.cache
    if cache.exists() and not args.rescan:
        rows = [json.loads(line) for line in cache.read_text().splitlines() if line]
        print(f"{len(rows)} labelled frames from {args.cache}")
    else:
        if not args.video:
            ap.error("no cache yet, so the video is needed")
        spans = [
            (s["start"] + 2, s["end"] - 2)
            for s in json.loads((ROOT / args.timeline).read_text())
            if s["state"] == "live" and s["end"] - s["start"] > 6
        ]
        rng = random.Random(args.seed)
        stamps = sorted(rng.uniform(*rng.choice(spans)) for _ in range(args.samples))
        with Pool(8, initializer=_init, initargs=(args.video,)) as pool:
            rows = [r for r in pool.map(_observe, stamps) if r]
        print(f"{len(rows)}/{len(stamps)} sampled LIVE frames carry a panel the "
              f"scoreboard can label")
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    # -- the value atlas, including the slash ----------------------------
    glyphs: dict[str, list[np.ndarray]] = defaultdict(list)
    for row in rows:
        for column, pairs in row["cells"].items():
            for char, mask in pairs:
                glyphs[char].append(_decode(mask))
    atlas = GlyphAtlas("spectated", canvas=VALUE_CANVAS)
    for char, masks in sorted(glyphs.items()):
        voted = _majority(masks)
        if voted is not None:
            atlas.add(char, voted)
    print(f"value atlas: {''.join(sorted(atlas.glyphs))} "
          f"from {sum(len(v) for v in glyphs.values())} observations "
          f"({', '.join(f'{c}:{len(v)}' for c, v in sorted(glyphs.items()))})")
    atlas.save(Path(args.out) / "atlas_spectated.json")

    # -- names, as a closed-set matcher ----------------------------------
    names = _name_matcher(rows)
    names.save(Path(args.out) / "spectated_names.json")

    # -- the stat label, as a whole region -------------------------------
    labels = _template_set("spectated_labels", rows, "label", "label_mask")
    labels.save(Path(args.out) / "spectated_labels.json")

    # -- the weapon name, clustered ---------------------------------------
    weapons = _weapon_templates(rows, Path(args.out) / "spectated_weapons.json",
                                ROOT / args.sheet)
    weapons.save(Path(args.out) / "spectated_weapons.json")

    _report_separation(atlas, glyphs)
    _report_names(names, rows)
    _report_templates(labels, rows, "label", "label_mask")
    _report_clock_agreement(atlas)
    return 0


def _name_matcher(rows: list[dict]) -> NameMatcher:
    """One template per (player, observed width), each a majority."""
    groups: dict[tuple[str, int], list[np.ndarray]] = defaultdict(list)
    teams: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        if not row.get("name"):
            continue
        mask = tight_crop(_decode(row["name_mask"]))
        if not mask.size:
            continue
        groups[(row["name"], mask.shape[1])].append(mask)
        if row.get("side"):
            teams[row["name"]][row["side"]] += 1
    matcher = NameMatcher(teams={n: c.most_common(1)[0][0] for n, c in teams.items()})
    for (label, _), masks in sorted(groups.items()):
        if len(masks) < 3:
            continue
        matcher.add(label, average_masks(masks))
    widths = Counter(label for label, _ in groups)
    print(f"spectated_names: {sum(len(v) for v in matcher.templates.values())} "
          f"templates over {len(matcher.templates)} players "
          f"({', '.join(f'{n}:{w}' for n, w in sorted(widths.items()))} widths seen)")
    return matcher


def _weapon_templates(rows: list[dict], previous: Path, sheet: Path,
                      min_members: int = 8) -> TemplateSet:
    """Cluster the weapon-name band and keep the clusters that recur.

    No oracle labels the weapon name, so the bitmaps are clustered and voted,
    and the strings are transcribed once from the sheet this writes. Names
    carry over across rebuilds by matching each new cluster against the
    previous config; an unmatched cluster gets a positional id.
    """
    masks = [tight_crop(_decode(r["weapon_mask"])) for r in rows]
    clusters: list[list[np.ndarray]] = []
    reps: list[np.ndarray] = []
    for mask in masks:
        if mask.size == 0 or mask.shape[0] < 8:
            continue
        for i, rep in enumerate(reps):
            if normalised_distance(mask, rep) < WEAPON_CLUSTER_RADIUS:
                clusters[i].append(mask)
                break
        else:
            reps.append(mask)
            clusters.append([mask])

    order = sorted(range(len(clusters)), key=lambda i: -len(clusters[i]))
    kept = [(len(clusters[i]), average_masks(clusters[i])) for i in order
            if len(clusters[i]) >= min_members]
    known = TemplateSet.load(previous) if previous.exists() else None

    out = TemplateSet("spectated_weapons")
    unnamed = 0
    for n, voted in kept:
        label = _carry_over(known, voted)
        if label is None:
            label, unnamed = f"weapon_{unnamed:02d}", unnamed + 1
        suffix = ""
        while label + suffix in out.templates:
            suffix = f"#{int(suffix[1:] or 0) + 1}"
        out.templates[label + suffix] = voted
    _write_sheet(sheet, kept)
    covered = sum(n for n, _ in kept)
    print(f"spectated_weapons: {len(out.templates)} templates covering "
          f"{covered}/{len(masks)} observations ({100 * covered / len(masks):.1f}%), "
          f"{len(clusters) - len(kept)} clusters below {min_members} members; "
          f"names {sorted({k.split('#')[0] for k in out.templates})}")
    print(f"  sheet: {sheet}")
    return out


def _carry_over(known: TemplateSet | None, voted: np.ndarray) -> str | None:
    if known is None:
        return None
    best, label = min(
        ((normalised_distance(voted, t), k) for k, t in known.templates.items()),
        default=(1.0, None),
    )
    if label is None or best >= WEAPON_CLUSTER_RADIUS:
        return None
    return label.split("#")[0]


def _write_sheet(path: Path, kept: list[tuple[int, np.ndarray]]) -> None:
    import cv2

    if not kept:
        return
    tiles = []
    for _, voted in kept:
        img = (voted * 255).astype(np.uint8)
        tiles.append(cv2.resize(img, (img.shape[1] * 3, img.shape[0] * 3),
                                interpolation=cv2.INTER_NEAREST))
    width = max(t.shape[1] for t in tiles)
    sheet = np.zeros((sum(t.shape[0] + 8 for t in tiles), width), np.uint8)
    y = 0
    for tile in tiles:
        sheet[y:y + tile.shape[0], :tile.shape[1]] = tile
        y += tile.shape[0] + 8
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), sheet)


def _report_names(matcher: NameMatcher, rows: list[dict]):
    right = wrong = unknown = 0
    for row in rows:
        if not row.get("name"):
            continue
        hit = matcher.match(_decode(row["name_mask"]))
        if hit.label is None:
            unknown += 1
        elif hit.label == row["name"]:
            right += 1
        else:
            wrong += 1
    total = right + wrong + unknown
    if total:
        print(f"spectated_names: {right}/{total} agree with the scoreboard "
              f"({100 * right / total:.1f}%), {wrong} disagree, {unknown} unknown")


def _template_set(name: str, rows: list[dict], key: str,
                  mask_key: str) -> TemplateSet:
    """One template per label, per distinct tight width.

    A name drawn mid-fade is a different bitmap, and `TemplateSet.match`
    requires shapes to agree within a pixel.
    """
    groups: dict[tuple[str, tuple[int, int]], list[np.ndarray]] = defaultdict(list)
    for row in rows:
        label = row.get(key)
        if not label:
            continue
        mask = tight_crop(_decode(row[mask_key]))
        if mask.size:
            groups[(label, mask.shape)].append(mask)
    out = TemplateSet(name)
    kept = Counter()
    for (label, _), masks in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        if len(masks) < 3:
            continue          # too few to outvote one frame's noise
        voted = np.mean(masks, axis=0) >= 0.5
        suffix = "" if kept[label] == 0 else f"#{kept[label]}"
        out.templates[label + suffix] = voted
        kept[label] += 1
    print(f"{name}: {len(out.templates)} templates over "
          f"{len(set(l for l, _ in groups))} labels")
    return out


def _report_separation(atlas: GlyphAtlas, glyphs: dict[str, list[np.ndarray]]):
    """Reports the in-class vs. misread distance gap for the acceptance limit."""
    worst_true, best_wrong = 0, 10**9
    for char, masks in glyphs.items():
        for mask in masks[:200]:
            hit = atlas.match(mask)
            if hit.char == char:
                worst_true = max(worst_true, hit.distance)
            else:
                best_wrong = min(best_wrong, hit.distance)
    print(f"value atlas: worst in-class distance {worst_true}, "
          f"best misread {best_wrong if best_wrong < 10**9 else 'none'}")


def _report_templates(templates: TemplateSet, rows: list[dict], key: str,
                      mask_key: str):
    right = wrong = unread = 0
    for row in rows:
        label = row.get(key)
        if not label:
            continue
        mask = _decode(row[mask_key])
        got = sp._template(templates, mask, 10**9)
        if got is None:
            unread += 1
        elif got.split("#")[0] == label:
            right += 1
        else:
            wrong += 1
    total = right + wrong + unread
    if total:
        print(f"{templates.name}: {right}/{total} agree with the scoreboard "
              f"({100 * right / total:.1f}%), {wrong} disagree, {unread} unread")


def _report_clock_agreement(atlas: GlyphAtlas):
    """Checks the panel's font against the header clock's atlas on shared glyphs."""
    clock = GlyphAtlas.load(ROOT / "src/cdlvision/config/atlas_header_clock.json")
    shared = sorted(set(atlas.glyphs) & set(clock.glyphs))
    distances = {}
    for char in shared:
        hit = clock.match(atlas.glyphs[char])
        distances[char] = (hit.char, hit.distance)
    agree = sum(1 for c, (got, _) in distances.items() if got == c)
    worst = max((d for _, d in distances.values()), default=0)
    print(f"vs the header clock atlas: {agree}/{len(shared)} of the shared "
          f"glyphs match, worst distance {worst}")
    print(f"  not in the clock's atlas: "
          f"{sorted(set(atlas.glyphs) - set(clock.glyphs))}")


if __name__ == "__main__":
    raise SystemExit(main())
