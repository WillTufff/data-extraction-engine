"""Cluster the scoreboard's gutter icons into a labellable template set.

Icons have no oracle, so identity comes from shape agreement: cluster the
observed bitmaps, average each cluster into a template, render a contact
sheet, and let a human attach a meaning to each.

The sheet shows three families: the skull (a death with no killfeed row is a
suicide), the unfilled objective disc, and the filled objective disc. The
filled disc doesn't separate by mode, but the mode is already read from the
stat label, so the gutter only needs to distinguish skull / objective / none.

    uv run python tools/cluster_scoreboard_icons.py <video> --sweep
    uv run python tools/cluster_scoreboard_icons.py <video> --threshold 0.30
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

from cdlvision import scoreboard as sb  # noqa: E402
from cdlvision import video  # noqa: E402
from cdlvision.glyphs import TemplateSet, tight_crop  # noqa: E402
from cdlvision.names import average_masks, normalised_distance  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# An icon that appears a handful of times is more likely a mid-animation frame
# than a class of its own.
MIN_CLUSTER = 10

_SOURCE: str | None = None
_INFO = None


def _init(source: str) -> None:
    global _SOURCE, _INFO
    _SOURCE, _INFO = source, video.probe(source)


def _one(ts: float):
    try:
        frame = video.frame_at(_SOURCE, ts, _INFO)
    except Exception:
        return []
    out = []
    for side in sb.SIDES:
        for slot in range(sb.ROW_COUNT):
            mask = sb.gutter_mask(side.cell(frame, slot, "gutter"))
            if mask.sum() < sb.MIN_ICON_INK:
                continue
            out.append((ts, side.name, slot, tight_crop(mask)))
    return out


def live_timestamps(timeline: Path) -> list[float]:
    stamps: list[float] = []
    for s in json.loads(timeline.read_text()):
        if s["state"] != "live":
            continue
        t = s["start"] + 2
        while t < s["end"] - 1:
            stamps.append(t)
            t += 1
    return stamps


def distances(masks: list[np.ndarray]) -> np.ndarray:
    """The pairwise distance matrix, computed once and reused by every threshold."""
    n = len(masks)
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d[i, j] = d[j, i] = normalised_distance(masks[i], masks[j])
    return d


def cluster(d: np.ndarray, threshold: float) -> list[list[int]]:
    """Greedy clustering around medoids (seeded on the densest remaining point,
    unlike single-link agglomerative, which lets a chain of near-misses merge
    distinct classes)."""
    n = len(d)
    unassigned = set(range(n))
    clusters: list[list[int]] = []
    while unassigned:
        idx = sorted(unassigned)
        seed = max(idx, key=lambda i: sum(d[i, j] < threshold for j in idx))
        members = [j for j in idx if d[seed, j] < threshold]
        clusters.append(members)
        unassigned -= set(members)
    return sorted(clusters, key=len, reverse=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--timeline", default="data/state_timeline.json")
    ap.add_argument("--samples", type=int, default=500)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threshold", type=float, default=0.30)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--sheet", default="data/scoreboard_icons.png")
    ap.add_argument(
        "--label", action="append", default=[],
        help="icon0=skull, repeatable. Clusters keep their positional id "
             "unless named, so the mapping stays visible in the config.",
    )
    args = ap.parse_args()

    stamps = live_timestamps(ROOT / args.timeline)
    random.Random(5).shuffle(stamps)
    stamps = sorted(stamps[: args.samples])

    observed: list[tuple] = []
    with Pool(args.workers, initializer=_init, initargs=(args.source,)) as pool:
        for batch in pool.imap(_one, stamps, chunksize=4):
            observed += batch
    masks = [m for *_, m in observed]
    print(f"{len(stamps)} LIVE frames sampled, {len(masks)} gutter cells hold an icon")
    print(f"shapes: {Counter(m.shape for m in masks).most_common(6)}")

    d = distances(masks)

    if args.sweep:
        for t in (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50):
            groups = cluster(d, t)
            big = [g for g in groups if len(g) >= MIN_CLUSTER]
            print(f"  threshold {t:.2f}: {len(groups):3d} clusters, "
                  f"{len(big)} with >={MIN_CLUSTER} members, "
                  f"covering {100 * sum(len(g) for g in big) / len(masks):.1f}%")
        return 0

    groups = [g for g in cluster(d, args.threshold) if len(g) >= MIN_CLUSTER]
    print(f"\nthreshold {args.threshold}: {len(groups)} clusters")

    names = dict(pair.split("=", 1) for pair in args.label)
    templates = TemplateSet("scoreboard_icons")
    for i, members in enumerate(groups):
        template = average_masks([masks[j] for j in members])
        label = names.get(f"icon{i}", f"icon{i}")
        templates.templates[label] = tight_crop(template)
        spread = max(
            normalised_distance(masks[j], template) for j in members
        )
        print(f"  {label:10s} (icon{i}): n={len(members):4d} "
              f"shape={template.shape} worst member distance {spread:.3f}")

    _sheet(groups, masks, ROOT / args.sheet)
    print(f"wrote {args.sheet} -- label the clusters, then rerun with --out")

    if args.out:
        templates.save(ROOT / args.out)
        print(f"wrote {args.out}")
    return 0


def _sheet(groups, masks, path: Path) -> None:
    """One row per cluster: its average, then a sample of its members."""
    cols = 13
    cell = 34
    sheet = np.zeros(((len(groups)) * (cell + 4), cols * (cell + 4), 3), np.uint8)
    for r, members in enumerate(groups):
        picks = [average_masks([masks[j] for j in members])] + [
            masks[j] for j in members[: cols - 1]
        ]
        for c, m in enumerate(picks):
            img = np.dstack([m.astype(np.uint8) * 255] * 3)
            if c == 0:  # tint the cluster average so it reads as the template
                img[..., 0] = 0
            y, x = r * (cell + 4), c * (cell + 4)
            sheet[y : y + m.shape[0], x : x + m.shape[1]] = img
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.resize(sheet, None, fx=3, fy=3,
                                      interpolation=cv2.INTER_NEAREST))


if __name__ == "__main__":
    raise SystemExit(main())
