"""Cluster killfeed weapon and medal icons into a labellable template set.

Icons have no oracle, so identity comes from shape agreement: cluster the
observed bitmaps, average each cluster into a template, and render a contact
sheet for a human to attach real weapon names to. Until then, clusters ship
with stable positional ids.

Clustering is greedy around medoids rather than agglomerative, to avoid a
chain of near-misses merging distinct icons.

    uv run python tools/cluster_killfeed_icons.py --sweep
    uv run python tools/cluster_killfeed_icons.py --threshold 0.28
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cdlvision.names import average_masks, normalised_distance  # noqa: E402
from scan_killfeed import decode  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# Minimum cluster size worth a template.
MIN_CLUSTER = 8


def load(path: Path, kind: str) -> list[np.ndarray]:
    out = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        payload = json.loads(line).get(f"{kind}_bmp")
        if payload:
            out.append(decode(payload))
    return out


def distances(masks: list[np.ndarray]) -> np.ndarray:
    """The pairwise distance matrix, computed once and reused by every threshold."""
    n = len(masks)
    dist = np.ones((n, n), dtype=np.float32)
    for i in range(n):
        dist[i, i] = 0.0
        for j in range(i + 1, n):
            d = normalised_distance(masks[i], masks[j])
            dist[i, j] = dist[j, i] = d
    return dist


def cluster(masks: list[np.ndarray], threshold: float, dist=None) -> list[dict]:
    """Greedy medoid clustering; returns averaged templates, largest first.

    Seeds are taken in order of neighbour count, so clusters form around a
    typical rendering rather than an arbitrary first sample.
    """
    n = len(masks)
    if dist is None:
        dist = distances(masks)

    neighbours = (dist <= threshold).sum(axis=1)
    order = np.argsort(-neighbours)

    assigned = np.full(n, -1)
    clusters: list[list[int]] = []
    for seed in order:
        if assigned[seed] != -1:
            continue
        members = [i for i in np.flatnonzero(dist[seed] <= threshold)
                   if assigned[i] == -1]
        if not members:
            continue
        cid = len(clusters)
        for i in members:
            assigned[i] = cid
        clusters.append(members)

    out = []
    for members in clusters:
        out.append({
            "n": len(members),
            "template": average_masks([masks[i] for i in members]),
            "widths": Counter(masks[i].shape[1] for i in members),
        })
    out.sort(key=lambda c: -c["n"])
    return out


def contact_sheet(clusters: list[dict], path: Path, scale: int = 4) -> None:
    """One row per cluster, so the shapes can be told apart by eye."""
    if not clusters:
        return
    h = max(c["template"].shape[0] for c in clusters)
    w = max(c["template"].shape[1] for c in clusters)
    pad = 4
    tile_h, tile_w = h + pad, w + pad
    canvas = np.zeros((tile_h * len(clusters), tile_w), np.uint8)
    for i, c in enumerate(clusters):
        t = c["template"]
        y = i * tile_h
        canvas[y : y + t.shape[0], : t.shape[1]] = t.astype(np.uint8) * 255
    big = cv2.resize(
        canvas, (tile_w * scale, tile_h * len(clusters) * scale),
        interpolation=cv2.INTER_NEAREST,
    )
    big = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)
    for i, c in enumerate(clusters):
        cv2.putText(big, f"{i}: n={c['n']}", (4, i * tile_h * scale + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), big)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="data/killfeed_rows.jsonl")
    ap.add_argument("--kind", choices=("weapon", "medal"), default="weapon")
    ap.add_argument("--threshold", type=float, default=0.28)
    ap.add_argument("--min-cluster", type=int, default=MIN_CLUSTER)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--sample", type=int, default=1200)
    ap.add_argument("--sheet", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    masks = load(ROOT / args.rows, args.kind)
    print(f"{len(masks)} {args.kind} bitmaps")
    if not masks:
        return 1
    if len(masks) > args.sample:
        step = len(masks) / args.sample
        masks = [masks[int(i * step)] for i in range(args.sample)]
        print(f"  clustering an even sample of {len(masks)}")

    dist = distances(masks)
    if args.sweep:
        print(f"\n{'thresh':>7} {'clusters':>9} {'>=min':>6} {'covered':>8}")
        for th in (0.16, 0.20, 0.24, 0.28, 0.32, 0.36, 0.40, 0.45):
            cs = cluster(masks, th, dist)
            big = [c for c in cs if c["n"] >= args.min_cluster]
            covered = sum(c["n"] for c in big) / len(masks)
            print(f"{th:7.2f} {len(cs):9d} {len(big):6d} {covered:8.1%}")
        return 0

    clusters = cluster(masks, args.threshold, dist)
    kept = [c for c in clusters if c["n"] >= args.min_cluster]
    covered = sum(c["n"] for c in kept) / len(masks)
    print(f"\n{len(clusters)} clusters, {len(kept)} with n>={args.min_cluster} "
          f"covering {covered:.1%} of observations")
    for i, c in enumerate(kept):
        widths = ",".join(f"{w}x{n}" for w, n in c["widths"].most_common(3))
        print(f"  {i:2d}  n={c['n']:4d}  widths {widths}")

    sheet = Path(args.sheet or f"data/killfeed_{args.kind}_clusters.png")
    contact_sheet(kept, ROOT / sheet)
    print(f"\nwrote {sheet}")

    if args.out:
        payload = {
            "name": f"killfeed_{args.kind}s",
            "threshold": args.threshold,
            "note": "positional ids; rename to real weapons from the sheet",
            "templates": {
                f"{args.kind}_{i:02d}": [
                    "".join("#" if v else "." for v in row)
                    for row in c["template"]
                ]
                for i, c in enumerate(kept)
            },
        }
        out = ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
