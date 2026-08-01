"""Build whole-bitmap templates for the killfeed's player names.

Player names are a closed set (eight per series), matched as one bitmap
rather than character by character. Name tokens are harvested from a sample
of LIVE frames and grouped by exact bitmap; recurring groups are the names.
The survey pass writes the clusters as a contact sheet for labelling, and the
labels are recorded in `LABELLED` below.

Two passes:

    uv run python tools/build_killfeed_templates.py <video> --survey
    # ... read the contact sheet, fill in LABELLED ...
    uv run python tools/build_killfeed_templates.py <video>
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

from cdlvision import video  # noqa: E402
from cdlvision.glyphs import TemplateSet, tight_crop  # noqa: E402
from cdlvision.killfeed import detect_rows  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# Cluster index -> player name, filled in from the survey contact sheet.
LABELLED: dict[int, str] = {
    0: "MERCULES",
    1: "SHOTZZY",
    2: "ABUZAH",
    3: "DRAZAH",
    4: "HUKE",
    5: "DASHY",
    6: "SIMP",
    7: "04",
}

_SOURCE: str | None = None
_INFO: video.VideoInfo | None = None


def _init(source: str) -> None:
    global _SOURCE, _INFO
    _SOURCE = source
    _INFO = video.probe(source)


def name_tokens(bgr: np.ndarray) -> list[np.ndarray]:
    """Attacker and victim name bitmaps from every well-formed row."""
    out: list[np.ndarray] = []
    for row in detect_rows(bgr):
        out.append(tight_crop(row.attacker.mask))
        out.append(tight_crop(row.victim.mask))
    return out


def _one(ts: float):
    try:
        bgr = video.frame_at(_SOURCE, ts, _INFO)
    except Exception:
        return []
    return name_tokens(bgr)


def live_timestamps(timeline: Path) -> list[float]:
    stamps: list[float] = []
    for s in json.loads(timeline.read_text()):
        if s["state"] != "live":
            continue
        t = s["start"] + 1
        while t < s["end"] - 1:
            stamps.append(t)
            t += 1
    return stamps


def cluster(masks: list[np.ndarray]) -> list[tuple[np.ndarray, int]]:
    """Group identical bitmaps, most frequent first."""
    buckets: Counter[tuple[int, int, bytes]] = Counter()
    for m in masks:
        buckets[(m.shape[0], m.shape[1], m.tobytes())] += 1
    out = []
    for (h, w, blob), n in buckets.most_common():
        out.append((np.frombuffer(blob, dtype=bool).reshape(h, w), n))
    return out


def contact_sheet(clusters, path: Path, limit: int = 24) -> None:
    tiles = []
    for i, (mask, n) in enumerate(clusters[:limit]):
        img = (mask.astype(np.uint8) * 255)
        img = cv2.copyMakeBorder(img, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=0)
        img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.putText(img, f"{i}:{n}", (2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 255, 0), 1)
        tiles.append(img)
    if not tiles:
        return
    h = max(t.shape[0] for t in tiles)
    w = max(t.shape[1] for t in tiles)
    canvas = np.zeros((h * len(tiles), w, 3), np.uint8)
    for i, t in enumerate(tiles):
        canvas[i * h : i * h + t.shape[0], : t.shape[1]] = t
    cv2.imwrite(str(path), canvas)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--timeline", default="data/state_timeline.json")
    ap.add_argument("--samples", type=int, default=600)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--survey", action="store_true",
                    help="cluster and write a contact sheet instead of saving")
    ap.add_argument("--sheet", default="data/killfeed_names.png")
    ap.add_argument("--out", default=ROOT / "src/cdlvision/config/killfeed_names.json")
    args = ap.parse_args()

    stamps = live_timestamps(ROOT / args.timeline)
    random.Random(args.seed).shuffle(stamps)
    stamps = sorted(stamps[: args.samples])

    masks: list[np.ndarray] = []
    with Pool(args.workers, initializer=_init, initargs=(args.source,)) as pool:
        for got in pool.imap(_one, stamps, chunksize=4):
            masks.extend(got)

    clusters = cluster(masks)
    print(f"{len(masks)} name tokens from {len(stamps)} frames "
          f"-> {len(clusters)} distinct bitmaps")
    total = sum(n for _, n in clusters)
    covered = sum(n for _, n in clusters[:8])
    print(f"top 8 clusters cover {covered}/{total} ({100 * covered / total:.1f}%)")
    for i, (m, n) in enumerate(clusters[:16]):
        print(f"  {i:2d}  {n:4d}  {m.shape[1]:3d}x{m.shape[0]:2d}"
              f"  {LABELLED.get(i, '?')}")

    if args.survey:
        sheet = ROOT / args.sheet
        sheet.parent.mkdir(parents=True, exist_ok=True)
        contact_sheet(clusters, sheet)
        print(f"\nwrote contact sheet -> {sheet}")
        return 0

    if not LABELLED:
        print("\nLABELLED is empty; run with --survey first")
        return 1

    templates = TemplateSet("killfeed_name")
    for index, name in LABELLED.items():
        if index >= len(clusters):
            print(f"  ! cluster {index} does not exist")
            return 1
        templates.add(name, clusters[index][0])
    templates.save(args.out)
    print(f"\nwrote {len(templates)} templates -> "
          f"{Path(args.out).relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
