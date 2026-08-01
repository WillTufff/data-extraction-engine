"""Classify sampled frames and montage them grouped by predicted state.

Usage:
    uv run python tools/classify_sweep.py <video> [--step 60] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cdlvision import video  # noqa: E402
from cdlvision.glyphs import TemplateSet  # noqa: E402
from cdlvision.state import State, classify  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

_SOURCE: str | None = None
_INFO: video.VideoInfo | None = None
_TAGS: TemplateSet | None = None


def _init(source: str) -> None:
    global _SOURCE, _INFO, _TAGS
    _SOURCE = source
    _INFO = video.probe(source)
    _TAGS = TemplateSet.load(ROOT / "src/cdlvision/config/tags.json")


def _one(ts: float):
    try:
        bgr = video.frame_at(_SOURCE, ts, _INFO)
    except Exception:
        return ts, None, None
    state, probes = classify(bgr, tags=_TAGS)
    return ts, state.value, {
        "name_rows": probes.name_rows,
        "name_pitches": list(probes.name_pitches),
        "chrome": probes.chrome,
        "tag": probes.tag,
        "tag_distance": probes.tag_distance,
        "series_orange": round(probes.series_orange, 4),
        "boxscore_columns": probes.boxscore_columns,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--step", type=float, default=60.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None, help="directory for montages + JSON")
    args = ap.parse_args()

    info = video.probe(args.source)
    stamps = [i * args.step for i in range(int(info.duration / args.step))]
    print(f"classifying {len(stamps)} frames every {args.step}s", flush=True)

    results = []
    with Pool(args.workers, initializer=_init, initargs=(args.source,)) as pool:
        for ts, state, probes in pool.imap(_one, stamps, chunksize=4):
            if state is not None:
                results.append({"t": ts, "state": state, "probes": probes})

    counts: dict[str, int] = defaultdict(int)
    for r in results:
        counts[r["state"]] += 1
    print("\npredicted state counts:")
    for s in State:
        print(f"  {s.value:9s} {counts.get(s.value, 0):4d}")

    if not args.out:
        return 0

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "sweep.json").write_text(json.dumps(results, indent=1) + "\n")

    by_state: dict[str, list[float]] = defaultdict(list)
    for r in results:
        by_state[r["state"]].append(r["t"])

    for state, times in by_state.items():
        tiles = out / f"tiles_{state}"
        tiles.mkdir(exist_ok=True)
        for f in tiles.glob("*.png"):
            f.unlink()
        for i, ts in enumerate(times):
            subprocess.run(
                ["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{ts:.3f}",
                 "-i", args.source, "-frames:v", "1",
                 "-vf", "scale=426:240", str(tiles / f"{i:04d}.png"), "-y"],
                check=True,
            )
        cols = 6
        subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error",
             "-pattern_type", "glob", "-i", str(tiles / "*.png"),
             "-vf", f"tile={cols}x{(len(times) + cols - 1) // cols}",
             "-frames:v", "1", str(out / f"montage_{state}.png"), "-y"],
            check=True,
        )
        print(f"  {state}: {len(times)} frames -> montage_{state}.png "
              f"({cols} per row, order = ascending timestamp)")
        print(f"     timestamps: {[int(t) for t in times]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
