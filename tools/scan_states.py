"""Classify every sampled frame of a VOD and cache the raw result.

The only step that touches the video; smoothing and validation are tuned by
re-reading this cache rather than re-decoding. Runs as one linear pass, and
stores the full probe output (not just the predicted state) so downstream
signatures can be re-derived from the cache.

Usage:
    uv run python tools/scan_states.py <video> [--fps 1.0] [--out data/states_raw.jsonl]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cdlvision import video  # noqa: E402
from cdlvision.glyphs import TemplateSet  # noqa: E402
from cdlvision.state import classify  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--fps", type=float, default=1.0, help="sampling rate in Hz")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--out", default="data/states_raw.jsonl")
    args = ap.parse_args()

    info = video.probe(args.source)
    tags = TemplateSet.load(ROOT / "src/cdlvision/config/tags.json")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    span = (args.end if args.end is not None else info.duration) - args.start
    expected = int(span * args.fps)
    print(
        f"scanning {info.duration:.0f}s at {args.fps} Hz -> ~{expected} frames",
        flush=True,
    )

    started = time.perf_counter()
    written = 0
    # Written incrementally so a long scan that dies partway is still usable
    # and its progress is inspectable while it runs.
    with out.open("w") as fh:
        for ts, bgr in video.stream_frames(
            args.source, args.fps, start=args.start, end=args.end, info=info
        ):
            state, p = classify(bgr, tags=tags)
            fh.write(
                json.dumps(
                    {
                        "t": round(ts, 3),
                        "state": state.value,
                        "name_rows": p.name_rows,
                        "name_pitches": list(p.name_pitches),
                        "chrome": p.chrome,
                        "tag": p.tag,
                        "tag_distance": p.tag_distance,
                        "series_orange": round(p.series_orange, 4),
                        "boxscore_columns": p.boxscore_columns,
                    }
                )
                + "\n"
            )
            written += 1
            if written % 250 == 0:
                fh.flush()
                rate = written / (time.perf_counter() - started)
                print(
                    f"  t={ts:7.1f}s  {written}/{expected}  {rate:.1f} frames/s",
                    flush=True,
                )

    elapsed = time.perf_counter() - started
    print(f"wrote {written} observations to {out} in {elapsed:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
