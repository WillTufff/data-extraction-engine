"""Cache a fixed region of the VOD as a lossless strip, in one decode pass.

Decode cost does not depend on sampling rate, so the strip is written densely
and the sampling decision is deferred to whatever reads it.

    uv run python tools/build_strip.py <video> --region killfeed --fps 12

The strip covers the whole VOD rather than only the LIVE segments, which
keeps `timestamp = start + index / fps` exactly true and survives a retune of
the state timeline.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cdlvision import killfeed, scoreboard, spectated, video  # noqa: E402
from cdlvision.progress import Progress  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# Region -> crop(s), derived from the reader that consumes them. A region can
# map to multiple crops sharing one decode pass (e.g. the scoreboard's three
# boxes at the frame's far edges).
REGIONS = {
    "killfeed": lambda: {"killfeed": killfeed.strip_crop()},
    "scoreboard": scoreboard.strip_crops,
    "spectated": lambda: {"spectated": spectated.strip_crop()},
}

DEFAULT_FPS = {"killfeed": 12.0, "scoreboard": 6.0, "spectated": 6.0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--region", choices=sorted(REGIONS), default="killfeed")
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    crops = REGIONS[args.region]()
    fps = args.fps if args.fps is not None else DEFAULT_FPS[args.region]
    outdir = Path(args.outdir) if args.outdir else ROOT / "data"
    outs = {outdir / f"strip_{name}.mp4": crop for name, crop in crops.items()}

    info = video.probe(args.source)
    span = (args.end if args.end is not None else info.duration) - args.start
    print(f"source   {info.width}x{info.height} @ {info.fps:.3f}fps, {info.duration:.0f}s")
    print(f"region   {args.region}, {len(outs)} crop(s)")
    for out, crop in outs.items():
        print(f"         {crop.filter:32s} -> {out}")
    print(f"         {fps}fps over {span:.0f}s -> ~{span * fps:.0f} frames each")

    began = time.monotonic()
    with Progress(
        f"strip_{args.region}",
        total=video.span_frames(info, fps, args.start, args.end),
        label=f"strip {args.region}",
    ) as bar:
        strips = video.extract_strips(
            args.source, outs, fps=fps,
            start=args.start, end=args.end, info=info,
            on_frame=lambda done, _total: bar.set(done),
        )
    elapsed = time.monotonic() - began

    total_mb = 0.0
    for out, strip in strips.items():
        size_mb = out.stat().st_size / 1024 / 1024
        total_mb += size_mb
        print(f"\n{out.name}: {strip.frames} frames, {size_mb:.0f} MB")
    print(f"\n{total_mb:.0f} MB total in {elapsed / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
