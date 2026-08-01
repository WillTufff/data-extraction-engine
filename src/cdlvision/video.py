"""Frame access for broadcast VODs.

Grab single frames by timestamp, or stream a sampled sequence for bulk
extraction. Frames are piped as raw BGR24 and never written to disk.

A strip is a small fixed crop of the source, written once as a lossless
video; everything downstream reads the strip instead of re-decoding the VOD.
Strips are encoded in the source's own pixel format (yuv420p) to stay
byte-identical to the source.

Strip timestamps are approximate, within about 34ms of `frame_at` at the
same nominal time, and not guaranteed to reference the same frame.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: float
    duration: float

    @property
    def frame_count(self) -> int:
        return int(self.duration * self.fps)


def probe(path: str | Path) -> VideoInfo:
    """Read stream metadata via ffprobe."""
    path = Path(path)
    out = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate",
            "-show_entries", "format=duration",
            "-of", "json", str(path),
        ],
        capture_output=True, text=True, check=True,
    ).stdout
    data = json.loads(out)
    stream = data["streams"][0]
    num, den = stream["r_frame_rate"].split("/")
    return VideoInfo(
        path=path,
        width=int(stream["width"]),
        height=int(stream["height"]),
        fps=int(num) / int(den),
        duration=float(data["format"]["duration"]),
    )


def frame_at(path: str | Path, timestamp: float, info: VideoInfo | None = None) -> np.ndarray:
    """Decode a single frame at `timestamp` seconds as a BGR uint8 array.

    Uses input seeking, so cost is roughly one keyframe interval regardless of
    how far into the file the timestamp sits.
    """
    info = info or probe(path)
    proc = subprocess.run(
        [
            "ffmpeg", "-nostdin", "-v", "error",
            "-ss", f"{timestamp:.3f}",
            "-i", str(path),
            "-frames:v", "1",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
        ],
        capture_output=True, check=True,
    )
    expected = info.width * info.height * 3
    if len(proc.stdout) < expected:
        raise ValueError(
            f"short read at t={timestamp}: got {len(proc.stdout)} bytes, want {expected}"
        )
    return np.frombuffer(proc.stdout[:expected], dtype=np.uint8).reshape(
        info.height, info.width, 3
    )


def stream_frames(
    path: str | Path,
    fps: float,
    start: float = 0.0,
    end: float | None = None,
    info: VideoInfo | None = None,
) -> Iterator[tuple[float, np.ndarray]]:
    """Yield `(timestamp, frame)` at a sampled rate from a single decode pass.

    One linear pass is dramatically cheaper than seeking per frame, so bulk
    extraction should always come through here rather than repeated `frame_at`.
    """
    info = info or probe(path)
    cmd = ["ffmpeg", "-nostdin", "-v", "error"]
    if start:
        cmd += ["-ss", f"{start:.3f}"]
    if end is not None:
        cmd += ["-to", f"{end:.3f}"]
    cmd += [
        "-i", str(path),
        "-vf", f"fps={fps}",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
    ]

    nbytes = info.width * info.height * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=nbytes * 4)
    try:
        index = 0
        while True:
            buf = proc.stdout.read(nbytes)
            if len(buf) < nbytes:
                break
            frame = np.frombuffer(buf, dtype=np.uint8).reshape(info.height, info.width, 3)
            yield start + index / fps, frame
            index += 1
    finally:
        proc.stdout.close()
        proc.wait()


@dataclass(frozen=True)
class Crop:
    """A region of the source frame, in source pixel coordinates.

    Offsets and extents must be even, since yuv420p subsamples chroma.
    """

    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        for name in ("x", "y", "width", "height"):
            value = getattr(self, name)
            if value % 2:
                raise ValueError(f"crop {name} must be even for yuv420p, got {value}")

    @property
    def filter(self) -> str:
        return f"crop={self.width}:{self.height}:{self.x}:{self.y}"

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Cut this region out of a full source frame."""
        return frame[self.y : self.y + self.height, self.x : self.x + self.width]


@dataclass(frozen=True)
class StripInfo:
    """A cached strip and the geometry needed to interpret it."""

    path: Path
    source: str
    crop: Crop
    fps: float
    start: float
    frames: int

    def timestamp(self, index: int) -> float:
        return self.start + index / self.fps


def _sidecar(path: Path) -> Path:
    return path.with_suffix(".json")


def span_frames(info: VideoInfo, fps: float, start: float, end: float | None) -> int:
    """How many frames a strip over this span should hold."""
    return int(round(((end if end is not None else info.duration) - start) * fps))


def _run_with_progress(cmd: list[str], on_frame, total: int) -> None:
    """Run ffmpeg, forwarding its `-progress` frame counter to `on_frame`."""
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    try:
        for line in proc.stdout:
            key, _, value = line.strip().partition("=")
            if key == "frame" and on_frame is not None:
                try:
                    on_frame(int(value), total)
                except ValueError:
                    pass
    finally:
        proc.stdout.close()
        stderr = proc.stderr.read()
        proc.stderr.close()
        code = proc.wait()
    if code != 0:
        raise subprocess.CalledProcessError(code, cmd, stderr=stderr)


def extract_strip(
    source: str | Path,
    out: str | Path,
    crop: Crop,
    fps: float,
    start: float = 0.0,
    end: float | None = None,
    info: VideoInfo | None = None,
    on_frame=None,
) -> StripInfo:
    """Write `crop` from `source` as a lossless strip, plus a JSON sidecar.

    The sidecar records the crop origin, rate and source so strip coordinates
    can be mapped back to full-frame ones.
    """
    source, out = Path(source), Path(out)
    info = info or probe(source)
    if crop.x + crop.width > info.width or crop.y + crop.height > info.height:
        raise ValueError(f"crop {crop} does not fit in {info.width}x{info.height}")
    out.parent.mkdir(parents=True, exist_ok=True)

    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y"]
    if start:
        cmd += ["-ss", f"{start:.3f}"]
    if end is not None:
        cmd += ["-to", f"{end:.3f}"]
    cmd += [
        "-i", str(source),
        "-vf", f"fps={fps},{crop.filter}",
        # Lossless, in the source's own pixel format.
        "-c:v", "libx264", "-qp", "0", "-preset", "medium", "-pix_fmt", "yuv420p",
        "-an",
        "-progress", "pipe:1", "-nostats",
        str(out),
    ]
    _run_with_progress(cmd, on_frame, span_frames(info, fps, start, end))

    strip = StripInfo(
        path=out,
        source=str(source),
        crop=crop,
        fps=fps,
        start=start,
        frames=_count_frames(out),
    )
    _write_sidecar(strip)
    return strip


def extract_strips(
    source: str | Path,
    regions: dict[str | Path, Crop],
    fps: float,
    start: float = 0.0,
    end: float | None = None,
    info: VideoInfo | None = None,
    on_frame=None,
) -> dict[Path, StripInfo]:
    """Write several crops of `source` as strips, in a single decode pass."""
    source = Path(source)
    info = info or probe(source)
    outs = {Path(p): c for p, c in regions.items()}
    for crop in outs.values():
        if crop.x + crop.width > info.width or crop.y + crop.height > info.height:
            raise ValueError(f"crop {crop} does not fit in {info.width}x{info.height}")

    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y"]
    if start:
        cmd += ["-ss", f"{start:.3f}"]
    if end is not None:
        cmd += ["-to", f"{end:.3f}"]
    cmd += ["-i", str(source)]

    labels = [f"s{i}" for i in range(len(outs))]
    graph = [f"[0:v]fps={fps},split={len(outs)}" + "".join(f"[{n}]" for n in labels)]
    for name, crop in zip(labels, outs.values()):
        graph.append(f"[{name}]{crop.filter}[o{name}]")
    cmd += ["-filter_complex", ";".join(graph)]
    for name, out in zip(labels, outs):
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd += [
            "-map", f"[o{name}]",
            "-c:v", "libx264", "-qp", "0", "-preset", "medium", "-pix_fmt", "yuv420p",
            "-an", str(out),
        ]
    cmd += ["-progress", "pipe:1", "-nostats"]
    _run_with_progress(cmd, on_frame, span_frames(info, fps, start, end))

    written: dict[Path, StripInfo] = {}
    for out, crop in outs.items():
        strip = StripInfo(
            path=out, source=str(source), crop=crop,
            fps=fps, start=start, frames=_count_frames(out),
        )
        _write_sidecar(strip)
        written[out] = strip
    return written


def _write_sidecar(strip: StripInfo) -> None:
    c = strip.crop
    _sidecar(strip.path).write_text(
        json.dumps(
            {
                "source": strip.source,
                "crop": {"x": c.x, "y": c.y, "width": c.width, "height": c.height},
                "fps": strip.fps,
                "start": strip.start,
                "frames": strip.frames,
            },
            indent=2,
        )
        + "\n"
    )


def _count_frames(path: Path) -> int:
    out = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-count_frames", "-show_entries", "stream=nb_read_frames",
            "-of", "default=nw=1:nk=1", str(path),
        ],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return int(out)


def load_strip(path: str | Path) -> StripInfo:
    """Read a strip's sidecar."""
    path = Path(path)
    data = json.loads(_sidecar(path).read_text())
    return StripInfo(
        path=path,
        source=data["source"],
        crop=Crop(**data["crop"]),
        fps=data["fps"],
        start=data["start"],
        frames=data["frames"],
    )


def stream_strip(
    strip: StripInfo, first: int = 0, count: int | None = None
) -> Iterator[tuple[float, np.ndarray]]:
    """Yield `(timestamp, crop)` for a strip, or for a frame range of one.

    The range lets a strip be read by several processes at once.
    """
    nbytes = strip.crop.width * strip.crop.height * 3
    cmd = ["ffmpeg", "-nostdin", "-v", "error"]
    if first:
        cmd += ["-ss", f"{first / strip.fps:.6f}"]
    cmd += ["-i", str(strip.path)]
    if count is not None:
        cmd += ["-frames:v", str(count)]
    cmd += ["-f", "rawvideo", "-pix_fmt", "bgr24", "-"]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=nbytes * 8)
    try:
        index = first
        while True:
            buf = proc.stdout.read(nbytes)
            if len(buf) < nbytes:
                break
            yield strip.timestamp(index), np.frombuffer(buf, dtype=np.uint8).reshape(
                strip.crop.height, strip.crop.width, 3
            )
            index += 1
    finally:
        proc.stdout.close()
        proc.wait()


class Canvas:
    """Pastes a strip back into full-frame coordinates.

    Lets readers address absolute source coordinates regardless of what was
    cropped. The buffer is reused across frames.
    """

    def __init__(self, crop: Crop, width: int, height: int) -> None:
        self.crop = crop
        self._buf = np.zeros((height, width, 3), dtype=np.uint8)

    def place(self, strip_frame: np.ndarray) -> np.ndarray:
        """Return a full-frame view with `strip_frame` at its source position.

        The returned array is reused on the next call; copy it to keep it.
        """
        c = self.crop
        self._buf[c.y : c.y + c.height, c.x : c.x + c.width] = strip_frame
        return self._buf
