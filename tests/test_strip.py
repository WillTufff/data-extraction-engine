"""Tests for the strip cache.

Checks strip geometry, and that reading a strip is byte-identical to decoding
the VOD directly. Tests needing the VOD skip without it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cdlvision import killfeed, video  # noqa: E402

SOURCE = next(ROOT.glob("*.mp4"), None)
needs_vod = pytest.mark.skipif(SOURCE is None, reason="source VOD not present")


# -- geometry ------------------------------------------------------------


def test_crop_covers_every_slot_band():
    """The crop must contain every band `segment_row` reads."""
    crop = killfeed.strip_crop()
    for top in killfeed.SLOTS:
        assert crop.y <= top - 1
        assert crop.y + crop.height >= top + killfeed.ROW_HEIGHT[1] - 1
    assert crop.x <= killfeed.SEARCH_X[0]
    assert crop.x + crop.width >= killfeed.SEARCH_X[1]


def test_crop_is_even_for_yuv420p():
    crop = killfeed.strip_crop()
    assert (crop.x, crop.y, crop.width, crop.height) == tuple(
        v - v % 2 for v in (crop.x, crop.y, crop.width, crop.height)
    )


def test_odd_crop_is_rejected():
    """An odd crop cannot survive yuv420p."""
    with pytest.raises(ValueError):
        video.Crop(x=49, y=446, width=384, height=142)


def test_canvas_restores_source_coordinates():
    crop = video.Crop(x=48, y=446, width=384, height=142)
    canvas = video.Canvas(crop, 1920, 1080)
    strip_frame = np.random.randint(0, 256, (142, 384, 3), dtype=np.uint8)
    full = canvas.place(strip_frame)
    assert full.shape == (1080, 1920, 3)
    assert np.array_equal(crop.apply(full), strip_frame)


# -- fidelity ------------------------------------------------------------


@pytest.fixture(scope="module")
def strip(tmp_path_factory):
    out = tmp_path_factory.mktemp("strip") / "killfeed.mp4"
    return video.extract_strip(
        SOURCE, out, killfeed.strip_crop(), fps=12.0, start=0.0, end=20.0
    )


@needs_vod
def test_strip_roundtrip_is_byte_identical(strip):
    """The encode must not change a single pixel."""
    info = video.probe(strip.source)
    direct = [
        strip.crop.apply(f)
        for _, f in video.stream_frames(strip.source, fps=strip.fps, end=20.0, info=info)
    ]
    got = [f for _, f in video.stream_strip(strip)]

    assert len(got) == len(direct)
    assert all(np.array_equal(a, b) for a, b in zip(got, direct))


@needs_vod
def test_strip_frames_read_as_rows_identically(strip):
    """Row detection must agree, not just the raw pixels."""
    info = video.probe(strip.source)
    canvas = video.Canvas(strip.crop, info.width, info.height)
    direct = video.stream_frames(strip.source, fps=strip.fps, end=20.0, info=info)

    for (_, sf), (_, df) in zip(video.stream_strip(strip), direct):
        assert [r.signature for r in killfeed.detect_rows(canvas.place(sf))] == [
            r.signature for r in killfeed.detect_rows(df)
        ]


@needs_vod
def test_sidecar_roundtrips(strip):
    loaded = video.load_strip(strip.path)
    assert loaded == strip


@needs_vod
def test_strip_timestamps_stay_within_a_few_source_frames(strip):
    """Strip frames land slightly later than `frame_at`'s seek; bound the drift."""
    info = video.probe(strip.source)
    tolerance = 4 / info.fps

    frames = list(video.stream_strip(strip))
    for index in (12, 120, len(frames) - 1):
        ts, sf = frames[index]
        window = [
            strip.crop.apply(f)
            for _, f in video.stream_frames(
                strip.source, fps=info.fps,
                start=max(0.0, ts - tolerance), end=ts + tolerance, info=info,
            )
        ]
        assert any(np.array_equal(sf, w) for w in window), (
            f"strip frame at t={ts:.3f} matches no source frame within "
            f"{tolerance * 1000:.0f}ms"
        )
