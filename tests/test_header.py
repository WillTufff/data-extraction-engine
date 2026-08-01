"""Tests for the centre header reader.

Covers the atlases (score, clock, rotation digits), reading against
hand-transcribed fixtures, and the structural checks (band bounds, token
gating). None is a valid reading in several places -- the mode strip during a
kill notification, the substrip in Hardpoint, the rotation timer outside
Hardpoint -- and those are pinned by the fixtures too.

Tests needing the VOD skip without it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cdlvision import header as hd  # noqa: E402
from cdlvision import video  # noqa: E402

SOURCE = next(ROOT.glob("*.mp4"), None)
needs_vod = pytest.mark.skipif(SOURCE is None, reason="source VOD not present")

FIXTURES = sorted((ROOT / "tests/fixtures").glob("scoreboard_t*.json"))
CONFIG = ROOT / "src/cdlvision/config"


@pytest.fixture(scope="module")
def reader():
    return hd.Reader.load()


@pytest.fixture(scope="module")
def info():
    return video.probe(SOURCE) if SOURCE else None


@pytest.fixture(scope="module")
def header_runs():
    from cdlvision import scoreboard_runs

    return scoreboard_runs.load(ROOT / "tests/fixtures/header_runs.jsonl.gz")


def _spec(path: Path) -> dict:
    return json.loads(path.read_text())


# -- the atlases ---------------------------------------------------------


def test_atlases_cover_every_digit(reader):
    """All ten digits, in all three cuts of the font."""
    for atlas in (reader.score, reader.clock, reader.rotation):
        assert set("0123456789") <= set(atlas.glyphs)


def test_only_the_countdowns_carry_a_colon(reader):
    """The colon is identified by position, so it must land only where it can."""
    assert ":" not in reader.score.glyphs
    assert ":" in reader.clock.glyphs
    assert ":" in reader.rotation.glyphs


def test_label_templates_are_the_closed_set(reader):
    """Mode titles and substrip titles, and nothing else."""
    labels = set(reader.labels.templates)
    assert set(hd.MODES) <= labels
    assert {"FIRST HALF", "SECOND HALF", "LIVES REMAINING"} <= labels


# -- the fixtures --------------------------------------------------------


@needs_vod
@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_header_matches_the_fixture(path, reader, info):
    """Every header field, against a hand transcription."""
    spec = _spec(path)
    frame = video.frame_at(SOURCE, spec["timestamp"], info)
    header = reader.read(frame)

    assert [header.score_left, header.score_right] == spec["score"]
    assert header.clock == spec["clock"]
    assert [header.maps_left, header.maps_right] == spec["maps"]
    assert header.rotation == spec["rotation"]
    assert header.mode == spec["mode"]
    assert header.substrip == spec["substrip"]
    assert header.best_of == spec["best_of"]
    # Lives are None outside Search and Destroy.
    assert [header.lives_left, header.lives_right] == (spec["lives"] or [None, None])


@needs_vod
@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_the_pip_bands_agree_on_the_best_of(path, reader, info):
    """Both bands draw the same series, so they must count the same blocks."""
    spec = _spec(path)
    frame = video.frame_at(SOURCE, spec["timestamp"], info)
    left = hd.read_pips(frame, "pips_left")
    right = hd.read_pips(frame, "pips_right")
    assert left is not None and right is not None
    assert left[1] == right[1] == (spec["best_of"] + 1) // 2


@needs_vod
def test_hardpoint_draws_no_substrip(reader, info):
    """Absence is a reading, and must not come back as a guess."""
    for spec in (_spec(p) for p in FIXTURES):
        if spec["mode"] != "CDL HARDPOINT":
            continue
        frame = video.frame_at(SOURCE, spec["timestamp"], info)
        assert reader.read_substrip(frame)[0] is None


@needs_vod
def test_the_mode_strip_is_not_always_naming_the_mode(reader, info):
    """The mode strip can show a notification or event title instead; t=3300 does."""
    frame = video.frame_at(SOURCE, 3300, info)
    assert reader.read_mode(frame)[0] is None


# -- structure -----------------------------------------------------------


def test_a_colon_fails_the_flat_height_gate():
    """`tokens` rejects a band with a colon; `clock_tokens` must be used instead."""
    mask = np.zeros((18, 40), dtype=bool)
    mask[0:18, 0:13] = True    # a digit, full height
    mask[6:17, 16:18] = True   # a colon, 11px of an 18px band
    mask[0:18, 21:34] = True   # another digit
    assert hd.tokens(mask, hd.CLOCK_HEIGHT) is None

    mask4 = np.zeros((18, 52), dtype=bool)
    mask4[0:18, 0:13] = True
    mask4[6:17, 16:18] = True
    mask4[0:18, 21:34] = True
    mask4[0:18, 37:50] = True
    got = hd.clock_tokens(mask4, hd.CLOCK_HEIGHT)
    assert got is not None and len(got) == 4
    assert got[1].shape[1] <= hd.COLON_MAX_WIDTH


def test_a_countdown_must_have_four_tokens():
    """"3:4" is not a clock."""
    mask = np.zeros((18, 34), dtype=bool)
    mask[0:18, 0:13] = True
    mask[6:17, 16:18] = True
    mask[0:18, 21:34] = True
    assert hd.clock_tokens(mask, hd.CLOCK_HEIGHT) is None


def test_seconds_parse_only_as_two_digits():
    assert hd.seconds("3:43") == 223
    assert hd.seconds("0:04") == 4
    assert hd.seconds("3:4") is None
    assert hd.seconds("343") is None
    assert hd.seconds(None) is None
    # The final thirty seconds are drawn as a different counter entirely.
    assert hd.seconds("28.5") == 28.5
    assert hd.seconds("6.5") == 6.5
    assert hd.seconds("0.0") == 0
    assert hd.seconds("28.55") is None
    assert hd.seconds(".5") is None


def test_the_header_bands_at_most_abut():
    """Bands may share a boundary pixel, but must not genuinely overlap."""
    from cdlvision.scoreboard import HEADER_REGIONS

    for a, (ax0, ax1, ay0, ay1) in HEADER_REGIONS.items():
        assert ax0 < ax1 and ay0 < ay1, a
        assert 0 <= ax0 and ax1 < 1920 and 0 <= ay0 and ay1 < 1080, a
        for b, (bx0, bx1, by0, by1) in HEADER_REGIONS.items():
            if a >= b:
                continue
            wide = min(ax1, bx1) - max(ax0, bx0) + 1
            tall = min(ay1, by1) - max(ay0, by0) + 1
            assert wide < 2 or tall < 2, f"{a} overlaps {b} by {wide}x{tall}"


def test_the_final_thirty_seconds_are_read(header_runs):
    """The clock below 0:30 is a different counter -- red, seconds and tenths,
    a decimal point for a colon. It read as None everywhere until the reader
    learned it, and absence looked legitimate because a clock that is not drawn
    is exactly what the end of a round looks like."""
    clock = [r for r in header_runs if r["field"] == "clock"]
    values = [hd.seconds(r["value"]) for r in clock]
    below = [v for v in values if v is not None and v < 30]
    unread = sum(r["samples"] for r in clock if r["value"] is None)
    assert len(below) > 2000
    assert unread < 100
    # Every sub-30 reading is the tenths form, and every tenths form is sub-30.
    for run in clock:
        if isinstance(run["value"], str) and "." in run["value"]:
            assert hd.seconds(run["value"]) < 30


@needs_vod
def test_the_red_counter_reads_and_the_white_one_is_untouched(reader, info):
    """Both formats off real frames. The white readings are the regression
    guard: the red path is chosen by colour before any matching, so it must
    never see a white band."""
    for t, want in ((1958.0, "28.5"), (1962.0, "24.5"), (1980.0, "6.5"),
                    (1986.0, "0.5"), (6545.0, "22.2")):
        frame = video.frame_at(SOURCE, t, info)
        assert reader.read_clock(frame)[0] == want, t
    for t, want in ((1950.8, "0:35"), (2450.0, "0:38"), (100.0, "4:27"),
                    (6600.0, "1:24")):
        frame = video.frame_at(SOURCE, t, info)
        assert reader.read_clock(frame)[0] == want, t


@needs_vod
def test_redness_separates_the_two_formats_with_room_to_spare(info):
    """-12..-5 for white against up to 98 for red, and the cut at 40 sits in
    the gap. Colour chooses the reader; the bitmaps still decide the value."""
    white = [1950.8, 2450.0, 100.0, 4200.0, 3300.0]
    red = [1958.0, 1962.0, 1980.0, 6545.0, 9645.0]
    got = {}
    for t in white + red:
        band = hd.region(video.frame_at(SOURCE, t, info), "clock")
        got[t] = hd.redness(band)
    assert max(got[t] for t in white) < hd.REDNESS_CUT
    assert min(got[t] for t in red) > hd.REDNESS_CUT
