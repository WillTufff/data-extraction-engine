"""Tests for the live scoreboard reader.

Covers panel geometry, glyph segmentation, row polarity, and reading against
hand-transcribed fixtures. Tests needing the VOD skip without it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cdlvision import scoreboard as sb  # noqa: E402
from cdlvision import video  # noqa: E402

SOURCE = next(ROOT.glob("*.mp4"), None)
needs_vod = pytest.mark.skipif(SOURCE is None, reason="source VOD not present")

FIXTURES = sorted((ROOT / "tests/fixtures").glob("scoreboard_t*.json"))


@pytest.fixture(scope="module")
def reader():
    return sb.Reader.load()


@pytest.fixture(scope="module")
def info():
    return video.probe(SOURCE) if SOURCE else None


# -- geometry ------------------------------------------------------------


def test_panels_are_an_exact_translation():
    """Every column of the right panel is MIRROR px from the left one."""
    for key in sb.COLUMNS:
        left = sb.LEFT.column(key)
        right = sb.RIGHT.column(key)
        assert right[0] - left[0] == sb.MIRROR
        assert right[1] - left[1] == sb.MIRROR


def test_columns_do_not_overlap():
    ordered = sorted(sb.COLUMNS.values())
    for (_, end), (start, _) in zip(ordered, ordered[1:]):
        assert start > end, f"columns overlap at x={end}"


def test_row_grid_is_evenly_pitched():
    assert sb.ROWS == (105, 137, 169, 201)
    assert all(b - a == sb.ROW_PITCH for a, b in zip(sb.ROWS, sb.ROWS[1:]))
    # The label row is above the grid and not on it.
    assert sb.LABEL_TOP < sb.ROWS[0]
    assert (sb.ROWS[0] - sb.LABEL_TOP) % sb.ROW_PITCH != 0


def test_strip_crops_cover_every_column_and_row():
    crops = sb.strip_crops()
    for side, name in ((sb.LEFT, "scoreboard_left"), (sb.RIGHT, "scoreboard_right")):
        crop = crops[name]
        for key in sb.COLUMNS:
            x0, x1 = side.column(key)
            assert crop.x <= x0 and x1 < crop.x + crop.width, key
        top = sb.ROWS[-1] + sb.ROW_HEIGHT + sb.GUTTER_OVERRUN[1]
        assert crop.y <= sb.LABEL_TOP and top <= crop.y + crop.height


# -- segmentation --------------------------------------------------------


def _mask(rows: list[str]) -> np.ndarray:
    return np.array([[c == "#" for c in row] for row in rows], dtype=bool)


def test_segment_cell_splits_touching_glyphs():
    """A token too wide to be one glyph is cut at its thinnest column."""
    # Two glyphs, 18px wide together, joined by a single-pixel bridge at x=9.
    merged = _mask([
        "##########....####",
        "#........#...##...",
        "#........#..##....",
        "#........####.....",
        "#........#..##....",
        "##########....####",
    ])
    assert merged.shape[1] > sb.MERGED_WIDTH
    spans = sb.segment_cell(merged)
    assert len(spans) == 2, spans
    assert spans[0][1] == spans[1][0], "the split must partition, not overlap"


def test_segment_cell_leaves_narrow_glyphs_alone():
    colon = _mask(["##", "##", "..", "..", "##", "##"])
    assert sb.segment_cell(colon) == [(0, 2)]


def test_token_spans_drop_thin_artefacts():
    """A one-pixel column of ink is the divider or noise, never a glyph."""
    mask = np.zeros((16, 40), dtype=bool)
    mask[:, 2:8] = True          # a real token, 96 ink pixels
    mask[:, 38] = True           # the column divider, 16 ink pixels
    assert sb.token_spans(mask) == [(2, 7)]


# -- polarity ------------------------------------------------------------


def test_row_ink_detects_both_polarities():
    dark = np.full((16, 60, 3), 35, dtype=np.uint8)
    dark[:, 10:20] = 250
    normal = sb.RowInk.measure(dark)
    assert not normal.inverted
    assert normal.apply(dark)[:, 10:20].all()
    assert not normal.apply(dark)[:, 0:5].any()

    bright = np.full((16, 60, 3), 235, dtype=np.uint8)
    bright[:, 10:20] = 10
    inverted = sb.RowInk.measure(bright)
    assert inverted.inverted
    assert inverted.apply(bright)[:, 10:20].all()
    assert not inverted.apply(bright)[:, 0:5].any()


# -- reading, against the hand-transcribed fixtures -----------------------


@needs_vod
@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda p: p.stem)
def test_fixture_rows_read_exactly(reader, info, fixture):
    spec = json.loads(fixture.read_text())
    frame = video.frame_at(SOURCE, spec["timestamp"], info)
    reading = reader.read(frame)

    assert reading.stat_label == spec["stat_label"]
    got = {(r.side, r.slot): r for r in reading.rows}
    for row in spec["rows"]:
        actual = got[(row["side"], row["slot"])]
        where = f"{row['side']}[{row['slot']}]"
        assert actual.number == int(row["number"]), where
        assert actual.name == row["name"], where
        assert f"{actual.kills}/{actual.deaths}" == row["kd"], where
        assert str(actual.streak) == row["strk"], where
        assert actual.stat == row["stat"], where
        assert actual.spectated == row["spectated"], where


@needs_vod
@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda p: p.stem)
def test_fixture_skulls_are_found(reader, info, fixture):
    """The skull marks a death; other gutter icons are checked only as "not a skull"."""
    spec = json.loads(fixture.read_text())
    frame = video.frame_at(SOURCE, spec["timestamp"], info)
    got = {(r.side, r.slot): r for r in reader.read(frame).rows}
    for row in spec["rows"]:
        actual = got[(row["side"], row["slot"])]
        where = f"{row['side']}[{row['slot']}]"
        assert actual.dead == (row["gutter"] == "skull"), where
        if row["gutter"] == "none":
            assert actual.gutter == sb.NO_ICON, where


@needs_vod
def test_mirror_holds_on_every_fixture(info):
    for fixture in FIXTURES:
        spec = json.loads(fixture.read_text())
        sb.check_mirror(video.frame_at(SOURCE, spec["timestamp"], info))


@needs_vod
def test_mirror_fails_loudly_when_a_panel_moves(info):
    """A layout change must raise, not parse into plausible wrong numbers."""
    spec = json.loads(FIXTURES[0].read_text())
    frame = video.frame_at(SOURCE, spec["timestamp"], info).copy()
    x0, x1 = sb.RIGHT.column("kd")
    band = frame[sb.LABEL_TOP : sb.LABEL_TOP + sb.ROW_HEIGHT, x0 - 8 : x1 + 1]
    frame[sb.LABEL_TOP : sb.LABEL_TOP + sb.ROW_HEIGHT, x0 : x1 + 9] = band
    with pytest.raises(sb.LayoutError):
        sb.check_mirror(frame)
