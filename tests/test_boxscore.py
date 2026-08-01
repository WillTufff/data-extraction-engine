"""Validation of the box-score parser.

Fixture frames are checked field by field against hand transcription;
held-out frames with no transcription are checked against game invariants.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from cdlvision import video
from cdlvision.boxscore import (
    MODE_SCHEMAS, NON_NUMERIC, detect_grid, detect_mode, parse,
)
from cdlvision.glyphs import GlyphAtlas, TemplateSet
from cdlvision.validate import run_all

ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = ROOT / "tests/fixtures"
ATLAS = ROOT / "src/cdlvision/config/atlas_boxscore.json"
MODES = ROOT / "src/cdlvision/config/modes.json"

FIXTURES = sorted(FIXTURE_DIR.glob("boxscore_t*.json"))

# Box scores located by tools/find_boxscores.py that have no transcription.
HELD_OUT = [760.0, 2800.0, 7350.0, 8230.0]


def _source() -> Path:
    sources = sorted(ROOT.glob("*.mp4"))
    if not sources:
        pytest.skip("no VOD present")
    return sources[0]


@pytest.fixture(scope="module")
def atlas() -> GlyphAtlas:
    return GlyphAtlas.load(ATLAS)


@pytest.fixture(scope="module")
def modes() -> TemplateSet:
    return TemplateSet.load(MODES)


@pytest.fixture(scope="module")
def info():
    return video.probe(_source())


def _gray(info, ts: float) -> np.ndarray:
    return cv2.cvtColor(video.frame_at(info.path, ts, info), cv2.COLOR_BGR2GRAY)


def test_atlas_covers_expected_charset(atlas):
    assert set(atlas.glyphs) == set("0123456789:./%")


def test_mode_templates_present(modes):
    assert set(modes.templates) == set(MODE_SCHEMAS)


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda p: p.stem)
def test_fixture_fields_match_ground_truth(fixture, atlas, modes, info):
    spec = json.loads(fixture.read_text())
    gray = _gray(info, spec["timestamp"])

    assert detect_mode(gray, modes).char == spec["mode"]

    grid = detect_grid(gray)
    assert grid.n_rows == 8
    assert grid.row_height == 23
    for block in (grid.rows[:4], grid.rows[4:]):
        assert {b - a for a, b in zip(block, block[1:])} == {60}

    result = parse(gray, atlas, mode=spec["mode"], grid=grid)
    errors = [
        f"{want['name']}.{name}: got {got.text(name)!r} want {want[name]!r}"
        for got, want in zip(result.players, spec["rows"])
        for name in MODE_SCHEMAS[spec["mode"]]
        if name not in NON_NUMERIC and got.text(name) != want[name]
    ]
    assert not errors, "\n".join(errors)


@pytest.mark.parametrize("ts", HELD_OUT, ids=lambda t: f"t{int(t)}")
def test_held_out_frames_satisfy_invariants(ts, atlas, modes, info):
    gray = _gray(info, ts)
    mode = detect_mode(gray, modes).char
    result = parse(gray, atlas, mode=mode, grid=detect_grid(gray))

    failures = [c for c in run_all(result) if not c.passed]
    assert not failures, "\n".join(f"{c.name}: {c.detail}" for c in failures)


@pytest.mark.parametrize("ts", HELD_OUT + [4920.0, 3940.0], ids=lambda t: f"t{int(t)}")
def test_confidence_separation(ts, atlas, modes, info):
    """A correct match must sit far from the runner-up."""
    gray = _gray(info, ts)
    mode = detect_mode(gray, modes).char
    result = parse(gray, atlas, mode=mode, grid=detect_grid(gray))

    matches = [m for p in result.players for f in p.fields.values() for m in f.matches]
    assert matches
    assert max(m.distance for m in matches) <= 12
    assert min(m.margin for m in matches) >= 12
