"""End-to-end checks over the spectated panel, scanned across the whole VOD.

Runs against the cached panel/scoreboard/header runs, without the video.
Checks the panel against independently-read regions -- the scoreboard for
the numbers, the roster for the team -- and that fields are read on nearly
every frame that draws them.
"""

from __future__ import annotations

import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pytest

from cdlvision import maps, scoreboard_runs
from cdlvision.glyphs import GlyphAtlas, TemplateSet
from cdlvision.names import NameMatcher

FIXTURES = Path(__file__).parent / "fixtures"
CONFIG = Path(__file__).parent.parent / "src" / "cdlvision" / "config"

STEP = 1 / 6
SHARED = {"name": "name", "kills": "kills", "deaths": "deaths",
          "streak": "streak", "stat": "stat"}


@pytest.fixture(scope="module")
def panel() -> dict[str, list[dict]]:
    with gzip.open(FIXTURES / "spectated_runs.jsonl.gz", "rt") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    by_field: dict[str, list[dict]] = defaultdict(list)
    for run in rows:
        by_field[run["field"]].append(run)
    for runs in by_field.values():
        runs.sort(key=lambda r: r["start"])
    return by_field


@pytest.fixture(scope="module")
def board() -> list[dict]:
    raw = scoreboard_runs.load(FIXTURES / "scoreboard_runs.jsonl.gz")
    ours, _ = scoreboard_runs.split_foreign(raw)
    header = scoreboard_runs.load(FIXTURES / "header_runs.jsonl.gz")
    ours, _ = maps.filter_to_series(ours, header)
    return ours


@pytest.fixture(scope="module")
def windows(board) -> list[maps.MapWindow]:
    header = scoreboard_runs.load(FIXTURES / "header_runs.jsonl.gz")
    return maps.derive(board, header)


@pytest.fixture(scope="module")
def paired(panel, board, windows) -> list[tuple[float, dict]]:
    """Sampled instants inside a map where the scoreboard marks a row spectated."""
    out = []
    for run in board:
        if not run["value"].get("spectated"):
            continue
        t = run["start"]
        while t <= run["end"]:
            if any(w.contains(t) for w in windows):
                out.append((t, run["value"]))
            t += STEP
    return out


def _at(runs: list[dict], t: float):
    for run in runs:
        if run["start"] <= t <= run["end"]:
            return run["value"]
    return None


@pytest.mark.parametrize("field", sorted(SHARED))
def test_panel_agrees_with_the_scoreboard(panel, paired, field):
    """The same number read twice, from regions 900px apart."""
    agree = total = 0
    for t, row in paired:
        want, got = row.get(SHARED[field]), _at(panel[field], t)
        if want is None or got is None:
            continue
        total += 1
        agree += str(want) == str(got)
    assert total > 20_000, f"only {total} comparisons for {field}"
    assert agree / total > 0.998, f"{field}: {agree}/{total}"


def test_the_bar_colour_and_the_name_never_disagree(panel, paired, board):
    """The bar's fill (team) and the name must agree with the scoreboard's side."""
    sides = {
        run["value"]["name"]: run["side"]
        for run in board if run["value"].get("name")
    }
    ok = bad = 0
    for t, _ in paired:
        team, name = _at(panel["team"], t), _at(panel["name"], t)
        if team is None or name is None or name not in sides:
            continue
        ok += team == sides[name]
        bad += team != sides[name]
    assert ok > 20_000
    assert bad == 0, f"{bad} frames put the colour and the name on opposite sides"


def test_every_field_is_read_on_nearly_every_frame_that_draws_one(panel):
    """Read rate per field, against `name`'s count of frames with a panel drawn."""
    drawn = sum(r["samples"] for r in panel["name"] if r["value"] is not None)
    assert drawn > 24_000
    for field, floor in (("hp", 0.99), ("kills", 0.99), ("deaths", 0.99),
                         ("streak", 0.95), ("stat", 0.99),
                         ("stat_label", 0.99), ("team", 0.99)):
        read = sum(r["samples"] for r in panel[field] if r["value"] is not None)
        assert read / drawn >= floor, f"{field}: {read}/{drawn}"


def test_hp_stays_inside_its_own_range(panel):
    """HP has no second witness, so only range and variety are checked."""
    values = Counter()
    for run in panel["hp"]:
        if run["value"] is not None:
            values[run["value"]] += run["samples"]
    assert min(values) >= 0 and max(values) <= 100
    assert len(values) > 50, "a stuck reading would show as a handful of values"
    assert values[100] / sum(values.values()) > 0.5


def test_the_weapon_is_one_of_the_three_the_series_used(panel):
    weapons = Counter(
        run["value"] for run in panel["weapon"] for _ in range(run["samples"])
    )
    assert set(weapons) - {None} == {"MPC-25", "M15 MOD 0", "JÄGER 45"}
    read = sum(n for w, n in weapons.items() if w is not None)
    assert read / sum(weapons.values()) > 0.85


def test_the_value_atlas_is_the_header_clocks_cut_plus_a_slash():
    """The panel and header-clock atlases share the same glyph cut, plus '/'."""
    panel_atlas = GlyphAtlas.load(CONFIG / "atlas_spectated.json")
    clock = GlyphAtlas.load(CONFIG / "atlas_header_clock.json")
    shared = sorted(set(panel_atlas.glyphs) & set(clock.glyphs))
    assert len(shared) == 11
    for char in shared:
        hit = clock.match(panel_atlas.glyphs[char])
        assert hit.char == char, f"{char} matched {hit.char}"
        assert hit.distance <= 2, f"{char} at distance {hit.distance}"
    assert set(panel_atlas.glyphs) - set(clock.glyphs) == {"/"}


def test_the_name_matcher_covers_the_whole_roster():
    matcher = NameMatcher.load(CONFIG / "spectated_names.json")
    assert len(matcher.players) == 8
    assert set(matcher.teams.values()) == {"left", "right"}


def test_the_stat_label_templates_are_the_three_modes():
    labels = TemplateSet.load(CONFIG / "spectated_labels.json")
    assert {k.split("#")[0] for k in labels.templates} == {"TIME", "PLNT", "OVLD"}


def test_the_panel_and_the_scoreboard_strips_are_aligned(panel, paired):
    """Agreement must peak at zero offset, not merely be high at it."""
    scores = {}
    for shift in (-2, -1, 0, 1, 2):
        agree = total = 0
        for t, row in paired:
            want, got = row.get("name"), _at(panel["name"], t + shift * STEP)
            if want is None or got is None:
                continue
            total += 1
            agree += str(want) == str(got)
        scores[shift] = agree / max(total, 1)
    assert max(scores, key=scores.get) == 0, scores
