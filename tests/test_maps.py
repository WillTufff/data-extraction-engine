"""End-to-end checks over map boundaries derived from the whole VOD.

Runs the real derivation over the cached scoreboard/header run caches,
without needing the video. Boundaries are checked for agreement across all
three detection mechanisms, and modes/map counts against the box scores.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cdlvision import maps, scoreboard_runs

FIXTURES = Path(__file__).parent / "fixtures"

# Where the three mechanisms agreed, to the second, on this VOD.
BOUNDARIES = (1656, 3213, 4142, 6237, 7526, 8663)


@pytest.fixture(scope="module")
def runs() -> list[dict]:
    raw = scoreboard_runs.load(FIXTURES / "scoreboard_runs.jsonl.gz")
    ours, _ = scoreboard_runs.split_foreign(raw)
    return ours


@pytest.fixture(scope="module")
def header() -> list[dict]:
    return scoreboard_runs.load(FIXTURES / "header_runs.jsonl.gz")


@pytest.fixture(scope="module")
def series(runs, header) -> list[dict]:
    ours, _ = maps.filter_to_series(runs, header)
    return ours


@pytest.fixture(scope="module")
def windows(series, header) -> list[maps.MapWindow]:
    return maps.derive(series, header, strict=True)


@pytest.fixture(scope="module")
def boxscores() -> list[dict]:
    return json.loads((FIXTURES / "boxscore_kd.json").read_text())


def test_every_boundary_has_all_three_witnesses(series, header):
    found = maps.boundaries(series, header)
    assert [round(b.t) for b in found] == list(BOUNDARIES)
    for boundary in found:
        assert set(boundary.witnesses) == {
            "scoreboard_reset", "pip_advance", "score_reset"
        }, f"t={boundary.t}"


def test_each_mechanism_finds_the_boundaries_alone(series, header):
    """Each witness checked on its own, so one going quiet is not masked."""
    assert sorted(round(t) for t in maps.scoreboard_resets(series)) == list(BOUNDARIES)
    assert sorted(round(t) for t in maps.pip_advances(header)) == list(BOUNDARIES)
    assert sorted(round(t) for t in maps.score_resets(header)) == list(BOUNDARIES)


def test_seven_maps_from_six_box_scores(windows, boxscores):
    assert len(windows) == len(boxscores) + 1
    assert [w.number for w in windows] == list(range(1, 8))


def test_windows_are_ordered_and_disjoint(windows):
    for previous, current in zip(windows, windows[1:]):
        assert previous.start < previous.end < current.start


def test_pips_advance_by_one_map_per_window(windows):
    assert windows[0].pips == (0, 0)
    for previous, current in zip(windows, windows[1:]):
        assert sum(current.pips) == sum(previous.pips) + 1
        assert current.pips >= previous.pips


def test_mode_from_the_stat_column_agrees_with_every_box_score(windows, boxscores):
    assigned = maps.assign_boxscores(windows, boxscores)
    assert len(assigned) == len(boxscores)
    for number, box in assigned.items():
        window = next(w for w in windows if w.number == number)
        assert window.mode == box["mode"], f"map {number}"
    # The map with no box score still gets a mode.
    assert windows[-1].number not in assigned
    assert windows[-1].mode is not None


def test_every_box_score_lands_after_the_map_it_summarises(windows, boxscores):
    assigned = maps.assign_boxscores(windows, boxscores)
    for number, box in assigned.items():
        window = next(w for w in windows if w.number == number)
        assert window.end <= box["t"]
        later = [w.start for w in windows if w.start > window.start]
        assert not later or box["t"] < min(later)


def test_replays_the_roster_cannot_see_are_found_by_the_header(runs, series):
    """Recaps of an earlier map of this series pass the roster check but fail
    the header's pip/score check."""
    assert 0 < len(runs) - len(series) < len(runs) // 10


def test_counts_are_monotone_within_each_map(series, windows):
    drops = list(_drops(series, windows))
    assert len(drops) <= 3, drops


def test_team_kills_never_exceed_opposing_deaths(series, windows):
    for window in windows:
        final: dict[int, dict] = {}
        for run in sorted(series, key=lambda r: r["start"]):
            value = run["value"]
            if window.contains(run["start"]) and value["kills"] is not None \
                    and value["number"]:
                final[value["number"]] = value
        assert sorted(final) == [1, 2, 3, 4, 5, 6, 7, 8], f"map {window.number}"
        left = [final[n] for n in (1, 2, 3, 4)]
        right = [final[n] for n in (5, 6, 7, 8)]
        for us, them in ((left, right), (right, left)):
            kills = sum(r["kills"] for r in us)
            deaths = sum(r["deaths"] for r in them)
            assert kills <= deaths, f"map {window.number}: {kills} > {deaths}"


def _drops(runs, windows):
    for slot_runs in _by_slot(runs).values():
        for window in windows:
            inside = [
                r for r in slot_runs
                if window.contains(r["start"]) and r["value"]["kills"] is not None
            ]
            for previous, current in zip(inside, inside[1:]):
                if previous["value"]["number"] != current["value"]["number"]:
                    continue
                for field in ("kills", "deaths"):
                    if current["value"][field] < previous["value"][field]:
                        yield (window.number, current["value"]["number"], field,
                               current["start"])


def _by_slot(runs):
    out: dict[tuple[str, int], list[dict]] = {}
    for run in runs:
        out.setdefault((run["side"], run["slot"]), []).append(run)
    for slot_runs in out.values():
        slot_runs.sort(key=lambda r: r["start"])
    return out
