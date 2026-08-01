"""HP against the killfeed, over the whole VOD and without the video.

The point of these is the *separation*: falls pair at a rate the negative
control and the null do not come near. Any one of the three numbers alone says
nothing, so they are asserted together and the bounds are set well clear of the
measured values, to catch a mechanism breaking rather than a decimal moving.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from cdlvision import hp, maps, scoreboard_runs  # noqa: E402
from cdlvision.events import DENSE_12HZ, build_events  # noqa: E402
from cdlvision.names import NameMatcher  # noqa: E402
from scan_killfeed import decode  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
CONFIG = Path(__file__).parent.parent / "src/cdlvision/config"

# One frame of the 6fps panel strip. The killfeed row is drawn after the death
# it reports, never before.
OFFSET = 1 / 6


@pytest.fixture(scope="module")
def panel() -> dict[str, list[dict]]:
    return hp.by_field(scoreboard_runs.load(FIXTURES / "spectated_runs.jsonl.gz"))


@pytest.fixture(scope="module")
def windows() -> list[maps.MapWindow]:
    board = scoreboard_runs.load(FIXTURES / "scoreboard_runs.jsonl.gz")
    header = scoreboard_runs.load(FIXTURES / "header_runs.jsonl.gz")
    ours, _ = scoreboard_runs.split_foreign(board)
    ours, _ = maps.filter_to_series(ours, header)
    return maps.derive(ours, header)


@pytest.fixture(scope="module")
def events():
    matcher = NameMatcher.load(CONFIG / "killfeed_names.json")
    rows = scoreboard_runs.load(FIXTURES / "killfeed_rows_dense.jsonl.gz")
    return build_events(matcher, rows, decode, **DENSE_12HZ.kwargs())


@pytest.fixture(scope="module")
def falls(panel, windows):
    return hp.falls(panel, windows)


def test_falls_are_found_and_all_land_inside_a_map(falls, windows):
    assert 200 < len(falls) < 400
    for fall in falls:
        assert any(w.contains(fall.t) for w in windows)
        assert fall.from_hp > 0
        assert fall.gap <= hp.MAX_READ_GAP


def test_a_panel_switch_is_not_counted_as_a_death(panel, windows):
    """Without the name-continuity rule the panel cutting to a dead team-mate
    reads as that team-mate dying at the moment of the cut."""
    strict = hp.falls(panel, windows)
    names = panel["name"]
    for fall in strict:
        assert hp.value_at(names, fall.t) == fall.player


def test_the_offset_is_one_panel_frame_and_never_negative(falls, events):
    swept = hp.sweep(falls, events, [n / 6 for n in range(-3, 5)])
    best = max(swept, key=lambda s: s[1])
    assert best[0] == pytest.approx(OFFSET, abs=1e-9)
    for offset, matched in swept:
        if offset < 0:
            assert matched <= 2, f"{matched} matches at offset {offset:+.3f}s"


def test_falls_pair_far_above_the_control_and_the_null(falls, panel, windows,
                                                       events):
    paired = sum(p.matched for p in hp.pair(falls, events, OFFSET))
    control = hp.survived_dips(panel, windows)
    control_rate = sum(
        p.matched for p in hp.pair(control, events, OFFSET)
    ) / len(control)
    null = max(
        sum(p.matched for p in hp.pair(_shift(falls, d), events, OFFSET))
        for d in (-120.0, -60.0, -30.0, 30.0, 60.0, 120.0)
    ) / len(falls)

    assert paired / len(falls) > 0.6
    assert control_rate < 0.10
    assert null < 0.10
    assert paired / len(falls) > 5 * max(control_rate, null)


def test_no_killfeed_event_accounts_for_two_falls(falls, events):
    """A double assignment would hide the duplicate the round check looks for."""
    used = [
        (p.event.t, p.fall.player)
        for p in hp.pair(falls, events, OFFSET) if p.matched
    ]
    assert len(used) == len(set(used))


def test_the_residual_offset_is_centred_once_the_offset_is_applied(falls, events):
    """The count and the residual must agree. Choosing the offset by match count
    alone does not guarantee that: with a window wide enough to be generous, the
    sweep slides up the late-chain tail and peaks where the residuals are not
    centred. This is the assertion that caught it."""
    deltas = sorted(p.delta for p in hp.pair(falls, events, OFFSET) if p.matched)
    assert abs(deltas[len(deltas) // 2]) <= 1 / 12


def test_deaths_of_the_spectated_player_are_almost_all_corroborated(
        events, panel, windows):
    stray = hp.unwitnessed_events(events, panel, windows)
    on_panel = [
        e for e in events
        if e.victim and any(w.contains(e.t) for w in windows)
        and hp.value_at(panel["name"], e.t) == e.victim
    ]
    assert len(on_panel) > 100
    assert len(stray) / len(on_panel) < 0.05


def _shift(falls_, delta):
    return [
        hp.Fall(t=f.t + delta, player=f.player, from_hp=f.from_hp, gap=f.gap)
        for f in falls_
    ]
