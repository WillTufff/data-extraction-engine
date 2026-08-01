"""Rounds derived over the whole VOD, from the committed run caches.

Asserts the game's own arithmetic rather than transcriptions: the score
staircase, rounds won against the map score, and the two exact upper bounds
Search and Destroy provides. The duplicate test is the sharpest one here --
it pins both the count and the *separation* between the two causes, because a
change that blurs them would still pass a bare count.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from cdlvision import maps, rounds, scoreboard_runs  # noqa: E402
from cdlvision.events import DENSE_12HZ, build_events  # noqa: E402
from cdlvision.names import NameMatcher  # noqa: E402
from scan_killfeed import decode  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
CONFIG = Path(__file__).parent.parent / "src/cdlvision/config"

# How each map divides, by mode. Eleven rounds on every Search and Destroy map,
# twelve hills on every Hardpoint one, two halves on every Overload one.
DIVISIONS = {1: 12, 2: 11, 3: 2, 4: 12, 5: 11, 6: 2, 7: 11}

# The one boundary short of a witness: the lives counters were not read as a
# pair across this break. The clock and the score both saw it.
KNOWN_THIN_BOUNDARY = (5, 6)

# The recap clip CLAUDE.md records as invisible to every header counter, since
# it replays the same map at the same round score.
REPLAY_T = 2580.0


@pytest.fixture(scope="module")
def header():
    return scoreboard_runs.load(FIXTURES / "header_runs.jsonl.gz")


@pytest.fixture(scope="module")
def board():
    raw = scoreboard_runs.load(FIXTURES / "scoreboard_runs.jsonl.gz")
    ours, _ = scoreboard_runs.split_foreign(raw)
    return ours


@pytest.fixture(scope="module")
def windows(board, header):
    ours, _ = maps.filter_to_series(board, header)
    return maps.derive(ours, header)


@pytest.fixture(scope="module")
def sides(board, header):
    ours, _ = maps.filter_to_series(board, header)
    return scoreboard_runs.player_sides(ours)


@pytest.fixture(scope="module")
def events():
    matcher = NameMatcher.load(CONFIG / "killfeed_names.json")
    rows = scoreboard_runs.load(FIXTURES / "killfeed_rows_dense.jsonl.gz")
    return build_events(matcher, rows, decode, **DENSE_12HZ.kwargs())


@pytest.fixture(scope="module")
def sd_rounds(windows, header, events, sides):
    out = {}
    winners = rounds.map_winners(windows)
    for window in windows:
        if window.mode == rounds.ROUNDS_MODE:
            derived = rounds.derive(window, header, events, sides)
            rounds.close_from_series(derived, winners.get(window.number))
            out[window.number] = derived
    return out


def test_only_search_and_destroy_has_rounds(windows, header, events, sides):
    for window in windows:
        derived = rounds.derive(window, header, events, sides)
        assert bool(derived) == (window.mode == rounds.ROUNDS_MODE), window.mode


def test_every_map_divides_into_what_its_mode_has(windows, header, events, sides):
    for window in windows:
        segments = rounds.subdivide(window, header, events, sides)
        assert len(segments) == DIVISIONS[window.number], window.number


def test_round_spans_are_ordered_and_cover_the_map(sd_rounds, windows):
    for number, derived in sd_rounds.items():
        window = next(w for w in windows if w.number == number)
        assert derived[0].start == window.start
        assert derived[-1].end == window.end
        for previous, current in zip(derived, derived[1:]):
            assert previous.start < previous.end == current.start


def test_boundaries_carry_all_three_witnesses(sd_rounds):
    thin = [
        (number, r.number) for number, derived in sd_rounds.items()
        for r in derived if r.number > 1 and len(r.witnesses) < 3
    ]
    assert thin == [KNOWN_THIN_BOUNDARY]


def test_the_score_entering_each_round_is_a_staircase(windows, header):
    """What rules out a *missing* round. Every map's last drawn score is 5-5,
    which looks like a failed read; a skipped round would show as a jump of two
    here, and none does."""
    for window in windows:
        if window.mode != rounds.ROUNDS_MODE:
            continue
        assert rounds.staircase_gaps(window, header) == [], window.number


def test_rounds_won_reproduce_the_map_score(sd_rounds):
    for number, derived in sd_rounds.items():
        won = Counter(r.winner for r in derived)
        before = derived[-1].score_before
        # The last round's own result is never drawn, so the rounds won before
        # it must equal the last score the broadcast did draw.
        earlier = Counter(r.winner for r in derived[:-1])
        assert (earlier["left"], earlier["right"]) == before, number
        assert won["left"] + won["right"] + won[None] == len(derived)


def test_the_winner_of_each_decided_map_agrees_with_the_pips(sd_rounds, windows):
    winners = rounds.map_winners(windows)
    for number, derived in sd_rounds.items():
        if winners.get(number) is None:
            continue
        won = Counter(r.winner for r in derived)
        assert won.most_common(1)[0][0] == winners[number], number


def test_duplicate_victims_split_cleanly_into_two_causes(sd_rounds):
    """The separation is the assertion, not the count. Fragments and replays
    have no overlap and are not close to it -- 4.33s against 29.75s."""
    dups = [d for derived in sd_rounds.values()
            for r in derived for d in rounds.duplicate_victims(r)]
    fragments = [d for d in dups if d[3] == "fragment"]
    replays = [d for d in dups if d[3] == "replay"]

    assert len(fragments) == 9
    assert len(replays) == 3
    assert max(b.t - a.t for _, a, b, _ in fragments) < \
        min(b.t - a.t for _, a, b, _ in replays)
    # Every replay is the one clip, and every one of its chains is a second
    # long at 12fps.
    for _, _, b, _ in replays:
        assert b.t == pytest.approx(REPLAY_T)
        assert b.frames <= 12


def test_a_fragment_always_has_a_short_chain_in_it(sd_rounds):
    """The evidence that these are chain fragmentation rather than real repeats:
    one half sits on `DENSE_12HZ`'s min_frames floor."""
    fragments = [d for derived in sd_rounds.values()
                 for r in derived for d in rounds.duplicate_victims(r)
                 if d[3] == "fragment"]
    for _, a, b, _ in fragments:
        assert min(a.frames, b.frames) <= 11


def test_no_side_loses_more_than_four_except_where_a_replay_explains_it(
        sd_rounds, sides):
    for number, derived in sd_rounds.items():
        for r in derived:
            over = rounds.over_four(r, sides)
            if not over:
                continue
            kinds = {d[3] for d in rounds.duplicate_victims(r)}
            assert "replay" in kinds, f"map {number} r{r.number} {over}"


def test_deaths_never_exceed_the_box_score(sd_rounds, windows, sides):
    boxscores = json.loads((FIXTURES / "boxscore_kd.json").read_text())
    assigned = maps.assign_boxscores(windows, boxscores)
    for number, derived in sd_rounds.items():
        box = assigned.get(number)
        if not box:
            continue
        for index, side in enumerate(rounds.SIDES):
            lost = sum(rounds.LIVES - r.lives_low[index] for r in derived
                       if r.lives_low[index] is not None)
            killfeed = sum(len(set(r.victims(sides, side))) for r in derived)
            deaths = sum(v["deaths"] for n, v in box["kd"].items()
                         if sides.get(n) == side)
            # Both are lower bounds -- the lives counter is not read at every
            # round's end and the killfeed loses rows over bright gameplay --
            # so the check is an ordering, never an equality.
            assert lost <= deaths, f"map {number} {side}"
            assert killfeed <= deaths, f"map {number} {side}"


def test_outcomes_are_only_the_ones_the_overlay_witnesses(sd_rounds):
    """Time expiry and detonation-against-defuse are not observable while the
    final-30 clock is unread, so they must never appear."""
    seen = {r.outcome for derived in sd_rounds.values() for r in derived}
    assert seen <= {"wipe", "plant", "detonation", "time"}
    # "defuse" must never appear: telling it from the planters being killed
    # needs to know which side planted, which the overlay does not draw.
    assert "defuse" not in seen


def test_every_round_gets_an_outcome(sd_rounds):
    """Only true once the final-thirty counter is read. While the clock stopped
    at 0:30 no round could be shown to have expired, and a third of them fell
    through unclassified."""
    outcomes = Counter(r.outcome for derived in sd_rounds.values()
                       for r in derived)
    assert outcomes[None] == 0
    # Both endings that need the recovered counter are actually present, so a
    # regression in the clock reader shows up here and not only in test_header.
    assert outcomes["time"] >= 4
    assert outcomes["detonation"] >= 3


def test_only_the_bomb_lets_a_wiped_side_win(sd_rounds):
    """A side that loses all four players and still wins can only have won on
    the bomb. This caught the first version asserting a wiped side never wins."""
    for number, derived in sd_rounds.items():
        for r in derived:
            if r.wiped and r.winner and r.winner == r.wiped:
                assert r.outcome == "detonation", f"map {number} r{r.number}"
                assert r.plant_t is not None


def test_a_plant_is_always_the_bomb_timer_starting_at_44(sd_rounds, windows,
                                                         header):
    """Nine plants across the series, every one an upward jump to exactly 44.0.
    A looser band admitted the t=2580 replay clip's 0:48 as a tenth."""
    planted = [r for derived in sd_rounds.values() for r in derived
               if r.plant_t is not None]
    assert len(planted) == 9
    for window in windows:
        if window.mode != rounds.ROUNDS_MODE:
            continue
        series = rounds.clock_series(header, window)
        at = dict(series)
        for r in sd_rounds[window.number]:
            if r.plant_t is not None:
                assert at[r.plant_t] == 44.0
