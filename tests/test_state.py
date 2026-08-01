"""Temporal smoothing of the per-frame state classification.

Uses synthetic observation sequences rather than video, since only the
smoother's handling of short runs is under test.
"""

from __future__ import annotations

import pytest

from cdlvision.state import (
    DEFAULT_MIN_DWELL,
    Probes,
    Segment,
    Smoothing,
    State,
    label_at,
    segment,
    state_from_probes,
)


def probes(**kw) -> Probes:
    base = dict(
        name_rows=4,
        name_pitches=(32, 32, 32),
        chrome=True,
        tag=None,
        tag_distance=None,
        series_orange=0.0,
        boxscore_columns=None,
    )
    return Probes(**{**base, **kw})


class TestStateFromProbes:
    def test_live_chrome_without_a_tag_is_live(self):
        assert state_from_probes(probes()) is State.LIVE

    def test_tag_over_live_chrome_is_replay(self):
        assert state_from_probes(probes(tag="REPLAY")) is State.REPLAY

    def test_tag_without_chrome_is_still_replay(self):
        # Scoreboard chrome can be absent at replay boundaries while the tag is on screen.
        assert state_from_probes(probes(chrome=False, tag="HIGHLIGHTS")) is State.REPLAY

    def test_highlights_and_replay_tags_both_mean_replay(self):
        for tag in ("REPLAY", "HIGHLIGHTS"):
            assert state_from_probes(probes(tag=tag)) is State.REPLAY

    def test_boxscore_grid_outranks_everything(self):
        assert state_from_probes(probes(boxscore_columns=9)) is State.BOXSCORE
        assert (
            state_from_probes(probes(boxscore_columns=9, tag="REPLAY"))
            is State.BOXSCORE
        )

    def test_series_needs_orange_and_no_chrome(self):
        assert state_from_probes(probes(chrome=False, series_orange=0.4)) is State.SERIES
        # Orange over live chrome is not the series overview.
        assert state_from_probes(probes(chrome=True, series_orange=0.4)) is State.LIVE

    def test_nothing_recognised_is_other(self):
        assert state_from_probes(probes(chrome=False)) is State.OTHER

    def test_untagged_frame_without_chrome_is_never_replay(self):
        assert state_from_probes(probes(chrome=False, series_orange=0.14)) is not State.REPLAY


def obs(spec: str, interval: float = 1.0, start: float = 0.0):
    """Build observations from a compact string, one character per sample.

    l=live r=replay b=boxscore s=series o=other
    """
    codes = {
        "l": State.LIVE,
        "r": State.REPLAY,
        "b": State.BOXSCORE,
        "s": State.SERIES,
        "o": State.OTHER,
    }
    return [(start + i * interval, codes[c]) for i, c in enumerate(spec)]


def states(segments):
    return [s.state for s in segments]


def test_empty_input_yields_no_segments():
    assert segment([]) == []


def test_uniform_run_is_one_segment():
    segs = segment(obs("l" * 10))
    assert states(segs) == [State.LIVE]
    assert segs[0].start == 0.0
    assert segs[0].n_samples == 10


def test_segments_tile_the_timeline_without_gaps():
    segs = segment(obs("l" * 10 + "b" * 10 + "l" * 10))
    assert len(segs) == 3
    for a, b in zip(segs, segs[1:]):
        assert a.end == b.start
    assert segs[0].start == 0.0
    # End extends one interval past the final sample.
    assert segs[-1].end == 30.0


def test_end_is_exclusive_and_covers_final_sample():
    segs = segment(obs("l" * 5))
    assert segs[0].end == 5.0
    assert segs[0].contains(4.9)
    assert not segs[0].contains(5.0)


def test_single_frame_flicker_is_absorbed():
    segs = segment(obs("l" * 10 + "o" + "l" * 10))
    assert states(segs) == [State.LIVE]
    assert segs[0].n_samples == 21


def test_run_at_the_dwell_threshold_survives():
    # OTHER's floor is 3s; three 1s samples is exactly 3s.
    assert DEFAULT_MIN_DWELL[State.OTHER] == 3.0
    segs = segment(obs("l" * 10 + "ooo" + "l" * 10))
    assert states(segs) == [State.LIVE, State.OTHER, State.LIVE]


def test_run_just_below_the_dwell_threshold_is_absorbed():
    segs = segment(obs("l" * 10 + "oo" + "l" * 10))
    assert states(segs) == [State.LIVE]


def test_adjacent_short_runs_collapse_together():
    # A cluster of one-sample runs must not survive by each protecting the next.
    segs = segment(obs("l" * 10 + "obso" + "l" * 10))
    assert states(segs) == [State.LIVE]


def test_short_run_joins_its_longer_neighbour():
    # A 2s OTHER between a long BOXSCORE and a short LIVE goes to BOXSCORE.
    segs = segment(obs("b" * 20 + "oo" + "l" * 6))
    assert states(segs) == [State.BOXSCORE, State.LIVE]
    assert segs[0].end == 22.0


def test_lone_replay_sample_is_preserved():
    # REPLAY is exempt from absorption.
    segs = segment(obs("l" * 10 + "r" + "l" * 10))
    assert states(segs) == [State.LIVE, State.REPLAY, State.LIVE]


def test_short_replay_dropout_is_bridged():
    # A brief drop mid-replay must not split one replay in two.
    segs = segment(obs("l" * 10 + "rrrr" + "ll" + "rrrr" + "l" * 10))
    assert states(segs) == [State.LIVE, State.REPLAY, State.LIVE]
    replay = segs[1]
    assert replay.start == 10.0
    assert replay.end == 20.0


def test_long_live_gap_between_replays_is_not_bridged():
    segs = segment(obs("l" * 6 + "rrrr" + "l" * 8 + "rrrr" + "l" * 6))
    assert states(segs) == [
        State.LIVE,
        State.REPLAY,
        State.LIVE,
        State.REPLAY,
        State.LIVE,
    ]


def test_bridging_beats_absorption_for_live_inside_replay():
    # Bridging runs before absorption, so a short LIVE gap resolves to REPLAY.
    segs = segment(obs("r" * 5 + "ll" + "r" * 5))
    assert states(segs) == [State.REPLAY]


def test_boundary_run_absorbs_into_its_only_neighbour():
    segs = segment(obs("o" + "l" * 20))
    assert states(segs) == [State.LIVE]
    assert segs[0].start == 0.0


def test_all_short_input_is_left_alone():
    # Nothing to absorb into; must not empty the timeline.
    segs = segment(obs("l"))
    assert states(segs) == [State.LIVE]


def test_sampling_interval_is_inferred_from_the_data():
    # At 2s spacing the same two-sample OTHER run is 4s and clears the floor.
    segs = segment(obs("l" * 10 + "oo" + "l" * 10, interval=2.0))
    assert states(segs) == [State.LIVE, State.OTHER, State.LIVE]


def test_interval_inference_tolerates_dropped_samples():
    o = obs("l" * 10 + "oo" + "l" * 10)
    del o[3]  # a missing sample
    segs = segment(o)
    assert states(segs) == [State.LIVE]


def test_observations_need_not_be_sorted():
    o = obs("l" * 10 + "b" * 10)
    segs = segment(list(reversed(o)))
    assert states(segs) == [State.LIVE, State.BOXSCORE]


def test_custom_parameters_override_defaults():
    params = Smoothing(min_dwell={State.LIVE: 30.0, State.BOXSCORE: 4.0})
    segs = segment(obs("b" * 20 + "l" * 6 + "b" * 20), params)
    assert states(segs) == [State.BOXSCORE]


def test_disabling_bridging_keeps_the_replay_split():
    params = Smoothing(min_dwell=DEFAULT_MIN_DWELL, bridge={})
    segs = segment(obs("r" * 5 + "lll" + "r" * 5), params)
    assert states(segs) == [State.REPLAY, State.LIVE, State.REPLAY]


@pytest.mark.parametrize(
    "spec",
    ["l" * 30, "l" * 10 + "o" + "l" * 10, "r" * 4 + "ll" + "r" * 4, "obso" + "l" * 20],
)
def test_smoothing_never_leaves_a_sub_dwell_run(spec):
    for seg in segment(obs(spec)):
        floor = DEFAULT_MIN_DWELL[seg.state]
        assert seg.duration >= floor or seg.n_samples == len(spec)


def test_label_at_reads_the_timeline():
    segs = segment(obs("l" * 10 + "b" * 10))
    assert label_at(segs, 0.0) is State.LIVE
    assert label_at(segs, 9.99) is State.LIVE
    assert label_at(segs, 10.0) is State.BOXSCORE
    assert label_at(segs, 100.0) is None


def test_segment_duration():
    s = Segment(State.LIVE, 10.0, 25.0, 15)
    assert s.duration == 15.0
