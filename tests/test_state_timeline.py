"""End-to-end checks over the full-VOD state scan.

Runs the real decision and smoothing code over the cached probe output for
all 9830 sampled seconds, without needing the video itself. Anchor timestamps
and the box-score count come from the box-score parser, an independent
mechanism.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from cdlvision.state import Probes, State, label_at, segment, state_from_probes

FIXTURES = Path(__file__).parent / "fixtures"

# Post-map box scores and the series overview, established independently.
BOXSCORE_ANCHORS = (760.0, 2800.0, 3940.0, 4920.0, 7350.0, 8230.0)
SERIES_ANCHOR = 4980.0


@pytest.fixture(scope="module")
def rows() -> list[dict]:
    with gzip.open(FIXTURES / "states_raw_fullvod.jsonl.gz", "rt") as fh:
        return [json.loads(line) for line in fh if line.strip()]


@pytest.fixture(scope="module")
def observations(rows) -> list[tuple[float, State]]:
    return [
        (
            r["t"],
            state_from_probes(
                Probes(
                    name_rows=r["name_rows"],
                    name_pitches=tuple(r["name_pitches"]),
                    chrome=r["chrome"],
                    tag=r["tag"],
                    tag_distance=r["tag_distance"],
                    series_orange=r["series_orange"],
                    boxscore_columns=r["boxscore_columns"],
                )
            ),
        )
        for r in rows
    ]


@pytest.fixture(scope="module")
def segments(observations):
    return segment(observations)


def test_scan_covers_the_whole_vod(rows):
    assert len(rows) == 9830
    assert rows[0]["t"] == 0.0
    assert rows[-1]["t"] == 9829.0


def test_every_box_score_anchor_lands_in_a_boxscore_segment(segments):
    for t in BOXSCORE_ANCHORS:
        assert label_at(segments, t) is State.BOXSCORE, f"anchor t={t}"


def test_series_anchor_lands_in_a_series_segment(segments):
    assert label_at(segments, SERIES_ANCHOR) is State.SERIES


def test_exactly_one_boxscore_segment_per_map(segments):
    box = [s for s in segments if s.state is State.BOXSCORE]
    assert len(box) == len(BOXSCORE_ANCHORS)
    # Every segment must be claimed by an anchor -- no spurious extras.
    for s in box:
        assert any(s.contains(t) for t in BOXSCORE_ANCHORS)


def test_box_score_segments_persist_for_seconds(segments):
    for s in (s for s in segments if s.state is State.BOXSCORE):
        assert 10.0 <= s.duration <= 60.0


def test_hand_labelled_frames_all_classify_correctly(segments):
    payload = json.loads((FIXTURES / "state_labels.json").read_text())
    for entry in payload["labels"]:
        got = label_at(segments, float(entry["t"]))
        assert got is State(entry["state"]), (
            f"t={entry['t']} ({entry['note']}): "
            f"labelled {entry['state']}, got {got.value if got else None}"
        )


def test_a_matched_tag_is_never_lost_to_a_missing_scoreboard(rows, observations):
    """Frames with a matched tag but no chrome (scoreboard animating) still read REPLAY."""
    by_t = dict(observations)
    tagged_no_chrome = [r for r in rows if r["tag"] and not r["chrome"]]
    assert tagged_no_chrome, "fixture should still contain this case"
    for r in tagged_no_chrome:
        assert by_t[r["t"]] is State.REPLAY, f"t={r['t']}"


def test_no_frame_has_both_a_tag_and_a_box_score_grid(rows):
    assert [r["t"] for r in rows if r["tag"] and r["boxscore_columns"] is not None] == []


def test_smoothing_removes_fragmentation_without_moving_anchors(observations, segments):
    raw_runs = (
        sum(1 for a, b in zip(observations, observations[1:]) if a[1] is not b[1]) + 1
    )
    assert len(segments) < raw_runs * 0.8
    changed = sum(1 for t, s in observations if label_at(segments, t) is not s)
    assert changed / len(observations) < 0.02


def test_segments_tile_the_scan_without_gaps(segments, rows):
    assert segments[0].start == rows[0]["t"]
    for a, b in zip(segments, segments[1:]):
        assert a.end == b.start
        assert a.state is not b.state


def test_every_state_occurs(segments):
    seen = {s.state for s in segments}
    assert seen == set(State)


def test_replay_occupancy_is_plausible(segments):
    """Loose sanity bound: replays are frequent but never dominant."""
    total = sum(s.duration for s in segments)
    replay = sum(s.duration for s in segments if s.state is State.REPLAY)
    assert 0.02 < replay / total < 0.20
