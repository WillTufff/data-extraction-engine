"""Tests for killfeed row deduplication: an entry that persists must collapse
into one event, and a pair that genuinely repeats must not."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cdlvision.events import (  # noqa: E402
    MAX_GAP,
    RowReading,
    dedupe,
    pooled_masks,
)
from cdlvision.names import NameMatch  # noqa: E402

SIG = (70, 46, 44)


def reading(t, slot, attacker="ABUZAH", victim="HUKE", sig=SIG, medal=False):
    def m(label):
        return NameMatch(label, 0.1, "OTHER", 0.5)

    return RowReading(
        t=float(t),
        slot=slot,
        attacker=m(attacker),
        victim=m(victim),
        medal=medal,
        signature=sig,
        attacker_mask=np.ones((17, 20), dtype=bool),
        victim_mask=np.ones((17, 20), dtype=bool),
    )


# -- persistence ---------------------------------------------------------


def test_one_entry_seen_repeatedly_is_one_event():
    """An entry sitting still across four frames is a single kill."""
    events = dedupe([reading(t, 569) for t in (10, 11, 12, 13)])
    assert len(events) == 1
    assert events[0].frames == 4
    assert events[0].t == 10.0


def test_entry_rising_up_the_grid_is_one_event():
    """Newer kills push it up; it is still the same kill."""
    events = dedupe([
        reading(10, 569), reading(11, 539), reading(12, 509), reading(13, 479),
    ])
    assert len(events) == 1
    assert events[0].slots == [569, 539, 509, 479]


def test_event_is_timestamped_at_first_sighting():
    events = dedupe([reading(t, 569) for t in (44, 45, 46)])
    assert events[0].t == 44.0


# -- genuine repeats -----------------------------------------------------


def test_same_pair_in_two_slots_in_one_frame_is_two_events():
    """The clearest repeat: both entries on screen at once."""
    events = dedupe([reading(10, 569), reading(10, 539)])
    assert len(events) == 2


def test_slot_moving_back_down_starts_a_new_event():
    """An entry is never redrawn lower, so this must be a second kill."""
    events = dedupe([reading(10, 509), reading(11, 539)])
    assert len(events) == 2


def test_repeat_after_the_entry_expires_is_two_events():
    events = dedupe([reading(10, 569), reading(10 + MAX_GAP + 1, 569)])
    assert len(events) == 2


def test_a_repeat_interleaved_with_persistence_splits_correctly():
    """First kill rises while the second appears beneath it."""
    events = dedupe([
        reading(10, 569),
        reading(11, 539), reading(11, 569),
        reading(12, 509), reading(12, 539),
    ])
    assert len(events) == 2
    assert sorted(e.frames for e in events) == [2, 3]


# -- identity ------------------------------------------------------------


def test_a_frame_missed_mid_life_does_not_split_the_event():
    """Bright gameplay swallows rows; a gap inside an entry's life is not a
    second kill."""
    events = dedupe([reading(10, 569), reading(12, 509)])
    assert len(events) == 1
    assert events[0].frames == 2


def test_two_concurrent_entries_stay_separate_when_one_frame_drops():
    """The oldest-first attachment must survive a missing row."""
    events = dedupe([
        reading(10, 569),
        reading(11, 539), reading(11, 569),
        reading(12, 509),  # the newer entry was missed this frame
        reading(13, 479), reading(13, 509),
    ])
    assert len(events) == 2
    first, second = sorted(events, key=lambda e: e.t)
    assert first.slots == [569, 539, 509, 479]
    assert second.slots == [569, 509]


def test_different_pairs_never_merge():
    events = dedupe([
        reading(10, 569, victim="HUKE"),
        reading(10, 539, victim="DASHY"),
    ])
    assert len(events) == 2
    assert {e.victim for e in events} == {"HUKE", "DASHY"}


def test_unresolved_rows_on_screen_together_stay_separate():
    """Two unknown rows are not one row just because both are unknown.

    They share a dedup key -- nothing distinguishes them by name -- so it is
    the chain logic that has to keep them apart.
    """
    events = dedupe([
        reading(10, 569, attacker=None, victim=None),
        reading(10, 539, attacker=None, victim=None),
    ])
    assert len(events) == 2


def test_width_jitter_does_not_split_an_entry():
    """Segmentation moves a name's edge by a pixel between frames; that must
    not read as a new kill."""
    events = dedupe([
        reading(10, 569, sig=(70, 46, 44)),
        reading(11, 569, sig=(71, 46, 44)),
        reading(12, 539, sig=(70, 47, 43)),
    ])
    assert len(events) == 1
    assert events[0].frames == 3


def test_medal_survives_a_frame_that_missed_it():
    """A medal fades with the row, so one sighting is enough."""
    events = dedupe([
        reading(10, 569, medal=False),
        reading(11, 569, medal=True),
        reading(12, 569, medal=False),
    ])
    assert len(events) == 1
    assert events[0].medal


# -- pooling -------------------------------------------------------------


def test_pooled_masks_returns_one_probe_per_name():
    run = [reading(10, 569), reading(11, 539)]
    a, v = pooled_masks(run)
    assert a.shape == (17, 20)
    assert v.shape == (17, 20)


def test_dedupe_of_nothing_is_nothing():
    assert dedupe([]) == []


def test_events_come_back_in_time_order():
    events = dedupe([
        reading(30, 569, victim="DASHY"),
        reading(10, 569, victim="HUKE"),
        reading(20, 569, victim="SIMP"),
    ])
    assert [e.t for e in events] == [10.0, 20.0, 30.0]
