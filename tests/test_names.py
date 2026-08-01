"""Tests for killfeed name classification: distance/averaging primitives, the
shipped matcher against the closed player set, and matching against the
labelled fixture."""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cdlvision.names import (  # noqa: E402
    MAX_DISTANCE,
    MIN_MARGIN,
    NameMatcher,
    average_masks,
    normalised_distance,
)

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "src/cdlvision/config/killfeed_names.json"
LABELS = ROOT / "tests/fixtures/killfeed_labels.jsonl.gz"

PLAYERS = {"ABUZAH", "DRAZAH", "04", "SIMP", "HUKE", "SHOTZZY", "MERCULES", "DASHY"}


def decode(payload: dict) -> np.ndarray:
    raw = np.frombuffer(
        __import__("base64").b64decode(payload["b"]), dtype=np.uint8
    )
    bits = np.unpackbits(raw)[: payload["h"] * payload["w"]]
    return bits.reshape(payload["h"], payload["w"]).astype(bool)


@pytest.fixture(scope="module")
def matcher() -> NameMatcher:
    return NameMatcher.load(CONFIG)


@pytest.fixture(scope="module")
def labelled() -> list[dict]:
    with gzip.open(LABELS, "rt") as fh:
        return [json.loads(line) for line in fh if line.strip()]


@pytest.fixture(scope="module")
def sampled(labelled) -> list[dict]:
    """Every fourth row, to keep the matching pass fast."""
    return labelled[::4]


# -- distance ------------------------------------------------------------


def test_normalised_distance_is_zero_for_identical():
    a = np.array([[True, False], [False, True]])
    assert normalised_distance(a, a) == 0.0


def test_normalised_distance_is_one_for_disjoint():
    a = np.array([[True, True, False, False]])
    b = np.array([[False, False, True, True]])
    assert normalised_distance(a, b, max_shift=0) == 1.0


def test_normalised_distance_is_size_independent():
    """A small name and a large one are scored on the same scale."""
    small = np.ones((3, 3), dtype=bool)
    large = np.ones((12, 40), dtype=bool)
    assert normalised_distance(small, small) == normalised_distance(large, large)


# -- averaging -----------------------------------------------------------


def test_average_masks_takes_the_majority():
    on = np.array([[True, True]])
    off = np.array([[True, False]])
    assert average_masks([on, on, off]).tolist() == [[True, True]]
    assert average_masks([on, off, off]).tolist() == [[True, False]]


def test_average_masks_tolerates_width_jitter():
    """Segmentation moves a name's right edge by a pixel between frames."""
    a = np.ones((5, 20), dtype=bool)
    b = np.ones((5, 21), dtype=bool)
    out = average_masks([a, b, b])
    assert out.shape == (5, 21)


def test_average_masks_rejects_empty():
    with pytest.raises(ValueError):
        average_masks([])


# -- the shipped matcher -------------------------------------------------


def test_config_covers_the_whole_roster(matcher):
    assert set(matcher.players) == PLAYERS


def test_config_carries_rosters_from_the_box_scores(matcher):
    assert set(matcher.teams) == PLAYERS
    assert len(set(matcher.teams.values())) == 2
    for team in set(matcher.teams.values()):
        assert sum(v == team for v in matcher.teams.values()) == 4


def test_names_carry_several_templates(matcher):
    assert len(matcher) > len(matcher.players)


def test_unknown_name_is_rejected_not_forced(matcher):
    """A shape unlike any player must come back unresolved."""
    noise = np.zeros((17, 60), dtype=bool)
    noise[::2, ::3] = True
    assert not matcher.match(noise).resolved


def test_empty_matcher_refuses_to_guess():
    with pytest.raises(ValueError):
        NameMatcher().match(np.ones((4, 4), dtype=bool))


# -- against the labelled fixture ---------------------------------------


def test_matches_the_oracle_labels(matcher, sampled):
    """Nearly everything resolves, and essentially nothing misreads."""
    correct = wrong = unknown = 0
    for r in sampled:
        for role in ("attacker", "victim"):
            truth = r.get(f"{role}_label")
            if truth is None:
                continue
            got = matcher.match(decode(r[f"{role}_bmp"]))
            if not got.resolved:
                unknown += 1
            elif got.label == truth:
                correct += 1
            else:
                wrong += 1

    total = correct + wrong + unknown
    assert total > 1400
    assert wrong <= 2
    assert correct / total > 0.98


def test_friendly_fire_rows_are_kept_not_corrected(matcher, labelled):
    """Same-team rows are real, so they must survive matching, not be rewritten."""
    same_team = [
        r
        for r in labelled
        if r.get("attacker_label")
        and r.get("victim_label")
        and matcher.teams[r["attacker_label"]] == matcher.teams[r["victim_label"]]
    ]
    assert same_team, "fixture should contain friendly-fire rows"

    for r in same_team:
        a = matcher.match(decode(r["attacker_bmp"]))
        v = matcher.match(decode(r["victim_bmp"]))
        if a.resolved and v.resolved:
            assert a.label == r["attacker_label"]
            assert v.label == r["victim_label"]
            assert matcher.same_team(a, v)


def test_enforcing_cross_team_loses_friendly_fire(matcher, labelled):
    """Forcing a cross-team match on a same-team row costs real kills."""
    kept = forced = 0
    for r in labelled:
        la, lv = r.get("attacker_label"), r.get("victim_label")
        if not (la and lv) or matcher.teams[la] != matcher.teams[lv]:
            continue
        plain = matcher.match_row(decode(r["attacker_bmp"]), decode(r["victim_bmp"]))
        tied = matcher.match_row(
            decode(r["attacker_bmp"]),
            decode(r["victim_bmp"]),
            enforce_cross_team=True,
        )
        kept += all(m.resolved for m in plain)
        forced += all(m.resolved for m in tied)

    assert kept > 0, "friendly-fire rows should read fine when left alone"
    assert forced < kept, "enforcing the constraint should cost real rows"


def test_thresholds_are_the_swept_values():
    assert (MAX_DISTANCE, MIN_MARGIN) == (0.70, 0.12)
