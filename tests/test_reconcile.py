"""End-to-end: cached killfeed rows must reconcile with the box-score K/D.

Runs from the cached scan, so it needs neither the video nor a decode pass.
Assertions are loose on undercounting and tight on overcounting.
"""

from __future__ import annotations

import gzip
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from cdlvision.events import build_events, chain_rows  # noqa: E402
from cdlvision.names import NameMatcher  # noqa: E402
from scan_killfeed import decode  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ROWS = ROOT / "tests/fixtures/killfeed_rows.jsonl.gz"
KD = ROOT / "tests/fixtures/boxscore_kd.json"
NAMES = ROOT / "src/cdlvision/config/killfeed_names.json"


@pytest.fixture(scope="module")
def rows() -> list[dict]:
    with gzip.open(ROWS, "rt") as fh:
        return [json.loads(line) for line in fh if line.strip()]


@pytest.fixture(scope="module")
def events(rows):
    return build_events(NameMatcher.load(NAMES), rows, decode)


def test_the_scan_covers_the_whole_series(rows):
    assert len(rows) > 3000
    assert {r["slot"] for r in rows} <= {569, 539, 509, 479, 449}


def test_chaining_collapses_persisting_rows(rows):
    """An entry lives about five seconds, so rows must outnumber chains."""
    chains = chain_rows(rows)
    assert len(chains) < len(rows)
    assert len(rows) / len(chains) > 2.0


def test_chains_never_move_back_down_the_grid(rows):
    """The invariant the whole dedup rests on."""
    for chain in chain_rows(rows):
        slots = [r["slot"] for r in chain]
        assert slots == sorted(slots, reverse=True)
        times = [r["t"] for r in chain]
        assert times == sorted(times)
        assert len(set(times)) == len(times), "one frame twice in one chain"


def test_most_events_resolve_both_names(events):
    resolved = [e for e in events if e.resolved]
    assert len(resolved) / len(events) > 0.65


def _totals(events):
    """Bucket events by map, using the bounds stored beside the K/D."""
    starts = [entry["start"] for entry in json.loads(KD.read_text())]

    def map_of(t):
        for i, start in enumerate(starts):
            if t < start:
                return i + 1
        return None

    kills: dict = {}
    deaths: dict = {}
    for e in events:
        if not e.resolved:
            continue
        m = map_of(e.t)
        if m is None:
            continue
        kills.setdefault(m, Counter())[e.attacker] += 1
        deaths.setdefault(m, Counter())[e.victim] += 1
    return kills, deaths


def test_reconciles_with_the_box_scores(events):
    """Kills recovered must stay high, and per-player excess bounded."""
    kills, deaths = _totals(events)
    maps = json.loads(KD.read_text())

    got_k = box_k = got_d = box_d = excess = 0
    for entry in maps:
        m = entry["map"]
        for name, box in entry["kd"].items():
            k = kills.get(m, Counter())[name]
            d = deaths.get(m, Counter())[name]
            got_k += k
            got_d += d
            box_k += box["kills"]
            box_d += box["deaths"]
            excess += max(0, k - box["kills"]) + max(0, d - box["deaths"])

    assert box_k > 900, "box scores should cover the whole series"
    assert got_k / box_k > 0.92
    assert got_d / box_d > 0.90
    # Never more kills than were actually scored, in total.
    assert got_k <= box_k
    assert excess < 80


def test_every_map_reads_a_full_roster():
    maps = json.loads(KD.read_text())
    assert len(maps) == 6
    for entry in maps:
        assert len(entry["kd"]) == 8, f"map {entry['map']} read {len(entry['kd'])}"
    rosters = [set(e["kd"]) for e in maps]
    assert all(r == rosters[0] for r in rosters), "roster changed between maps"
