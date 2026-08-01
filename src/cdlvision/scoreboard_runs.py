"""Loading and cleaning the run-length scoreboard timeline.

`tools/scan_scoreboard.py` writes one record per (slot, reading, time span).
Everything that consumes it -- validation, killfeed localisation -- needs the
same two things first, so they live here rather than in each tool:

- the **series roster**, derived from the runs themselves rather than written
  down. Number 1 is whoever number 1 is in the overwhelming majority of LIVE
  samples, which needs no config file and no assumption about which team is on
  which side.
- **foreign runs removed**. The state classifier marks recap clips LIVE, and
  those clips are gameplay from earlier matches in the tournament with a fully
  drawn scoreboard for a different series. They are not misreads and cannot be
  found by any confidence measure; they are found by the roster contradicting
  itself, which only works because identity is read rather than inferred.
"""

from __future__ import annotations

import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

from .scoreboard import foreign


def load(path: str | Path) -> list[dict]:
    """Read a run cache, plain or gzipped -- the fixtures are gzipped."""
    path = Path(path)
    text = (
        gzip.decompress(path.read_bytes()).decode()
        if path.suffix == ".gz" else path.read_text()
    )
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def series_roster(runs: list[dict]) -> dict[int, str]:
    """Number -> name, taken as the modal reading weighted by frames seen.

    Weighting by `samples` rather than by run count is what makes this robust:
    a recap window produces as many runs as a real map produces in the same
    span, but the real series occupies 99% of the LIVE frames.
    """
    votes: dict[int, Counter] = defaultdict(Counter)
    for run in runs:
        value = run["value"]
        if value["number"] is not None and value["name"] is not None:
            votes[value["number"]][value["name"]] += run["samples"]
    return {number: names.most_common(1)[0][0] for number, names in votes.items()}


def split_foreign(runs: list[dict], roster: dict[int, str] | None = None):
    """Partition runs into (this series, some other one)."""
    roster = roster or series_roster(runs)
    ours, theirs = [], []
    for run in runs:
        value = run["value"]
        pair = (
            {value["number"]: value["name"]}
            if value["number"] is not None and value["name"] is not None
            else {}
        )
        (theirs if foreign(pair, roster) else ours).append(run)
    return ours, theirs


def by_slot(runs: list[dict]) -> dict[tuple[str, int], list[dict]]:
    out: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for run in runs:
        out[(run["side"], run["slot"])].append(run)
    for slot_runs in out.values():
        slot_runs.sort(key=lambda r: r["start"])
    return out
