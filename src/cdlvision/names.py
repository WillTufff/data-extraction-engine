"""Deterministic identification of killfeed player names.

Names are classified against the closed set of players in the series rather
than read character by character: several ink-normalised templates per
player, matched by Hamming distance with a margin against the runner-up. A
same-team attacker/victim pair is reported as a flag (friendly fire is legal)
but never used to override a match.

No model, no training. Templates are bitmaps in JSON.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .glyphs import _decode, _encode, tight_crop

# A name is accepted only if it is this close to its own template (on the
# ink-normalised scale) and this far clear of the best rival. The margin is
# the primary decision; the distance ceiling guards against a name outside
# the roster entirely (e.g. a substitute).
MAX_DISTANCE = 0.70
MIN_MARGIN = 0.12

UNKNOWN = None


@dataclass(frozen=True)
class NameMatch:
    """Outcome of classifying one name region against the roster."""

    label: str | None
    distance: float
    runner_up: str | None
    runner_up_distance: float

    @property
    def margin(self) -> float:
        return self.runner_up_distance - self.distance

    @property
    def resolved(self) -> bool:
        return self.label is not None


def normalised_distance(a: np.ndarray, b: np.ndarray, max_shift: int = 1) -> float:
    """Hamming distance over the union of ink, best over small alignments.

    0 is identical and 1 is disjoint, independent of how much ink the name
    has, so "04" and "MERCULES" are scored on the same scale.
    """
    h = max(a.shape[0], b.shape[0])
    w = max(a.shape[1], b.shape[1])
    pa = np.zeros((h, w), dtype=bool)
    pb = np.zeros((h, w), dtype=bool)
    pa[: a.shape[0], : a.shape[1]] = a
    pb[: b.shape[0], : b.shape[1]] = b

    best = 1.0
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            shifted = np.roll(np.roll(pa, dy, axis=0), dx, axis=1)
            union = int(np.count_nonzero(shifted | pb))
            if not union:
                continue
            best = min(best, int(np.count_nonzero(shifted ^ pb)) / union)
    return best


def average_masks(masks: list[np.ndarray]) -> np.ndarray:
    """Pixelwise majority over observations of the same name.

    Used both to build a template from many frames and, at read time, to
    collapse the frames one killfeed entry persists for into a single probe.
    Masks are padded to the largest shape before voting.
    """
    if not masks:
        raise ValueError("no masks to average")
    h = max(m.shape[0] for m in masks)
    w = max(m.shape[1] for m in masks)
    stack = np.zeros((len(masks), h, w), dtype=np.uint8)
    for i, m in enumerate(masks):
        stack[i, : m.shape[0], : m.shape[1]] = m
    return stack.sum(axis=0) * 2 > len(masks)


class NameMatcher:
    """Classifies a name bitmap as one of the eight players, or as unknown."""

    def __init__(
        self,
        templates: dict[str, list[np.ndarray]] | None = None,
        teams: dict[str, str] | None = None,
    ):
        self.templates: dict[str, list[np.ndarray]] = {
            k: list(v) for k, v in (templates or {}).items()
        }
        self.teams: dict[str, str] = dict(teams or {})

    def add(self, label: str, mask: np.ndarray) -> None:
        self.templates.setdefault(label, []).append(tight_crop(mask))

    @property
    def players(self) -> list[str]:
        return sorted(self.templates)

    # -- matching ---------------------------------------------------------

    def rank(self, mask: np.ndarray) -> list[tuple[float, str]]:
        """Every player scored against this region, best first.

        Unlike `TemplateSet`, a shape mismatch is not a hard filter, just
        distance: the ink-normalised distance already penalizes a wrong width.
        """
        if not self.templates:
            raise ValueError("name matcher has no templates")
        probe = tight_crop(mask)
        scored = [
            (min(normalised_distance(probe, t) for t in variants), label)
            for label, variants in self.templates.items()
        ]
        scored.sort()
        return scored

    def match(
        self,
        mask: np.ndarray,
        max_distance: float = MAX_DISTANCE,
        min_margin: float = MIN_MARGIN,
    ) -> NameMatch:
        """Classify one name, returning `label=None` when it is not certain."""
        scored = self.rank(mask)
        distance, label = scored[0]
        runner_distance, runner = (
            (scored[1][0], scored[1][1]) if len(scored) > 1 else (1.0, None)
        )
        ok = distance <= max_distance and (runner_distance - distance) >= min_margin
        return NameMatch(
            label if ok else UNKNOWN, distance, runner, runner_distance
        )

    def same_team(self, a: NameMatch, v: NameMatch) -> bool:
        """Whether a resolved row puts both names on one team.

        Reported for the caller's information, never used to alter a match.
        """
        if not (a.resolved and v.resolved and self.teams):
            return False
        return self.teams.get(a.label) == self.teams.get(v.label)

    def match_row(
        self,
        attacker: np.ndarray,
        victim: np.ndarray,
        max_distance: float = MAX_DISTANCE,
        min_margin: float = MIN_MARGIN,
        enforce_cross_team: bool = False,
    ) -> tuple[NameMatch, NameMatch]:
        """Classify both names in a row.

        By default the two names are read independently. `enforce_cross_team`
        picks the lowest-total-distance opposite-team pair instead, for a
        ruleset where friendly fire cannot occur. Either way, an uncertain
        name is left unresolved rather than guessed at.
        """
        if not enforce_cross_team or not self.teams:
            return (
                self.match(attacker, max_distance, min_margin),
                self.match(victim, max_distance, min_margin),
            )
        ra, rv = self.rank(attacker), self.rank(victim)

        best: tuple[float, tuple[float, str], tuple[float, str]] | None = None
        for da, la in ra:
            for dv, lv in rv:
                if self.teams.get(la) == self.teams.get(lv):
                    continue
                total = da + dv
                if best is None or total < best[0]:
                    best = (total, (da, la), (dv, lv))
        if best is None:
            return (
                NameMatch(UNKNOWN, 1.0, None, 1.0),
                NameMatch(UNKNOWN, 1.0, None, 1.0),
            )

        _, (da, la), (dv, lv) = best
        return (
            self._constrained(ra, da, la, max_distance, min_margin),
            self._constrained(rv, dv, lv, max_distance, min_margin),
        )

    def _constrained(
        self,
        scored: list[tuple[float, str]],
        distance: float,
        label: str,
        max_distance: float,
        min_margin: float,
    ) -> NameMatch:
        """Wrap a constraint-chosen candidate, scored against its best rival."""
        rivals = [(d, s) for d, s in scored if s != label]
        runner_distance, runner = rivals[0] if rivals else (1.0, None)
        ok = distance <= max_distance and (runner_distance - distance) >= min_margin
        return NameMatch(label if ok else UNKNOWN, distance, runner, runner_distance)

    # -- persistence ------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "name": "killfeed_names",
            "teams": self.teams,
            "templates": {
                label: [_encode(t) for t in variants]
                for label, variants in sorted(self.templates.items())
            },
        }
        path.write_text(json.dumps(payload, indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> NameMatcher:
        payload = json.loads(Path(path).read_text())
        return cls(
            {
                label: [_decode(rows) for rows in variants]
                for label, variants in payload["templates"].items()
            },
            payload.get("teams", {}),
        )

    def __len__(self) -> int:
        return sum(len(v) for v in self.templates.values())

    def __contains__(self, label: str) -> bool:
        return label in self.templates

    def __repr__(self) -> str:
        return (
            f"NameMatcher(players={len(self.templates)}, "
            f"templates={len(self)})"
        )
