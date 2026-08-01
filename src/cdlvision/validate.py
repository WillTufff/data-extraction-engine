"""Structural consistency checks over parsed box scores.

These check extracted numbers against invariants the game itself guarantees,
so they validate frames for which no hand transcription exists. That matters:
hand-labelled ground truth is expensive and covers a handful of frames, while
invariants cover every frame for free.

The load-bearing one is kill/death reconciliation. Every kill credited to one
team is a death for the other, so a team's kill total must equal the opposing
team's death total minus that team's non-player deaths (suicides, explosives,
fall damage). Kills can therefore never exceed opposing deaths — that direction
is a hard error, while a small shortfall is expected and is reported as the
implied non-player death count.
"""

from __future__ import annotations

from dataclasses import dataclass

from .boxscore import BoxScore


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


def _split_kd(text: str) -> tuple[int, int]:
    kills, _, deaths = text.partition("/")
    return int(kills), int(deaths)


def reconcile_kills_deaths(score: BoxScore, team_size: int = 4) -> list[Check]:
    """Check that each team's kills are accounted for by opposing deaths."""
    rows = [_split_kd(p.text("kills_deaths")) for p in score.players]
    if len(rows) != team_size * 2:
        return [Check("kills_deaths_shape", False,
                      f"expected {team_size * 2} players, got {len(rows)}")]

    a, b = rows[:team_size], rows[team_size:]
    checks: list[Check] = []
    for label, us, them in (("team_a", a, b), ("team_b", b, a)):
        kills = sum(k for k, _ in us)
        opposing_deaths = sum(d for _, d in them)
        if kills > opposing_deaths:
            checks.append(Check(
                f"{label}_kills_vs_opposing_deaths", False,
                f"{kills} kills exceeds {opposing_deaths} opposing deaths "
                f"by {kills - opposing_deaths}",
            ))
        else:
            checks.append(Check(
                f"{label}_kills_vs_opposing_deaths", True,
                f"{kills} kills, {opposing_deaths} opposing deaths "
                f"({opposing_deaths - kills} non-player)",
            ))
    return checks


def numeric_wellformed(score: BoxScore) -> list[Check]:
    """Check that every parsed cell is syntactically valid for its column."""
    checks: list[Check] = []
    for i, player in enumerate(score.players):
        for name, f in player.fields.items():
            if not f.matches:
                continue
            text = f.text
            ok = (
                text.replace("/", "", 1).isdigit() if "/" in text
                else text.replace(":", "", 1).isdigit() if ":" in text
                else text.rstrip("%").replace(".", "", 1).isdigit()
            )
            if not ok:
                checks.append(Check(f"row{i}.{name}", False, f"unparseable {text!r}"))
    if not checks:
        checks.append(Check("numeric_wellformed", True, "all cells well-formed"))
    return checks


def confidence(score: BoxScore, margin_floor: int = 8) -> list[Check]:
    """Check that no glyph was decided on a narrow margin."""
    weak = [
        (i, f.name, m.char, m.margin)
        for i, p in enumerate(score.players)
        for f in p.fields.values()
        for m in f.matches
        if m.margin < margin_floor
    ]
    if weak:
        head = ", ".join(f"row{i}.{n}={c!r}@{g}" for i, n, c, g in weak[:5])
        return [Check("confidence", False,
                      f"{len(weak)} glyphs below margin {margin_floor}: {head}")]
    return [Check("confidence", True, f"all glyphs above margin {margin_floor}")]


def run_all(score: BoxScore, margin_floor: int = 8) -> list[Check]:
    return (
        confidence(score, margin_floor)
        + numeric_wellformed(score)
        + reconcile_kills_deaths(score)
    )
