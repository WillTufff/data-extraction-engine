"""Rounds, derived from the header and populated from the killfeed.

The first module here that produces *game semantics* rather than screen
contents. Everything below it answers "what was drawn"; this answers "what
happened" -- who died in what order, which side won the round and how.

**Only Search and Destroy has rounds, and that was measured rather than
assumed.** The premise going in was that the two non-Hardpoint modes were both
round-based. Overload is not: its clock counts down from 5:00 across the whole
map, its substrip reads FIRST HALF / SECOND HALF, it has no lives counter, and
its score band is drawn continuously -- two runs over a map, not one per round.
Two of its score steps sit 19.5 seconds apart, which no round could be. So
Overload is a timed mode with halves and scoring events, and it is described
here as such rather than forced into a shape it does not have.

Three witnesses find a Search and Destroy round boundary, and the useful thing
is that they are three different *fields*:

- **The round clock jumps back to 1:30.** The sharpest, exact to the second, and
  on its own it recovers every round on all three maps.
- **Both lives counters return to 4.**
- **The map score steps by one.**

The tolerance between them is 25 seconds, where `maps.py` uses 2. That is not
slack, it is latency: the three fire at genuinely different moments of the same
break -- the clock when the next round's timer appears, the score at some point
during the break, the lives counter when the panel repaints. A boundary is a
break, not an instant, and pretending otherwise would make two of the three
witnesses look broken.

Header runs must be gated on `maps.live_series_spans` before any of this. The
one-second replay clips at t=2580 and t=8761 draw a valid round clock and a
valid score, and each manufactures a phantom round if let through.

Two findings that came out of building it:

- **No round is missing, and the proof is arithmetic rather than a witness.**
  Every map's last drawn score is 5-5, which looks like a reading failure. It is
  not: the score *before* each round forms a staircase stepping by exactly one
  with no jumps of two, so no round is unaccounted for. All three Search and
  Destroy maps ran 11 rounds and finished 6-5. What is genuinely missing is only
  the final round's score update, which the broadcast never draws in a LIVE
  frame because the map ends on it.
- **One death per player per round is an exact upper bound the killfeed can
  violate**, and unlike the box score -- which can only report excess as a total
  of 50 -- it names the offending pair of events. See `duplicate_victims`.
- **Every round now has an outcome, because the reader changed rather than the
  reasoning.** While the clock stopped dead at 0:30 no round could be shown to
  have run out of time, and a third of them fell through unclassified. Once
  `header.read_clock` learned the final-thirty counter, "time" and "detonation"
  became witnessable and the trichotomy -- a round ends by bomb, by wipe or by
  time and by nothing else -- closed the rest. 19 wipes, 6 plants whose ending is
  not witnessed, 5 expiries and 3 detonations, with nothing unknown. See
  `_outcome` for the one distinction still deliberately not drawn.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from .header import seconds as header_seconds
from .maps import MIN_RUN_SECONDS, live_series_spans, pair_series

ROUNDS_MODE = "search_and_destroy"
HALVES_MODE = "overload"
HILLS_MODE = "hardpoint"

# A jump in the clock to at least this many seconds is the next round's timer
# appearing. The round is 1:30 and the first reading after a reset lands at
# 1:25-1:28, so the bar sits well below that and well above the bomb.
ROUND_CLOCK = 80

# The bomb timer replaces the round clock in the same band in the same font,
# starting at 0:44. An upward jump into this band is a plant, not a round.
#
# Tight around 44 because it can afford to be: measured, all nine plants in the
# series jump to exactly 44.0, and the only other upward jump below 1:20 anywhere
# is the t=2580 replay clip landing on 0:48. A band of 40-50 admitted that clip
# as a tenth plant, in the round it was already corrupting.
PLANT_CLOCK = (43, 45)

# Below 0:30 the clock is a different counter -- red, seconds and tenths, a
# decimal point for a colon. `header.read_clock` reads it now; it did not when
# this module was first written, and every round therefore ended at an apparent
# 0:30 whether it ran to time or not. A counter this close to zero is expiring.
EXPIRY_CLOCK = 0.6

# How far apart two witnesses may fire and still be the same boundary. Twenty
# five seconds where `maps.WITNESS_TOLERANCE` is two, because these three
# mechanisms mark different moments of the same between-round break rather than
# the same instant read three ways.
WITNESS_TOLERANCE = 25.0

# Lives each side starts a round with.
LIVES = 4

SIDES = ("left", "right")


@dataclass
class RoundBoundary:
    """A candidate round start and which mechanisms saw it."""

    t: float
    witnesses: dict[str, float] = field(default_factory=dict)

    @property
    def confirmed(self) -> bool:
        return len(self.witnesses) >= 3


@dataclass
class Round:
    """One Search and Destroy round."""

    map_number: int
    number: int
    start: float
    end: float
    score_before: tuple[int | None, int | None] = (None, None)
    winner: str | None = None
    decided_by: str | None = None
    outcome: str | None = None
    wiped: str | None = None
    plant_t: float | None = None
    lives_low: tuple[int | None, int | None] = (None, None)
    kills: list = field(default_factory=list)
    witnesses: tuple[str, ...] = ()

    def contains(self, t: float) -> bool:
        return self.start <= t < self.end

    @property
    def first_blood(self):
        """The first kill of the round, which is the one that matters most."""
        return self.kills[0] if self.kills else None

    def victims(self, sides: dict[str, str], side: str) -> list[str]:
        return [e.victim for e in self.kills
                if e.victim and sides.get(e.victim) == side]


# -- the three witnesses ---------------------------------------------------


def parse_clock(value) -> float | None:
    """Either drawn form of the countdown, in seconds.

    Delegates to the header, which knows there are two of them: "1:27" for the
    ordinary clock and "6.5" for the final thirty seconds.
    """
    return header_seconds(value) if isinstance(value, str) else None


def clock_series(header_runs: list[dict], window) -> list[tuple[float, int]]:
    """(time, seconds) for the clock inside a window, in order.

    Restricted to runs the header itself calls the live series, and -- for the
    ordinary clock only -- to runs that lasted longer than a frame. See
    `_long_enough` for why that qualifier is load-bearing.
    """
    live = live_series_spans(header_runs)
    out = []
    for run in sorted(header_runs, key=lambda r: r["start"]):
        if run["field"] != "clock" or not _long_enough(run):
            continue
        if not window.contains(run["start"]):
            continue
        if not any(a <= run["start"] <= b for a, b in live):
            continue
        seconds = parse_clock(run["value"])
        if seconds is not None:
            out.append((run["start"], seconds))
    return out


def _long_enough(run: dict) -> bool:
    """Whether a clock run lasted long enough to be a state rather than a slip.

    The two-frame rule does not apply to the final-thirty counter, and applying
    it silently discarded every reading of it. The rule exists because at 6fps a
    real state is drawn at least six times, so a value that appears once and
    disagrees with identical neighbours is a misread. That premise is exactly
    false here: the tenths counter changes ten times a second and is sampled six
    times a second, so *every* reading is a run of one frame and there are no
    identical neighbours for it to disagree with. Filtering on duration would
    keep only the value it rests on at zero -- which is what it did, and which
    looked like the counter jumping from 30 straight to 0.
    """
    if isinstance(run["value"], str) and "." in run["value"]:
        return True
    return run["end"] - run["start"] >= MIN_RUN_SECONDS


def clock_resets(header_runs: list[dict], window) -> list[float]:
    """Times the round clock jumps back up to the round length.

    The sharpest of the three witnesses and the only one that needs no
    tolerance: on this VOD it lands to the second on the score band restarting.
    """
    series = clock_series(header_runs, window)
    return [
        t for (_, before), (t, after) in zip(series, series[1:])
        if after > before and after >= ROUND_CLOCK
    ]


def plants(header_runs: list[dict], window) -> list[float]:
    """Times the bomb timer replaces the round clock.

    In Search and Destroy the clock is not one counter: on a plant the round
    clock is replaced by the bomb timer, drawn in the same band in the same font,
    starting at 0:44. That is why the header's monotonicity check is written as
    an invariant to be tested rather than assumed -- and it is what makes the
    plant free to detect here.
    """
    series = clock_series(header_runs, window)
    return [
        t for (_, before), (t, after) in zip(series, series[1:])
        if after > before and PLANT_CLOCK[0] <= after <= PLANT_CLOCK[1]
    ]


def lives_series(header_runs: list[dict], window) -> list[tuple[float, tuple]]:
    live = live_series_spans(header_runs)
    return [
        (t, pair) for t, _, pair in pair_series(header_runs, "lives_left",
                                                "lives_right")
        if window.contains(t) and any(a <= t <= b for a, b in live)
    ]


def lives_resets(header_runs: list[dict], window) -> list[float]:
    """Times both lives counters are back to full having not been.

    Read off the pair rather than each side, for the reason `maps.score_resets`
    gives about the map score: a side that lost nothing last round never steps,
    so testing the two separately misses the boundary entirely.
    """
    series = lives_series(header_runs, window)
    out = []
    for (_, before), (t, after) in zip(series, series[1:]):
        if after == (LIVES, LIVES) and before != (LIVES, LIVES):
            out.append(t)
    return out


def score_series(header_runs: list[dict], window) -> list[tuple[float, tuple]]:
    live = live_series_spans(header_runs)
    return [
        (t, pair) for t, _, pair in pair_series(header_runs, "score_left",
                                                "score_right")
        if window.contains(t) and any(a <= t <= b for a, b in live)
    ]


def score_steps(header_runs: list[dict], window) -> list[tuple[float, tuple, str]]:
    """Times the map score gains a point, with the side that gained it."""
    out = []
    for (_, before), (t, after) in zip(score_series(header_runs, window),
                                       score_series(header_runs, window)[1:]):
        if after[0] == before[0] + 1 and after[1] == before[1]:
            out.append((t, after, "left"))
        elif after[1] == before[1] + 1 and after[0] == before[0]:
            out.append((t, after, "right"))
    return out


def filter_to_series(events, header_runs: list[dict]):
    """Drop killfeed events drawn by recap footage.

    A map window is not enough. `maps.filter_to_series` removes recap runs from
    the *scoreboard*, and the map windows it produces still span the replays
    inside them -- the one-second cut at t=2580 sits in the middle of map 2's
    round 9. That clip draws a killfeed of an earlier moment of the same map,
    and its rows enter as events 30 seconds displaced.

    Left in, they look exactly like the thing this module is built to detect:
    three duplicate victims in one round and a side losing six players. The
    tell is the interval. A fragmented chain repeats within five seconds; these
    repeated at 29 to 33. A duplicate whose two halves are half a minute apart
    is not one entry read twice, it is two different moments of the map on
    screen -- which is the signature CLAUDE.md records for recap footage
    manufacturing errors in both directions at once.
    """
    live = live_series_spans(header_runs)
    return [e for e in (events or [])
            if any(a <= e.t <= b for a, b in live)]


# -- putting them together -------------------------------------------------


def boundaries(header_runs: list[dict], window) -> list[RoundBoundary]:
    """Every candidate round start in a map, annotated with its witnesses.

    Clustered on the clock, which is the mechanism that fires first and most
    precisely. A witness with no clock reset beside it still gets a boundary of
    its own, so a missing reset shows up as an unconfirmed boundary rather than
    as nothing at all -- the same shape as `maps.boundaries`.
    """
    seen = {
        "clock_reset": clock_resets(header_runs, window),
        "lives_reset": lives_resets(header_runs, window),
        "score_step": [t for t, _, _ in score_steps(header_runs, window)],
    }
    out: list[RoundBoundary] = []
    for name in ("clock_reset", "lives_reset", "score_step"):
        for t in seen[name]:
            match = next(
                (b for b in out
                 if abs(b.t - t) <= WITNESS_TOLERANCE and name not in b.witnesses),
                None,
            )
            if match is None:
                match = RoundBoundary(t=float(t))
                out.append(match)
            match.witnesses[name] = float(t)
    out.sort(key=lambda b: b.t)
    return out


def derive(window, header_runs: list[dict], events=None,
           sides: dict[str, str] | None = None,
           strict: bool = False) -> list[Round]:
    """Rounds for one map, or an empty list for a mode that has none.

    `sides` maps a player name to "left" or "right"
    (`scoreboard_runs.player_sides`). It is optional because the boundaries do
    not need it, and used only to let the killfeed close the final round.
    """
    if window.mode != ROUNDS_MODE:
        return []

    found = boundaries(header_runs, window)
    unconfirmed = [b for b in found if not b.confirmed]
    if unconfirmed and strict:
        raise ValueError(
            f"map {window.number}: round boundaries disagree between "
            "mechanisms: " + ", ".join(
                f"t={b.t:.0f} seen only by {sorted(b.witnesses)}"
                for b in unconfirmed
            )
        )

    edges = [window.start] + [b.t for b in found]
    events = filter_to_series(events, header_runs)
    steps = score_steps(header_runs, window)
    scores = score_series(header_runs, window)
    planted = plants(header_runs, window)
    clock = clock_series(header_runs, window)
    lives = lives_series(header_runs, window)

    out: list[Round] = []
    for number, start in enumerate(edges, start=1):
        end = edges[number] if number < len(edges) else window.end
        witnesses = (("map_start",) if number == 1
                     else tuple(sorted(found[number - 2].witnesses)))
        inside = [e for e in (events or []) if start <= e.t < end]
        inside.sort(key=lambda e: e.t)
        # The side that scored during *this* round's break is the side that won
        # it, and that step is recorded at the next round's boundary.
        won = [side for t, _, side in steps if end - WITNESS_TOLERANCE
               <= t <= end + WITNESS_TOLERANCE]
        plant = next((t for t in planted if start <= t < end), None)
        low = _lives_low(lives, start, end)
        rnd = Round(
            map_number=window.number,
            number=number,
            start=start,
            end=end,
            score_before=_score_at(scores, start),
            winner=won[0] if len(won) == 1 else None,
            decided_by="score_step" if len(won) == 1 else None,
            plant_t=plant,
            lives_low=low,
            kills=inside,
            witnesses=witnesses,
        )
        rnd.wiped, rnd.outcome = _outcome(rnd, clock)
        out.append(rnd)
    _close_last(out, sides)
    return out


def _score_at(series, t: float) -> tuple[int | None, int | None]:
    """The map score as it stood entering a round.

    The last pair at or before the round starts, not the first pair inside it.
    The score updates during the break, a few seconds *before* the next round's
    clock appears, so a reading taken from inside the round is either missing --
    the step already happened outside the span -- or is the step that ends the
    round, which is the score *after* it. Both were wrong in the first version
    and both looked plausible: rounds reported either (None, None) or a score one
    point ahead of the truth.
    """
    before = [pair for start, pair in series if start <= t]
    return before[-1] if before else (None, None)


def _lives_low(series, start: float, end: float) -> tuple[int | None, int | None]:
    """The lowest lives reading each side showed during a round.

    A *lower bound* on deaths, not a count, and the distinction matters: the
    counter is not read at every round's end, so `LIVES - low` under-reports
    whenever the last reading came before the last death. Summed over map 2 it
    gives 23 where the box score says 27. It is used as a bound and never as an
    equality.
    """
    inside = [pair for t, pair in series if start <= t < end]
    if not inside:
        return (None, None)
    return (min(p[0] for p in inside), min(p[1] for p in inside))


def _outcome(rnd: Round, clock) -> tuple[str | None, str | None]:
    """How the round ended.

    All four Search and Destroy endings are observable now that the final thirty
    seconds are read; before that, two of them were not, and the first version of
    this function said so rather than guessing. What changed is the *reader*, not
    the reasoning.

    - **detonation** -- a plant whose bomb timer runs to zero, or a plant where
      the side that *won* is the side that was wiped, since a team with nobody
      left can only have won on the bomb. The second path found the case on map 5
      round 4 that caught this function asserting a wiped side never wins.
    - **time** -- no plant, and the round clock itself expires. Newly available:
      this could not be seen at all until `header.read_clock` learned the
      final-thirty counter, because every round appeared to stop at 0:30.
    - **wipe** -- no plant, and the clock did *not* expire. A Search and Destroy
      round ends by bomb, by wipe, or by time and by nothing else, so ruling out
      two names the third. The lives counter confirms it where it was read, and
      the inference no longer depends on its having been.
    - **plant** -- a plant whose ending is not witnessed.

    "Defuse" is deliberately absent. Telling a defuse from the planters being
    killed needs to know *which side planted*, and the overlay never draws it: a
    plant with the bomb still running and the non-wiped side winning is
    consistent with both. Naming one would invent a distinction the pixels do not
    carry, which is the call `name_killfeed_weapons.py` made for JAGER 45 /
    MPC-25.
    """
    wiped = next(
        (side for side, low in zip(SIDES, rnd.lives_low) if low == 0), None
    )
    if rnd.plant_t is not None:
        bomb = [s for t, s in clock if rnd.plant_t <= t < rnd.end]
        if (bomb and min(bomb) <= EXPIRY_CLOCK) or \
                (wiped is not None and rnd.winner == wiped):
            return wiped, "detonation"
        return wiped, "plant"
    seen = [s for t, s in clock if rnd.start <= t < rnd.end]
    if seen and min(seen) <= EXPIRY_CLOCK:
        return wiped, "time"
    if wiped is not None or seen:
        return wiped, "wipe"
    return wiped, None


def _close_last(rounds: list[Round], sides: dict[str, str] | None) -> None:
    """Name the winner of the final round, which no score step witnesses.

    The map ends on it, so the broadcast never draws the winning score in a LIVE
    frame -- every map's last reading is 5-5. The round is real and the killfeed
    can often close it: a side losing all four players was wiped, and the other
    side won. Where even that is silent the winner stays None rather than being
    guessed, and `decided_by` says which mechanism spoke.
    """
    if not rounds:
        return
    last = rounds[-1]
    if last.winner is not None:
        return
    if last.wiped is not None:
        last.winner = _other(last.wiped)
        last.decided_by = "lives_wipe"
        return
    # The killfeed is the second way in, and it closes rounds the lives counter
    # does not: it names four distinct victims on a side where the counter was
    # simply not read at the moment it hit zero.
    if sides:
        for side in SIDES:
            if len(set(last.victims(sides, side))) >= LIVES:
                last.winner = _other(side)
                last.decided_by = "killfeed_wipe"
                last.wiped = side
                last.outcome = last.outcome or "wipe"
                return


def _other(side: str) -> str:
    return SIDES[1] if side == SIDES[0] else SIDES[0]


def close_from_series(rounds: list[Round], map_winner: str | None) -> None:
    """Last resort for the final round: the map's own winner.

    A map is won by taking its last round, so if the series pips say who won the
    map they have said who won the round. This is weaker than the other paths --
    it derives the round from the map rather than the map from its rounds -- so
    it is applied explicitly by the caller and recorded as such, never folded
    into `derive`.
    """
    if rounds and rounds[-1].winner is None and map_winner is not None:
        rounds[-1].winner = map_winner
        rounds[-1].decided_by = "series_pips"


def map_winners(windows) -> dict[int, str | None]:
    """Which side won each map, from the series pips advancing."""
    out: dict[int, str | None] = {}
    for previous, current in zip(windows, windows[1:]):
        if previous.pips is None or current.pips is None:
            out[previous.number] = None
        elif current.pips[0] == previous.pips[0] + 1:
            out[previous.number] = "left"
        elif current.pips[1] == previous.pips[1] + 1:
            out[previous.number] = "right"
        else:
            out[previous.number] = None
    return out


# -- the two modes that have no rounds -------------------------------------


@dataclass
class Segment:
    """A map subdivision that is not a round: a Hardpoint hill or an Overload
    half. Kept apart from `Round` on purpose -- a hill has no winner, no lives
    and no plant, and giving it those fields as None would invite code to treat
    the three modes as one thing when they are not."""

    map_number: int
    number: int
    kind: str
    start: float
    end: float
    kills: list = field(default_factory=list)
    witnesses: tuple[str, ...] = ()


def hills(window, header_runs: list[dict], events=None) -> list[Segment]:
    """Hardpoint hill rotations, off the rotation timer.

    Hardpoint has no rounds; it has hills, and the header already carries the
    countdown to the next one. The cadence is exact: on both Hardpoint maps the
    timer resets at 61-second intervals, eleven times, so a map is twelve hills.
    One witness rather than the three a Search and Destroy round gets -- there is
    no second mechanism drawing the hill index -- and it is written here as one
    witness rather than dressed up as agreement.
    """
    if window.mode != HILLS_MODE:
        return []
    live = live_series_spans(header_runs)
    series = [
        (r["start"], parse_clock(r["value"]))
        for r in sorted(header_runs, key=lambda r: r["start"])
        if r["field"] == "rotation" and r["end"] - r["start"] >= MIN_RUN_SECONDS
        and window.contains(r["start"])
        and any(a <= r["start"] <= b for a, b in live)
    ]
    edges = [window.start] + [
        t for (_, before), (t, after) in zip(series, series[1:])
        if before is not None and after is not None and after > before
    ]
    return _segments(window, edges, "hill", ("rotation_reset",), events)


def halves(window, header_runs: list[dict], events=None) -> list[Segment]:
    """Overload halves, off the substrip naming them.

    Overload is a timed mode: one 5:00 clock across the map, no lives counter, a
    continuously drawn score, and a substrip that says which half it is. The
    substrip is the only thing that marks the division, and it says so in words.
    """
    if window.mode != HALVES_MODE:
        return []
    live = live_series_spans(header_runs)
    edges: list[float] = []
    seen: set[str] = set()
    for run in sorted(header_runs, key=lambda r: r["start"]):
        if run["field"] != "substrip" or run["end"] - run["start"] < MIN_RUN_SECONDS:
            continue
        if not window.contains(run["start"]):
            continue
        if not any(a <= run["start"] <= b for a, b in live):
            continue
        label = run["value"]
        if isinstance(label, str) and label.endswith("HALF") and label not in seen:
            seen.add(label)
            edges.append(run["start"] if edges else window.start)
    return _segments(window, edges or [window.start], "half", ("substrip",), events)


def _segments(window, edges: list[float], kind: str, witnesses: tuple,
              events) -> list[Segment]:
    out = []
    for number, start in enumerate(edges, start=1):
        end = edges[number] if number < len(edges) else window.end
        inside = sorted((e for e in (events or []) if start <= e.t < end),
                        key=lambda e: e.t)
        out.append(Segment(
            map_number=window.number, number=number, kind=kind,
            start=start, end=end, kills=inside,
            witnesses=("map_start",) if number == 1 else witnesses,
        ))
    return out


def subdivide(window, header_runs: list[dict], events=None,
              sides: dict[str, str] | None = None):
    """Whatever this map's mode divides into: rounds, hills or halves.

    One entry point so a caller does not have to know which mode does what, and
    so adding a mode is a change in one place.
    """
    if window.mode == ROUNDS_MODE:
        return derive(window, header_runs, events, sides)
    if window.mode == HILLS_MODE:
        return hills(window, header_runs, events)
    if window.mode == HALVES_MODE:
        return halves(window, header_runs, events)
    return []


# -- what the killfeed's ordering buys -------------------------------------


# Below this a duplicate is one entry read twice; above it, it is two different
# moments of the map on screen. Measured, the two populations do not overlap and
# are not close to overlapping: fragments run 0.75 to 4.33 seconds and replays
# 29.75 to 33.00, so the cut is placed in the empty middle rather than fitted.
FRAGMENT_GAP = 10.0


def duplicate_victims(rnd: Round) -> list[tuple]:
    """Pairs of events naming the same victim twice in one round, classified.

    In Search and Destroy a player dies at most once per round, so this is an
    exact upper bound rather than a heuristic -- and the killfeed violates it,
    which makes it a detector. The box score can only report this class as a
    total of 50 across the series and `localise_killfeed.py` can only bracket it
    in time; this names the pair of events.

    Yields `(victim, first, second, kind)`. The interval separates two unrelated
    causes completely, which is the part worth keeping:

    - **"fragment"**, 0.75 to 4.33s apart, nine of them. Same victim, almost
      always the same attacker, and at least one member is a chain of 5 to 7
      frames sitting right on `events.DENSE_12HZ`'s `min_frames=5` floor. Chain
      fragmentation caught in the act.
    - **"replay"**, 29.75 to 33.00s apart, three of them, and all three have
      their second event at *exactly* t=2580.0 in 12-frame chains -- a
      one-second clip at 12fps. That is the recap CLAUDE.md records as the one
      no counter in the header can distinguish, because it replays the same map
      at the same round score. A rule about the game rather than about the
      overlay finds it anyway.

    Both are reported and neither is tuned away. Moving `min_frames` off nine
    observations of one mode would be fitting a global threshold to a corner.
    """
    seen: dict[str, list] = defaultdict(list)
    for event in rnd.kills:
        if event.victim:
            seen[event.victim].append(event)
    out = []
    for victim, group in seen.items():
        group.sort(key=lambda e: e.t)
        for a, b in zip(group, group[1:]):
            kind = "fragment" if b.t - a.t <= FRAGMENT_GAP else "replay"
            out.append((victim, a, b, kind))
    return out


def over_four(rnd: Round, sides: dict[str, str]) -> list[str]:
    """Sides the killfeed says lost more than four players in one round."""
    return [
        side for side in SIDES
        if len(rnd.victims(sides, side)) > LIVES
    ]


def staircase_gaps(window, header_runs: list[dict]) -> list[tuple]:
    """Places the score before consecutive rounds jumps by more than one.

    The argument that no round is missing. Every map's last drawn score is 5-5,
    which looks like a failure to read the winning point; what rules out a
    *missing round* is that the score entering each round steps by exactly one
    throughout. A skipped round would show up here as a jump of two, and none
    does on any of the three maps.
    """
    series = score_series(header_runs, window)
    out = []
    for (_, before), (t, after) in zip(series, series[1:]):
        if sum(after) - sum(before) not in (0, 1):
            out.append((t, before, after))
    return out
