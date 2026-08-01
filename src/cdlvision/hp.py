"""Deaths read off the spectated panel's HP, and matched against the killfeed.

HP was the one field in this project with no second witness. It was reported
rather than checked -- 0..100 with no out-of-range readings, 100 on 65.9% of
samples and 0 on 5.7% -- which says the *reader* is sane but says nothing about
whether a 0 means what it looks like it means.

The killfeed answers that. The panel draws one player's HP; the killfeed draws a
row naming a victim. When the panel's player dies both must fire, a fraction of
a second apart, and neither region can derive the other: different strips,
different atlases, different code. So the pairing rate is evidence in the same
sense `validate_spectated.py` makes it.

What makes this a check rather than a coincidence, all measured on this VOD:

- **The peak is sharp and signed the right way.** Matching within a 0.35s window
  peaks at 72.0% with the killfeed *later* by 0.167s, and is 0.0% at any
  negative offset. A killfeed row cannot precede the death it reports, and it
  does not. 0.167s is exactly one frame of the 6fps panel strip.
- **The null is 2%.** Shifting the falls 30, 60 or 120 seconds -- same players,
  wrong times -- pairs at 0.7-2.2%. 72% against that floor is not chance.
- **Damage that is survived does not pair.** 169 drops of 25 HP or more where
  the player did not then die pair at 2.4%, which is the null.

That last one took two attempts and is the reason `survived_dips` exists rather
than a plain "big drop" control. Scored over *every* drop of 25 or more, the
control pairs at 37% -- because the drop immediately preceding a death is the
damage that caused it, so the naive control is contaminated with the very thing
it is controlling for. A negative control has to exclude the positive case
explicitly; one that merely looks different does not.

The offset is measured per map and does not drift: +0.167s on five of seven maps,
+0.083s on map 4 and +0.250s on map 7, which is one strip frame either way. It is
reported and used as a window centre, never silently applied to the timestamps.

The residual is the honest part. 18.8% of falls have no killfeed row within a
second, which is the killfeed's known shortfall over bright gameplay seen from a
new angle, and HP cannot separate a lost row from a suicide any more than the
gutter skull could -- both mark every death. What HP adds is *when*, to a fifth
of a second, for the player the broadcast happened to be watching.

The reverse direction is stronger and is the part worth keeping: of 198 killfeed
events whose victim is the player currently on the panel, 195 have an HP fall.
With 114 invented kills still outstanding, that says the invention is not
concentrated on the spectated player.
"""

from __future__ import annotations

import bisect
from collections import defaultdict
from dataclasses import dataclass

# A fall separated from its last positive reading by longer than this was not
# observed falling -- the panel stopped being drawn in between, and what happened
# in the gap is unknown. Kept short: at 6fps a legitimate step is 0.167s.
MAX_READ_GAP = 1.0

# Half-width of the pairing window. Two panel frames either side of the measured
# offset, which absorbs the strip drift CLAUDE.md documents without being wide
# enough to reach a neighbouring death.
PAIR_WINDOW = 0.35

# Half-width used when *locating* the offset, as against counting matches at it.
# Narrower than one panel frame, and it has to be: the delta distribution has a
# long right tail of events whose chain started late over bright gameplay, so a
# window wide enough to be generous is also wide enough to slide up that tail and
# report an offset later than the truth. Measured, a 0.35s window peaks at
# +0.333s while the residuals at that offset sit at a median of -0.167s -- the
# count says one thing and the residual says the count is wrong. A 0.125s window
# peaks where the residual is zero. Localise narrow, then count wide.
SWEEP_WINDOW = 0.125

# Where the killfeed sits relative to the panel, measured by `sweep` and equal
# to one frame of the 6fps panel strip. A default, not a correction: callers
# sweep and report before they use it.
KILLFEED_OFFSET = 1 / 6

# What counts as damage worth using as a negative control, and how long a player
# has to survive it to qualify. Three seconds is well past the 0.167s a real
# death takes to reach the killfeed.
CONTROL_DROP = 25
SURVIVAL_WINDOW = 3.0


@dataclass(frozen=True)
class Fall:
    """One observed transition of the spectated player's HP to zero."""

    t: float
    player: str
    from_hp: int
    gap: float

    @property
    def key(self) -> tuple[float, str]:
        return (self.t, self.player)


@dataclass(frozen=True)
class Pairing:
    """A fall and the killfeed event that reported it, if one did."""

    fall: Fall
    event: object | None
    delta: float | None

    @property
    def matched(self) -> bool:
        return self.event is not None


def by_field(runs: list[dict]) -> dict[str, list[dict]]:
    """Group panel runs by field name, each sorted by time."""
    out: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        out[run["field"]].append(run)
    for field_runs in out.values():
        field_runs.sort(key=lambda r: r["start"])
    return out


def value_at(runs: list[dict], t: float):
    """The value of a run-length field at a time, or None if nothing covers it."""
    lo, hi = 0, len(runs) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        run = runs[mid]
        if t < run["start"]:
            hi = mid - 1
        elif t > run["end"]:
            lo = mid + 1
        else:
            return run["value"]
    return None


def falls(panel: dict[str, list[dict]], windows=None) -> list[Fall]:
    """Every observed fall of the panel's HP to zero.

    Three conditions, and each rules out a different way of counting something
    that is not a death:

    - The previous reading must be positive. A run of zeros is one death, not
      several, and the panel holds 0 while the death cam plays.
    - The panel must be showing the *same player* either side of the fall.
      Without this, the panel cutting to a team-mate who is already dead reads
      as that team-mate dying at the moment of the cut. It costs 3 falls here,
      which is small, but the ones it costs are exactly the wrong ones.
    - The two readings must be close in time. A fall either side of a gap in
      which the panel was not drawn was not observed; the player did die, but
      not then, and pairing it against a timestamp would be inventing precision.
    """
    return _transitions(panel, windows, to_zero=True)


def survived_dips(panel: dict[str, list[dict]], windows=None,
                  min_drop: int = CONTROL_DROP,
                  survival_window: float = SURVIVAL_WINDOW) -> list[Fall]:
    """Large HP drops the player *survived* -- the negative control.

    Excluding the ones followed by a death is the whole point, and is not a
    refinement of the control but the difference between a control and a
    restatement of the hypothesis. Scored over every drop of `min_drop` or more,
    including those that precede a death, this pairs at 37%; the drop that
    precedes a death is the damage that caused it. Restricted to drops the
    player walked away from, it pairs at 2.4%, which is the null.
    """
    zeros: dict[str, list[float]] = defaultdict(list)
    for run in panel.get("hp", []):
        if run["value"] == 0:
            name = value_at(panel.get("name", []), run["start"])
            if name is not None:
                zeros[name].append(run["start"])
    for times in zeros.values():
        times.sort()

    def died_soon(player: str, t: float) -> bool:
        times = zeros.get(player, [])
        i = bisect.bisect_left(times, t)
        return i < len(times) and times[i] <= t + survival_window

    return [
        dip for dip in _transitions(panel, windows, to_zero=False,
                                    min_drop=min_drop)
        if not died_soon(dip.player, dip.t)
    ]


def _transitions(panel: dict[str, list[dict]], windows, to_zero: bool,
                 min_drop: int = 0) -> list[Fall]:
    hp = panel.get("hp", [])
    names = panel.get("name", [])
    out: list[Fall] = []
    previous = None
    for run in hp:
        value = run["value"]
        if previous is not None and previous["value"] not in (None, 0):
            drop = _drop(previous["value"], value, to_zero, min_drop)
            gap = run["start"] - previous["end"]
            if drop and gap <= MAX_READ_GAP and _inside(windows, run["start"]):
                before = value_at(names, previous["end"])
                after = value_at(names, run["start"])
                if before is not None and before == after:
                    out.append(Fall(t=run["start"], player=before,
                                    from_hp=previous["value"], gap=gap))
        if value is not None:
            previous = run
    return out


def _drop(before: int, after, to_zero: bool, min_drop: int) -> bool:
    if after is None:
        return False
    if to_zero:
        return after == 0
    return after > 0 and before - after >= min_drop


def _inside(windows, t: float) -> bool:
    return windows is None or any(w.contains(t) for w in windows)


def deaths_by_player(events) -> dict[str, list[float]]:
    """Killfeed event times, grouped by the victim's name."""
    out: dict[str, list[float]] = defaultdict(list)
    for event in events:
        if event.victim:
            out[event.victim].append(event.t)
    for times in out.values():
        times.sort()
    return out


def pair(falls_: list[Fall], events, offset: float = KILLFEED_OFFSET,
         window: float = PAIR_WINDOW) -> list[Pairing]:
    """Match each fall to at most one killfeed event, nearest first.

    Assigned globally by closeness rather than fall by fall, so one event cannot
    account for two falls. That matters in Search and Destroy, where a player
    dies once a round and a double assignment would hide exactly the duplicate
    the round check is looking for.
    """
    by_victim = deaths_by_player(events)
    by_time: dict[str, dict[float, object]] = defaultdict(dict)
    for event in events:
        if event.victim:
            by_time[event.victim][event.t] = event

    candidates = []
    for fall in falls_:
        times = by_victim.get(fall.player, [])
        centre = fall.t + offset
        i = bisect.bisect_left(times, centre - window)
        while i < len(times) and times[i] <= centre + window:
            candidates.append((abs(times[i] - centre), fall, fall.player, times[i]))
            i += 1
    candidates.sort(key=lambda c: c[0])

    taken_fall: set[tuple[float, str]] = set()
    taken_event: set[tuple[str, float]] = set()
    matched: dict[tuple[float, str], tuple[object, float]] = {}
    for _, fall, player, t in candidates:
        if fall.key in taken_fall or (player, t) in taken_event:
            continue
        taken_fall.add(fall.key)
        taken_event.add((player, t))
        matched[fall.key] = (by_time[player][t], t - (fall.t + offset))

    out = []
    for fall in falls_:
        event, delta = matched.get(fall.key, (None, None))
        out.append(Pairing(fall=fall, event=event, delta=delta))
    return out


def sweep(falls_: list[Fall], events, offsets, window: float = SWEEP_WINDOW):
    """Matched fraction at each candidate offset, for locating the offset.

    The two strips were written in separate decode passes, so the offset between
    them is a fact to be measured rather than assumed to be zero -- the same
    argument `validate_spectated.py` makes. A peak anywhere other than a frame or
    two would be a fact about the strips, not a licence to shift anything.

    Defaults to the narrow `SWEEP_WINDOW`, because a sweep is asking where the
    offset *is*, which a generous window answers badly.
    """
    return [
        (offset, sum(p.matched for p in pair(falls_, events, offset, window)))
        for offset in offsets
    ]


# How far either side of a killfeed event the panel is allowed to show the
# victim at zero HP for the death to count as corroborated. Wider and more
# forgiving than `PAIR_WINDOW` on purpose: this asks whether HP ever agreed the
# player died, not whether a fall was *observed* at the instant. The panel drops
# frames, so a death can be real, read, and still have no fall to point at.
CORROBORATION = (-1.0, 2.0)


def unwitnessed_events(events, panel: dict[str, list[dict]], windows=None,
                       span: tuple[float, float] = CORROBORATION):
    """Killfeed deaths of the player on screen that HP never corroborated.

    The direction with no innocent explanation. A fall with no event is a row the
    killfeed lost, which is documented and expected; an event naming a victim the
    panel was watching, at a moment that player's HP never read zero at all, is a
    kill that may not have happened.

    Asked as "did the panel ever show this player dead near here", not as "did a
    fall pair". The distinction is worth 8 of 198: a fall needs a *positive*
    reading immediately before it, so a death the panel joins late -- already at
    zero, no transition drawn -- has no fall to offer and is not thereby
    uncorroborated. Answering the stricter question would have reported those
    eight as suspect kills, and they are nothing of the kind.
    """
    names = panel.get("name", [])
    zeros: dict[str, list[float]] = defaultdict(list)
    for run in panel.get("hp", []):
        if run["value"] == 0:
            player = value_at(names, run["start"])
            if player is not None:
                zeros[player].append(run["start"])
    for times in zeros.values():
        times.sort()

    out = []
    for event in events:
        if not event.victim or not _inside(windows, event.t):
            continue
        if value_at(names, event.t) != event.victim:
            continue
        times = zeros.get(event.victim, [])
        i = bisect.bisect_left(times, event.t + span[0])
        if not (i < len(times) and times[i] <= event.t + span[1]):
            out.append(event)
    return out
