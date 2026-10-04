"""What a musical position and a note value are allowed to be.

These sets are the stage's contract with itself: the model can only ever write a
duration or a position that appears here, so their coverage of real scores is an
upper bound on how right any output can be.

Measured on a composer-balanced sample -- one score per composer for all sixteen
ASAP composers, picked with `random.Random(0).choice(sorted(found))` rather than
by alphabetical prefix, 54,458 notes. An earlier alphabetical sample read 98.43%
because ASAP's first entries are all Bach, which barely uses the figures this
vocabulary cannot write; that bug has appeared four times in this project, so
every figure below is from the seeded pick.

                        writable durations  onsets   durations  bar positions
    (1,2,3,4,6,8,12,16)                 96     145      96.85%         97.97%
    (1,2,3,4,6,8,10,12,16,24)          160     241      98.51%         99.41%

Denominator **10** buys 898 of those 907 newly writable notes on its own: it is
the quintuplet family (1/5, 1/10, 3/10, 2/5, 3/5, 9/10), and it is worth more
per token than anything else available -- 11.2 notes per token added, against
5.6 for 20 and 2.8 for 40, which cover the same notes with more slots.
Rachmaninoff's Prelude op. 32/5 goes from 55.25% of its durations to 96.10% and
from 67.32% of its positions to 100%; Glinka 92.42% -> 97.52%, Scriabin 94.91%
-> 97.34%.

Denominator **24** buys only 9 notes in 54,458, and it is here anyway, because
of which 9. A 64th-note triplet lasts 1/24 of a quarter; with no slot for it,
each note snapped to 1/16 -- LONGER than the 1/24 gap to the next onset -- so
the three notes of Beethoven Op. 7 No. 2's bar 11 overlapped inside one voice,
music21's exporter laid them end to end, the bar exported as 49/16 instead of 3
and all 76 bars after it shifted. Coverage counts notes; this counts files. It
also makes 5/24 a position, which the same bar needs.

Widening further was measured and rejected: adding 14 and 15 (septuplets, and
1/15) costs 80 more duration slots and 120 more onset slots -- every one of them
a class the model's output head has to learn -- for 0.18 points of coverage.

Coverage is **not uniform across composers**, so a per-composer result is read
against its own ceiling. Under the wider grid Mozart still loses 6.9% of its
durations and Debussy 2.4%; the remainder is written-out cadenza figuration with
onsets like 109/40, 40/63 and 33/47 that no fixed grid carries. `build.py` does
not depend on this grid being adequate -- it clamps a note that would overrun
the next onset in its voice -- and that guard, not this table, is what stops the
overlap class of failure.

Of the residual gap, 1.19 points are **grace notes**, which carry duration 0 and
are not a rounding error: a grace note is the only zero-duration thing in the
corpus, and written as if it lasted a real note value it lands in the same voice
at the same onset as the note it decorates, where the exporter lays the two end
to end. Slot 0 of `DURATIONS` is reserved for them, so "no written duration" is
distinguishable from the shortest real value. Counting that slot as coverage the
wider grid reaches 99.70% of durations.

Raising the duration LIMIT does nothing at all: only 4 notes in the sample last
longer than four quarters. The onset limit is a different matter -- `ONSETS` and
`BAR_LENGTHS` both stop at 6 quarters, so a 4/2 or 12/4 bar clamps. That is a
known limitation of its own and is not addressed here.
"""

from fractions import Fraction

DENOMINATORS = (1, 2, 3, 4, 6, 8, 10, 12, 16, 24)
DURATION_LIMIT = Fraction(4)
ONSET_LIMIT = Fraction(6)

VELOCITY_BUCKETS = 8
UNKNOWN_VELOCITY = 0


def family(denominators, limit: Fraction, include_zero: bool) -> tuple[Fraction, ...]:
    """Every n/d at or below `limit`, sorted, without duplicates."""
    values = set()
    for denominator in denominators:
        numerator = 0 if include_zero else 1
        while Fraction(numerator, denominator) <= limit:
            values.add(Fraction(numerator, denominator))
            numerator += 1
    return tuple(sorted(values))


# Slot 0 is "no written duration at all" -- a grace note. It has to be a slot of
# its own: 1/16 was both the smallest duration and the entry at index 0, so one
# index meant either "grace" or "sixteenth" and nothing downstream could tell
# them apart. `index_of` skips the reserved slot, so no real duration lands on
# it, exactly as `BAR_LENGTHS` reserves its own slot 0 below.
DURATIONS = (None, *family(DENOMINATORS, DURATION_LIMIT, include_zero=False))
GRACE_DURATION = 0
# The shortest value that can actually be written, used wherever a duration has
# to be floored at something that is still a note.
SHORTEST_DURATION = min(value for value in DURATIONS if value is not None)
ONSETS = family(DENOMINATORS, ONSET_LIMIT, include_zero=True)
# Slot 0 means "this note does not begin a bar"; the rest are bar durations.
BAR_LENGTHS = (None, *family(DENOMINATORS, ONSET_LIMIT, include_zero=False))


def index_of(value: Fraction, table) -> int:
    """The slot whose value is nearest to `value`.

    Never raises. A quintuplet has no exact slot in this family and must still be
    written somewhere -- badly placed beats unwritten, and the coverage figures
    above say how often that happens.

    A `None` entry is a RESERVED slot -- "no bar starts here" in `BAR_LENGTHS`,
    "this is a grace note" in `DURATIONS` -- and is never a candidate, so a real
    value can never be snapped onto one and the reserved meaning stays
    unambiguous.
    """
    candidates = [(abs(entry - value), index) for index, entry in enumerate(table)
                  if entry is not None]
    return min(candidates)[1]


# Slot 0 is "something else", so an unseen signature lands somewhere rather than
# raising. The rest are every signature a composer-balanced sample of ASAP
# actually uses -- only thirteen across all sixteen composers -- plus a few
# common ones it happened not to contain.
TIME_SIGNATURES = (None, "1/2", "2/2", "2/4", "3/2", "3/4", "3/8", "4/4", "4/8",
                   "5/4", "5/8", "6/4", "6/8", "7/8", "9/8", "12/8", "12/16")


def metre_index(ratio: str | None) -> int:
    """A time signature's slot, or 0 for one the table does not carry.

    A bar LENGTH cannot name a metre: 6/8 and 3/4 are both three quarters, and
    12/8 and 6/4 are both six, and a rebuilt score has to declare one of them.
    6/8 is the second most common signature in this corpus, so this is not a
    corner.

    This stream was added to explain the Barcarolle's round trip, MeanER 54.5
    against Bach's 15.0, and it did not: with the rebuilt file correctly
    declaring 12/8 the figure stayed 54.513, to three decimals. The cause was
    accidental SPELLING -- `notation.build._spell` -- which moved every note
    carrying an accidental by a semitone or two. The stream is kept because a
    bar length genuinely cannot name the metre, not because it was measured to
    help.
    """
    try:
        return TIME_SIGNATURES.index(ratio)
    except ValueError:
        return 0


def velocity_bucket(velocity: int | None) -> int:
    """A velocity in one of seven buckets, or the reserved unknown slot.

    Slot 0 is unknown and is what inference always passes: a note list decoded
    from audio carries no velocity, so the model must not learn to rely on one.
    """
    if velocity is None:
        return UNKNOWN_VELOCITY
    clipped = max(0, min(127, int(velocity)))
    return 1 + min(VELOCITY_BUCKETS - 2, clipped * (VELOCITY_BUCKETS - 1) // 128)
