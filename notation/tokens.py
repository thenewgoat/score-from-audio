"""Performance and score as parallel streams of compound tokens.

A flat vocabulary over every (pitch, position, duration, staff, voice) combination
would be enormous and mostly unused. Parallel streams keep each vocabulary small
and each prediction a separate small classification, which is what makes the
sequences short enough to train on a thousand pieces.

The input is what was played: pitch, onset and duration in seconds, velocity. The
output is what is written: pitch, position within the bar, note value, whether a
bar starts here and how long it is, which hand, which voice, the accidental, and
the metre in force.

An EMPTY BAR -- a bar of rests, with no note in either staff -- gets a position
of its own, carrying the reserved pitch `PITCH_REST` and the bar's marker. That
is a choice, and these are the alternatives it was chosen over.

The bar stream marks a new bar by putting a non-zero length on the FIRST NOTE of
that bar, so a bar with no notes has nothing to hang its marker on. Left as it
was, such a bar simply vanished and every later bar slid up one: Chopin's
Scherzo Op. 20 rebuilt with 621 measures against the source's 625, and from bar
68 on each bar held the next bar's content.

The obvious alternative is to fold a COUNT of skipped empty bars into the
following bar's marker. It was rejected on three counts. A trailing empty bar
has no following bar, so the count has nowhere to live and the defect survives
at the end of every piece that closes on silence. The count needs either a
second stream or a bar vocabulary multiplied by the number of empty bars it can
express, and every extra slot is a class the model has to learn from almost no
examples. And it makes the bar stream non-positional: every other stream is read
one slot per position, and a marker that silently stands for several bars breaks
that invariant for the decoder and for anything that inspects the streams.

A position costs one token per empty bar -- empty bars are rare, so the sequences
barely lengthen -- and it needs NO change in the builder. The builder already
reads the bar marker before it judges the pitch, and already drops a position
whose pitch is not a real MIDI note; that rule exists so a batch padded mid-piece
keeps its bar structure, and an empty bar rides on it unchanged. Two consecutive
empty bars are two positions, and an empty bar at the end of a piece is a
position like any other.

`PITCH_REST` is a token of its own rather than a reused `PITCH_PAD`, because PAD
means "there is no position here": a trainer masks its loss and a decoder stops
at the first one, so an empty bar spelt PAD would be unlearnable and would
truncate every piece at its first bar of silence.

Scores with no empty bar emit no such position, so their streams are byte-for-byte
what they were before.
"""

from fractions import Fraction

import numpy as np

from notation.vocab import (BAR_LENGTHS, DURATIONS, GRACE_DURATION, ONSETS,
                            index_of, metre_index, velocity_bucket)

PITCH_PAD = 128
PITCH_BOS = 129
PITCH_EOS = 130
# A bar that writes no notes at all -- see the module docstring. It is a real
# position in the sequence, unlike PAD/BOS/EOS, and the builder drops it as a
# note while still honouring its bar marker.
PITCH_REST = 131
# The two slots the aligned representation needs (`notation.align`), reserved
# here rather than in that module so every consumer of a pitch value -- the
# model's embeddings and head among them -- sees one vocabulary. A PLAYED note
# the aligner could not match to any score note becomes this OUTPUT pitch: the
# decoder is trained to write it rather than invent or drop a note.
PITCH_SPACE = 132
# A score note nobody played gets its own INPUT slot, inserted so the aligned
# input and target sequences stay the same length position-for-position; the
# model is fed this pitch there and is never asked to write it.
PITCH_UNPLAYED = 133
PITCH_VOCAB = 134

INPUT_STREAMS = ("pitch", "onset", "duration", "velocity")
# `key` is last so every stream before it keeps its position in any ordered use.
OUTPUT_STREAMS = ("pitch", "onset", "duration", "bar", "hand", "voice", "accidental", "metre",
                  "key")

# Which layout a checkpoint's OUTPUT streams follow, recorded in
# `notation.model.Config.targets`. A checkpoint that records neither --
# everything trained before the aligned representation, run4 and recipe16
# included -- predates the field and is LEGACY: one slot per bar-ordered score
# position (`encode_score`), no `PITCH_SPACE` or `PITCH_UNPLAYED` in sight.
# ALIGNED is one slot per performance-aligned position (`notation.align`),
# including spaces and unplayed slots. `notation.predict.load` says which,
# the same way it already says which input encoding applies.
LEGACY_TARGETS = 1
ALIGNED_TARGETS = 2

# Both input encodings share these 200 slots, so a checkpoint's embedding
# tables are the same shape whichever it was trained on.
SECONDS_BUCKETS = 200
# The LEGACY encoding's reach: seconds since the window start, linear over
# [0, 8 s]. Measured on 150 seeded performances a 16-bar window runs 23 s
# median and 85 s p99, so at 16 bars most notes would pile onto the last slot.
SECONDS_LIMIT = 8.0

# Which input encoding a checkpoint was trained on. It is recorded in the
# checkpoint's model config (`notation.model.Config.input_encoding`) and a
# checkpoint that records none is LEGACY: run4 and everything before it.
#
#   1  onset = seconds since the window start, duration = seconds, both linear
#      over [0, SECONDS_LIMIT] and clipped there.
#   2  onset = gap since the previous note (the first note's gap is 0),
#      duration = seconds, both log-squashed over [0, GAP_LIMIT].
LEGACY_ENCODING = 1
INPUT_ENCODING = 2
# The log squash's reach and knee. 60 s is past any gap or pedalled note in
# ASAP (the longest notes measured are about 16 s), so the top slot is a
# guard, and every value that reaches it is counted. The knee at 20 ms puts
# the fine slots where rhythm lives: about 4 ms a slot at a 0.1 s gap and
# 38 ms at 1 s, against 40 ms everywhere for the legacy linear 8 s.
GAP_LIMIT = 60.0
GAP_KNEE = 0.02
VOICES = 8
# A grand staff, and only ever a grand staff: the output type is two staves,
# so a third part folds onto the left hand below and the dataset excludes the
# scores where that would matter. Named so the model's `hand` head is sized
# from the encoder rather than from a literal 2 written twice.
HANDS = 2

# Index 0 is "nothing written", which is what most notes carry.
ACCIDENTALS = (None, "flat", "double-flat", "sharp", "double-sharp", "natural")

# The key signature in force, one slot per note: 0 is "none" (BOS, padding, a
# space slot), and 1..15 are 7 flats .. 7 sharps. A signature, not a key: G
# major and E minor are one value, as in MusicXML's <key><fifths> and in Beyer
# & Dai's stream. A score that declares no signature is C major / A minor,
# fifths 0 -- a real value, not "none".
#
# This numbering is ours, not theirs: value = fifths + 8, 0 = none, against
# their released tokenizer's fifths + 7 with 15 = none/ignore.
KEY_NONE = 0
KEY_VOCAB = 16


def key_index(fifths: int) -> int:
    """The token for a signature of `fifths` (negative = flats), clamped to 7."""
    return int(max(-7, min(7, int(fifths)))) + 8


def key_fifths(index: int) -> int | None:
    """The fifths a key token names, or None for "none"."""
    return None if int(index) == KEY_NONE else int(index) - 8


def seconds_bucket(value: float) -> int:
    """A time in seconds as one of `SECONDS_BUCKETS` slots over [0, SECONDS_LIMIT].

    Clipped at both ends: a chunk should never contain a note beyond the limit,
    and if one appears it belongs at the edge rather than raising.
    """
    clamped = min(max(float(value), 0.0), SECONDS_LIMIT)
    return min(SECONDS_BUCKETS - 1,
               int(clamped / SECONDS_LIMIT * (SECONDS_BUCKETS - 1) + 0.5))


def gap_bucket(value: float) -> int:
    """A gap or a duration in seconds as one of `SECONDS_BUCKETS` log slots.

    Monotone, 0 at 0 s and the top slot at `GAP_LIMIT`, clipped at both ends.
    A log rather than a line because what matters about a gap is its RATIO to
    its neighbours -- a sixteenth against an eighth -- at every tempo.
    """
    clamped = min(max(float(value), 0.0), GAP_LIMIT)
    squashed = np.log1p(clamped / GAP_KNEE) / np.log1p(GAP_LIMIT / GAP_KNEE)
    return min(SECONDS_BUCKETS - 1, int(squashed * (SECONDS_BUCKETS - 1) + 0.5))


def encode_performance(notes, velocities=None, start: float = 0.0,
                       encoding: int = INPUT_ENCODING,
                       counts: dict | None = None) -> dict[str, np.ndarray]:
    """Input streams for a performance chunk, ordered by (onset, pitch).

    `velocities` is optional and parallel to `notes`. Where it is absent the
    reserved unknown bucket is used -- which is what inference always passes,
    because a note list decoded from audio carries no velocity.

    `encoding` is one of the numbered input encodings above, and must be the
    one the model was trained on: `notation.predict` reads it off the
    checkpoint. `start` is the window start: the legacy encoding measures every
    onset from it, the gap encoding only the first note's.

    `counts`, if given, gains `clipped`: gaps and durations the gap encoding put
    on its top slot because they reached `GAP_LIMIT`. The legacy encoding's
    clip at `SECONDS_LIMIT` is the training filter's business and is not
    counted here -- see `notation.dataset.examples`.
    """
    if encoding not in (LEGACY_ENCODING, INPUT_ENCODING):
        raise ValueError(f"unknown input encoding {encoding!r}")
    indexed = list(range(len(notes)))
    indexed.sort(key=lambda i: (notes[i][0], notes[i][2]))
    streams = {name: [] for name in INPUT_STREAMS}
    previous = None
    clipped = 0
    for i in indexed:
        onset, offset, pitch = notes[i]
        streams["pitch"].append(int(pitch))
        length = max(offset - onset, 0.0)
        if encoding == LEGACY_ENCODING:
            streams["onset"].append(seconds_bucket(onset - start))
            streams["duration"].append(seconds_bucket(length))
        else:
            # Sorted by onset, so every gap is >= 0. The first note's gap is
            # measured from the window start: every window starts on a
            # downbeat, so that distance is where the note falls in the opening
            # bar, and a bar that opens with a rest is invisible without it.
            # Training stores window notes relative to their window, so start
            # is 0 there; prediction passes the real start. Clamped at 0 for a
            # note jittered a few milliseconds before the bar line.
            gap = max(onset - (start if previous is None else previous), 0.0)
            previous = onset
            clipped += (gap >= GAP_LIMIT) + (length >= GAP_LIMIT)
            streams["onset"].append(gap_bucket(gap))
            streams["duration"].append(gap_bucket(length))
        streams["velocity"].append(
            velocity_bucket(None if velocities is None else velocities[i]))
    if counts is not None:
        counts["clipped"] = counts.get("clipped", 0) + int(clipped)
    return {name: np.array(values, dtype=np.int64) for name, values in streams.items()}


def encode_aligned(notes, slots, start: int = 0, stop: int | None = None,
                   encoding: int = INPUT_ENCODING,
                   counts: dict | None = None, velocity: bool = False) -> dict[str, np.ndarray]:
    """Input streams for `slots[start:stop]` of an aligned sequence, in SLOT order.

    `slots[k]` is an index into `notes` -- a played note -- or -1, a score note
    nobody played (`notation.align`'s inserted slot). Slots are NOT re-sorted:
    each input position is paired with the target at the same position, so the
    order is the alignment's, and a played slot's onset is never earlier than
    the previous played slot's except by jitter (clamped to a 0 gap).

    The ONE rule for a crop's or a chunk's first gap, shared by training
    (`notation.dataset.Batches`) and decoding: a played slot's gap is measured
    from the previous PLAYED slot in the whole sequence -- before `start` too
    -- and is 0 only for the piece's first played note. A crop that measured
    its first gap from itself would always see 0 there and could not tell
    where in the bar it opens, the defect `dca52d1` fixed for bar windows.
    An unplayed slot is pitch `PITCH_UNPLAYED` with gap, duration and velocity
    0, and does not reset the gap: the played note after it is measured from
    the played note before it, as the performer's timing says.

    Only the gap encoding: the legacy one measures from a window start, which
    an aligned crop does not have.

    `velocity` reads each played note's own velocity (its notes' fourth
    element) into the velocity stream, bucketed; without it every slot gets
    the reserved "unknown" bucket, which is what a note list with no velocity
    -- decoded audio, or a checkpoint from before this option -- always was.
    Asking for velocity from notes that do not carry it is refused rather than
    silently degrading to unknown, since that would hide a model reading the
    wrong config.
    """
    if encoding != INPUT_ENCODING:
        raise ValueError(f"aligned slots are encoded as gaps only, not encoding {encoding!r}")
    stop = len(slots) if stop is None else stop
    previous = None
    for k in range(start - 1, -1, -1):
        if slots[k] >= 0:
            previous = float(notes[slots[k]][0])
            break
    streams = {name: [] for name in INPUT_STREAMS}
    clipped = 0
    for k in range(start, stop):
        index = slots[k]
        if index < 0:
            streams["pitch"].append(PITCH_UNPLAYED)
            streams["onset"].append(0)
            streams["duration"].append(0)
            streams["velocity"].append(0)
            continue
        onset, offset, pitch = notes[index][:3]
        onset = float(onset)
        gap = 0.0 if previous is None else max(onset - previous, 0.0)
        previous = onset
        length = max(float(offset) - onset, 0.0)
        clipped += (gap >= GAP_LIMIT) + (length >= GAP_LIMIT)
        streams["pitch"].append(int(pitch))
        streams["onset"].append(gap_bucket(gap))
        streams["duration"].append(gap_bucket(length))
        if velocity:
            if len(notes[index]) < 4:
                raise ValueError("this model reads velocity, and the notes carry none")
            streams["velocity"].append(velocity_bucket(int(notes[index][3])))
        else:
            streams["velocity"].append(velocity_bucket(None))
    if counts is not None:
        counts["clipped"] = counts.get("clipped", 0) + int(clipped)
    return {name: np.array(values, dtype=np.int64) for name, values in streams.items()}


def _voice_of(element) -> int:
    """The element's voice number, 0-based and clipped, defaulting to the first."""
    import music21

    for parent in element.sites.get():
        if isinstance(parent, music21.stream.Voice):
            try:
                return max(0, min(VOICES - 1, int(parent.id) - 1))
            except (TypeError, ValueError):
                return 0
    return 0


def _written_span(measure) -> Fraction:
    """How many quarters this measure actually WRITES, not what its metre says.

    `measure.barDuration` is the nominal length the time signature implies.
    music21 carries the shortfall of a pickup or a truncated bar in
    `paddingLeft`/`paddingRight`, so the written span is the difference.
    Measured across the seeded per-composer sample: Brahms Op. 118 No. 2
    barDuration 3, paddingLeft 2 -> span 1; Debussy, Chopin, Beethoven and
    Bach all pad 0, so their spans are unchanged by this.

    A file whose padding meets or exceeds the bar (nothing left to write) is
    not a bar we can represent, so it falls back to the nominal length rather
    than recording a zero or negative one that no `BAR_LENGTHS` slot carries.
    """
    nominal = Fraction(measure.barDuration.quarterLength).limit_denominator(64)
    padding = (Fraction(measure.paddingLeft or 0).limit_denominator(64)
               + Fraction(measure.paddingRight or 0).limit_denominator(64))
    span = nominal - padding
    return span if span > 0 else nominal


def encode_score(score) -> dict[str, np.ndarray]:
    """Output streams for a score, ordered by (measure, position, hand, pitch).

    Only the first note of each measure carries a bar length; every other note
    carries slot 0, meaning "no bar starts here". Together with the position
    within the bar, that is enough to put the bar lines back.

    Every slot, an empty bar's included, carries the key signature in force at
    its bar (`key_index`), fifths 0 until a signature is declared.
    """
    import music21

    rows = []
    # Every bar seen, in first-seen order, against its length and the metre and
    # key signature in force there; and the bars that write at least one note
    # in EITHER staff. The difference is the empty bars, which get a position
    # of their own below.
    bars: dict = {}
    voiced: set = set()
    for hand, part in enumerate(score.parts):
        metre = 0
        # No signature declared yet is C major / A minor, a real key (fifths
        # 0), not KEY_NONE -- see `key_index`.
        fifths = 0
        # A measure NUMBER is not a bar: a first and second ending repeat the
        # same number, and keying on it alone merges two physical bars into one.
        # Counting occurrences within the part separates them while keeping the
        # two staves' bars aligned, which is what the marker depends on.
        occurrences: dict = {}
        for measure in part.getElementsByClass(music21.stream.Measure):
            occurrences[measure.number] = occurrences.get(measure.number, 0) + 1
            bar = (measure.number, occurrences[measure.number])
            declared = measure.getElementsByClass(music21.meter.TimeSignature)
            if declared:
                metre = metre_index(declared[0].ratioString)
            # A non-traditional <key> (spelt with <key-step>/<key-alter>, not
            # <fifths>) parses with `sharps` None: it names no place on the
            # circle of fifths, so it is read as no declaration and the key
            # already in force carries on, rather than crashing the encode.
            signatures = [signature for signature in
                          measure.getElementsByClass(music21.key.KeySignature)
                          if signature.sharps is not None]
            if signatures:
                fifths = signatures[0].sharps
            # The bar's WRITTEN span, not its nominal one. `barDuration` is what
            # the time signature says a full bar holds; an anacrusis or a
            # truncated final bar is shorter, and music21 records the
            # difference as padding. Brahms Op. 118 No. 2 opens with a
            # 1-quarter pickup whose barDuration is 3: recording 3 made the
            # builder pad it out to a full bar and shifted every later bar by
            # 2 quarters, permanently -- 301 of its 1722 note events matched on
            # (offset, pitch), with every bar length individually "correct".
            length = _written_span(measure)
            # Hand 0 is read first, so a bar's recorded metre and key are the
            # ones the right hand declares -- the same staff whose notes would
            # carry them.
            bars.setdefault(bar, (length, metre, key_index(fifths)))
            for element in measure.recurse().notes:
                voiced.add(bar)
                position = Fraction(
                    element.getOffsetInHierarchy(measure)).limit_denominator(64)
                duration = Fraction(element.duration.quarterLength).limit_denominator(64)
                voice = _voice_of(element)
                pitches = element.pitches if hasattr(element, "pitches") else [element.pitch]
                for pitch in pitches:
                    accidental = pitch.accidental.name if pitch.accidental else None
                    rows.append({
                        "number": bar, "length": length, "position": position,
                        # Any part past the second folds onto the left hand. The
                        # output is a two-stave grand staff, so a third stave has
                        # no faithful target; the dataset excludes those scores
                        # rather than teaching this fold as if it were correct.
                        "duration": duration, "hand": min(hand, HANDS - 1), "voice": voice,
                        "metre": metre, "key": key_index(fifths),
                        "pitch": int(pitch.midi),
                        "accidental": ACCIDENTALS.index(accidental)
                        if accidental in ACCIDENTALS else 0,
                    })

    # A bar with no note in either staff has nothing to hang its marker on, so
    # it would emit no marker at all and the builder would never create it --
    # Chopin's Scherzo Op. 20 rebuilt with 621 measures against the source's
    # 625, and from bar 68 on every bar held the next bar's content. It gets one
    # position of its own instead; the module docstring says why this rather
    # than a count folded into the following bar's marker.
    for bar, (length, metre, key) in bars.items():
        if bar in voiced:
            continue
        rows.append({
            "number": bar, "length": length, "position": Fraction(0),
            "duration": Fraction(0), "hand": 0, "voice": 0, "metre": metre,
            "key": key, "pitch": PITCH_REST, "accidental": 0,
        })

    rows.sort(key=lambda row: (row["number"], row["position"], row["hand"], row["pitch"]))
    seen = set()
    streams = {name: [] for name in OUTPUT_STREAMS}
    for row in rows:
        streams["pitch"].append(row["pitch"])
        streams["onset"].append(index_of(row["position"], ONSETS))
        # A zero duration is a GRACE note -- the ornament sounds, but takes no
        # time from its bar -- and it gets the reserved slot, not the nearest
        # real value. Written as a 1/16 it collided with the note it decorates
        # and lengthened the bar: see `notation.vocab`. `PITCH_REST`, the empty
        # bar's marker, also carries duration 0 and rides on the same slot; the
        # builder drops it as a note before any duration is read.
        streams["duration"].append(index_of(row["duration"], DURATIONS)
                                   if row["duration"] > 0 else GRACE_DURATION)
        first = row["number"] not in seen
        seen.add(row["number"])
        streams["bar"].append(index_of(row["length"], BAR_LENGTHS) if first else 0)
        streams["hand"].append(row["hand"])
        streams["voice"].append(row["voice"])
        streams["accidental"].append(row["accidental"])
        # Every note carries the metre in force, not just the first of a bar:
        # the metre is a property of a passage, and a decoder that takes the
        # majority over a bar cannot be derailed by one stray prediction.
        streams["metre"].append(row["metre"])
        streams["key"].append(row["key"])
    return {name: np.array(values, dtype=np.int64) for name, values in streams.items()}
