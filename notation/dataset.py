"""ASAP's aligned pairs, cut into chunks a model can read.

No audio is opened here. That is the point of this stage: all 1,067 performances
are usable, not the 519 that happen to have a recording, and the detector's error
is out of the loop during training. Both counts are `metadata.csv` rows -- 1,067
in total, 519 with a non-empty `audio_performance`.

Chunks are whole bars. ASAP annotates beats and downbeats on both the score and
the performance side with the same indices, so a bar is the one unit that can be
cut identically from both.
"""

import csv
import hashlib
import os
import xml.etree.ElementTree as ET

import numpy as np

from notation.tokens import (INPUT_ENCODING, INPUT_STREAMS, LEGACY_ENCODING, OUTPUT_STREAMS,
                             PITCH_BOS, PITCH_EOS, PITCH_PAD, PITCH_SPACE, PITCH_UNPLAYED,
                             SECONDS_LIMIT, ALIGNED_TARGETS, LEGACY_TARGETS,
                             encode_aligned, encode_performance, encode_score)

# Measured on stage 3's validation results at theta 0.4: recall 0.75 and
# precision 0.89, so a quarter of played notes go missing and about an eighth of
# what survives was never played. The reference work trains on Disklavier MIDI
# and has no reason to model this; every note this project will ever notate
# comes through it.
DROP_RATE = 0.25
INSERT_RATE = 0.12
# The performer who is not a metronome, and the tempo they chose. Beyer &
# Dai's recipe: onsets jittered by N(0, 0.05 s) and a uniform global tempo
# factor over 0.8-1.2. This project had 0.02 s and 0.95-1.05 through run4;
# both are settings (`notation.train.TrainConfig`), so that run is still
# reproducible. The jitter is applied in validation too, with a fixed seed --
# see `Batches`.
ONSET_SIGMA = 0.05
TEMPO_LOW = 0.8
TEMPO_HIGH = 1.2
# Beyer & Dai's other two augmentations, used by the aligned batches only: a
# transposition of up to an octave either way (`transpose`) and each note's
# duration scaled by U(0.95, 1.05) (`stretch`'s `duration_jitter`).
#
# Transposition is applied to a row with probability TRANSPOSE_PROBABILITY,
# and then by a shift uniform over +-1..TRANSPOSE_RANGE semitones (never 0).
# A transposed row is not scored on `accidental` (its labels spell the old
# key), and a shift uniform over -12..12 transposed 24 rows in 25, which left
# about 4% of the corpus to learn spelling from. At 0.5 half the rows keep
# it (Ruling E); both numbers are settings (`notation.train.TrainConfig`).
TRANSPOSE_PROBABILITY = 0.5
TRANSPOSE_RANGE = 12
DURATION_JITTER = 0.05
# Beyer & Dai's recipe differs from the above in four ways, each a setting
# that defaults to the above so every earlier run is still what it was
# (`notation.train.TrainConfig`): every row is transposed, by -6..6 INCLUDING
# 0 (`transposition_step`'s `include_zero`); a transposed row's accidentals
# are re-spelled rather than unscored (`respell`); the duration jitter is one
# factor per row tied to the tempo (`stretch`'s `duration_mode="tempo"`); and
# velocities are jittered (`jitter_velocity`).
# The share of a performance's in-range played notes the aligner must match to
# a score note for the performance to be trained on. 0.8 is the plan's
# threshold, not a tuned one. On Task 1's seeded one-per-composer sample of
# nine accepted training performances the matched share was median 0.946;
# 0.8 drops the two one-bar-out pairings there (Haydn 49-1 at 0.69, which
# `notation.align.paired_well` does not catch, and Kreisleriana 4 at 0.53,
# which it does) and also Schubert D.899/1 at 0.71, whose shortfall has not
# been diagnosed. The corpus-wide count this rejects is in the Task 3 report.
MIN_MATCHED = 0.8

# How a performance is cut into windows. Recorded in the checkpoint
# (`TrainConfig.windowing`) because decoding has to cut the same way, and a
# checkpoint that records none is LEGACY: run4 and everything before it.
#
#   1  fixed `bars`-bar windows. The trailing part-window is dropped, and so
#      is any window too long for the model.
#   2  the same windows, but the trailing part-window is kept as a window of
#      its own, and a window too long for the model is halved on a bar line,
#      recursively, until each half fits (see `fit_windows`).
#
# 2 exists for 16-bar windows. Measured on 150 seeded performances: 16 bars
# is median 293 notes, p90 487, p99 747, and 7.6% of windows exceed the 512
# positions; at 8 bars 0.2% do. Dropping 7.6% of the corpus -- the densest
# 7.6% -- is not an option, and neither is a decode that never notates a
# piece's last fifteen bars.
LEGACY_WINDOWING = 1
WINDOWING = 2

SPLITS = ("train", "validation", "test")

# The scores ASAP writes on more than two staves. The output of this whole stage
# is a two-stave grand staff, so a third stave has no faithful target:
# `notation.tokens.encode_score` folds any part past the second onto the left
# hand, and training on that would teach the fold as if it were the right
# answer. These pieces are left out of the dataset entirely rather than half
# taught.
#
# Measured over all 222 ASAP compositions with `staff_count` below: 215 are on
# two staves, five on three and two on four (Liszt's Gondoliera and Hungarian
# Rhapsody 6, which add a third and a fourth for cross-hand writing). music21
# imports each of these seven as that many `PartStaff` objects, so `len(
# score.parts) > 2` agrees file for file. `ASAP_DIR=... pytest` re-measures it.
MULTI_STAFF = frozenset({
    "Liszt/Annees_de_pelerinage_2_1_Gondoliera",
    "Liszt/Hungarian_Rhapsodies_6",
    "Liszt/Transcendental_Etudes_4",
    "Ravel/Gaspard_de_la_Nuit_1_Ondine",
    "Ravel/Miroirs_3_Une_Barque",
    "Ravel/Miroirs_4_Alborada_del_gracioso",
    "Scriabin/Sonatas_5",
})


def staff_count(xml_score: str) -> int:
    """How many staves a MusicXML score is written on.

    A part declares `<staves>` in its attributes when it holds more than one --
    a piano grand staff is one part of two -- and a part that never declares it
    is on one. The total is the sum over parts, which is what music21 turns into
    `PartStaff` objects and what `notation.tokens.encode_score` then iterates.

    Parsed with `iterparse`, not music21: this reads a fact out of the header of
    222 files on a Windows mount, and a full music21 import of the corpus is
    minutes and gigabytes for one integer per file.
    """
    staves: dict[str, int] = {}
    current = None
    for event, element in ET.iterparse(xml_score, events=("start", "end")):
        if event == "start" and element.tag == "part":
            current = element.get("id") or f"part{len(staves)}"
            staves.setdefault(current, 1)
        elif event == "end" and element.tag == "staves" and current is not None:
            staves[current] = max(staves[current], int(element.text))
        if event == "end":
            # The whole corpus does not fit in memory -- three OOM kills this
            # session -- and nothing here needs the tree once it is counted.
            element.clear()
    return sum(staves.values())


def splits(dataset: str, seed: int = 0) -> dict[str, str]:
    """Composition key to split, following the reference paper's protocol.

    For each composer one piece is held out with all of its performances; of the
    rest, a tenth validate. Split by piece, never by performance -- two
    performances of the same piece are the same music, and separating them would
    leak the answer.

    Pieces in `MULTI_STAFF` are not in the returned map at all: they are out of
    the dataset, not held out of training.

    The order within a composer is a hash of the key, never the order the rows
    happen to arrive in. An alphabetical prefix is not a sample -- taking one
    has produced a wrong published number twice in this project -- and a hash
    also keeps the split stable when ASAP adds a performance of a piece that is
    already here.
    """
    with open(os.path.join(dataset, "metadata.csv")) as handle:
        rows = csv.DictReader(handle)

        by_composer: dict[str, set[str]] = {}
        for row in rows:
            key = f"{row['composer']}/{row['title']}"
            if key in MULTI_STAFF:
                continue
            by_composer.setdefault(row["composer"], set()).add(key)

    def rank(key: str) -> int:
        return int(hashlib.md5(f"{seed}:{key}".encode()).hexdigest(), 16)

    assignment: dict[str, str] = {}
    for composer, keys in sorted(by_composer.items()):
        ordered = sorted(keys, key=rank)
        # Held out first, so a composer with only one piece gives it to test and
        # keeps nothing. Seven of ASAP's sixteen composers end with no training
        # piece: Balakirev, Brahms, Glinka and Prokofiev have exactly one piece;
        # Ravel and Scriabin come down to one once the wide scores are out; and
        # Debussy's two become one test and one validation. That is the protocol
        # working as written, and it means those seven measure generalisation to
        # a composer never trained on rather than to an unseen piece.
        #
        # Over ASAP as a whole: 215 compositions of 222 survive the exclusion,
        # and split 179 / 20 / 16 -- 888 / 73 / 60 of the 1,021 performances
        # that remain of 1,067.
        assignment[ordered[0]] = "test"
        remaining = ordered[1:]
        validating = max(1, len(remaining) // 10) if remaining else 0
        for key in remaining[:validating]:
            assignment[key] = "validation"
        for key in remaining[validating:]:
            assignment[key] = "train"
    return assignment


def splits_from_file(path: str, dataset: str) -> tuple[dict[str, str], set[str]]:
    """`(assignment, performances)` from a `{midi_performance: split}` JSON file.

    For a split someone else chose -- Beyer & Dai's (`notation/splits/`) --
    rather than `splits`' hash. The file names PERFORMANCES, because theirs
    leaves individual performances out (their `TO_IGNORE_INDICES`); a piece
    must still sit on one side only, or two performances of the same music
    would leak the answer, and a file that splits one is refused. A
    performance `metadata.csv` does not have is refused too: a typo would
    otherwise shrink the corpus silently. `MULTI_STAFF` pieces are left out
    exactly as `splits` leaves them out.
    """
    import json

    with open(path) as handle:
        wanted = json.load(handle)
    with open(os.path.join(dataset, "metadata.csv")) as handle:
        keys = {row["midi_performance"]: f"{row['composer']}/{row['title']}"
                for row in csv.DictReader(handle) if row.get("midi_performance")}
    unknown = sorted(set(wanted) - set(keys))
    if unknown:
        raise ValueError(f"{path} names performances metadata.csv does not have: {unknown[:5]}")
    assignment: dict[str, str] = {}
    performances: set[str] = set()
    for performance, side in sorted(wanted.items()):
        if side not in SPLITS:
            raise ValueError(f"{path}: {performance} is on no known side ({side!r})")
        key = keys[performance]
        if key in MULTI_STAFF:
            continue
        if assignment.setdefault(key, side) != side:
            raise ValueError(f"{path} puts {key} on both {assignment[key]} and {side}")
        performances.add(performance)
    return assignment, performances


def untrained_composers(assignment: dict[str, str]) -> set[str]:
    """Composers with no `"train"` piece in this `splits()` assignment.

    Computed from the assignment itself rather than by re-reading
    `metadata.csv`: that keeps it true for any seed `splits()` is called with,
    instead of a second, separately-computed number that could drift from the
    split it is supposed to describe.

    Measured on the real corpus (`ASAP_DIR`-gated test below): seven of
    sixteen composers -- Balakirev, Brahms, Debussy, Glinka, Prokofiev, Ravel
    and Scriabin -- which is 27 of the 60 test performances and 7 of the 16
    test pieces. That is `splits()`'s protocol working as written (see its
    docstring), not a bug; this function exists so a report of the test
    result can break those rows out rather than averaging them into the rest.
    """
    composers = {key.split("/", 1)[0] for key in assignment}
    trained = {key.split("/", 1)[0] for key, side in assignment.items() if side == "train"}
    return composers - trained


def score_bar_numbers(xml_score: str) -> list[tuple[int, int]]:
    """The score's own bar identity, `(measure.number, occurrence)`, in order.

    This is the SAME convention `notation.tokens.encode_score` bars by, and
    for the same reason: music21 numbers a pickup measure 0 rather than 1, and
    a first/second ending repeats a measure number, so the number alone is not
    a bar. `occurrence` counts how many times a number has been seen so far in
    this part, which tells the two endings apart while keeping the pickup's 0
    intact -- reversing that (numbering the pickup 1, or keying on the bare
    number) is exactly the mistake this function exists to not make; see
    `tests/test_notation_dataset.py`'s pickup and volta tests.

    Read from the first part only: a grand staff's two parts share one set of
    measure numbers bar for bar (`encode_score` relies on the same fact to
    align both hands' bars), so a second part would only recount them.
    """
    import music21

    return bar_numbers_of(music21.converter.parse(xml_score))


def bar_numbers_of(score) -> list[tuple[int, int]]:
    """`score_bar_numbers` for a score that is already parsed.

    Split out so the loaders can take the bar numbers and the output streams
    from ONE music21 parse: parsing is by far the most expensive thing in
    preparing this corpus, and doing it twice per score doubled the wait for
    nothing.
    """
    import music21

    part = score.parts[0]
    occurrences: dict[int, int] = {}
    numbers = []
    for measure in part.getElementsByClass(music21.stream.Measure):
        occurrences[measure.number] = occurrences.get(measure.number, 0) + 1
        numbers.append((measure.number, occurrences[measure.number]))
    return numbers


def bar_offset(score_annotations: str) -> int:
    """How many score bars come BEFORE the first annotated downbeat: 0 or 1.

    ASAP annotates a downbeat at every BAR LINE, so the first `db` in a file is
    the start of the first FULL bar. A score that opens with an ANACRUSIS has a
    bar before that line, and the score's own annotation says so in two ways --
    it lists beats before its first `db` (Bach BWV 854: `b,,4` then `b` then
    `db,4/4` at 2.0), or its first `db` is not at time 0 (Beethoven Op. 27 No. 2
    at 0.42857, Chopin Op. 25 No. 4 at 0.375). Either way the i-th performance
    downbeat names `score_bars[i + 1]`, not `score_bars[i]`.

    This is read from `midi_score_annotations.txt`, the SCORE's own grid, not
    from a performance's: it is a property of the music, one file per
    composition, and every one of ASAP's 235 compositions has it.

    Measured: 95 of those 235 compositions (430 of 1,067 performances) open
    this way. Pairing them at offset 0 -- which is what this function was
    missing -- put the PREVIOUS bar's score in front of every window. Over the
    20 validation compositions the mean share of target pitches present in the
    paired performance rose from 0.64 to 0.93 on exactly those pieces, and the
    rule picks the better offset on 18 of the 20 (both exceptions are pieces
    the repeat-mismatch test below throws out anyway).
    """
    try:
        with open(score_annotations) as handle:
            for line in handle:
                if not line.strip():
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) > 2 and parts[2].split(",")[0] == "db":
                    return 1 if float(parts[0]) > 1e-6 else 0
                return 1          # something is annotated before any downbeat
    except OSError:
        # No score annotation: assume the common case rather than dropping the
        # piece. Measured as never happening on ASAP, so this is a guard, not a
        # path the corpus takes.
        return 0
    return 0


def repeat_mismatch(downbeats, score_bars: list[tuple[int, int]], offset: int) -> bool:
    """Whether a performance plays a structural repeat this score does not write out.

    A performance's downbeats and the score's bars are counted from the same
    place -- once `offset` has taken the anacrusis off the score's count -- so
    they should differ by at most the trailing downbeat that closes the last
    bar. More than that means the performance played a repeat the score does
    not write out, and from the first repeat onwards EVERY later window is
    paired with the wrong bar. See `examples`'s `mismatched` counter, which
    this backs, for the measurement: dropping these avoided training 301 of
    888 performances against mispaired windows.
    """
    return abs(len(downbeats) - (len(score_bars) - offset)) > 1


def bar_windows(annotations: str, bars: int,
                 score_bars: list[tuple[int, int]],
                 offset: int = 0, tail: bool = False) -> list[tuple[float, float, tuple, tuple]]:
    """`(start seconds, end seconds, first bar, last bar)` for each whole window.

    Bar numbers are the SCORE's, `(measure.number, occurrence)` from
    `score_bar_numbers` -- not the position of a downbeat in the performance
    annotation, which is what this counted before. The two disagree exactly
    where `score_bar_numbers` says they can: a pickup, which music21 numbers 0
    rather than 1, and a volta, whose occurrence rather than its bare number
    tells two endings apart. The score's numbering wins because Task 8 cuts
    the score side by measure number and matches it to these windows -- a
    window that reported its own downbeat count instead would silently pair
    with the wrong bar wherever a pickup shifted the two by one.

    The performance's downbeats and the score's bars are both counted from the
    start of the piece, so the i-th downbeat names `score_bars[i + offset]`.
    `offset` is `bar_offset`'s answer: 0 normally, 1 where the score opens with
    an anacrusis, because ASAP's first annotated downbeat is then the start of
    the SECOND bar. Written down here as a parameter rather than inferred,
    because `windowed_score` has to cut the score side at the same offset or
    the two sides key their windows differently and every pair is a miss.

    That the i-th downbeat names the i-th bar stops
    holding once a performance actually plays a structural repeat this score
    does not write out separately (ASAP gives such pieces their own
    `_no_repeat` score variant, but not every performance is guaranteed to
    match the variant passed in) -- rather than guess past the point the two
    can still agree, windows are cut off at whichever of the annotation or
    `score_bars` runs out first, the same way a trailing part-bar is dropped.

    `tail` keeps the trailing part-window -- the bars left over when the
    piece is not a whole number of windows -- as a shorter window of its own,
    which `WINDOWING` 2 does; see `window_spans`.
    """
    lines = bar_lines(annotations, score_bars, offset)
    return [(float(lines[first]), float(lines[first + count]),
             score_bars[first + offset], score_bars[first + offset + count - 1])
            for first, count in window_spans(len(lines) - 1, bars, tail)]


def bar_lines(annotations: str, score_bars: list[tuple[int, int]] | None = None,
              offset: int = 0) -> np.ndarray:
    """The performance's downbeat times that both sides agree on.

    Bar `i` of the performance runs from `lines[i]` to `lines[i + 1]` and is
    `score_bars[i + offset]`. With `score_bars` the lines stop where either
    side runs out -- see `bar_windows` for why -- and without one (notating a
    performance no score is paired with) every downbeat is used.
    """
    from evaluation.pairs import downbeat_map

    downbeats = np.asarray(downbeat_map(annotations), dtype=float)
    if score_bars is None:
        return downbeats
    return downbeats[:max(0, min(len(downbeats), len(score_bars) - offset + 1))]


def window_spans(count: int, bars: int, tail: bool) -> list[tuple[int, int]]:
    """`(first bar, bars)` for each window over `count` bars, in order.

    Whole `bars`-bar windows from bar 0. The bars left over at the end are a
    window of their own when `tail` is set, and are dropped otherwise --
    which is what every window this project cut before `WINDOWING` 2 did.
    """
    spans = [(first, bars) for first in range(0, count - bars + 1, bars)]
    covered = len(spans) * bars
    if tail and covered < count:
        spans.append((covered, count - covered))
    return spans


def fit_windows(first: int, count: int, fits, counts: dict) -> list[tuple[int, int]]:
    """`(first, count)` halved on bar lines until every piece `fits`.

    `fits(first, count)` says whether that many bars from `first` fit the
    model. A window that does not is cut into two at its middle bar line --
    the larger half first when `count` is odd -- and each half is tried in
    turn, so 16 -> 8 -> 4 -> 2 -> 1, and every piece still starts and ends on
    a bar line. A single bar that still does not fit is dropped, never
    truncated: a truncated bar stops mid-way and shifts everything after it.

    `counts` gains `split` (windows halved) and `too_long` (single bars
    dropped), which is how a run reports both.
    """
    counts.setdefault("split", 0)
    counts.setdefault("too_long", 0)
    if fits(first, count):
        return [(first, count)]
    if count <= 1:
        counts["too_long"] += 1
        return []
    counts["split"] += 1
    half = (count + 1) // 2
    return (fit_windows(first, half, fits, counts)
            + fit_windows(first + half, count - half, fits, counts))


def stretch(notes, rng, low: float = TEMPO_LOW, high: float = TEMPO_HIGH,
            jitter: float = 0.05, duration_jitter: float = 0.0,
            duration_mode: str = "note") -> list[tuple[float, float, int]]:
    """A different tempo, and a performer who is not a metronome.

    The scale changes the whole window, uniformly over `[low, high]`; the
    jitter changes each gap between notes, so the model cannot learn that a
    fixed number of seconds means a fixed note value. (`jitter` is a
    multiplicative factor on each gap; the onset jitter in seconds is
    `corrupt`'s `sigma`.)

    `duration_jitter` multiplies each note's (scaled) duration by
    U(1 - d, 1 + d) and leaves its onset where it is: how long a key is held
    is the least reliable thing a pianist plays, and Beyer & Dai jitter it by
    5% (`DURATION_JITTER`, which the aligned batches pass). It defaults to 0
    here, and draws nothing then, so the bar-window batches -- run4 and
    recipe16 -- see exactly the random stream they always did.

    `duration_mode` says what that jitter is relative to. "note" (the above)
    draws a factor per note. "tempo" is Beyer & Dai's: ONE factor per call,
    `scale + U(-d, d)`, drawn before the notes -- every duration in the row
    moves with the tempo, near it but not exactly at it, and the durations
    keep their proportions to one another. Anything else raises.

    The notes come back in `sorted()` order, which is the order they arrive in
    when they arrive sorted -- the aligned batches rely on that.
    """
    if duration_mode not in ("note", "tempo"):
        raise ValueError(f"no such duration mode: {duration_mode!r} (note or tempo)")
    if not notes:
        return []
    scale = float(rng.uniform(low, high))
    # Drawn once, after the tempo and before any gap: "note" draws nothing
    # here, so its random stream is exactly what it always was.
    factor = (scale + float(rng.uniform(-duration_jitter, duration_jitter))
              if duration_mode == "tempo" else None)
    # `sorted()` on a NOTE tuple sorts on the whole tuple, but two notes never
    # share `(onset, offset, pitch)` in an aligned performance, so a velocity
    # tail never reaches the comparison and never reorders anything -- the
    # ordering below is exactly today's for 3-tuples.
    ordered = sorted(notes, key=lambda note: tuple(note[:3]))
    out = []
    shift = 0.0
    previous = ordered[0][0]
    for note in ordered:
        onset, offset, pitch = note[:3]
        gap = (onset - previous) * scale
        if jitter > 0 and gap > 0:
            gap *= max(0.1, float(rng.normal(1.0, jitter)))
        shift += gap
        previous = onset
        if factor is not None:
            length = (offset - onset) * factor
        else:
            length = (offset - onset) * scale
            if duration_jitter > 0:
                length *= float(rng.uniform(1.0 - duration_jitter, 1.0 + duration_jitter))
        out.append((ordered[0][0] + shift, ordered[0][0] + shift + length, pitch, *note[3:]))
    return out


def _fold(pitch: int) -> int:
    """A MIDI pitch brought into 0..127 by whole octaves."""
    while pitch > 127:
        pitch -= 12
    while pitch < 0:
        pitch += 12
    return pitch


def transposition_step(rng, reach: int, probability: float,
                       include_zero: bool = False) -> int:
    """The semitones `transpose` moves a row by: 0 for a row left as it is.

    With probability `probability` a step is drawn. By default it is uniform
    over -`reach`..-1 and 1..`reach` -- never 0, so a row whose step is not 0
    really moved. `include_zero` is Beyer & Dai's: uniform over
    -`reach`..`reach`, 0 included, so at probability 1 one row in 2*reach + 1
    stays where it was by the draw itself rather than by a second setting.

    Both draws are made whatever they decide, so the random stream after a
    row does not depend on whether it was transposed; `include_zero` False
    makes exactly the draws `transpose` always made, in the same order.
    """
    applied = float(rng.random()) < probability
    if include_zero:
        drawn = int(rng.integers(-reach, reach + 1))
        return drawn if applied else 0
    drawn = int(rng.integers(0, 2 * reach)) if reach > 0 else 0
    return (drawn - reach if drawn < reach else drawn - reach + 1) if applied and reach > 0 else 0


def key_letters(key, step: int) -> int | None:
    """Letter steps for moving a crop up `step` semitones, chosen by its key.

    Classical practice writes a transposed passage in whichever enharmonic key
    needs fewer sharps or flats -- C-sharp major (7 sharps) as D-flat major
    (5 flats) -- and spells every note by that one interval. A key signature
    of `f` fifths moved by `d` letters and `step` semitones becomes
    `f + 7*step - 12*d`. The candidates are the two nearest letter counts
    (`respell`'s), and the choice ranks: first, keys in the crop left needing
    more than 7 accidentals (a crop can straddle a key change); then the
    crop's majority key's accidental count; then flats on a tie (F-sharp
    major and G-flat major both have 6 -> G-flat). The majority is the most
    common non-"none" key token, a tie going to the flatter key. A crop with
    no key (all "none") returns None, and callers fall back to counting
    accidentals.
    """
    from collections import Counter

    from notation.tokens import KEY_NONE, key_fifths

    values = [int(v) for v in np.asarray(key).ravel() if int(v) != KEY_NONE]
    if not values:
        return None
    counts = Counter(values)
    top = max(counts.values())
    majority = min(key_fifths(v) for v, c in counts.items() if c == top)
    present = sorted({key_fifths(v) for v in counts})
    exact = step * 7 / 12
    best = None
    for letters in sorted({int(np.floor(exact)), int(np.ceil(exact))}):
        landed = majority + 7 * step - 12 * letters
        overflow = sum(abs(f + 7 * step - 12 * letters) > 7 for f in present)
        rank = (overflow, abs(landed), 0 if landed < 0 else 1)
        if best is None or rank < best[0]:
            best = (rank, letters)
    return best[1]


def shift_pitches(notes, target: dict, step: int, letters: int | None = None) -> tuple[list, dict]:
    """`(notes, target)` with every pitch moved by `step` semitones.

    The shift applies to every played pitch and every target pitch that is a
    MIDI note. The reserved values -- `PITCH_REST`, `PITCH_SPACE` and the rest
    above 127 -- are not pitches and are left alone. A pitch pushed out of
    0..127 is folded back by octaves rather than clipped, so it stays the same
    pitch class as its neighbours. Only `pitch` changes: the accidentals are
    the caller's (`respell`, or unscored). `target` is not modified.

    A note's velocity, if it carries one, is untouched: transposition moves
    pitch, not how hard the note was struck.

    The key stream moves with the notes: a signature of `f` fifths becomes
    `f + 7*step - 12*d` fifths, where `d` (`letters`) is the letter-step count
    chosen for the crop by `key_letters` when not given directly; "none"
    stays "none". Nothing moves when `target` carries no key stream at all
    (an older, hand-built target).
    """
    out = dict(target)
    if step == 0:
        return list(notes), out
    moved = [(onset, offset, _fold(int(pitch) + step), *rest)
             for onset, offset, pitch, *rest in notes]
    pitch = np.asarray(target["pitch"]).copy()
    real = (pitch >= 0) & (pitch <= 127)
    pitch[real] = [_fold(int(value) + step) for value in pitch[real]]
    out["pitch"] = pitch
    if "key" in target:
        from notation.tokens import KEY_NONE, key_fifths, key_index

        if letters is None:
            letters = key_letters(target["key"], step)
        if letters is not None:
            key = np.asarray(target["key"]).copy()
            present = key != KEY_NONE
            key[present] = [key_index(key_fifths(v) + 7 * step - 12 * letters)
                            for v in key[present]]
            out["key"] = key
    return moved, out


def transpose(notes, target: dict, rng, reach: int = TRANSPOSE_RANGE,
              probability: float = TRANSPOSE_PROBABILITY):
    """`(notes, target, shifted)`: input and target moved by one integer together.

    With probability `probability` the row is shifted, by a number of
    semitones uniform over -`reach`..-1 and 1..`reach` -- never 0, so a row
    that is flagged `shifted` really moved (`transposition_step`). Otherwise
    it is returned as it came. How pitches move is `shift_pitches`.

    `shifted` is whether the row was moved: a transposed target's ACCIDENTAL
    stream is no longer the one the score spells (C major up a semitone is
    D-flat or C-sharp), so unless the batches re-spell it (`respell`) the
    trainer masks that stream on shifted rows. `target` is not modified.
    """
    step = transposition_step(rng, reach, probability)
    moved, out = shift_pitches(notes, target, step)
    return moved, out, step != 0


# Semitones above C of each letter C D E F G A B, and each accidental's
# alteration, by its name in `notation.tokens.ACCIDENTALS`.
_LETTER_SEMITONES = (0, 2, 4, 5, 7, 9, 11)
_ALTER = {None: 0, "flat": -1, "double-flat": -2, "sharp": 1, "double-sharp": 2,
          "natural": 0}
_ALTERED = {-2: "double-flat", -1: "flat", 1: "sharp", 2: "double-sharp"}


def respell(pitch, accidental, step: int, letters: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """`(accidentals, valid)` for target notes moved up `step` semitones.

    Beyer & Dai re-spell a transposed score rather than leave its accidentals
    unscored: each note keeps its letter moved by one interval, the
    alteration follows from the new pitch, and of the intervals that make
    `step` semitones -- the two nearest diatonic step counts, `step * 7 / 12`
    rounded down and up -- the one that can write the most notes wins, and
    of those the one writing the fewest accidentals (then the smaller letter
    move). A note the winner cannot write -- it would need a triple sharp or
    flat, or its letter and pitch disagree to begin with -- is invalid: its
    label is left unscored, not guessed. Written from that description; none
    of their code.

    A token is the note's SPELLING (music21's `pitch.accidental`, which a key
    signature does not hide), so its letter is `pitch - alteration`. A
    "natural" token that stays on its letter's natural keeps its "natural";
    one that moves off it takes the new alteration's name, as any note does.

    `pitch` is the UNtransposed MIDI pitch of each note; the reserved values
    (outside 0..127) are not notes and come back valid and unchanged. `step`
    0 changes nothing. Octave folding (`shift_pitches`) moves a pitch by 12,
    which changes neither letter nor alteration, so the unfolded pitch spells
    the same.

    `letters` is the letter-step count to spell EVERY note by -- the crop's
    key (`key_letters`), chosen once for the whole crop so its notes and its
    key signature agree. When given, only that one candidate is tried;
    otherwise (no key stream) today's per-note count-of-accidentals choice
    between the two nearest candidates stands.
    """
    from notation.tokens import ACCIDENTALS

    pitch = np.asarray(pitch, dtype=np.int64)
    accidental = np.asarray(accidental, dtype=np.int64)
    if step == 0:
        return accidental.copy(), np.ones(len(pitch), dtype=bool)
    notes = np.flatnonzero((pitch >= 0) & (pitch <= 127))
    exact = step * 7 / 12
    candidates = [letters] if letters is not None else sorted(
        {int(np.floor(exact)), int(np.ceil(exact))})
    best = None
    for letters in candidates:
        tokens = accidental.copy()
        valid = np.ones(len(pitch), dtype=bool)
        written = 0
        for k in notes:
            name = ACCIDENTALS[int(accidental[k])]
            natural = int(pitch[k]) - _ALTER[name]
            if natural % 12 not in _LETTER_SEMITONES:
                valid[k] = False
                continue
            # A diatonic number: seven letters to the octave, C0 = 0.
            number = (natural // 12) * 7 + _LETTER_SEMITONES.index(natural % 12) + letters
            landed = (number // 7) * 12 + _LETTER_SEMITONES[number % 7]
            moved = int(pitch[k]) + step - landed
            if abs(moved) > 2:
                valid[k] = False
                continue
            spelled = _ALTERED.get(moved, "natural" if name == "natural" else None)
            tokens[k] = ACCIDENTALS.index(spelled)
            written += moved != 0
        rank = (int((~valid).sum()), written, abs(letters))
        if best is None or rank < best[0]:
            best = (rank, tokens, valid)
    return best[1], best[2]


def jitter_velocity(notes, rng, sigma: float) -> list:
    """Each note's velocity moved by a whole step drawn from N(0, `sigma`).

    Beyer & Dai's loader jitters velocity so the model does not lean on the
    exact dynamics one recording happened to have. The result is clipped to
    1..127: 0 is a note-off in MIDI, not a very quiet note. A note without a
    velocity (a 3-tuple) comes back as it was, drawing nothing.

    `sigma` 0 or less returns the notes unchanged and draws nothing, so a run
    without it keeps exactly its random stream.
    """
    if sigma <= 0:
        return notes
    out = []
    for note in notes:
        if len(note) < 4:
            out.append(note)
            continue
        shifted = int(note[3]) + int(np.round(rng.normal(0.0, sigma)))
        out.append((*note[:3], int(np.clip(shifted, 1, 127)), *note[4:]))
    return out


def corrupt(notes, rng, drop: float = DROP_RATE, insert: float = INSERT_RATE,
            sigma: float = ONSET_SIGMA) -> list[tuple]:
    """The input as this project's decoder would have produced it.

    Notes go missing, notes appear that were never played, and onsets wobble.
    Rates default to what stage 3 actually measures.

    Each note's tail beyond `(onset, offset, pitch)` -- velocity, for a
    velocity checkpoint -- is carried through, and an inserted note copies its
    anchor's, as `corrupt_aligned` does. The tail draws nothing, so the random
    stream and the first three fields are exactly what 3-tuples give.
    """
    if not notes:
        return []
    kept = [note for note in notes if drop <= 0 or rng.random() >= drop]

    spurious = []
    if insert > 0 and kept:
        for _ in range(int(round(len(kept) * insert))):
            anchor = kept[int(rng.integers(len(kept)))]
            onset = float(anchor[0] + rng.normal(0.0, 0.08))
            pitch = int(np.clip(anchor[2] + rng.integers(-12, 13), 21, 108))
            spurious.append((onset, onset + max(0.05, float(anchor[1] - anchor[0])), pitch,
                             *anchor[3:]))

    out = []
    for onset, offset, pitch, *rest in kept + spurious:
        moved = onset + (float(rng.normal(0.0, sigma)) if sigma > 0 else 0.0)
        length = max(0.03, offset - onset)
        out.append((moved, moved + length, int(pitch), *rest))
    out.sort()
    return out


def corrupt_aligned(notes, slots, target: dict, rng, start: int = 0,
                    drop: float = 0.0, insert: float = 0.0, sigma: float = 0.0):
    """`corrupt` for an aligned crop: `(notes, slots, target)` after decoder noise.

    `slots[start:]` is the crop and `target` is parallel to it; `slots[:start]`
    is context (the previous played note, which sets the first gap) and is
    never touched. A dropped played note that the score writes becomes an
    unplayed slot -- the score note is still there, nobody "played" it -- and
    a dropped space simply goes. An inserted note is a space, placed before the
    first played slot that starts after it. `sigma` moves each played onset,
    which can leave a gap negative; `encode_aligned` clamps it to 0.

    All three rates default to 0, and so does every aligned run the plan
    makes: the aligned model is trained on clean performances, as Beyer & Dai
    train, and the detector's error is a setting (`drop`, `insert`) to switch
    on, not a default.
    """
    notes = [tuple(note) for note in notes]
    kept_slots = list(slots[:start])
    rows: list[int] = []
    for k in range(start, len(slots)):
        index = slots[k]
        if index >= 0 and drop > 0 and rng.random() < drop:
            if int(target["pitch"][k - start]) == PITCH_SPACE:
                continue
            kept_slots.append(-1)
        else:
            kept_slots.append(index)
        rows.append(k - start)
    target = {name: np.asarray(value)[rows] for name, value in target.items()}

    if sigma > 0:
        for k in range(start, len(kept_slots)):
            index = kept_slots[k]
            if index >= 0:
                onset, offset, pitch, *rest = notes[index]
                moved = float(onset + rng.normal(0.0, sigma))
                notes[index] = (moved, moved + (offset - onset), pitch, *rest)

    played = [k for k in range(start, len(kept_slots)) if kept_slots[k] >= 0]
    if insert > 0 and played:
        for _ in range(int(round(len(played) * insert))):
            anchor = notes[kept_slots[played[int(rng.integers(len(played)))]]]
            onset = float(anchor[0] + rng.normal(0.0, 0.08))
            pitch = int(np.clip(anchor[2] + rng.integers(-12, 13), 21, 108))
            # An inserted note is fictitious -- copying the anchor's velocity
            # (rather than "unknown") is a guess, but a consistent one: it
            # comes from a real nearby note rather than from nothing.
            notes.append((onset, onset + max(0.05, float(anchor[1] - anchor[0])), pitch,
                         *anchor[3:]))
            at = next((k for k in range(start, len(kept_slots))
                       if kept_slots[k] >= 0 and notes[kept_slots[k]][0] > onset),
                      len(kept_slots))
            kept_slots.insert(at, len(notes) - 1)
            for name, value in target.items():
                space = PITCH_SPACE if name == "pitch" else 0
                target[name] = np.insert(value, at - start, space)
    return notes, kept_slots, target


def noise_defaults(targets: int) -> dict[str, float]:
    """The decoder-noise and onset-jitter rates a layout trains with when unset.

    Bar windows: stage 3's measured detector rates (`DROP_RATE`, `INSERT_RATE`)
    and `ONSET_SIGMA`, what every run through recipe16 trained on. Aligned
    performances: all 0 -- they train clean, as Beyer & Dai train, and their
    timing noise is `stretch`'s multiplicative gap jitter. One function, so
    `Batches` and the checkpoint `notation.train` writes cannot disagree about
    what "unset" meant.
    """
    if targets == ALIGNED_TARGETS:
        return {"drop": 0.0, "insert": 0.0, "onset_sigma": 0.0}
    return {"drop": DROP_RATE, "insert": INSERT_RATE, "onset_sigma": ONSET_SIGMA}


def crop_inputs(notes, slots, start: int, stop: int):
    """`(played, local_slots, begin)`: what `encode_aligned` reads for one crop.

    `notes` is the whole performance in `sorted()` order and `slots` indexes it
    (-1 for an unplayed slot) over the whole aligned sequence; the crop is
    `slots[start:stop]`. `played` is the crop's played notes plus, first, the
    CONTEXT note -- the performance note just before the crop's earliest played
    note -- and `local_slots` is `[0] * begin + crop`, the crop re-indexed into
    `played`; `begin` is 1 with a context note, 0 without. Then
    `encode_aligned(played, local_slots, start=begin)` measures the crop's
    first gap from the context note (Ruling B), and 0 only at the piece's
    first note.

    The context is the previous note in the PERFORMANCE, not the previous
    played slot in `slots`: 546 of the 616 training performances play notes
    before the paired range (`outside_played`), which have no slot, so a crop
    at slot 0 measured from the slots saw a first gap of 0 wherever the
    performance actually had a note before it. Decoding calls this too, with
    `slots = range(len(notes))` -- every note played, none unplayed -- so a
    chunk's first gap is measured exactly as a training crop's.

    `played` is in index order, so `sorted()` order: `stretch` re-sorts what
    it is given, and anything else would point `local_slots` at the wrong
    notes after it.

    Each note's tail beyond `(onset, offset, pitch)` -- velocity, when `notes`
    carries it -- comes along unchanged.
    """
    crop = [int(index) for index in slots[start:stop]]
    wanted = sorted({index for index in crop if index >= 0})
    context = [wanted[0] - 1] if wanted and wanted[0] > 0 else []
    wanted = context + wanted
    local = {index: at for at, index in enumerate(wanted)}
    played = [tuple(float(x) for x in notes[index][:2]) + (int(notes[index][2]),) +
             tuple(int(v) for v in notes[index][3:]) for index in wanted]
    begin = len(context)
    return played, [0] * begin + [local[index] if index >= 0 else -1 for index in crop], begin


def windowed_score(streams: dict[str, np.ndarray], score_bars: list[tuple[int, int]],
                   bars: int,
                   offset: int = 0) -> tuple[dict[tuple, dict[str, np.ndarray]], int]:
    """Output streams cut into `bars`-bar windows, and how many bars were found.

    The cut is made on the BAR STREAM's non-zero slots, which is what the
    encoder promises a bar line is: `encode_score` marks the first position of
    every bar, including an empty bar's `PITCH_REST` position, and marks
    nothing else. Counting those markers therefore recovers the score's own bar
    sequence, in the same order `bar_numbers_of` reads it, without a second
    pass over the music21 objects.

    Windows are keyed by `(first bar, last bar)` -- the same `(number,
    occurrence)` pairs `bar_windows` returns -- so pairing a performance window
    with its score window is a dictionary lookup, and a window the two sides
    disagree about is a miss rather than a silent off-by-one.

    The returned marker count is the caller's cross-check: it should equal
    `len(score_bars)`, and where it does not the score's two staves disagree
    about their bars and the pairing cannot be trusted.

    `offset` skips that many bars before the first window, and must be the
    same `bar_offset` the performance side is cut with: an anacrusis is never
    a whole window of its own -- no downbeat opens it -- so the score side has
    to start at the bar the first downbeat actually names.
    """
    starts = np.flatnonzero(streams["bar"] != 0)
    windows: dict[tuple, dict[str, np.ndarray]] = {}
    usable = min(len(starts), len(score_bars))
    for first in range(offset, usable - bars + 1, bars):
        stop = (int(starts[first + bars]) if first + bars < len(starts)
                else len(streams["bar"]))
        cut = slice(int(starts[first]), stop)
        windows[(score_bars[first], score_bars[first + bars - 1])] = {
            name: value[cut] for name, value in streams.items()}
    return windows, len(starts)


def aligned_targets(streams: dict[str, np.ndarray], alignment) -> dict[str, np.ndarray]:
    """One target slot per aligned position, with the bar markers re-derived.

    A slot paired with score position `s` carries `s`'s nine streams; a space
    (`score == -1`, a played note the score does not write) carries
    `PITCH_SPACE` and 0 everywhere else.

    The bar stream is NOT copied. The builder reads bar lines positionally --
    a non-zero `bar` opens a bar and everything after it belongs there -- and
    `encode_score` put each marker on the bar's first note in (bar, position,
    hand, pitch) order. In performance order that note need not come first
    (a chord's top note played first is enough), and every slot before it
    would be built into the previous bar. So the marker goes on the first
    non-space slot, in SLOT order, whose own bar index (`score_positions`) is
    that bar's; every other slot carries 0. Each slot's own bar index is used,
    never the running maximum of score indices: after a swap across a bar
    line the score indices are not monotone. That this placement then builds
    every slot in its own bar needs the slots' bars to be non-decreasing,
    which `notation.align.bar_ordered` makes true first.
    """
    from notation.align import score_positions

    positions = score_positions(streams)
    marker_of: dict[int, int] = {}
    for index, (bar, _, _) in enumerate(positions):
        if int(streams["bar"][index]) and bar not in marker_of:
            marker_of[bar] = int(streams["bar"][index])
    slots = len(alignment.score)
    out = {name: np.zeros(slots, dtype=np.int64) for name in OUTPUT_STREAMS}
    out["pitch"][:] = PITCH_SPACE
    marked: set[int] = set()
    for slot, s in enumerate(alignment.score):
        if s < 0:
            continue
        for name in OUTPUT_STREAMS:
            if name != "bar":
                out[name][slot] = int(streams[name][s])
        bar = positions[s][0]
        if bar not in marked:
            marked.add(bar)
            out["bar"][slot] = marker_of.get(bar, 0)
    return out


def _misbarred(streams, alignment, target) -> int:
    """Non-space slots the builder would put in a bar other than their own."""
    from notation.align import score_positions

    positions = score_positions(streams)
    opened, wrong, first = -1, 0, None
    for slot, s in enumerate(alignment.score):
        if s < 0:
            continue
        if first is None:
            first = positions[s][0]
        opened += target["bar"][slot] != 0
        wrong += positions[s][0] - first != opened
    return int(wrong)


def beat_starts(notes, slots, beats) -> np.ndarray:
    """Slot indices where a new inter-beat interval begins; the first is 0.

    A played slot's beat is the last annotated beat at or before its onset;
    an unplayed slot takes the beat of the slot before it (the aligner puts
    an interval's unplayed slots after its played ones). The running maximum
    is used, so a note swapped across a beat line (`notation.align`'s
    `moved`) never opens an earlier beat again. Beyer & Dai start every
    training crop on one of these (`Batches`, `crop="beat"`).
    """
    beats = np.asarray(beats, dtype=float)
    out, current = [0], None
    for k, index in enumerate(slots):
        if index < 0:
            continue
        beat = int(np.searchsorted(beats, float(notes[index][0]), side="right")) - 1
        if current is None:
            current = beat
        elif beat > current:
            out.append(k)
            current = beat
    return np.asarray(out, dtype=np.int32)


def aligned_examples(dataset: str, want: str, assignment: dict[str, str] | None = None,
                     recordings: int = 0, seed: int = 0, stats: dict | None = None,
                     min_matched: float = MIN_MATCHED,
                     performances: set[str] | None = None):
    """Yield `(notes, target, rows, starts)` for every performance aligned to its score.

    `notes` is the whole performance, `(onset, offset, pitch, velocity)`
    float32 in `sorted()` order on the first three columns -- the aligner
    itself is given only those three (Ruling: velocity never decides an
    alignment); `rows` is an `(slots, 2)` int32 array of `(performance index
    or -1, score index or -1)` in slot order -- the alignment, after
    `notation.align.bar_ordered` -- and `target` the nine output streams per
    slot (`aligned_targets`), int16. `starts` is `beat_starts` over that same
    slot order: where `Batches` (`crop="beat"`) may start a training crop.

    One music21 parse per composition, as `examples`. A performance is left
    out, and counted, under each of three rules, applied in this order so each
    performance is counted once, under the first it fails:

    * `mismatched`: `repeat_mismatch`, the rule the bar-window data has always
      applied -- a structural repeat the score does not write out.
    * `pairing_rejected`: `notation.align.paired_well` says a neighbouring bar
      offset matches more notes than the pipeline's own (Ruling D) -- a score
      that writes one performed bar as two short measures.
    * `alignments_rejected`: under `min_matched` of the in-range played notes
      are matched to a score note.

    `performances`, when given, is `splits_from_file`'s second element and is
    passed straight to `_performance_rows`: a piece `assignment` puts on
    `want`'s side can still have a performance that file left out.

    `stats` also accumulates, over the performances used: `played`,
    `matched`, `spaces`, `unplayed`, `moved`, `outside_played` and
    `outside_score` (the aligner's own counts), `repaired` and `relocated`
    (`bar_ordered`'s), `misbarred` (slots the builder would still put in the
    wrong bar -- 0 by construction, counted to prove it) and `slots`. Nothing
    is dropped without a count.
    """
    import music21

    from evaluation.notation_eval import notes_from_midi
    from evaluation.pairs import beat_map, downbeat_map
    from notation.align import align, bar_ordered, paired_well

    counters = stats if stats is not None else {}
    for name in ("performances", "pieces", "failed", "mismatched", "pairing_rejected",
                 "alignments_rejected", "unpaired", "bar_disagreement", "anacrusis",
                 "played", "matched", "spaces", "unplayed", "moved", "outside_played",
                 "outside_score", "repaired", "relocated", "misbarred", "slots"):
        counters.setdefault(name, 0)
    counters.setdefault("pieces_seen", set())
    counters.setdefault("performances_seen", set())

    assignment = assignment if assignment is not None else splits(dataset)
    rows = _performance_rows(dataset, assignment, want, recordings, seed,
                             performances=performances)
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(f"{row['composer']}/{row['title']}", []).append(row)

    for key, group in sorted(grouped.items()):
        folder = os.path.join(dataset, os.path.dirname(group[0]["midi_performance"]))
        xml_score = os.path.join(folder, "xml_score.musicxml")
        if not os.path.exists(xml_score):
            counters["failed"] += len(group)
            continue
        offset = bar_offset(os.path.join(folder, "midi_score_annotations.txt"))
        counters["anacrusis"] += offset
        try:
            score = music21.converter.parse(xml_score)
            score_bars = bar_numbers_of(score)
            streams = encode_score(score)
        except Exception:                      # noqa: BLE001 -- as in `examples`
            counters["failed"] += len(group)
            continue
        finally:
            score = None
        if int(np.count_nonzero(streams["bar"])) != len(score_bars):
            counters["bar_disagreement"] += 1
        counters["pieces_seen"].add(key)

        for row in group:
            annotations = os.path.join(dataset, row["performance_annotations"])
            midi = os.path.join(dataset, row["midi_performance"])
            if not (os.path.exists(annotations) and os.path.exists(midi)):
                counters["failed"] += 1
                continue
            if repeat_mismatch(downbeat_map(annotations), score_bars, offset):
                counters["mismatched"] += 1
                continue
            try:
                played = notes_from_midi(midi, velocity=True)
            except Exception:                  # noqa: BLE001
                counters["failed"] += 1
                continue
            # Rounded to float32 BEFORE sorting and aligning, so the order the
            # rows index is exactly the `sorted()` order of what is stored:
            # `stretch` re-sorts what it is given, and two onsets float32 made
            # equal would otherwise swap places under it and every later row
            # would point at the wrong note. Velocity rides along on the same
            # order -- it is never itself a sort key -- and is not rounded:
            # it is an int already, and float32 has nothing to lose there.
            rounded = np.array([note[:3] for note in played],
                               dtype=np.float32).reshape(-1, 3)
            order = sorted(range(len(played)), key=lambda k: (float(rounded[k][0]),
                           float(rounded[k][1]), int(rounded[k][2])))
            notes = [(float(rounded[k][0]), float(rounded[k][1]), int(rounded[k][2]),
                     int(played[k][3])) for k in order]
            lines = bar_lines(annotations, score_bars, offset)
            if len(lines) < 2 or not notes:
                counters["unpaired"] += 1
                continue
            beats = beat_map(annotations)
            # The aligner sees exactly today's 3-column input; the velocity
            # column is carried alongside it, never fed to the aligner.
            triples = [note[:3] for note in notes]
            ok, _, _, _ = paired_well(triples, beats, lines, streams, offset)
            if not ok:
                counters["pairing_rejected"] += 1
                continue
            alignment = align(triples, beats, lines, streams, offset)
            if alignment.matched < min_matched * max(alignment.played, 1):
                counters["alignments_rejected"] += 1
                continue
            alignment, repaired, relocated = bar_ordered(alignment, streams)
            starts = beat_starts(notes, alignment.performance, beats)
            target = aligned_targets(streams, alignment)
            # The builder's reading of the re-marked targets, checked against
            # each slot's own bar: `bar_ordered` promises this is 0, and the
            # count is how a corpus-wide build shows it.
            counters["misbarred"] += _misbarred(streams, alignment, target)
            counters["performances"] += 1
            counters["performances_seen"].add(row["midi_performance"])
            for name in ("played", "matched", "spaces", "unplayed", "moved",
                         "outside_played", "outside_score"):
                counters[name] += getattr(alignment, name)
            counters["repaired"] += repaired
            counters["relocated"] += relocated
            counters["slots"] += len(alignment.score)
            yield (np.array(notes, dtype=np.float32).reshape(-1, 4),
                   {name: value.astype(np.int16) for name, value in target.items()},
                   np.array([alignment.performance, alignment.score],
                            dtype=np.int32).T.reshape(-1, 2),
                   starts)
    counters["pieces"] = len(counters["pieces_seen"])


def _performance_rows(dataset: str, assignment: dict[str, str], want: str,
                      recordings: int = 0, seed: int = 0,
                      performances: set[str] | None = None) -> list[dict]:
    """Every metadata row on `want`'s side, optionally subsampled.

    The subsample is a seeded permutation, never the first N. An alphabetical
    prefix of ASAP is Bach and nothing else, and taking one has produced a
    wrong published number twice in this project.

    `performances`, when given, is `splits_from_file`'s second element: the
    midi performances that file actually names. A piece it assigns can still
    have a performance it left out (their `TO_IGNORE_INDICES`), and that
    performance must not silently train or validate anyway.
    """
    with open(os.path.join(dataset, "metadata.csv")) as handle:
        rows = [row for row in csv.DictReader(handle)
                if row.get("midi_performance")
                and assignment.get(f"{row['composer']}/{row['title']}") == want
                and (performances is None or row["midi_performance"] in performances)]
    rows.sort(key=lambda row: row["midi_performance"])
    if recordings and recordings < len(rows):
        chosen = np.random.default_rng(seed).permutation(len(rows))[:recordings]
        rows = [rows[index] for index in sorted(chosen)]
    return rows


def examples(dataset: str, want: str, bars: int, max_length: int,
             assignment: dict[str, str] | None = None, recordings: int = 0,
             seed: int = 0, stats: dict | None = None,
             windowing: int = WINDOWING, encoding: int = INPUT_ENCODING):
    """Yield `(performance notes, score streams)` for every paired window.

    One music21 parse per COMPOSITION, reused by all of its performances, and
    released before the next: holding the corpus has been OOM-killed three
    times in this project, and 215 parsed scores do not fit beside a model.

    `stats` accumulates what was skipped and why. Nothing here is allowed to
    drop a window silently -- a short window list is the signature of a
    performance that plays a repeat its score does not write out, and it has to
    read as a pairing mismatch rather than as a slightly worse model. Under
    `WINDOWING` 2 it also counts `tail` (trailing part-windows kept) and
    `split` (windows halved to fit); `too_long` is then single bars that did
    not fit even alone.

    `encoding` is the input encoding the windows will be fed as. Only the
    legacy one drops a window longer than `SECONDS_LIMIT`: seconds since the
    window start clip there, so every later note would sit on one slot. The
    gap encoding does not clip below `GAP_LIMIT` (60 s), so the filter has no
    reason to exist -- and at 16 bars, 23 s median, it would drop nearly
    every window.
    """
    import music21

    from evaluation.notation_eval import notes_from_midi
    from evaluation.pairs import downbeat_map

    counters = stats if stats is not None else {}
    for name in ("performances", "pieces", "windows", "too_long", "unpaired",
                 "mismatched", "bar_disagreement", "failed", "anacrusis", "tail", "split"):
        counters.setdefault(name, 0)
    counters.setdefault("pieces_seen", set())
    counters.setdefault("performances_seen", set())
    tail = windowing != LEGACY_WINDOWING
    seconds_limit = encoding == LEGACY_ENCODING

    assignment = assignment if assignment is not None else splits(dataset)
    rows = _performance_rows(dataset, assignment, want, recordings, seed)
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(f"{row['composer']}/{row['title']}", []).append(row)

    for key, group in sorted(grouped.items()):
        folder = os.path.join(dataset, os.path.dirname(group[0]["midi_performance"]))
        xml_score = os.path.join(folder, "xml_score.musicxml")
        if not os.path.exists(xml_score):
            counters["failed"] += 1
            continue
        # Where this score's first DOWNBEAT falls. A score that opens with an
        # anacrusis has a bar before it, and both sides have to skip that bar
        # or every window is paired with the bar before the one it played.
        offset = bar_offset(os.path.join(folder, "midi_score_annotations.txt"))
        counters["anacrusis"] += offset
        try:
            score = music21.converter.parse(xml_score)
            score_bars = bar_numbers_of(score)
            streams = encode_score(score)
        except Exception:                      # noqa: BLE001 -- see below
            # music21 refuses a handful of real ASAP sources outright. One
            # unusable score must cost that score, not the run, and the count
            # is reported so a run that quietly lost half the corpus cannot
            # read as a clean one.
            counters["failed"] += 1
            continue
        finally:
            score = None
        # The bar lines on the score side, exactly as `windowed_score` finds
        # them: the bar stream's non-zero slots. A window is cut from the
        # starts of its first bar to the start of the bar after its last.
        starts = np.flatnonzero(streams["bar"] != 0)
        markers = len(starts)
        score_usable = min(markers, len(score_bars))
        if markers != len(score_bars):
            counters["bar_disagreement"] += 1
        counters["pieces_seen"].add(key)

        def target_of(first: int, count: int) -> dict[str, np.ndarray]:
            at = first + offset
            stop = (int(starts[at + count]) if at + count < markers
                    else len(streams["bar"]))
            return {name: value[int(starts[at]):stop] for name, value in streams.items()}

        for row in group:
            annotations = os.path.join(dataset, row["performance_annotations"])
            midi = os.path.join(dataset, row["midi_performance"])
            if not (os.path.exists(annotations) and os.path.exists(midi)):
                counters["failed"] += 1
                continue
            downbeats = downbeat_map(annotations)
            # See `repeat_mismatch`: more than one bar's disagreement means the
            # performance played a structural repeat this score does not write
            # out, and from the first repeat onwards EVERY later window is
            # paired with the wrong bar. Measured over the 20 validation
            # compositions: the six this flags hold 0.50 of their target
            # pitches in the paired performance against 0.91 for the other
            # fourteen, even after each is given its best offset.
            if repeat_mismatch(downbeats, score_bars, offset):
                counters["mismatched"] += 1
                continue
            try:
                notes = notes_from_midi(midi)
            except Exception:                  # noqa: BLE001
                counters["failed"] += 1
                continue
            counters["performances"] += 1
            counters["performances_seen"].add(row["midi_performance"])
            # `notes_from_midi` sorts by onset, so a window's notes are one
            # contiguous run and a search finds it -- the same notes, in the
            # same order, as testing every note against every window.
            onsets = [note[0] for note in notes]
            lines = bar_lines(annotations, score_bars, offset)

            def inside_of(first: int, count: int) -> tuple[np.ndarray, float, float]:
                start, end = float(lines[first]), float(lines[first + count])
                low = int(np.searchsorted(onsets, start, side="left"))
                high = int(np.searchsorted(onsets, end, side="left"))
                return np.array(notes[low:high], dtype=np.float32).reshape(-1, 3), start, end

            def fits(first: int, count: int) -> bool:
                inside, start, end = inside_of(first, count)
                # +2 for the BOS and EOS the batcher brackets the target with.
                return (len(inside) <= max_length
                        and len(target_of(first, count)["pitch"]) + 2 <= max_length
                        and not (seconds_limit and end - start > SECONDS_LIMIT))

            spans = window_spans(len(lines) - 1, bars, tail)
            if spans and spans[-1][1] < bars:
                counters["tail"] += 1
            for first, count in spans:
                # The score side has run out: the key `windowed_score` would
                # have looked up does not exist.
                if first + offset + count > score_usable:
                    counters["unpaired"] += 1
                    continue
                # A window with nothing in it teaches the model to write an
                # empty bar for a bar that was played -- the performance and
                # the score have come apart here, so it is dropped, not kept.
                # Tested before fitting, as it always was, so an empty window
                # is `unpaired` and never `too_long`.
                if len(inside_of(first, count)[0]) == 0:
                    counters["unpaired"] += 1
                    continue
                if windowing == LEGACY_WINDOWING:
                    if fits(first, count):
                        pieces = [(first, count)]
                    else:
                        counters["too_long"] += 1
                        pieces = []
                else:
                    pieces = fit_windows(first, count, fits, counters)
                for piece in pieces:
                    inside, start, _ = inside_of(*piece)
                    if len(inside) == 0:          # a half of a window can be empty
                        counters["unpaired"] += 1
                        continue
                    counters["windows"] += 1
                    inside[:, :2] -= start
                    yield inside, {name: value.astype(np.int16)
                                   for name, value in target_of(*piece).items()}
    counters["pieces"] = len(counters["pieces_seen"])


class AlignedBatch(tuple):
    """`(source, target, mask, accidental_keep)`, carrying its own counts.

    `accidental_keep` is shaped like `target["pitch"]`, BOS column included
    (always True): False where an `accidental` label is not to be scored --
    every slot of a row transposed without re-spelling, and the notes
    `respell` could not write.

    A tuple, so it unpacks and indexes as the plain four-tuple does; `counts`
    is what preparing it counted, for `Batches.commit`.
    """

    def __new__(cls, items, counts: dict):
        made = super().__new__(cls, items)
        made.counts = counts
        return made


class Batches:
    """Padded batches over a fixed set of windows.

    Augmentation happens here rather than in `examples` so that a training
    epoch sees a different tempo and a different set of decoder errors each
    time, while validation sees the same ones every time -- a validation loss
    that moves because its noise moved cannot be compared across evaluations,
    and early stopping reads exactly that comparison.
    """

    def __init__(self, held, batch: int, seed: int, device: str, augment: bool,
                 stats: dict, drop: float | None = None, insert: float | None = None,
                 tempo: tuple[float, float] = (TEMPO_LOW, TEMPO_HIGH),
                 sigma: float | None = None, encoding: int = INPUT_ENCODING,
                 targets: int = LEGACY_TARGETS, max_length: int | None = None,
                 duration_jitter: float = DURATION_JITTER,
                 transpose_probability: float = TRANSPOSE_PROBABILITY,
                 transpose_range: int = TRANSPOSE_RANGE,
                 velocity_input: bool = False, transpose_zero: bool = False,
                 respelled: bool = False, duration_mode: str = "note",
                 velocity_jitter: float = 0.0, sampling: str = "uniform",
                 crop: str = "slot", crop_shift: int = 0, min_beats: int = 16,
                 synthetic_from: int | None = None, synthetic_share: float = 0.5):
        self.held = held
        self.batch = batch
        self.seed = seed
        self.device = device
        self.augment = augment
        self.stats = stats
        # Settable so that "is the decoder noise what makes this unlearnable?"
        # is a run, not an opinion. At the measured rates a quarter of every
        # target's notes were never shown to the model, which is an
        # irreducible term in the loss.
        #
        # Unset (None), each takes its layout's default: stage 3's measured
        # rates for bar windows, as every run through recipe16 trained, and 0
        # for aligned performances, which train clean as Beyer & Dai do.
        defaults = noise_defaults(targets)
        self.drop = drop if drop is not None else defaults["drop"]
        self.insert = insert if insert is not None else defaults["insert"]
        # The tempo range `stretch` draws from (training only) and the onset
        # jitter `corrupt` applies (both sides), from the run's config.
        self.tempo = tempo
        # `corrupt`'s additive onset jitter; see `noise_defaults`.
        self.sigma = sigma if sigma is not None else defaults["onset_sigma"]
        # Which layout `held` is in: bar windows `(notes, target)`, or aligned
        # performances `(notes, target, rows)` from `aligned_examples`, served
        # in crops of up to `max_length - 1` slots -- the target's BOS takes
        # the remaining position -- and yielded as `(source, target, mask,
        # accidental_keep)`, the last a per-position mask (`AlignedBatch`).
        # Training (`augment`) serves one crop per performance per pass, from
        # a random start; validation tiles every performance whole (`tiles`).
        self.targets = targets
        if targets == ALIGNED_TARGETS and not max_length:
            raise ValueError("aligned batches need the model's max_length to crop to")
        self.max_length = max_length
        self.duration_jitter = duration_jitter
        # Aligned batches only: see `TRANSPOSE_PROBABILITY`.
        self.transpose_probability = transpose_probability
        self.transpose_range = transpose_range
        # Whether `encode_aligned` reads each note's own velocity or the
        # reserved "unknown" bucket -- see `Config.velocity_input`. False for
        # everything through aligned1.
        self.velocity_input = velocity_input
        # Beyer & Dai's augmentation values, aligned training only; each
        # default is what every run before them trained with. `transpose_zero`:
        # the step is drawn over -range..range with 0 in it
        # (`transposition_step`). `respelled`: a transposed row's accidentals
        # are re-spelled (`respell`) and scored, rather than unscored.
        # `duration_mode`: see `stretch`. `velocity_jitter`: the sigma of
        # `jitter_velocity`, 0 for none.
        self.transpose_zero = transpose_zero
        self.respelled = respelled
        self.duration_mode = duration_mode
        self.velocity_jitter = velocity_jitter
        # Beyer & Dai's other two loader settings, aligned training only.
        # `sampling`: "length" draws performances WITH REPLACEMENT in
        # proportion to their played-note count, so a long performance
        # contributes as many crops to an epoch as its length warrants,
        # rather than exactly one like every shorter one ("uniform", every
        # run before them). `crop`: "beat" starts a training crop on one of
        # `starts` (a held row's beat-start slots), uniform over all but the
        # last `min_beats` beats so a crop still has somewhere to run, then
        # drops 0..`crop_shift` - 1 leading slots -- Beyer & Dai vary the
        # phase within the beat too. "slot" (default) keeps the plain
        # uniform slot draw every run before them used.
        if sampling not in ("uniform", "length"):
            raise ValueError(f"no such sampling: {sampling!r} (uniform or length)")
        if crop not in ("slot", "beat"):
            raise ValueError(f"no such crop: {crop!r} (slot or beat)")
        self.sampling = sampling
        self.crop = crop
        self.crop_shift = crop_shift
        self.min_beats = min_beats
        # Each held performance's played-note count, normalised to weights
        # for `sampling == "length"` -- computed once here rather than per
        # epoch, since `held` never changes after construction. Bar windows
        # have no `rows` to count (`held` is `(notes, target)` pairs), and
        # never use it: `sampling` only reads this at `targets ==
        # ALIGNED_TARGETS` (`_aligned`).
        if held and targets == ALIGNED_TARGETS:
            counts = np.array([int((np.asarray(row[2])[:, 0] >= 0).sum()) for row in held],
                              dtype=float)
            self._weights = counts / counts.sum() if counts.sum() > 0 else None
            if synthetic_from is not None and 0 < synthetic_from < len(held):
                # Rows from `synthetic_from` on are made-up performances
                # (`notation.synthetic`): each group gets its share of the
                # draws, and within a group a row still weighs its length.
                if sampling != "length":
                    raise ValueError("a synthetic share needs sampling='length'")
                real, made = counts[:synthetic_from], counts[synthetic_from:]
                self._weights = np.concatenate([real / real.sum() * (1 - synthetic_share),
                                                made / made.sum() * synthetic_share])
        else:
            self._weights = None
        if respelled and (self.drop > 0 or self.insert > 0):
            # Detector noise deletes and inserts target slots after the
            # re-spelling, which would have to carry the keep mask through
            # them. The recipe that re-spells trains clean, so the pairing is
            # refused rather than half-supported.
            raise ValueError("respell needs drop and insert at 0: detector noise "
                             "reorders the slots the re-spelled accidentals label")
        if targets == ALIGNED_TARGETS:
            # `crops`: rows served as a crop of a longer performance (the rest
            # waits for another pass's start). `trimmed`: crops that inserted
            # detector noise pushed past the limit, which lose their tail.
            #
            # These, and `emptied`, `clipped`, `rows` and `transposed`, are
            # counted per SERVED batch for aligned batches: each batch carries
            # its own counts (`AlignedBatch.counts`) and they reach `stats`
            # only when the consumer calls `commit` -- the trainer does for
            # every batch it steps on, and for validation once, on the first
            # evaluation. A prefetching worker prepares batches that are never
            # trained on, and validation serves the same tiles every
            # evaluation; counting at preparation over-counted both.
            for name in ("crops", "trimmed", "rows", "transposed"):
                stats.setdefault(name, 0)
        # The input encoding the model is trained on; see `notation.tokens`.
        self.encoding = encoding
        # Windows `corrupt` emptied and `__iter__` therefore skipped, counted
        # over every pass: an epoch for training, one evaluation for
        # validation. Kept in `stats`, which is what `train` writes to
        # `training.json`, so a skip is reported beside the other skips rather
        # than disappearing. At run4's rates (drop 0) it cannot happen; at the
        # default 25% drop a one-note window loses its only note a quarter of
        # the time.
        stats.setdefault("emptied", 0)
        # Gaps and durations the gap encoding put on its top slot, counted the
        # same way. `GAP_LIMIT` is 60 s, so this should stay at 0; it is
        # counted so that is a measurement, not an assumption.
        stats.setdefault("clipped", 0)
        self.pieces = set(stats.get("pieces_seen", ()))
        self.performances = set(stats.get("performances_seen", ()))
        self.mismatched = stats.get("mismatched", 0)

    def __len__(self) -> int:
        rows = len(self.tiles()) if self._tiled() else len(self.held)
        return (rows + self.batch - 1) // self.batch

    def _tiled(self) -> bool:
        return self.targets == ALIGNED_TARGETS and not self.augment

    def tiles(self) -> list[tuple[int, int]]:
        """`(performance, start)` for every consecutive crop of every performance.

        What validation serves, in this order, every evaluation. Best-checkpoint
        selection reads validation loss, and one crop per performance covered
        28k of the 97k validation slots (Task 3's cache): the other 71% --
        every long performance's middle and end -- never counted. Consecutive
        crops of `max_length - 1` from slot 0 cover every slot exactly once;
        a performance's last tile is whatever is left.
        """
        width = self.max_length - 1
        return [(index, start) for index, (_, _, rows, *_) in enumerate(self.held)
                for start in range(0, max(len(rows), 1), width)]

    def __iter__(self):
        import torch

        rng = np.random.default_rng(self.seed if not self.augment
                                    else self.seed + self._epoch())
        order = rng.permutation(len(self.held))
        if self.targets == ALIGNED_TARGETS:
            yield from self._aligned(order, rng, torch)
            return
        for at in range(0, len(order), self.batch):
            chosen = order[at:at + self.batch]
            sources, targets = [], []
            for index in chosen:
                notes, target = self.held[index]
                played = [(float(a), float(b), int(c)) for a, b, c in notes]
                if self.augment:
                    played = stretch(played, rng, low=self.tempo[0], high=self.tempo[1])
                played = corrupt(played, rng, drop=self.drop, insert=self.insert,
                                 sigma=self.sigma)
                if not played:
                    # Everything was dropped. The score is still the right
                    # answer for what was played, but an empty input cannot
                    # teach it, so the window sits this epoch out -- counted.
                    self.stats["emptied"] += 1
                    continue
                sources.append(encode_performance(played, encoding=self.encoding,
                                                  counts=self.stats))
                targets.append(target)
            if sources:
                yield self._collate(sources, targets, torch)

    def commit(self, batch) -> None:
        """Add a served batch's counts to `stats`; see `__init__`.

        A batch without counts (a bar window's, or a test's hand-built one)
        commits nothing: bar-window counts are still taken at preparation.
        """
        for name, value in getattr(batch, "counts", {}).items():
            self.stats[name] = self.stats.get(name, 0) + value

    def _aligned(self, order, rng, torch):
        # Training: one random crop per performance, in the epoch's order --
        # or, at `sampling == "length"`, performances drawn WITH REPLACEMENT
        # in proportion to their played-note count, so a long performance
        # gets as many crops in an epoch as its length warrants rather than
        # exactly one, like every short one. Validation is untouched: every
        # tile, in a fixed order (`tiles`).
        if not self._tiled() and self.sampling == "length" and self._weights is not None:
            order = rng.choice(len(self.held), size=len(self.held), replace=True,
                               p=self._weights)
        wanted = self.tiles() if self._tiled() else [(int(index), None) for index in order]
        for at in range(0, len(wanted), self.batch):
            sources, targets, keeps, shifted = [], [], [], []
            counts = {"crops": 0, "trimmed": 0, "emptied": 0, "clipped": 0}
            for index, start in wanted[at:at + self.batch]:
                served = self._aligned_row(self.held[index], rng, start, counts)
                if served is None:
                    counts["emptied"] += 1
                    continue
                sources.append(served[0]); targets.append(served[1])
                keeps.append(served[2]); shifted.append(served[3])
            counts["rows"] = len(sources)
            counts["transposed"] = int(sum(shifted))
            if not sources:
                # Nothing to serve, so nothing will be committed: the rows it
                # emptied are counted here instead of being lost.
                self.commit(AlignedBatch((), counts))
                continue
            source, target, mask = self._collate(sources, targets, torch, eos=False)
            # Column 0 is BOS; a row's slots follow it, and padding past them
            # is already out of the loss by `mask`, so it is left True.
            accidental_keep = torch.ones_like(mask)
            for row, keep in enumerate(keeps):
                accidental_keep[row, 1:1 + len(keep)] = torch.from_numpy(keep).to(mask.device)
            yield AlignedBatch((source, target, mask, accidental_keep), counts)

    def _aligned_row(self, held, rng, start: int | None = None, counts: dict | None = None):
        """`(source, target, keep, shifted)` for one crop of one performance, or None.

        `keep` has one bool per target slot: whether its `accidental` label is
        scored. `shifted` is whether the row was transposed at all (counted).

        The crop is `max_length - 1` consecutive slots from `start`. When
        `start` is None it is drawn: uniform over every slot (`crop ==
        "slot"`, every run before Beyer & Dai), or ON A BEAT (`crop ==
        "beat"`) -- one of the row's `starts` (`beat_starts`), uniform over
        all but its last `min_beats` beats so a crop still has somewhere to
        run, then shifted by 0..`crop_shift` - 1 leading slots dropped, so
        training does not always start a beat on the crop's first position.
        A performance shorter than the crop is served whole under "slot".
        Under "beat" the crop still starts at the drawn beat plus its shift
        and runs to the end, so it loses the slots before the drawn beat and
        0..`crop_shift` - 1 more -- even when the drawn beat is slot 0.

        Only the notes the crop reads are augmented: its played notes and the
        performance note before it, which sets its first gap (`crop_inputs`).
        """
        notes, target, rows, *more = held
        starts = more[0] if more else np.arange(len(rows), dtype=np.int32)
        slots = np.asarray(rows)[:, 0]
        width = self.max_length - 1
        if start is None:
            if self.crop == "beat":
                last = max(len(starts) - self.min_beats, 0)
                begin = int(starts[int(rng.integers(0, last + 1))])
                if self.crop_shift > 0:
                    begin += int(rng.integers(0, self.crop_shift))
                start = min(begin, max(len(slots) - 1, 0))
            else:
                start = 0
                if len(slots) > width:
                    start = int(rng.integers(0, len(slots) - width + 1))
        counts = counts if counts is not None else self.stats
        if len(slots) > width:
            counts["crops"] += 1
        stop = min(len(slots), start + width)
        played, local_slots, begin = crop_inputs(notes, slots, start, stop)
        cut = {name: np.asarray(value[start:stop]) for name, value in target.items()}

        keep = np.ones(len(cut["pitch"]), dtype=bool)
        shifted = False
        if self.augment and played:
            played = stretch(played, rng, low=self.tempo[0], high=self.tempo[1],
                             duration_jitter=self.duration_jitter,
                             duration_mode=self.duration_mode)
            # Draws nothing at 0, so a run without it keeps its random stream.
            played = jitter_velocity(played, rng, self.velocity_jitter)
            step = transposition_step(rng, self.transpose_range, self.transpose_probability,
                                      include_zero=self.transpose_zero)
            original_pitch = np.asarray(cut["pitch"]).copy()
            # Chosen once from the UNshifted key, before `shift_pitches` moves
            # it, so notes and key agree on the same interval.
            letters = key_letters(cut["key"], step) if (step != 0 and "key" in cut) else None
            played, cut = shift_pitches(played, cut, step, letters=letters)
            if step == 0:
                pass
            elif self.respelled:
                spelled, keep = respell(original_pitch, cut["accidental"], step, letters=letters)
                # An unspellable note's label is unscored; the decoder still
                # reads its slot, so it gets a real token (0), never an invalid one.
                spelled[~keep] = 0
                cut["accidental"] = spelled.astype(np.asarray(cut["accidental"]).dtype)
            else:
                # Not re-spelled: the labels spell the old key, so none is scored.
                keep = np.zeros(len(cut["pitch"]), dtype=bool)
            shifted = step != 0
        if self.drop > 0 or self.insert > 0 or self.sigma > 0:
            played, local_slots, cut = corrupt_aligned(
                played, local_slots, cut, rng, start=begin, drop=self.drop,
                insert=self.insert, sigma=self.sigma)
            if len(local_slots) - begin > width:
                counts["trimmed"] += 1
                local_slots = local_slots[:begin + width]
                cut = {name: value[:width] for name, value in cut.items()}
            if not self.respelled:
                # Drops and inserts re-slot the target; a keep mask that is
                # not re-spelled is one value for the whole row, so it is
                # simply rebuilt at the row's new length. (Re-spelling with
                # drops or inserts is refused in `__init__`.)
                keep = np.full(len(cut["pitch"]), not shifted)
        if not any(index >= 0 for index in local_slots[begin:]):
            return None
        source = encode_aligned(played, local_slots, start=begin, encoding=self.encoding,
                                counts=counts, velocity=self.velocity_input)
        return source, cut, keep, shifted

    def _epoch(self) -> int:
        self._epochs = getattr(self, "_epochs", -1) + 1
        return self._epochs

    def _collate(self, sources, targets, torch, eos: bool = True):
        """Pad to the longest member, PITCH_PAD in the pitch streams.

        The pad token matters: `NotationModel.forward` derives its source
        padding mask from `pitch == PITCH_PAD`, so padding with 0 -- a real
        MIDI note -- would let the encoder attend to padding and change a short
        window's prediction depending on who it was batched with, with nothing
        failing.
        """
        width = max(len(source["pitch"]) for source in sources)
        source = {}
        for name in INPUT_STREAMS:
            block = np.zeros((len(sources), width), dtype=np.int64)
            if name == "pitch":
                block[:] = PITCH_PAD
            for row, one in enumerate(sources):
                block[row, :len(one[name])] = one[name]
            source[name] = torch.from_numpy(block).to(self.device)

        # BOS at the front and EOS at the end, in the pitch stream only: it is
        # the only stream with a slot for them, and it is the stream greedy
        # decoding reads to decide when to stop. An aligned target has no EOS
        # (`eos=False`): it writes exactly one slot per input slot, so there
        # is nothing to decide.
        length = max(len(target["pitch"]) for target in targets) + (2 if eos else 1)
        target_block = {}
        for name in OUTPUT_STREAMS:
            block = np.zeros((len(targets), length), dtype=np.int64)
            if name == "pitch":
                block[:] = PITCH_PAD
            for row, one in enumerate(targets):
                inner = len(one["pitch"])
                block[row, 1:1 + inner] = one[name]
                if name == "pitch":
                    block[row, 0] = PITCH_BOS
                    if eos:
                        block[row, 1 + inner] = PITCH_EOS
                else:
                    block[row, 0] = 0
                    if eos:
                        block[row, 1 + inner] = 0
            target_block[name] = torch.from_numpy(block).to(self.device)
        mask = target_block["pitch"] != PITCH_PAD
        return source, target_block, mask


# The format of a prepared window: how `examples` cuts and pairs it. Bump it
# whenever that changes. The settings in `cache_key` name what went IN, so on
# their own they cannot tell that the code turning them into windows has
# changed: `61872ca` re-paired every window with the bar the performance
# actually played, the settings were identical, and the only thing that kept
# the old cache -- which trained 40% of the corpus against the previous bar --
# from being served was renaming the pickle by hand. 2 is that pairing; 1 was
# everything before it. 3 is `examples` rebuilt on `window_spans` and
# `fit_windows`: the trailing part-window, halving, and the 8-second filter
# confined to the legacy encoding. 4 is the aligned performances of
# `aligned_examples` beside them, keyed by `targets` so that neither layout's
# cache can ever be served to a run of the other. The precedent is
# `REFERENCE_VERSION` in `evaluation.notation_eval`, for the same reason.
#
# How a window is ENCODED -- the vocabulary tables and the encoder -- is not
# this number's job. `cache_key` derives that part (`_encoding_digest`), so it
# cannot be forgotten the way a hand-bumped number can.
WINDOW_VERSION = 4


def _encoding_digest() -> str:
    """A digest of everything that decides which INDEX a cached token holds.

    The cache stores token indices (int16), not values. A change to
    `DENOMINATORS`, `TIME_SIGNATURES`, `ACCIDENTALS`, the encoder, the
    multi-stave exclusion or the split would otherwise leave a cache that
    loads cleanly and serves indices into the old tables -- and it fails
    loudly only against a model whose head size happened to change, which a
    reordered table does not do.

    The tables are digested by value, read from the modules at call time.
    The encoder and the split are digested by their SOURCE: that is
    conservative -- editing a comment in `notation.tokens` rebuilds the cache,
    a quarter of an hour -- but the failure it can cause is a rebuild, never a
    stale cache, which is the direction to be wrong in.
    """
    import inspect

    from notation import tokens, vocab

    tables = (vocab.DENOMINATORS, vocab.DURATION_LIMIT, vocab.ONSET_LIMIT,
              vocab.DURATIONS, vocab.ONSETS, vocab.BAR_LENGTHS, vocab.TIME_SIGNATURES,
              vocab.VELOCITY_BUCKETS, vocab.UNKNOWN_VELOCITY, tokens.ACCIDENTALS,
              tokens.INPUT_STREAMS, tokens.OUTPUT_STREAMS, tokens.PITCH_PAD,
              tokens.PITCH_BOS, tokens.PITCH_EOS, tokens.PITCH_REST,
              tokens.SECONDS_BUCKETS, tokens.SECONDS_LIMIT, tokens.GAP_LIMIT,
              tokens.GAP_KNEE, tokens.VOICES, tokens.HANDS,
              sorted(MULTI_STAFF))
    digest = hashlib.sha256(repr(tables).encode())
    for code in (inspect.getsource(tokens), inspect.getsource(vocab),
                 inspect.getsource(splits)):
        digest.update(code.encode())
    return digest.hexdigest()[:16]


def _alignment_digest() -> str:
    """A digest of the code that turns a performance into aligned rows.

    The same conservative rule as `_encoding_digest`: the aligner, the bar
    repair and the target builder are digested by source, so a change to any
    of them rebuilds the aligned cache rather than serving rows the current
    code would not make.

    Every function `aligned_examples` reads a performance or a score through
    is in it, not only the aligner: which rows are sampled
    (`_performance_rows`), the score's bar identities (`bar_numbers_of`), the
    beat and downbeat times (`evaluation.pairs.beat_map`, `downbeat_map`), the
    MIDI reader (`notes_from_midi`) and the misbar count (`_misbarred`). A
    change to any of those changes the rows as surely as a change to the
    aligner does, and this project has lost a whole model to a cache that
    outlived the code which made it (`WINDOW_VERSION`). The encoder is
    `_encoding_digest`'s, which the key carries beside this.
    """
    import inspect

    from evaluation.notation_eval import notes_from_midi
    from evaluation.pairs import beat_map, downbeat_map
    from notation import align

    digest = hashlib.sha256(inspect.getsource(align).encode())
    for function in (aligned_examples, aligned_targets, repeat_mismatch, bar_lines,
                     bar_offset, bar_numbers_of, _performance_rows, _misbarred, beat_starts,
                     beat_map, downbeat_map, notes_from_midi):
        digest.update(inspect.getsource(function).encode())
    return digest.hexdigest()[:16]


def cache_key(config) -> dict:
    """Everything about a `TrainConfig` that changes what the windows ARE.

    `batch` and `device` are deliberately absent: they change how the windows
    are served, not which ones exist, so a cache is reusable across them. A
    cache whose key does not match is rebuilt rather than trusted -- a stale
    one would train the next run on the previous run's split. The key carries
    `WINDOW_VERSION` too, so a cache written by other window code is stale as
    well, and a cache from before the version existed has none and never
    matches. And it carries `_encoding_digest()`, so a cache encoded against
    other tables is stale without anyone having to remember to say so.
    """
    targets = getattr(config, "targets", LEGACY_TARGETS)
    if targets == ALIGNED_TARGETS:
        # An aligned cache holds whole performances and their raw notes: the
        # crop to `max_length` and the input encoding happen per batch, and
        # `bars` and `windowing` do not apply, so none of them is in the key
        # -- one cache serves every run that differs only in those.
        key = {name: getattr(config, name) for name in ("dataset", "recordings", "seed")}
        key["min_matched"] = getattr(config, "min_matched", MIN_MATCHED)
        key["alignment"] = _alignment_digest()
        # Which performances the cache holds. Empty (the default: `splits`'
        # hash) leaves the key exactly as before, so every existing cache
        # still matches; a split file changes which performances the cache
        # must hold, keyed by the FILE'S CONTENTS rather than its path, so a
        # renamed but identical file still matches and an edited one does not.
        split_file = getattr(config, "split_file", "")
        if split_file:
            with open(split_file, "rb") as handle:
                key["split_file"] = hashlib.sha256(handle.read()).hexdigest()[:16]
    else:
        key = {name: getattr(config, name)
               for name in ("dataset", "bars", "max_length", "recordings", "seed",
                            # Both change which windows exist: the windowing
                            # halves and keeps the tail, and the legacy encoding
                            # drops a window past `SECONDS_LIMIT`.
                            "windowing", "input_encoding")}
    key["targets"] = targets
    key["version"] = WINDOW_VERSION
    key["encoding"] = _encoding_digest()
    return key


def build_loaders(config) -> tuple[Batches, Batches]:
    """`(training, validation)` batches for a `TrainConfig`.

    Both sides are materialised once as raw window notes plus encoded score
    streams -- a couple of hundred megabytes for the whole training split --
    and re-augmented per epoch. The corpus itself is streamed and never held:
    see `examples`. Measured on the real checkout, preparing it costs about a
    quarter of an hour, nearly all of it music21 parsing 179 scores and mido
    reading the performances that survive the pairing check (646 of the
    split's 888 for run4), so `config.cache` stores the result and a rerun
    starts in seconds.

    `config.targets` picks the layout: bar windows from `examples` (a config
    that names none, as every run through recipe16), or whole aligned
    performances from `aligned_examples`, cropped per batch to
    `config.max_length`. The cache key records which (`cache_key`).
    """
    import pickle

    # Which target layout to build: a config that names none is a bar-window
    # run, as everything through recipe16 was.
    aligned = getattr(config, "targets", LEGACY_TARGETS) == ALIGNED_TARGETS
    cache = getattr(config, "cache", "")
    if cache and os.path.exists(cache):
        with open(cache, "rb") as handle:
            held = pickle.load(handle)
        stored = (held.get("key") or {}).get("targets", LEGACY_TARGETS)
        if stored != getattr(config, "targets", LEGACY_TARGETS):
            # A cache of the OTHER layout is never rebuilt over. `targets`
            # defaults to aligned since recipe16, so a bar-window command
            # re-run without `--targets 1` would otherwise replace
            # `windows-bars16.pkl` -- a quarter of an hour to rebuild, and the
            # data a published checkpoint was trained on -- with an aligned
            # pickle under the bar-window name. A cache from before `targets`
            # was keyed holds bar windows.
            raise ValueError(
                f"{cache} holds targets={stored} but this run asks for "
                f"targets={getattr(config, 'targets', LEGACY_TARGETS)}; pass the "
                "matching --targets, or a different --cache")
        if held.get("key") != cache_key(config):
            # Said out loud: rebuilding costs a quarter of an hour, and a run
            # that silently takes that long looks like a hang.
            import sys
            print(f"{cache}: built for {held.get('key')}, not {cache_key(config)}; "
                  "rebuilding", file=sys.stderr)
            held = None
    else:
        held = None

    if held is None:
        performances = None
        if getattr(config, "split_file", ""):
            # Bar windows have no `performances` argument to pass this to --
            # `splits_from_file`'s leaving individual performances out only
            # means something once a whole performance is one training row,
            # which is the aligned layout.
            if not aligned:
                raise ValueError("--split-file needs aligned targets")
            assignment, performances = splits_from_file(config.split_file, config.dataset)
        else:
            assignment = splits(config.dataset, config.seed)
        held = {"key": cache_key(config), "sides": {}}
        for want in ("train", "validation"):
            stats: dict = {}
            recordings = config.recordings if want == "train" else 0
            if aligned:
                windows = list(aligned_examples(
                    config.dataset, want, assignment=assignment, recordings=recordings,
                    seed=config.seed, stats=stats,
                    min_matched=getattr(config, "min_matched", MIN_MATCHED),
                    performances=performances))
            else:
                windows = list(examples(config.dataset, want, config.bars, config.max_length,
                                        assignment=assignment, recordings=recordings,
                                        seed=config.seed, stats=stats,
                                        windowing=config.windowing,
                                        encoding=config.input_encoding))
            held["sides"][want] = (windows, stats)
        if cache:
            os.makedirs(os.path.dirname(os.path.abspath(cache)), exist_ok=True)
            with open(cache + ".tmp", "wb") as handle:
                pickle.dump(held, handle, protocol=4)
            os.replace(cache + ".tmp", cache)

    synthetic_from = None
    if getattr(config, "synthetic", ""):
        # Not in the cache: made-up rows are built once by
        # `notation.synthetic.build`, and a run takes them or does not.
        if not aligned:
            raise ValueError("--synthetic needs aligned targets")
        with open(config.synthetic, "rb") as handle:
            synthetic = pickle.load(handle)
        windows, stats = held["sides"]["train"]
        synthetic_from = len(windows)
        stats = dict(stats, synthetic_rows=len(synthetic["rows"]),
                     synthetic_stats=synthetic["stats"])
        held["sides"]["train"] = (list(windows) + list(synthetic["rows"]), stats)

    made = []
    for want, augment in (("train", True), ("validation", False)):
        windows, stats = held["sides"][want]
        extra = ({"targets": ALIGNED_TARGETS, "max_length": config.max_length,
                  "transpose_probability": getattr(config, "transpose_probability",
                                                   TRANSPOSE_PROBABILITY),
                  "transpose_range": getattr(config, "transpose_range", TRANSPOSE_RANGE),
                  # None (a config `notation.train.resolved` has not seen) is
                  # the aligned default, what aligned1 trained with.
                  "duration_jitter": (DURATION_JITTER
                                      if getattr(config, "duration_jitter", None) is None
                                      else config.duration_jitter),
                  "velocity_input": getattr(config, "velocity_input", False),
                  "transpose_zero": getattr(config, "transpose_zero", False),
                  "respelled": getattr(config, "respell", False),
                  "duration_mode": getattr(config, "duration_mode", "note"),
                  "velocity_jitter": getattr(config, "velocity_jitter", 0.0),
                  "sampling": getattr(config, "sampling", "uniform"),
                  "crop": getattr(config, "crop", "slot"),
                  "crop_shift": getattr(config, "crop_shift", 0)}
                 if aligned else {})
        if aligned and augment and synthetic_from is not None:
            extra.update(synthetic_from=synthetic_from,
                         synthetic_share=getattr(config, "synthetic_share", 0.5))
        made.append(Batches(windows, config.batch, config.seed + len(made), config.device,
                            augment, stats, config.drop, config.insert,
                            tempo=(config.tempo_low, config.tempo_high),
                            sigma=config.onset_sigma, encoding=config.input_encoding,
                            **extra))
    return made[0], made[1]
