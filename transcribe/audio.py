"""Turning detected note events into a readable grand-staff score.

A pitch detector gives you (onset, offset, pitch) triples. A score is a
different kind of object, and three things stand between them. Each was found
by measuring a real 50-second piano recording, and each is handled here
explicitly:

  clustering  Notes struck together do not arrive together. A hand spreads a
              chord, and onset detection has finite resolution. Quantised
              independently, two members of one chord can round into different
              slots -- and then a notation library, seeing two elements at
              different offsets, writes them as separate voices. On the
              measured recording, 108 of 262 onset gaps were under 60ms, and
              11 of those pairs landed on different sixteenth slots.

  filling     A detector reports when a note stopped SOUNDING, not when the
              next one starts. Piano tone decays from the instant of the
              strike, so in a legato passage the note is marked over while the
              key is still down. Taken literally that leaves a hole: on the
              measured recording, 124 of 262 notes ended before the next
              attack, totalling 28.6 seconds of hole in a 49.7-second piece.
              Each note is therefore held to the next attack IN ITS OWN STAFF
              -- the global next attack is not enough, because then a treble
              note stops whenever the bass happens to move.

  quantising  Onsets snap to a grid whose phase is fitted to the data. A piece
              need not begin on a beat, and assuming it does rounds every note
              in the piece against the wrong slots.

The grid size is a genuine trade-off with no right answer, so it is a
parameter rather than a constant. Measured on the same recording: a sixteenth
grid keeps all 155 attacks but leaves 44% of them off the eighth, which reads
as a page of dotted eighths; an eighth grid gives clean note values but rounds
14 attacks (9%) into the chord before them. Of the 31 sixteenth-gaps, only 9
were unambiguous split eighth-pairs; the rest divided 17/14 between rolled
chords and genuine stepwise runs, so no automatic rule separates them.

Everything down to `plan` is pure stdlib, so it can be tested without audio,
models or a notation library. `build_score` imports music21, `detect_events`
imports basic-pitch, and `estimate_beats`/`estimate_tempo` import a beat
tracker (see `transcribe.beats`), all lazily -- basic-pitch pulls TensorFlow,
and no part of the planning logic needs any of them.
"""

import csv
import math
import os
import sys
from dataclasses import dataclass
from fractions import Fraction

SIXTEENTH = Fraction(1, 4)
EIGHTH = Fraction(1, 2)
GRIDS = {"sixteenth": SIXTEENTH, "eighth": EIGHTH, "quarter": Fraction(1)}

CLUSTER_WINDOW = 0.060       # seconds; notes this close were struck together
MAX_FILL = Fraction(2)       # quarter notes; a longer gap is a real rest
SPLIT_PITCH = 60             # middle C and above -> treble staff


@dataclass(frozen=True)
class Event:
    """One detected note: when it started and stopped sounding, and its pitch."""

    onset: float
    offset: float
    pitch: int


@dataclass(frozen=True)
class Placement:
    """One notated chord or note, positioned in quarter-note units."""

    offset: Fraction
    duration: Fraction
    pitches: tuple[int, ...]
    staff: int


def read_events(path: str) -> list[Event]:
    """Read basic-pitch's CSV export."""
    events = []
    with open(path, newline="") as handle:
        reader = csv.reader(handle)
        next(reader)
        for row in reader:
            events.append(Event(float(row[0]), float(row[1]), int(row[2])))
    return sorted(events, key=lambda event: (event.onset, event.pitch))


def cluster(events: list[Event], window: float = CLUSTER_WINDOW) -> list[list[Event]]:
    """Group events struck together; one list per attack.

    The window runs from the last event admitted, not from the first, so a
    rolled chord spread over more than one window still forms a single attack.
    """
    if not events:
        return []
    events = sorted(events, key=lambda event: (event.onset, event.pitch))
    groups = [[events[0]]]
    for event in events[1:]:
        if event.onset - groups[-1][-1].onset <= window:
            groups[-1].append(event)
        else:
            groups.append([event])
    return groups


def fit_phase(times: list[float], step: float, trials: int = 60) -> float:
    """The grid offset that minimises mean rounding error.

    Without this the grid is assumed to start at t=0, which is only true if the
    recording begins exactly on a beat. When it does not, every note in the
    piece is rounded against the wrong slots.
    """
    if not times:
        return 0.0

    def error(offset: float) -> float:
        total = 0.0
        for time in times:
            units = (time - offset) / step
            total += abs(units - round(units))
        return total / len(times)

    return min((step * index / trials for index in range(trials)), key=error)


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def plan(
    events: list[Event],
    bpm: float,
    grid: Fraction = EIGHTH,
    max_fill: Fraction = MAX_FILL,
    window: float = CLUSTER_WINDOW,
    split_pitch: int = SPLIT_PITCH,
) -> list[Placement]:
    """Lay detected events out on a grid, ready to notate.

    Returns placements ordered by staff then offset. Pure: no audio, no
    notation library, no network.
    """
    if bpm <= 0:
        raise ValueError("bpm must be positive")
    if grid <= 0:
        raise ValueError("grid must be positive")
    groups = cluster(events, window)
    if not groups:
        return []

    step = (60.0 / bpm) * float(grid)
    onsets = [_median([event.onset for event in group]) for group in groups]
    phase = fit_phase(onsets, step)

    placements: list[Placement] = []
    for staff in (1, 2):
        entries: list[dict] = []
        for group, onset in zip(groups, onsets):
            wanted = [event for event in group
                      if (event.pitch >= split_pitch) == (staff == 1)]
            if not wanted:
                continue
            offset = max(Fraction(0), Fraction(round((onset - phase) / step)) * grid)
            held = max(event.offset for event in wanted)
            # Two attacks can round into one slot. They are a single chord, not
            # two elements competing for the same offset -- left unmerged, a
            # notation library resolves the overlap by inventing a voice, which
            # is what scatters stray rests across the page.
            if entries and entries[-1]["offset"] == offset:
                entries[-1]["pitches"] |= {event.pitch for event in wanted}
                entries[-1]["held"] = max(entries[-1]["held"], held)
                continue
            entries.append({
                "offset": offset,
                "seconds": onset,
                "pitches": {event.pitch for event in wanted},
                "held": held,
            })

        for index, entry in enumerate(entries):
            detected = max(grid, Fraction(round((entry["held"] - entry["seconds"]) / step)) * grid)
            if index + 1 == len(entries):
                duration = detected
            else:
                gap = entries[index + 1]["offset"] - entry["offset"]
                duration = gap if gap <= max_fill else min(detected, gap)
            placements.append(Placement(
                offset=entry["offset"],
                duration=duration,
                pitches=tuple(sorted(entry["pitches"])),
                staff=staff,
            ))

    return placements


def build_score(placements: list[Placement], bpm: float, beats_per_bar: int = 4):
    """Render placements as a music21 grand staff.

    Two `PartStaff` objects inside a braced `StaffGroup`, which is what makes
    one piano part rather than two unrelated instruments.
    """
    from music21 import chord as m21chord
    from music21 import clef, layout, meter, note, stream, tempo

    if not placements:
        raise ValueError("no placements to notate")

    score = stream.Score()
    score.insert(0, tempo.MetronomeMark(number=round(bpm)))
    parts = []
    for staff in (1, 2):
        part = stream.PartStaff()
        part.insert(0, clef.TrebleClef() if staff == 1 else clef.BassClef())
        part.insert(0, meter.TimeSignature(f"{beats_per_bar}/4"))
        for placement in placements:
            if placement.staff != staff:
                continue
            element = (m21chord.Chord(list(placement.pitches))
                       if len(placement.pitches) > 1
                       else note.Note(placement.pitches[0]))
            element.quarterLength = float(placement.duration)
            part.insert(float(placement.offset), element)
        parts.append(part)

    for part in parts:
        score.insert(0, part)
    score.insert(0, layout.StaffGroup(parts, symbol="brace", barTogether=True))
    written = score.makeNotation()

    # Fill the silences with rests that are actually drawn.
    #
    # Left alone, music21's MusicXML exporter invents the filler rests itself
    # and hardcodes `hideRests=True`, so every rest is written
    # `print-object="no"`. A reader honouring that -- verovio does -- draws
    # nothing, and a bar of silence engraves as an empty bar. The notation is
    # correct in the file and absent on the page, which is worse than either.
    #
    # Filling them here leaves the exporter nothing to invent.
    for part in written.parts:
        part.makeRests(fillGaps=True, inPlace=True, hideRests=False,
                       timeRangeFromBarDuration=True)
    return written


def detect_events(audio_path: str, **thresholds) -> list[Event]:
    """Detect notes in an audio or video file.

    Imported lazily so the planning logic above stays free of numpy, the model
    and the audio stack.
    """
    from transcribe.detect import detect

    return sorted(
        (Event(d.onset, d.offset, d.pitch) for d in detect(audio_path, **thresholds)),
        key=lambda event: (event.onset, event.pitch),
    )


def estimate_beats(audio_path: str, tracker=None):
    """Track the audio. Returns a `transcribe.beats.Beats`."""
    from transcribe.beats import track

    return track(audio_path, tracker)


def estimate_tempo(audio_path: str, tracker=None) -> float:
    """Beat-track the audio for a tempo, in BPM.

    Kept as a name because the workbench and the Collier pair generator want
    only the number. The tempo is derived from beat spacing rather than a
    tempogram -- see `transcribe.beats`.
    """
    return estimate_beats(audio_path, tracker).tempo


def transcribe(
    source: str,
    out_path: str,
    bpm: float | None = None,
    grid: Fraction = EIGHTH,
    beats_per_bar: int | None = None,
    tracker=None,
) -> list[Placement]:
    """Audio (or a basic-pitch CSV) to a MusicXML file."""
    if source.lower().endswith(".csv"):
        events = read_events(source)
        if bpm is None:
            raise ValueError("--bpm is required when the source is a CSV")
        if beats_per_bar is None:
            beats_per_bar = 4
    else:
        # One tracking pass supplies both: the tempo sets how short a note may
        # be and still be real, and the downbeats say how many beats to a bar.
        if bpm is None or beats_per_bar is None:
            tracked = estimate_beats(source, tracker)
            if bpm is None:
                bpm = tracked.tempo
                # librosa always returned a number; a tracker that reports
                # fewer than two beats gives NaN (see `Beats.derive`), and
                # frames_for_tempo(nan) does not raise -- it would reach
                # `plan` and `build_score` as a silent NaN bpm.
                if not math.isfinite(bpm):
                    raise ValueError(
                        f"beat tracking found too few beats in {source!r} to "
                        "derive a tempo; pass --bpm explicitly")
            if beats_per_bar is None:
                # A tracker without downbeats reports None; 4/4 remains the
                # fallback, but it is now a fallback rather than an assumption.
                beats_per_bar = tracked.beats_per_bar or 4
        from transcribe.detect import frames_for_tempo

        # Derived from the tempo, NOT from the notation grid. How short a note
        # may be and still be real is a fact about the playing; how it is
        # written down is a separate choice. Tying the floor to an eighth-note
        # grid would discard a staccato eighth for being brief.
        events = detect_events(source, min_note_frames=frames_for_tempo(bpm))
    if not events:
        raise ValueError(f"no notes detected in {source}")

    placements = plan(events, bpm, grid=grid)
    build_score(placements, bpm, beats_per_bar).write("musicxml", fp=out_path)
    return placements


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m transcribe.audio",
        description="Transcribe audio into a grand-staff MusicXML score.",
    )
    parser.add_argument("source", help="audio/video file, or a basic-pitch CSV export")
    parser.add_argument("--out", required=True, help="MusicXML file to write")
    parser.add_argument("--bpm", type=float, default=None,
                        help="tempo; beat-tracked from the audio when omitted")
    parser.add_argument("--grid", choices=sorted(GRIDS), default="eighth",
                        help="quantisation grid (default: eighth). A finer grid keeps "
                             "fast passages but scatters dotted values; see the module "
                             "docstring for the measured trade-off.")
    parser.add_argument("--beats-per-bar", type=int, default=None,
                        help="beats to a bar; read from the tracked downbeats when omitted")
    args = parser.parse_args(argv)

    if not os.path.exists(args.source):
        parser.error(f"source not found: {args.source}")

    try:
        placements = transcribe(
            args.source, args.out, bpm=args.bpm,
            grid=GRIDS[args.grid], beats_per_bar=args.beats_per_bar,
        )
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    chords = sum(1 for placement in placements if len(placement.pitches) > 1)
    print(f"{len(placements)} placements ({chords} chords) -> {args.out}")

    from transcribe.validate import validate

    for problem in validate(args.out):
        print(f"warning: {problem}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
