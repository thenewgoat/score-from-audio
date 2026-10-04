"""How long a score note should be written, in beats.

The training pair's duration target comes from ASAP's score MIDI, and the raw
lengths there are not note values. Across all 178 scores and 517,073 notes the
median distance from a raw length to the nearest clean value is 0.83%, at the
75th percentile too: a uniform gate applied by whatever rendered the MIDI.
Training on the raw length would teach the corrector to reproduce that
artefact, so the target is snapped -- the same argument `pairs.label` makes
about position.

The lattice is every multiple of an eighth or a sixth of a beat, up to 32
beats. It was not the first choice. Snapping to single note values from an
eighth to eight beats left 20.4% of the unsnapped notes longer than half a
beat, and they were not ornaments: tied notes, which the MIDI stores as one
long note (1.25, 2.5, 5 beats), and notes rendered short by a fixed gap -- the
commonest failure was a one-beat note at 0.917. A lattice holds any tied sum;
the gap-shortened notes it leaves unsnapped, flagged rather than mislabelled.

Measured with this lattice (`audit`, 2026-09-13): 93.1% snapped, 6.9% of the
unsnapped longer than half a beat, every note under a tenth of a beat left
unsnapped.
"""

from collections import defaultdict, deque

import numpy as np

RELATIVE = 0.02
FLOOR = 0.02            # beats
LONGEST = 32.0          # beats

LATTICE = np.array(sorted({m / 8 for m in range(1, int(LONGEST * 8) + 1)}
                          | {m / 6 for m in range(1, int(LONGEST * 6) + 1)}))


def snap(duration: float) -> tuple[float, bool]:
    """(written length, whether it snapped). A length that does not snap is
    returned raw, so the residual survives for anything that wants it.

    Relative with an absolute floor: a purely relative tolerance over-penalises
    short notes, and a purely absolute one under-serves long ones.
    """
    duration = float(duration)
    if not np.isfinite(duration) or duration <= 0:
        return duration, False
    index = int(np.clip(np.searchsorted(LATTICE, duration), 1, len(LATTICE) - 1))
    low, high = LATTICE[index - 1], LATTICE[index]
    nearest = low if abs(duration - low) <= abs(high - duration) else high
    if abs(duration - nearest) <= max(RELATIVE * nearest, FLOOR):
        return float(nearest), True
    return duration, False


def midi_spans(path: str) -> list[tuple[float, float, int]]:
    """(note-on seconds, note-off seconds, pitch) for every terminated note.

    A note ends at a `note_off` or at a `note_on` with velocity 0 -- both
    spellings occur. Notes of the same pitch on the same channel are paired
    first in, first out. A note-on that is never ended has no length and is
    left out, and so is a span of zero length -- an exporter's strike, release
    and re-strike of one key at a single instant. Those are the two ways this
    differs from `pairs.midi_notes`.
    """
    import mido

    spans, clock, sounding = [], 0.0, defaultdict(deque)
    for message in mido.MidiFile(path):
        clock += message.time
        if message.type == "note_on" and message.velocity > 0:
            sounding[(message.channel, message.note)].append(clock)
        elif message.type == "note_off" or (message.type == "note_on" and message.velocity == 0):
            started = sounding[(message.channel, message.note)]
            if started:
                begun = started.popleft()
                # A span with no length is not a note. Two voices sharing one key
                # are exported as strike, instant release, re-strike at a single
                # instant; kept, that zero-length span sorts ahead of the real
                # note at the same onset and wins its match in `pairs.label`,
                # discarding a measurable duration.
                if clock > begun:
                    spans.append((begun, clock, message.note))
    return sorted(spans)


def length_in_beats(on: float, off: float, beats: np.ndarray) -> float:
    """A note's length in fractional beats through a beat map, NaN outside it.

    Through `pairs.beat_index`, the one place seconds become beats, so a length
    and a position can never disagree about the conversion.
    """
    from evaluation.pairs import beat_index

    if len(beats) < 2 or on < beats[0] or off > beats[-1] or off <= on:
        return float("nan")
    start, end = beat_index([on, off], beats)
    return float(end - start)


def audit(dataset: str) -> dict:
    """Snap every note of every ASAP score, re-running the numbers above.

    Each score is counted once, though several performances share it. Lengths
    under 0.005 beats are MIDI debris and are not counted.
    """
    import csv
    import os

    from evaluation.pairs import beat_map

    with open(os.path.join(dataset, "metadata.csv")) as handle:
        rows = [r for r in csv.DictReader(handle) if r.get("audio_performance")]
    scores: dict[str, str] = {}
    for row in rows:
        scores.setdefault(row["midi_score"], row["midi_score_annotations"])

    lengths, snapped_flags, per_score = [], [], {}
    for score, annotations in scores.items():
        beats = beat_map(os.path.join(dataset, annotations))
        found = [length_in_beats(on, off, beats)
                 for on, off, _ in midi_spans(os.path.join(dataset, score))]
        found = [x for x in found if np.isfinite(x) and x > 0.005]
        flags = [snap(x)[1] for x in found]
        lengths.extend(found)
        snapped_flags.extend(flags)
        per_score[score] = (1 - sum(flags) / len(flags)) if flags else float("nan")

    length = np.array(lengths)
    snapped = np.array(snapped_flags)
    unsnapped = length[~snapped]
    return {
        "scores": len(scores),
        "notes": int(len(length)),
        "snapped": float(snapped.mean()),
        "unsnapped_median": float(np.median(unsnapped)),
        "unsnapped_p90": float(np.percentile(unsnapped, 90)),
        "unsnapped_over_half_beat": float(np.mean(unsnapped > 0.5)),
        "short_left_unsnapped": float(np.mean(~snapped[length < 0.1])),
        "unsnapped_rate_per_score": per_score,
    }


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m evaluation.durations",
        description="Audit the duration lattice against every ASAP score.")
    parser.add_argument("dataset", help="an asap-dataset checkout")
    args = parser.parse_args(argv)

    result = audit(args.dataset)
    rates = np.array([r for r in result["unsnapped_rate_per_score"].values() if np.isfinite(r)])
    print(f"scores {result['scores']}, notes {result['notes']:,}")
    print(f"snapped                         {100 * result['snapped']:.1f}%")
    print(f"unsnapped: median length        {result['unsnapped_median']:.3f} beats")
    print(f"unsnapped: 90th percentile      {result['unsnapped_p90']:.3f} beats")
    print(f"unsnapped: longer than 1/2 beat {100 * result['unsnapped_over_half_beat']:.1f}%")
    print(f"notes under 0.1 beat unsnapped  {100 * result['short_left_unsnapped']:.1f}%")
    print(f"unsnapped rate per score        median {100 * np.median(rates):.1f}%, "
          f"p90 {100 * np.percentile(rates, 90):.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
