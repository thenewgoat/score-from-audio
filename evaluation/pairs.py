"""Training pairs for the layer that turns detected notes into notated ones.

Every pair is made by running the REAL detector over REAL audio. Nothing is
synthetically corrupted. A model of the detector's errors is a guess about
which mistakes it makes and how often; running it is not. The cost is that a
pair requires audio, so the corpus is whatever has recordings -- but the
corruption is then exactly the corruption the model will meet at inference,
including the failures we have measured and the ones we have not thought of.

A pair is one detected note and what it should have become:

    input   onset and offset in seconds, pitch, amplitude, and how the detector
            admitted it: the onset peak, whether the melodia pass found it, and
            the note map a semitone either side
    target  position and duration in fractional beat numbers, and whether it
            is a real note at all
    key     the composition, which is what a train/test split must group on

Position is the output that matters most. It is what rubato, swing and
quantisation all come down to, and it is the thing the current rules get wrong
because they impose one global grid on a performance that does not keep one.

Two sources, differing in how position is known:

  ASAP      beat and downbeat annotations for the performance AND the score,
            the same beats in the same order. That is a piecewise-linear map
            from performance seconds to score beats, exact and human-made,
            requiring no note-level alignment.

  Collier   no annotations, so the map comes from aligning the score to the
            audio. Weaker -- alignment cost varies, and that variance becomes
            label noise -- but it carries something ASAP cannot: the notation
            conventions of the transcribers whose scores are the target.
"""

import json
import os
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class Pair:
    """One detected note, and what it should have been."""

    onset: float          # seconds, as detected
    offset: float
    pitch: int
    amplitude: float
    landed: float         # through the ANNOTATED beats -- an oracle, never a model input
    position: float       # where it should be WRITTEN -- the notated position
    real: bool            # was there a notated note here at all
    tracked: float = float("nan")   # through the REAL beat tracker -- the model's input
    onset_confidence: float = float("nan")   # the onset peak that admitted it; NaN iff from_melodia
    from_melodia: bool = False               # admitted by the energy pass, with no onset evidence
    pitch_below: float = float("nan")        # note map at the onset frame, a semitone down
    pitch_above: float = float("nan")        # note map at the onset frame, a semitone up
    # Written length in beats. NaN when not real -- and also when real but the
    # matched score note's length is unknown, because it ends past the last
    # annotated beat. Filter training targets on a finite duration, not on
    # `real` and `unsnapped` alone.
    duration: float = float("nan")
    unsnapped: bool = False                  # the length did not snap, so duration is raw
    composition: str = ""                    # composer/title: what a split must group on


def beat_map(annotation_path: str) -> np.ndarray:
    """Beat times from an ASAP annotation file, in order."""
    times = []
    with open(annotation_path) as handle:
        for line in handle:
            if not line.strip():
                continue
            times.append(float(line.split("\t")[0]))
    return np.array(times)


def downbeat_map(annotation_path: str) -> np.ndarray:
    """Downbeat times from an ASAP annotation file, in order.

    The third column is the label: `b` for a beat, `db` for a downbeat, `bR`
    where ASAP could not establish the position (rubato, mid-score pickups). A
    downbeat may carry more: `db,4/4,-5` is a downbeat, a time signature and a
    key signature.
    """
    times = []
    with open(annotation_path) as handle:
        for line in handle:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) > 2 and parts[2].split(",")[0] == "db":
                times.append(float(parts[0]))
    return np.array(times)


def anchored_positions(times, beats: np.ndarray, downbeats: np.ndarray) -> np.ndarray:
    """Fractional beat numbers measured from the first downbeat -- bar 1 beat 1.

    `beat_index` counts from `beats[0]`, wherever that happens to be. For the
    annotated grid that is ASAP's first beat; for a tracked grid it is whatever
    the tracker found first. The two differ by an unknown per-piece constant,
    and a per-piece constant is pure noise to a model that has to map one onto
    the other. Anchoring both at their own first downbeat gives them a shared
    origin.

    Times outside the grid get NaN rather than the nearest endpoint, for the
    same reason as `position_from_alignment`.
    """
    if len(downbeats) == 0 or len(beats) < 2:
        return np.full(len(np.atleast_1d(times)), np.nan)
    anchor = float(np.interp(downbeats[0], beats, np.arange(len(beats))))
    return positions_in_beats(times, beats) - anchor


def position_from_beats(performance: np.ndarray, score: np.ndarray, seconds):
    """Performance seconds to score quarters, through matched beats.

    The two annotation files list the same beats in the same order -- verified
    on ASAP: equal counts, identical labels -- so beat i gives a corresponding
    pair of times. Interpolating between them maps any moment in the
    performance to its place in the score, which is exactly a rubato-aware
    grid and needs no note-level alignment.
    """
    if len(performance) != len(score):
        raise ValueError(
            f"beat counts differ ({len(performance)} vs {len(score)}); "
            "these annotations do not describe the same piece")
    return np.interp(seconds, performance, score)


def position_from_alignment(alignment, seconds):
    """Audio seconds to score quarters, through a chroma alignment.

    The inverse of `Alignment.to_seconds`. Used where there are no beat
    annotations, and weaker for it: the alignment is inferred rather than
    annotated, so its error becomes label noise.

    Outside the aligned span the answer is NaN, not the nearest endpoint.
    `np.interp` clamps by default, which silently pinned every detection before
    or after the aligned music to position 0 or to the final beat -- labels
    that look valid and are fabricated.
    """
    seconds = np.asarray(seconds, dtype=float)
    inside = (seconds >= alignment.audio_seconds[0]) & (seconds <= alignment.audio_seconds[-1])
    out = np.full(seconds.shape, np.nan)
    out[inside] = np.interp(seconds[inside], alignment.audio_seconds,
                            alignment.score_quarters)
    return out


def alignment_offset(detections, performance_midi, span: float = 4.0,
                     tolerance: float = 0.05) -> tuple[float, float]:
    """How far the recording sits from its own performance MIDI, and how well
    they agree once shifted. Returns (offset seconds, agreement at that offset).

    ASAP places each recording by a `start` value, and for 513 of 519 that is
    right to within ~10ms. For six it is out by 0.8-1.3s -- recordings whose
    performance MIDI came from a different capture than the audio -- and pairs
    made from them are misplaced wholesale. They could be spotted by a naming
    convention, but a measurement does not depend on one and catches anything
    else that drifts the same way.
    """
    by_pitch: dict[int, np.ndarray] = {}
    for time, pitch in performance_midi:
        by_pitch.setdefault(pitch, []).append(time)
    by_pitch = {p: np.array(sorted(v)) for p, v in by_pitch.items()}
    times = np.array([d.onset for d in detections])
    pitches = [d.pitch for d in detections]

    def residuals(shift):
        """Signed gap to the nearest same-pitch MIDI note, for each match."""
        out = []
        for time, pitch in zip(times, pitches):
            candidates = by_pitch.get(pitch)
            if candidates is None:
                continue
            index = np.searchsorted(candidates, time + shift)
            for j in (index - 1, index):
                if 0 <= j < len(candidates) and abs(candidates[j] - time - shift) < tolerance:
                    out.append(candidates[j] - time)
                    break
        return out

    def agreement(shift):
        return len(residuals(shift)) / max(1, len(times))

    # Search finds the right neighbourhood but cannot fix the offset precisely:
    # every shift within the match tolerance of the truth matches the same
    # notes, so the agreement curve is flat-topped and its argmax sits at the
    # plateau's EDGE -- a bias of up to the tolerance itself. So search only to
    # get close, then read the offset off the matched notes directly.
    coarse = np.arange(-span, span, 0.05)
    near = coarse[int(np.argmax([agreement(x) for x in coarse]))]
    matched = residuals(near)
    if not matched:
        return float(near), 0.0
    shift = float(np.median(matched))
    return shift, agreement(shift)


class AdmissionPathError(AssertionError):
    """`onset_confidence` and `from_melodia` disagree: the detector's two
    admission paths have been conflated. Stops generation -- never skipped."""


def check_admission_paths(detections) -> None:
    """Every detection's confidence is NaN exactly when the melodia pass found it.

    A violation means `decode` has mixed up its two loops, and every pair made
    from its output would describe its input wrongly. That is a bug in the
    generator, not a property of one recording, so it must not be recorded as a
    skipped piece and walked past.
    """
    for detection in detections:
        if bool(np.isnan(detection.onset_confidence)) != bool(detection.from_melodia):
            raise AdmissionPathError(
                f"detection at {detection.onset:.3f}s, pitch {detection.pitch}: "
                f"onset_confidence={detection.onset_confidence} but "
                f"from_melodia={detection.from_melodia}")


def _rounded(value) -> float:
    value = float(value)
    return round(value, 4) if np.isfinite(value) else float("nan")


def label(detections, positions, reference, tolerance: float = 0.25,
          tracked=None, composition: str = "") -> list[Pair]:
    """Attach a target to each detection.

    A detection is `real` when the score has a note of the same pitch within
    `tolerance` beats of where it landed, and its target is then that
    score note's OWN position -- which sits exactly on the notated grid.

    The distinction matters and was got wrong once. Where a note landed is
    continuous, because the performance has rubato; where it should be written
    is not. Training on where it landed would teach the layer to reproduce the
    performance's timing, which is the thing being corrected.

    A real detection also takes that score note's written length, snapped by
    `evaluation.durations.snap`; a length that does not snap is kept raw and
    flagged `unsnapped`. A real detection whose score note has no known length
    -- one ending past the last annotated beat -- gets a NaN duration and is
    not flagged. The detector's admission evidence rides along as it came, read
    with defaults so a detection that lacks it still labels.

    Notes in the score with no detection are deliberately absent. The layer is
    not permitted to invent notes: a missing note is the detector's failure,
    and hiding it here would make it unmeasurable.
    """
    from evaluation.durations import snap

    by_pitch: dict[int, list] = {}
    for note in reference:
        by_pitch.setdefault(note.pitch, []).append(note)
    for notes in by_pitch.values():
        notes.sort(key=lambda note: note.onset)
    onsets = {pitch: [note.onset for note in notes] for pitch, notes in by_pitch.items()}

    tracked = [float("nan")] * len(detections) if tracked is None else list(tracked)
    pairs = []
    for detection, landed, heard in zip(detections, positions, tracked):
        if not np.isfinite(landed):
            continue
        candidates = by_pitch.get(detection.pitch, [])
        written, real, match = float(landed), False, None
        if candidates:
            index = np.searchsorted(onsets[detection.pitch], landed)
            best = None
            for near in candidates[max(0, index - 2):index + 3]:
                gap = abs(near.onset - landed)
                if gap <= tolerance and (best is None or gap < best[0]):
                    best = (gap, near)
            if best is not None:
                match = best[1]
                written, real = float(match.onset), True

        duration, unsnapped = float("nan"), False
        if match is not None:
            duration, snapped = snap(getattr(match, "length", float("nan")))
            unsnapped = bool(np.isfinite(duration) and not snapped)

        pairs.append(Pair(
            onset=round(float(detection.onset), 4),
            offset=round(float(detection.offset), 4),
            pitch=int(detection.pitch),
            amplitude=round(float(detection.amplitude), 4),
            landed=round(float(landed), 4),
            position=round(float(written), 4),
            real=real,
            tracked=_rounded(heard),
            onset_confidence=_rounded(getattr(detection, "onset_confidence", float("nan"))),
            from_melodia=bool(getattr(detection, "from_melodia", False)),
            pitch_below=_rounded(getattr(detection, "pitch_below", float("nan"))),
            pitch_above=_rounded(getattr(detection, "pitch_above", float("nan"))),
            duration=_rounded(duration),
            unsnapped=unsnapped,
            composition=composition,
        ))
    return pairs


def write(pairs: list[Pair], path: str, label_name: str) -> None:
    with open(path, "a") as handle:
        for pair in pairs:
            handle.write(json.dumps({"source": label_name, **asdict(pair)}) + "\n")


def midi_notes(path: str) -> list[tuple[float, int]]:
    """(seconds, pitch) for every note-on in a MIDI file."""
    import mido

    out, clock = [], 0.0
    for message in mido.MidiFile(path):
        clock += message.time
        if message.type == "note_on" and message.velocity > 0:
            out.append((clock, message.note))
    return out


def beat_index(times, beats: np.ndarray) -> np.ndarray:
    """Seconds to fractional beat number, against a list of beat times.

    Working in beats rather than seconds or quarters sidesteps every tempo
    conversion: beat i of the performance and beat i of the score are the same
    musical moment by construction, so both sides land in one coordinate.
    """
    return np.interp(np.asarray(times, dtype=float), beats, np.arange(len(beats)))


def positions_in_beats(onsets, performance_beats: np.ndarray) -> np.ndarray:
    """Detection times to fractional beat numbers, in ASAP's own frame.

    No offset is added. The recording has already been cropped as ASAP crops
    it, so its clock IS the annotations' clock. Detections outside the
    annotated span get NaN rather than the nearest endpoint, for the same
    reason as `position_from_alignment`.
    """
    onsets = np.asarray(onsets, dtype=float)
    inside = (onsets >= performance_beats[0]) & (onsets <= performance_beats[-1])
    return np.where(inside, beat_index(onsets, performance_beats), np.nan)


def provenance(extra: dict | None = None) -> dict:
    """What produced a set of pairs, so the pairs can be regenerated exactly.

    Pairs are only meaningful relative to the detector that made them. Train on
    pairs generated at one threshold and run the layer behind another, and it
    corrects for a detector that is no longer there -- silently, since nothing
    in the pairs says which detector they came from. So they say.
    """
    import datetime
    import hashlib
    import inspect
    import subprocess

    from transcribe import detect
    from evaluation import durations

    defaults = {name: parameter.default
                for name, parameter in inspect.signature(detect.decode).parameters.items()
                if parameter.default is not inspect.Parameter.empty}
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                text=True, cwd=os.path.dirname(__file__)).stdout.strip()
    except Exception:                                          # noqa: BLE001
        commit = None
    return {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "generator_commit": commit,
        "model_sha256": hashlib.sha256(open(detect.MODEL_PATH, "rb").read()).hexdigest(),
        "decoder_defaults": defaults,
        "min_note_frames_rule": "frames_for_tempo(tempo from the annotated beats)",
        "position_unit": "fractional beat number, measured from the first downbeat",
        "duration_unit": "fractional beat number, snapped to the lattice below",
        "duration_lattice": "multiples of 1/8 and 1/6 beat, up to 32 beats",
        "duration_tolerance": {"relative": durations.RELATIVE, "floor_beats": durations.FLOOR},
        **(extra or {}),
    }


def _fold_tracker(stem: str, cache: dict):
    """One Beat This! instance per checkpoint, reused across every piece in it.

    Nine checkpoints cover the corpus -- eight cross-validation folds plus
    `final0` for the pieces Beat This! never trained on. A fresh instance per
    piece would load a 78 MB checkpoint once per recording instead of once per
    checkpoint.
    """
    from evaluation.folds import checkpoint_for
    from transcribe.beats import get_tracker

    checkpoint = checkpoint_for(stem)
    if checkpoint not in cache:
        cache[checkpoint] = get_tracker("beat_this", checkpoint=checkpoint)
    return cache[checkpoint]


def from_asap(dataset: str, audio_dir: str, out_path: str, limit: int | None = None,
              cache_dir: str | None = None, progress=None, tracker=None) -> dict:
    """Build pairs from ASAP, running the detector over its recordings.

    `audio_dir` is an ASAP tree whose recordings were written by
    `evaluation.asap_audio --canonical`: ASAP's own crop and padding, at
    MAESTRO's full rate and channels, at the paths the metadata names. That puts
    the recording in the same time frame as the annotations, so detection times
    are compared with beat times DIRECTLY -- no offset added, none subtracted.

    The detector resamples on load and caches its probability maps, so it reads
    the full-rate recording itself. No second, downsampled copy is kept.

    An earlier version added ASAP's `start` to every detection and ignored the
    0.5s of padding ASAP prepends, which would have misplaced every note by
    about half a second. Verified the frame is now right: on Bach BWV 846 the
    first detected onset lands at 0.499s against the performance MIDI's 0.500s.

    ASAP annotates the same beats in performance and score, so a detection's
    true metrical position needs no note-level alignment -- map its time to a
    fractional beat number through the performance's beats, and that number
    means the same thing in the score.
    """
    # A tracker without downbeats has no anchor, so it cannot produce the
    # shared origin `anchored_positions` depends on. Refused loudly rather than
    # falling back to the first beat, which would reintroduce the very constant
    # the anchor removes -- silently, and in 1.67 M rows.
    if tracker is not None and not tracker.has_downbeats:
        raise ValueError(
            f"tracker {tracker.name!r} reports no downbeats, so pairs would have "
            "no phase anchor; pass a downbeat-capable tracker")

    import csv

    from transcribe.detect import decode_to_notes, frames_for_tempo
    from evaluation.asap_audio import canonical_path
    from evaluation.folds import stem_for
    from evaluation.workbench import maps_for
    from evaluation.durations import length_in_beats, midi_spans

    cache_dir = cache_dir or os.path.join(dataset, ".pairs")
    os.makedirs(cache_dir, exist_ok=True)
    rows = [r for r in csv.DictReader(open(os.path.join(dataset, "metadata.csv")))
            if r.get("audio_performance")][:limit]
    # Grouped by checkpoint so each of the eight loads once rather than once per
    # piece. `limit` is applied before the sort so a limited run stays a prefix
    # of the same corpus.
    from evaluation.folds import checkpoint_for as _checkpoint_for
    rows.sort(key=lambda r: _checkpoint_for(stem_for(r["audio_performance"])))

    if os.path.exists(out_path):
        os.remove(out_path)
    written, skipped, per_source, fold_cache = 0, [], {}, {}
    for index, row in enumerate(rows, 1):
        name = row["audio_performance"]
        if progress:
            progress(index, len(rows), name)
        try:
            audio = canonical_path(audio_dir, row)
            if not os.path.exists(audio):
                skipped.append((name, "not extracted")); continue

            performance_beats = beat_map(os.path.join(dataset, row["performance_annotations"]))
            score_beats = beat_map(os.path.join(dataset, row["midi_score_annotations"]))
            if len(performance_beats) != len(score_beats):
                skipped.append((name, "beat counts differ")); continue

            maps = maps_for(audio, cache_dir)
            # The tempo comes from the annotated beats, not a beat tracker:
            # they are ground truth here, and the floor depends on the tempo.
            tempo = 60.0 / float(np.median(np.diff(performance_beats)))
            detections = decode_to_notes(maps, min_note_frames=frames_for_tempo(tempo))
            # Refuses, and is re-raised below rather than skipped: a conflation
            # of the two admission paths is a generator bug, not a bad piece.
            check_admission_paths(detections)

            shift, agreement = alignment_offset(
                detections, midi_notes(os.path.join(dataset, row["midi_performance"])))
            if abs(shift) > MAX_OFFSET:
                skipped.append((name, f"audio {shift:+.2f}s from its own performance MIDI"))
                continue

            performance_downbeats = downbeat_map(
                os.path.join(dataset, row["performance_annotations"]))
            if len(performance_downbeats) == 0:
                skipped.append((name, "no annotated downbeats")); continue

            piece = tracker or _fold_tracker(stem_for(name), fold_cache)
            heard = piece.track(audio)
            # `has_downbeats` is only a class-level promise -- a tracker that
            # generally reports downbeats can still return none for one
            # particular recording. Without this, that piece would slip
            # through with `tracked` all-NaN and nothing would count it,
            # exactly the asymmetry the annotated-side guard above prevents.
            if len(heard.downbeats) == 0:
                skipped.append((name, "tracker found no downbeats")); continue
            tracked = (anchored_positions([d.onset for d in detections],
                                          heard.beats, heard.downbeats)
                       if len(heard.beats) > 1 else
                       np.full(len(detections), np.nan))

            landed = anchored_positions([d.onset for d in detections],
                                        performance_beats, performance_downbeats)

            score_downbeats = downbeat_map(
                os.path.join(dataset, row["midi_score_annotations"]))
            if len(score_downbeats) == 0:
                skipped.append((name, "no score downbeats")); continue
            spans = midi_spans(os.path.join(dataset, row["midi_score"]))
            placed = anchored_positions([on for on, _, _ in spans], score_beats, score_downbeats)
            reference = [_ScoreNote(pitch, float(position), length_in_beats(on, off, score_beats))
                         for (on, off, pitch), position in zip(spans, placed)]
            composition = f"{row['composer']}/{row['title']}"
            pairs = label(detections, landed, reference, tolerance=0.25, tracked=tracked,
                          composition=composition)
            write(pairs, out_path, name)
            written += len(pairs)
            heard_tempo = heard.tempo
            real_count = sum(1 for p in pairs if p.real)
            unsnapped_count = sum(1 for p in pairs if p.unsnapped)
            per_source[name] = {"tempo": round(tempo, 2),
                                "tracker": heard.tracker,
                                "tracked_tempo": round(heard_tempo, 2)
                                                 if np.isfinite(heard_tempo) else None,
                                "tempo_ratio": round(heard_tempo / tempo, 3)
                                               if np.isfinite(heard_tempo) else None,
                                "beats_per_bar": heard.beats_per_bar,
                                "min_note_frames": frames_for_tempo(tempo),
                                "midi_offset": round(shift, 3),
                                "midi_agreement": round(agreement, 3),
                                "pairs": len(pairs),
                                "real": real_count,
                                "composition": composition,
                                "unsnapped_rate": round(unsnapped_count / real_count, 4)
                                                  if real_count else None,
                                "tracked_nan": sum(1 for p in pairs
                                                   if not np.isfinite(p.tracked))}
        except AdmissionPathError:
            raise
        except Exception as error:                             # noqa: BLE001
            skipped.append((name, f"{type(error).__name__}: {error}"))

    with open(out_path + ".meta.json", "w") as handle:
        json.dump(provenance({"source": "ASAP", "dataset": dataset,
                              "trackers": sorted({r["tracker"] for r in per_source.values()}),
                              "pieces": len(per_source), "pairs": written,
                              "skipped": skipped, "per_source": per_source}),
                  handle, indent=1)
    return {"pairs": written, "pieces": len(rows) - len(skipped), "skipped": skipped}


MAX_OFFSET = 0.1    # seconds; the six misplaced recordings sit at 0.8-1.3s


class _ScoreNote:
    """A score note positioned in beats, which is the unit `label` compares in.

    `length` is its raw length in beats; `label` snaps it, so the snap happens
    in one place.
    """

    __slots__ = ("pitch", "onset", "length")

    def __init__(self, pitch: int, onset: float, length: float = float("nan")):
        self.pitch, self.onset, self.length = pitch, onset, length


def from_collier(dataset: str, out_path: str, limit: int | None = None,
                 min_correlation: float = 0.8, cache_dir: str | None = None,
                 progress=None) -> dict:
    """Build pairs from a corpus of recordings with published scores.

    The detector is run over the real audio; the score is aligned to that audio
    and gives each detection its metrical position.
    """
    from evaluation.align import align
    from evaluation.measure import prepare_audio
    from evaluation.reference import read_score
    from evaluation.workbench import maps_for
    from transcribe.detect import decode_to_notes, frames_for_tempo

    cache_dir = cache_dir or os.path.join(dataset, ".pairs")
    os.makedirs(cache_dir, exist_ok=True)
    rows = json.load(open(os.path.join(dataset, "pairing_evidence.json")))
    chosen = [r for r in rows
              if r.get("score_path") and r.get("audio_path")
              and (r.get("correlation") or -9) >= min_correlation][:limit]

    if os.path.exists(out_path):
        os.remove(out_path)
    written, skipped = 0, []
    for index, row in enumerate(chosen, 1):
        name = row["label"]
        if progress:
            progress(index, len(chosen), name)
        try:
            source = [os.path.join(dataset, "audio_src", f)
                      for f in os.listdir(os.path.join(dataset, "audio_src"))
                      if os.path.splitext(f)[0] == name]
            if not source:
                skipped.append((name, "no audio")); continue
            audio = prepare_audio(source[0], os.path.join(cache_dir, "wav"), 22050)
            reference = read_score(os.path.join(dataset, row["score_path"]))
            if not reference:
                skipped.append((name, "empty score")); continue

            alignment = align(reference, audio)
            maps = maps_for(audio, cache_dir)
            from transcribe.audio import estimate_tempo

            tempo = estimate_tempo(audio)
            detections = decode_to_notes(maps, min_note_frames=frames_for_tempo(tempo))

            positions = position_from_alignment(
                alignment, np.array([d.onset for d in detections]))
            pairs = label(detections, positions, reference)
            write(pairs, out_path, name)
            written += len(pairs)
        except Exception as error:                             # noqa: BLE001
            skipped.append((name, f"{type(error).__name__}: {error}"))
    return {"pairs": written, "pieces": len(chosen) - len(skipped), "skipped": skipped}


def main(argv=None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="python -m evaluation.pairs",
        description="Build training pairs by running the real detector over real audio.")
    parser.add_argument("dataset")
    parser.add_argument("--asap-audio", default=None,
                        help="treat DATASET as an ASAP checkout and read cropped audio from here")
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-correlation", type=float, default=0.8)
    parser.add_argument("--cache", default=None)
    parser.add_argument("--tracker", default=None,
                        help="beat tracker to use; default is one Beat This! "
                             "checkpoint per ASAP fold, which is the only honest "
                             "choice for pairs (see evaluation/folds.py)")
    args = parser.parse_args(argv)

    def progress(index, total, name):
        print(f"[{index}/{total}] {name[:58]}", file=sys.stderr)

    if args.asap_audio:
        if args.tracker == "beat_this":
            # get_tracker("beat_this") with no checkpoint builds "final0",
            # which trained on 471 of ASAP's 519 pieces. Tracking every piece
            # with that checkpoint bypasses `_fold_tracker`'s per-piece
            # held-out selection and would fill `tracked` from a model that
            # memorised the beats it is meant to be tracking blind.
            parser.error(
                "--tracker beat_this carries no fold, so it would track ASAP "
                "with checkpoint final0 -- trained on 471 of ASAP's 519 pieces. "
                "ASAP pairs must use the held-out fold checkpoints; omit "
                "--tracker to select them automatically (see evaluation/folds.py).")
        chosen = None
        if args.tracker:
            from transcribe.beats import get_tracker
            chosen = get_tracker(args.tracker)
        result = from_asap(args.dataset, args.asap_audio, args.out, args.limit,
                           args.cache, progress, chosen)
    else:
        result = from_collier(args.dataset, args.out, args.limit,
                              args.min_correlation, args.cache, progress)
    print(f"\n{result['pairs']:,} pairs from {result['pieces']} pieces -> {args.out}")
    for name, why in result["skipped"][:8]:
        print(f"  skipped {name[:44]}: {why}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
