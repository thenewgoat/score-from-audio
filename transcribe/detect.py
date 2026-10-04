"""Detecting notes with basic-pitch's model, without basic-pitch's dependencies.

The `basic-pitch` package pulls TensorFlow -- about 1.4 GB -- to run a network
of 230 KB. The model is also published as ONNX, which runs under a runtime two
orders of magnitude smaller, so the model file is vendored here (Apache-2.0,
Spotify AB; see `models/NOTICE`) and driven directly.

That is not only about size. It makes the whole project installable from one
set of requirements, and it is the same file the phone app runs, so the desktop
and the phone execute the same weights rather than two ports of them.

The model takes RAW AUDIO. The constant-Q transform and the harmonic stacking
are layers inside the graph, so there is no signal processing to reimplement
and nothing to get subtly wrong.

What did have to be reimplemented is everything around it: the overlapping
windows, the trimming that stitches their outputs back together, and the
decoder that turns three probability maps into notes. Those are ported from
basic-pitch and checked against it.
"""

import os
from dataclasses import dataclass

import numpy as np

AUDIO_SAMPLE_RATE = 22050
ANNOTATIONS_FPS = 86        # the model's own figure, not 22050/256 = 86.13
FFT_HOP = 256
ANNOT_N_FRAMES = 172
AUDIO_N_SAMPLES = 43844
N_OVERLAPPING_FRAMES = 30
MIDI_OFFSET = 21
MAX_FREQ_IDX = 87
CONTOURS_BINS_PER_SEMITONE = 3

MIN_NOTE_FRAMES = 11        # basic-pitch's default: 128 ms
SHORTEST_FRAMES = 3         # ~35 ms; below this the decoder admits blips
LONGEST_FRAMES = 11

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "nmp.onnx")
_SESSION = None


@dataclass(frozen=True)
class Detection:
    """A detected note, and the evidence that admitted it.

    `amplitude` is the mean frame activation while it sounded. The last four
    fields record how `decode` came to admit the note, which is known only
    inside `decode`: once it returns, which of its two loops produced an event
    is gone. They default, so a Detection built without them still works.
    """

    onset: float
    offset: float
    pitch: int
    amplitude: float
    onset_confidence: float = float("nan")   # the onset-map peak that admitted it
    from_melodia: bool = False               # found by the energy pass; no onset peak
    pitch_below: float = float("nan")        # note map at the onset frame, pitch - 1
    pitch_above: float = float("nan")        # note map at the onset frame, pitch + 1


def session():
    """The ONNX session, made once. Loading dominates the cost of a short clip."""
    global _SESSION
    if _SESSION is None:
        import onnxruntime

        options = onnxruntime.SessionOptions()
        options.log_severity_level = 3
        _SESSION = onnxruntime.InferenceSession(
            MODEL_PATH, options, providers=["CPUExecutionProvider"])
    return _SESSION


def load_audio(path: str) -> np.ndarray:
    """Mono float32 at the model's sample rate."""
    import librosa

    samples, _ = librosa.load(path, sr=AUDIO_SAMPLE_RATE, mono=True)
    return samples.astype(np.float32)


def windows(samples: np.ndarray):
    """Overlapping windows, padded at the front as basic-pitch does.

    The half-overlap of silence at the start matters: without it the first
    window has no history and its opening frames -- which are then trimmed as
    overlap -- would be taken from real audio instead.
    """
    overlap = N_OVERLAPPING_FRAMES * FFT_HOP
    hop = AUDIO_N_SAMPLES - overlap
    padded = np.concatenate([np.zeros(overlap // 2, dtype=np.float32), samples])
    for start in range(0, max(1, len(padded)), hop):
        chunk = padded[start:start + AUDIO_N_SAMPLES]
        if len(chunk) < AUDIO_N_SAMPLES:
            chunk = np.pad(chunk, (0, AUDIO_N_SAMPLES - len(chunk)))
        yield chunk
        if start + AUDIO_N_SAMPLES >= len(padded):
            break


def posteriorgrams(samples: np.ndarray) -> dict:
    """Run the model over a whole signal; returns onset/note/contour maps.

    Windows overlap, so half the overlap is trimmed from each end of every
    window's output before they are concatenated -- the frames a window
    predicts at its edges are the ones it has least context for.
    """
    run = session()
    name = run.get_inputs()[0].name
    outputs = run.get_outputs()
    # The graph names its outputs opaquely; identify them by width instead.
    parts: dict[str, list] = {"onset": [], "note": [], "contour": []}
    for chunk in windows(samples):
        result = run.run(None, {name: chunk.reshape(1, -1, 1)})
        wide = [r for r in result if r.shape[-1] > 100]
        narrow = [r for r in result if r.shape[-1] <= 100]
        parts["contour"].append(wide[0][0])
        # Of the two 88-wide maps, the onset map is the sparser one: onsets fire
        # at note starts, while the note map stays high for a note's whole length.
        first, second = narrow[0][0], narrow[1][0]
        onset, note = ((first, second) if first.mean() < second.mean()
                       else (second, first))
        parts["onset"].append(onset)
        parts["note"].append(note)

    half = N_OVERLAPPING_FRAMES // 2
    frames = int(np.floor(len(samples) * (ANNOTATIONS_FPS / AUDIO_SAMPLE_RATE)))
    return {
        key: np.concatenate([w[half:-half] for w in value])[:frames]
        for key, value in parts.items()
    }


def frames_for_tempo(bpm: float, grid_of_a_quarter: float = 0.25) -> int:
    """The shortest note worth keeping, in frames, at a given tempo.

    A fixed floor is the wrong shape. The default of 11 frames is 128ms, while
    a sixteenth note is 125ms at 120 BPM, 105ms at 143 and 81ms at 185 -- so at
    every tempo in the corpus this project was built against, every sixteenth
    note was discarded before anything downstream could see it.

    Measured on a real recording at the default: recall in the 100-150ms band
    was 0.667, and 0.963 with the floor at 35ms, while notes over 300ms barely
    moved (0.862 to 0.884). That is the signature of a filter, not of a
    detector that cannot hear.

    Half the grid unit: short enough to admit the fastest note the grid can
    represent, long enough to reject a blip. Clamped, because a very fast tempo
    would otherwise drive the floor into the noise and a very slow one would
    reintroduce the problem this exists to fix.
    """
    if bpm <= 0:
        return MIN_NOTE_FRAMES
    seconds = (60.0 / bpm) * grid_of_a_quarter / 2.0
    return int(min(LONGEST_FRAMES, max(SHORTEST_FRAMES, round(seconds * ANNOTATIONS_FPS))))


def infer_onsets(onsets: np.ndarray, frames: np.ndarray, n_diff: int = 2) -> np.ndarray:
    """Add onsets implied by a sharp rise in frame energy.

    The onset head misses notes that swell rather than strike. Taking the
    maximum of the two never removes an onset the network was sure about.
    """
    diffs = []
    for n in range(1, n_diff + 1):
        padded = np.concatenate([np.zeros((n, frames.shape[1])), frames])
        diffs.append(padded[n:, :] - padded[:-n, :])
    difference = np.min(diffs, axis=0)
    difference[difference < 0] = 0
    difference[:n_diff, :] = 0
    peak = difference.max()
    if peak > 0:
        difference = onsets.max() * difference / peak
    return np.max([onsets, difference], axis=0)


def frame_times(n_frames: int) -> np.ndarray:
    """Frame index to seconds, undoing the per-window offset.

    Not simply frame/rate: the windows overlap, and the correction includes a
    constant the original calls magic. Dropping it drifts the whole
    transcription against the audio.
    """
    index = np.arange(n_frames)
    offset = (FFT_HOP / AUDIO_SAMPLE_RATE) * (
        ANNOT_N_FRAMES - (AUDIO_N_SAMPLES / FFT_HOP)) + 0.0018
    return index * FFT_HOP / AUDIO_SAMPLE_RATE - offset * np.floor(index / ANNOT_N_FRAMES)


def sustain_end(energy: np.ndarray, start: int, key: int, frame_threshold: float,
                energy_tolerance: int = 11) -> int:
    """The frame a note starting at `start` on `key` stops sounding, exclusive.

    Walk forward while the note map holds at or above `frame_threshold`, allowing
    up to `energy_tolerance` consecutive frames below it before giving up, then
    step back over those. Extracted from `decode` so a decoder that picks its own
    onsets ends its notes by the same rule rather than inventing a second one.

    `energy` is whatever map the caller is willing to consume: `decode` passes a
    working copy it clears as it goes, so a note cannot be ended by energy another
    note already claimed, while a caller that picks non-overlapping onsets can pass
    the note map itself.
    """
    n_frames = energy.shape[0]
    i, k = start + 1, 0
    while i < n_frames - 1 and k < energy_tolerance:
        k = k + 1 if energy[i, key] < frame_threshold else 0
        i += 1
    return i - k


def decode(note: np.ndarray, onset: np.ndarray, onset_threshold: float = 0.5,
           frame_threshold: float = 0.3, min_note_frames: int = MIN_NOTE_FRAMES,
           infer: bool = True, melodia: bool = True,
           energy_tolerance: int = 11) -> list[tuple[int, int, int, float, float, bool, float, float]]:
    """Probability maps to (start frame, end frame, midi, amplitude, onset
    confidence, from melodia, note map below, note map above).

    This is where most of the transcription quality lives. The network only
    says that something is sounding; where a note begins, how long it lasts and
    whether it exists at all is decided entirely here, which is why the
    thresholds are worth exposing rather than fixing.

    The last four say how each note was admitted. The onset-driven loop starts
    a note ON an onset peak, so its confidence is that peak's value -- read from
    the map actually peak-picked, the inferred one when `infer` is set. The
    melodia pass starts a note where sustained energy faded in and has no onset
    evidence, so its confidence is NaN. Measured on ASAP the two populations err
    in opposite directions, +4.8 ms against -35 ms, which is why a corrector
    needs to know which it is looking at.
    """
    import scipy.signal

    if infer:
        onset = infer_onsets(onset, note)
    n_frames = note.shape[0]

    peaks = np.zeros_like(onset)
    local = scipy.signal.argrelmax(onset, axis=0)
    peaks[local] = onset[local]
    times, freqs = np.where(peaks >= onset_threshold)
    times, freqs = times[::-1], freqs[::-1]          # walk backwards in time

    energy = note.copy()
    events: list[tuple[int, int, int, float, float, bool, float, float]] = []

    def clear(lo: int, hi: int, f: int) -> None:
        energy[lo:hi, f] = 0
        if f < MAX_FREQ_IDX:
            energy[lo:hi, f + 1] = 0
        if f > 0:
            energy[lo:hi, f - 1] = 0

    def neighbours(frame: int, f: int) -> tuple[float, float]:
        """The note map a semitone either side, NaN off the edge of the range."""
        below = float(note[frame, f - 1]) if f > 0 else float("nan")
        above = float(note[frame, f + 1]) if f < MAX_FREQ_IDX else float("nan")
        return below, above

    for start, f in zip(times, freqs):
        if start >= n_frames - 1:
            continue
        i = sustain_end(energy, start, f, frame_threshold, energy_tolerance)
        if i - start <= min_note_frames:
            continue
        clear(start, i, f)
        events.append((int(start), int(i), int(f + MIDI_OFFSET), float(note[start:i, f].mean()),
                       float(onset[start, f]), False, *neighbours(start, f)))

    if melodia:
        while energy.max() > frame_threshold:
            mid, f = np.unravel_index(np.argmax(energy), energy.shape)
            energy[mid, f] = 0

            i, k = mid + 1, 0
            while i < n_frames - 1 and k < energy_tolerance:
                k = k + 1 if energy[i, f] < frame_threshold else 0
                clear(i, i + 1, f)
                i += 1
            end = i - 1 - k

            i, k = mid - 1, 0
            while i > 0 and k < energy_tolerance:
                k = k + 1 if energy[i, f] < frame_threshold else 0
                clear(i, i + 1, f)
                i -= 1
            start = i + 1 + k

            if end - start <= min_note_frames:
                continue
            events.append((int(start), int(end), int(f + MIDI_OFFSET),
                           float(note[start:end, f].mean()),
                           float("nan"), True, *neighbours(start, f)))

    # Sorted on the first four fields only: a melodia note's confidence is NaN,
    # and NaN has no order.
    events.sort(key=lambda event: event[:4])
    return events


def detect(path_or_samples, **thresholds) -> list[Detection]:
    """Audio (path or samples) to detected notes."""
    samples = (load_audio(path_or_samples) if isinstance(path_or_samples, str)
               else np.asarray(path_or_samples, dtype=np.float32))
    maps = posteriorgrams(samples)
    return decode_to_notes(maps, **thresholds)


def decode_to_notes(maps: dict, **thresholds) -> list[Detection]:
    """Decode cached probability maps. Separate from `detect` because the maps
    are the expensive part and the thresholds are the part worth changing."""
    events = decode(maps["note"], maps["onset"], **thresholds)
    times = frame_times(maps["note"].shape[0])
    last = len(times) - 1
    return [
        Detection(float(times[min(s, last)]), float(times[min(e, last)]), p, a,
                  confidence, found_by_melodia, below, above)
        for s, e, p, a, confidence, found_by_melodia, below, above in events
    ]
