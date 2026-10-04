"""Beat tracking as a stage, with a pluggable backend.

For most of this project's life the beat tracker was
`librosa.beat.beat_track` called with every default, in four places. Its tempo
comes from a tempogram under a log-normal prior centred on 120 BPM, and
measured across the 495 ASAP pieces we generate training pairs from, that prior
dominates the music: the correlation between log(annotated tempo) and
log(tracked/annotated) is -0.91. Slow pieces were doubled or tripled, fast ones
halved, and the 45% it got right were the ones already near 120. It was not
mistracking the music; it was reporting its prior.

Two things follow, and both are structural rather than a change of library.

First, **tempo is derived here, never reported by a backend**. It is
`60 / median(diff(beats))` in `Beats.derive`, so a backend that happens to
offer its own prior-laden tempo cannot leak it into the pipeline.

Second, the tracker is **named and swappable**. Validating this change means
running two trackers over the same corpus, and the fold policy in
`evaluation/folds.py` runs Beat This! in eight configurations, one per
cross-validation fold. Both want a tracker to be a value you pass, not a
library a module imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

import numpy as np

DEFAULT_TRACKER = "beat_this"


@dataclass(frozen=True)
class Beats:
    """A tracked beat grid, and what can be read off it."""

    beats: np.ndarray          # seconds
    downbeats: np.ndarray      # seconds; a subset of `beats`, possibly empty
    tempo: float               # BPM, derived -- see the module docstring
    beats_per_bar: int | None  # None when the backend gives no downbeats
    tracker: str               # which backend and configuration produced this

    @classmethod
    def derive(cls, beats, downbeats, tracker: str) -> "Beats":
        beats = np.asarray(beats, dtype=float)
        downbeats = np.asarray(downbeats, dtype=float)
        return cls(
            beats=beats,
            downbeats=downbeats,
            tempo=_tempo(beats),
            beats_per_bar=_beats_per_bar(beats, downbeats),
            tracker=tracker,
        )

    @property
    def anchor(self) -> float:
        """The first downbeat, as a fractional beat number: the phase origin.

        Positions measured from here count from bar 1 beat 1. Without it, a
        grid's beat 0 is wherever the tracker happened to start, which differs
        from the annotations' beat 0 by an unknown per-piece constant -- and a
        per-piece constant is noise the correction layer cannot recover from
        its input. NaN when there are no downbeats, which is why a
        downbeat-less tracker cannot generate pairs.
        """
        if len(self.downbeats) == 0 or len(self.beats) < 2:
            return float("nan")
        return float(np.interp(self.downbeats[0], self.beats,
                               np.arange(len(self.beats))))


def _tempo(beats: np.ndarray) -> float:
    """BPM from beat spacing. The median, so one misplaced beat cannot move it."""
    if len(beats) < 2:
        return float("nan")
    spacing = float(np.median(np.diff(beats)))
    return 60.0 / spacing if spacing > 0 else float("nan")


def _beats_per_bar(beats: np.ndarray, downbeats: np.ndarray) -> int | None:
    """How many beats to a bar, from the median gap between downbeats.

    An aggregate on purpose. Beat This! scores 61.2 downbeat F1 on classical
    piano, so roughly two individual downbeats in five are wrong -- but
    deciding "this piece is in 3" only needs the spacing right on average
    across a whole piece. A missed downbeat at bar 37 costs nothing here,
    whereas drawing a bar line at each detected downbeat would cost a bar line.
    """
    if len(downbeats) < 2 or len(beats) < 2:
        return None
    indices = np.interp(downbeats, beats, np.arange(len(beats)))
    spacing = float(np.median(np.diff(indices)))
    if not np.isfinite(spacing) or spacing < 1:
        return None
    return int(round(spacing))


@runtime_checkable
class Tracker(Protocol):
    """What every backend provides."""

    name: str
    has_downbeats: bool

    def track(self, source: str) -> Beats: ...


_TRACKERS: dict[str, Callable[..., Tracker]] = {}


def register(name: str, factory: Callable[..., Tracker]) -> None:
    _TRACKERS[name] = factory


def get_tracker(name: str = DEFAULT_TRACKER, **options) -> Tracker:
    # Imported here rather than at module scope: the backends pull in torch and
    # librosa, and `Beats` itself must stay importable without either.
    import transcribe.trackers  # noqa: F401

    if name not in _TRACKERS:
        raise ValueError(f"unknown tracker {name!r}; have {sorted(_TRACKERS)}")
    return _TRACKERS[name](**options)


def track(source: str, tracker: Tracker | str | None = None) -> Beats:
    """Track `source`, with the default backend, a named one, or one given."""
    if tracker is None or isinstance(tracker, str):
        tracker = get_tracker(tracker or DEFAULT_TRACKER)
    return tracker.track(source)
