"""Beat This! -- Foscarin, Schlüter and Widmer, ISMIR 2024.

A rotary transformer over mel spectrograms, alternating attention over
frequency and time, with no DBN postprocessing. The missing DBN is the point:
a DBN constrains metre and tempo, which is the same class of constraint that
made librosa's 120 BPM prior so damaging on this corpus.

MIT for both code and the published weights, which is why it is preferred over
madmom (CC BY-NC-SA).

It takes 22.05 kHz mono, which is what `evaluation/asap_audio.py` already
caches, and returns beat and downbeat times in seconds.

One caution that is handled in `evaluation/folds.py`, not here: ASAP is one of
the 18 datasets Beat This! trains on, so the shipped `final0/1/2` checkpoints
have seen most of our corpus. Which checkpoint is honest for a given piece is
that module's job.
"""

from transcribe.beats import Beats, register


def _load(checkpoint: str, device: str | None):
    """Build the underlying File2Beats. Separated so tests can stub it."""
    from beat_this.inference import File2Beats

    if device is None:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    return File2Beats(checkpoint_path=checkpoint, device=device, dbn=False)


class BeatThisTracker:
    """One checkpoint of Beat This!, loaded on first use and then reused.

    The instance holds the loaded model because the fold policy runs eight
    checkpoints over 466 pieces; loading per call would mean 466 loads of a
    78 MB checkpoint rather than eight.
    """

    has_downbeats = True

    def __init__(self, checkpoint: str = "final0", device: str | None = None):
        self.checkpoint = checkpoint
        self.device = device
        self._model = None

    @property
    def name(self) -> str:
        # The checkpoint is part of the identity: fold3 and fold4 are different
        # models, and a set of pairs has to record which one made its grid.
        return f"beat_this:{self.checkpoint}"

    def track(self, source: str) -> Beats:
        if self._model is None:
            self._model = _load(self.checkpoint, self.device)
        beats, downbeats = self._model(source)
        return Beats.derive(beats, downbeats, self.name)


register("beat_this", BeatThisTracker)
