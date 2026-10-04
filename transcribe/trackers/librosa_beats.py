"""The librosa baseline.

Retained deliberately, as the thing Beat This! is measured against. The
acceptance bar for replacing it is a comparison over all 495 ASAP pieces, and a
baseline that is not in the tree cannot be re-run.

Named `librosa_beats` rather than `librosa` so it cannot shadow the library it
imports.
"""

from transcribe.beats import Beats, register


class LibrosaTracker:
    """Ellis's dynamic-programming tracker, as librosa ships it.

    `start_bpm` is exposed because it is the whole problem: it is the centre of
    the log-normal prior the tempogram is read under, and leaving it at
    librosa's 120.0 is what produced the -0.91 correlation. Exposed so the
    baseline can be run at its default -- which is what the historical
    measurements used -- rather than quietly improved into something that is no
    longer the baseline.
    """

    name = "librosa"
    has_downbeats = False

    def __init__(self, start_bpm: float = 120.0):
        self.start_bpm = start_bpm

    def track(self, source: str) -> Beats:
        import librosa

        samples, rate = librosa.load(source, sr=22050, mono=True)
        # The tempo librosa returns alongside these beats is discarded: the
        # contract derives tempo from beat spacing, for every backend alike.
        _, beats = librosa.beat.beat_track(y=samples, sr=rate, units="time",
                                           start_bpm=self.start_bpm)
        return Beats.derive(beats, [], self.name)


register("librosa", LibrosaTracker)
