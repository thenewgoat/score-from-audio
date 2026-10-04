"""Fixtures the Kotlin pipeline is tested against, from Transkun's own code.

Run in transkun-env:  python -m export.transkun_fixtures
Committed output: small, and derived from MIT code and seeded noise only. The
note lists come from MAESTRO recordings (CC BY-NC-SA) but contain only note
events, not audio; the audio stays in ASAP_AUDIO and gated tests read it there.
"""

import json
import os
import shutil

import numpy as np
import torch

from export import paths
from export.transkun_export import (frontend_constants, load_model, make_frames, normalise,
                                    reference_features, write_frontend)


def synthetic_audio(seconds=2.0, fs=44100):
    rng = np.random.default_rng(7)
    t = np.arange(int(seconds * fs)) / fs
    left = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 659.25 * t)
    right = 0.25 * np.sin(2 * np.pi * 523.25 * t) + 0.05 * rng.standard_normal(t.size)
    return np.stack([left, right]).astype(np.float32)


def frontend_fixture(model, out):
    audio = synthetic_audio()
    frames = normalise(make_frames(audio))
    with torch.no_grad():
        features = model.framewiseFeatureExtractor(torch.from_numpy(frames)[None]).numpy()[0, 0]
    windows, mel = frontend_constants(model)
    assert np.allclose(features, reference_features(frames, windows, mel), atol=2e-4)
    np.save(os.path.join(out, "frontend_audio.npy"), audio)
    np.save(os.path.join(out, "frontend_features.npy"), features.astype(np.float32))
    write_frontend(os.path.join(out, "transkun_frontend.bin"), model)


def viterbi_fixture(out):
    from transkun.CRF.NeuralSemiCRFInterval import viterbiBackward
    g = torch.Generator().manual_seed(11)
    score = torch.randn(60, 60, 5, generator=g) - 0.8
    forced = [0, 3, 0, 10, 59]
    intervals = viterbiBackward(score, torch.zeros(59, 5), forced)
    np.save(os.path.join(out, "viterbi_score.npy"), score.numpy().astype(np.float32))
    with open(os.path.join(out, "viterbi_expected.json"), "w") as f:
        json.dump({"forced": forced, "intervals": [[list(i) for i in track] for track in intervals]}, f)


def notes_fixture(model, out):
    import soundfile
    clips = []
    for rel in paths.PARITY_CLIPS:
        audio, fs = soundfile.read(os.path.join(paths.ASAP_AUDIO, rel), dtype="int16", always_2d=True,
                                   frames=paths.PARITY_SECONDS * 44100)
        assert fs == 44100
        x = torch.from_numpy(audio.astype(np.float32) / 2 ** 15)          # as transcribe.readAudio
        with torch.no_grad():
            notes = model.transcribe(x)
        clips.append({"path": rel, "seconds": paths.PARITY_SECONDS,
                      "notes": [[n.start, n.end, int(n.pitch), int(n.velocity)] for n in notes]})
        print(rel, len(notes), "notes")
    with open(os.path.join(out, "transkun_notes.json"), "w") as f:
        json.dump({"clips": clips}, f)


def main() -> int:
    os.makedirs(paths.FIXTURES, exist_ok=True)
    model = load_model()
    frontend_fixture(model, paths.FIXTURES)
    viterbi_fixture(paths.FIXTURES)
    notes_fixture(model, paths.FIXTURES)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
