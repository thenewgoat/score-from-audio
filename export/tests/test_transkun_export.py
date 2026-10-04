import numpy as np
import torch

from export.transkun_export import (frontend_constants, load_model, make_frames, normalise,
                                    reference_features)


def test_frames_match_transkun():
    from transkun.Util import makeFrame
    x = np.random.default_rng(0).standard_normal((2, 50_000)).astype(np.float32)
    theirs = makeFrame(torch.from_numpy(x), 1024, 4096).numpy()
    np.testing.assert_array_equal(make_frames(x), theirs)


def test_a_segment_has_691_frames():
    assert make_frames(np.zeros((1, 705_600), np.float32)).shape[1] == 691


def test_numpy_features_match_torch():
    model = load_model()
    x = np.random.default_rng(1).standard_normal((2, 44_100)).astype(np.float32) * 0.1
    frames = normalise(make_frames(x))
    windows, mel = frontend_constants(model)
    ours = reference_features(frames, windows, mel)
    with torch.no_grad():
        theirs = model.framewiseFeatureExtractor(torch.from_numpy(frames)[None]).numpy()[0, 0]
    assert ours.shape == theirs.shape == (frames.shape[1], 229, 6)
    np.testing.assert_allclose(ours, theirs, atol=2e-4)


def test_normalise_matches_transkun():
    frames = make_frames(np.random.default_rng(2).standard_normal((1, 30_000)).astype(np.float32))
    t = torch.from_numpy(frames)[None]
    theirs = ((t - t.mean(dim=[1, 2, 3], keepdim=True)) / (t.std(dim=[1, 2, 3], keepdim=True) + 1e-8))[0]
    np.testing.assert_allclose(normalise(frames), theirs.numpy(), atol=1e-5)


import os

import onnxruntime as ort
import pytest

from export import paths
from export.transkun_export import (Core, Heads, INT8_FP32_NODES, SEGMENT_FRAMES, export_graphs,
                                    write_int8)


def _segment_features(model, seconds_offset=30.0):
    """Features of one real 16 s segment, or of seeded noise when ASAP is absent."""
    import soundfile
    clip = os.path.join(paths.ASAP_AUDIO, paths.PARITY_CLIPS[0])
    if os.path.exists(clip):
        audio, fs = soundfile.read(clip, dtype="float32", always_2d=True,
                                   start=int(seconds_offset * 44100), frames=705_600)
        x = audio.T
    else:
        x = np.random.default_rng(3).standard_normal((2, 705_600)).astype(np.float32) * 0.05
    windows, mel = frontend_constants(model)
    return reference_features(normalise(make_frames(x)), windows, mel)[None]


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    out = tmp_path_factory.mktemp("models")
    model = load_model()
    export_graphs(model, str(out))
    return model, out


def test_noise_score_is_zero_so_kotlin_may_omit_it():
    model = load_model()
    ctx = torch.randn(1, 90, 50, 256)
    _, skip = model.scorer(ctx)
    assert torch.count_nonzero(skip) == 0


def test_core_onnx_decodes_the_same_intervals_as_torch(exported):
    from transkun.CRF.NeuralSemiCRFInterval import viterbiBackward
    model, out = exported
    features = _segment_features(model)
    with torch.no_grad():
        score_t, ctx_t = Core(model)(torch.from_numpy(features))
    session = ort.InferenceSession(str(out / "transkun_core.onnx"))
    score_o, ctx_o = session.run(["score", "ctx"], {"features": features})
    assert score_o.shape == (SEGMENT_FRAMES, SEGMENT_FRAMES, 90)
    np.testing.assert_allclose(ctx_o, ctx_t.numpy(), rtol=1e-3, atol=1e-3)
    zeros = torch.zeros(SEGMENT_FRAMES - 1, 90)
    start = [344] * 90
    assert viterbiBackward(torch.from_numpy(score_o), zeros, start) == viterbiBackward(score_t, zeros, start)


def _intervals(path):
    return {(track, tuple(iv)) for track, ivs in enumerate(path) for iv in ivs}


def test_int8_core_keeps_the_scorer_and_attention_position_embeddings_fp32(exported):
    import onnx
    _, out = exported
    write_int8(str(out / "transkun_core.onnx"), str(out / "transkun_core_int8.onnx"))
    graph = onnx.load(str(out / "transkun_core_int8.onnx")).graph
    ops = {node.name: node.op_type for node in graph.node}
    for name in INT8_FP32_NODES:
        assert ops[name] == "MatMul", name
    assert sum(op == "MatMulInteger" for op in ops.values()) == 75
    assert "ConvInteger" not in ops.values()


def test_int8_core_decodes_nearly_the_same_intervals_as_fp32(exported):
    """Measured on this segment (PARITY_CLIPS[0] at 30 s): 52 of fp32's 55 intervals come
    back identical, the other 3 with the same start within 2 frames, and int8 adds one
    sustain-pedal interval -- start-matched F 0.991. Over 12 recordings x 60 s, note onset F
    is 0.989 against fp32's 0.991. The bounds leave room for one or two more such moves."""
    from transkun.CRF.NeuralSemiCRFInterval import viterbiBackward
    model, out = exported
    write_int8(str(out / "transkun_core.onnx"), str(out / "transkun_core_int8.onnx"))
    features = _segment_features(model)
    zeros = torch.zeros(SEGMENT_FRAMES - 1, 90)
    start = [344] * 90
    paths_ = {}
    for name in ("transkun_core.onnx", "transkun_core_int8.onnx"):
        score, ctx = ort.InferenceSession(str(out / name)).run(["score", "ctx"], {"features": features})
        assert score.shape == (SEGMENT_FRAMES, SEGMENT_FRAMES, 90)
        assert ctx.shape == (90, SEGMENT_FRAMES, 256)
        paths_[name] = _intervals(viterbiBackward(torch.from_numpy(score), zeros, start))
    fp32, int8 = paths_["transkun_core.onnx"], paths_["transkun_core_int8.onnx"]
    assert len(fp32 & int8) >= 0.9 * len(fp32)
    matched = sum(any(t == u and abs(iv[0] - jv[0]) <= 2 for u, jv in int8) for t, iv in fp32)
    f = 2 * matched / (len(fp32) + len(int8))
    assert f >= 0.97, f


def test_heads_onnx_match_torch(exported):
    model, out = exported
    attr = torch.randn(300, 768)
    with torch.no_grad():
        vel_t, of_t, pres_t = Heads(model)(attr)
    session = ort.InferenceSession(str(out / "transkun_heads.onnx"))
    vel_o, of_o, pres_o = session.run(None, {"attr": attr.numpy()})
    np.testing.assert_array_equal(vel_o, vel_t.numpy())
    # of_value is in frames (1 frame = 1024/44100 s ~= 23 ms), so 1e-2 frame ~= 232 us --
    # still well under the spec's 1 ms end-to-end note-time tolerance. The residual here
    # is cross-backend GEMM noise in refinedOFPredictor's logit, amplified by the
    # ContinuousBernoulli mean just outside its own (0.499, 0.501) stability cutoff; it's
    # heavy-tailed (150-trial max observed 3.6e-3), hence the margin over that tail.
    np.testing.assert_allclose(of_o, of_t.numpy(), atol=1e-2)
    np.testing.assert_array_equal(pres_o, pres_t.numpy())


def test_heads_reproduce_transcribe_frames_arithmetic():
    """Heads must equal the inline code in TransKun.transcribeFrames."""
    model = load_model()
    attr = torch.randn(40, 768)
    with torch.no_grad():
        vel, of_value, presence = Heads(model)(attr)
        logits_v = model.velocityPredictor(attr)
        value, pres = model.refinedOFPredictor(attr).chunk(2, dim=-1)
        dist = torch.distributions.ContinuousBernoulli(logits=value)
        expected = torch.clamp((dist.mean - 0.5) / 0.99, -0.5, 0.5)
    np.testing.assert_array_equal(vel.numpy(), logits_v.softmax(-1).argmax(-1).numpy())
    np.testing.assert_allclose(of_value.numpy(), expected.numpy(), atol=1e-6)
    np.testing.assert_array_equal(presence.numpy(), (pres > 0).numpy())
