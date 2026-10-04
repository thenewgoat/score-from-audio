"""Transkun, split where ONNX can follow it.

The graph starts at the features the backbone reads. Everything before that --
framing, gain normalisation, six windows, the FFT, the mel projection and the
log -- runs in Kotlin, from constants written here: `torch.fft.rfft` produces
complex tensors ONNX handles poorly, and a matrix-multiply DFT would put 67 MB of
cos/sin tables in the graph. The learned Gaussian windows are rebuilt from their
parameters on every torch call; here they are evaluated once and frozen.

Run in transkun-env:  python -m export.transkun_export
"""

import json
import math
import os
import struct

import numpy as np
import torch
from torch.onnx import symbolic_helper
from torch.onnx import symbolic_opset9 as opset9

from export import paths

SEGMENT_FRAMES = 691
TRACKS = [-64, -67] + list(range(21, 109))

# MatMuls the int8 core leaves in fp32: the scorer's projection, and the learned
# position embeddings the transformer encoder adds to its time-frequency grid
# (posEmbedBuilderAttnTF) and uses as its 90 output tracks' queries (...AttnTE).
# Quantising them costs accuracy (see write_int8). The attention products
# MatMul_3/_4 multiply two activations, so quantize_dynamic leaves them fp32 anyway,
# and Conv stays fp32 because onnxruntime has no ConvInteger kernel on Android.
INT8_FP32_NODES = ["/scorer/map/map.0/MatMul"] + [
    f"/backbone/posEmbedBuilderAttn{axis}/{layer}/MatMul"
    for axis in ("TF", "TE") for layer in ("proj", "mlp/mlp.0", "mlp/mlp.3")]


@symbolic_helper.parse_args("v", "i", "i", "i")
def _diag_embed_symbolic(g, self, offset, dim1, dim2):
    """`aten::diag_embed` has no ONNX opset-17 lowering (torch.onnx.errors.UnsupportedOperatorError);
    ScaledInnerProductIntervalScorer calls it with the defaults below to build its diagonal term.
    Same trick as opset13's `diagonal` symbolic, run in reverse: multiply by an EyeLike mask."""
    assert offset == 0 and dim1 == -2 and dim2 == -1, (offset, dim1, dim2)
    shape = g.op("Shape", self)                                      # 1-D, e.g. [1, 90, 691]
    n = g.op("Slice", shape, g.op("Constant", value_t=torch.LongTensor([-1])),
             g.op("Constant", value_t=torch.LongTensor([9223372036854775807])),
             g.op("Constant", value_t=torch.LongTensor([0])))         # 1-D, [691]: last dim, kept
    mask_shape = g.op("Concat", n, n, axis_i=0)
    mask = opset9.zeros(g, mask_shape, None, None, None)
    eye = g.op("EyeLike", mask)
    self_unsq = symbolic_helper._unsqueeze_helper(g, self, [-1])
    return g.op("Mul", self_unsq, eye)


torch.onnx.register_custom_op_symbolic("aten::diag_embed", _diag_embed_symbolic, 17)


@symbolic_helper.parse_args("v")
def _sigmoid_symbolic(g, self):
    """onnxruntime's native Sigmoid op rounds sloppily in the last 1-2 ULPs when it
    saturates near 0 or 1 (verified: for a logit around 15.8, it lands on 1 - 2*eps
    where torch's own kernel correctly rounds to the nearer 1 - eps). Heads.forward's
    velocity/offset mean is a ratio of logs of `cut` and `1 - cut`, so that ULP
    disagreement blows up into a visible mismatch between the exported graph and
    eager torch. The textbook `1 / (1 + exp(-x))`, spelled out, rounds the same way
    torch's kernel does."""
    exp_neg = g.op("Exp", g.op("Neg", self))
    one = g.op("Constant", value_t=torch.tensor(1.0))
    return g.op("Div", one, g.op("Add", one, exp_neg))


torch.onnx.register_custom_op_symbolic("aten::sigmoid", _sigmoid_symbolic, 17)


def load_model():
    """Transkun 2.0 exactly as its own `transcribe` loads it, on the CPU."""
    import moduleconf
    from importlib.resources import files

    conf_path = str(files("transkun") / "pretrained" / "2.0.conf")
    weight_path = str(files("transkun") / "pretrained" / "2.0.pt")
    manager = moduleconf.parseFromFile(conf_path)
    model = manager["Model"].module.TransKun(conf=manager["Model"].config)
    checkpoint = torch.load(weight_path, map_location="cpu")
    state = checkpoint.get("best_state_dict", checkpoint.get("state_dict"))
    model.load_state_dict(state, strict=False)
    return model.eval()


def make_frames(x: np.ndarray, hop: int = 1024, win: int = 4096) -> np.ndarray:
    """[C, N] samples -> [C, T, win] frames, padded as Transkun's makeFrame pads."""
    n = x.shape[-1]
    n_frames = math.ceil(n / hop) + 1
    left = win // 2
    right = (n_frames - 1) * hop + win // 2 - n
    padded = np.pad(x, ((0, 0), (left, right)))
    index = np.arange(win)[None, :] + hop * np.arange(n_frames)[:, None]
    return padded[:, index]


def normalise(frames: np.ndarray) -> np.ndarray:
    """Gain normalisation over every sample of every frame, overlaps counted each time."""
    mean = frames.mean(dtype=np.float64)
    std = frames.std(dtype=np.float64, ddof=1)
    return ((frames - mean) / (std + 1e-8)).astype(np.float32)


def frontend_constants(model):
    spectrum = model.framewiseFeatureExtractor.spectrogramExtractor
    with torch.no_grad():
        windows = torch.cat([spectrum.win.unsqueeze(0), spectrum.winGen.get().t()], dim=0)
        mel = model.framewiseFeatureExtractor.freq2mels
    return windows.numpy().astype(np.float32), mel.numpy().astype(np.float32)


def reference_features(frames: np.ndarray, windows: np.ndarray, mel: np.ndarray) -> np.ndarray:
    """What the Kotlin frontend must reproduce: [C, T, 4096] normalised frames -> [T, 229, 6]."""
    spectrum = np.fft.rfft(frames[:, :, None, :].astype(np.float64) * windows[None, None], norm="ortho")
    power = (np.abs(spectrum) ** 2).mean(axis=0)                  # [T, 6, 2049], channel mean
    mels = power @ mel.astype(np.float64)                          # [T, 6, 229]
    eps = 1e-5
    mels = (np.log(mels + eps) - math.log(eps)) / (-math.log(eps))
    return mels.transpose(0, 2, 1).astype(np.float32)


def write_frontend(path: str, model=None) -> None:
    windows, mel = frontend_constants(model or load_model())
    with open(path, "wb") as f:
        f.write(b"TKF1")
        f.write(struct.pack("<4i", windows.shape[0], windows.shape[1], mel.shape[0], mel.shape[1]))
        f.write(windows.astype("<f4").tobytes())
        f.write(mel.astype("<f4").tobytes())


class Core(torch.nn.Module):
    """Backbone and interval scorer: features [1, T, 229, 6] -> score [T, T, 90], ctx [90, T, 256].

    The scorer's second output, the non-event score, is identically zero
    (`b = diag * 0.0` in ScaledInnerProductIntervalScorer), so it is not exported
    and the Kotlin Viterbi uses zeros.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        # Gradient checkpointing re-enters autograd in a way `torch.onnx.export`'s
        # tracer cannot follow (`RuntimeError: _Map_base::at`). Backbone exposes this
        # flag for exactly this case; `self.training` is already False under `.eval()`.
        self.model.backbone.useGradientCheckpoint = False
        self.register_buffer("tracks", torch.tensor(TRACKS))

    def forward(self, features):
        ctx = self.model.backbone(features, outputIndices=self.tracks)   # [1, 90, T, 256]
        score, _ = self.model.scorer(ctx)                                 # [T, T, 1, 90]
        return score[:, :, 0, :], ctx[0]


class Heads(torch.nn.Module):
    """Velocity and onset/offset refinement for each decoded interval, as transcribeFrames does it."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, attr):
        velocity = self.model.velocityPredictor(attr).argmax(-1)
        value, presence = self.model.refinedOFPredictor(attr).chunk(2, dim=-1)
        # ContinuousBernoulli(logits=value).mean, written out so it exports.
        # `probs` is `clamp_probs(sigmoid(logits))` there (torch/distributions/utils.py),
        # keeping it inside (eps, 1 - eps); skipping that clamp lets sigmoid saturate to
        # exactly 0.0 or 1.0 for large |value|, where log1p(-cut)/log(cut) hit -inf.
        p = torch.sigmoid(value)
        eps = torch.finfo(p.dtype).eps
        p = torch.clamp(p, min=eps, max=1.0 - eps)
        unstable = (p > 0.499) & (p < 0.501)
        cut = torch.where(unstable, torch.full_like(p, 0.499), p)
        mean = cut / (2.0 * cut - 1.0) + 1.0 / (torch.log1p(-cut) - torch.log(cut))
        x = p - 0.5
        taylor = 0.5 + (1.0 / 3.0 + 16.0 / 45.0 * x * x) * x
        mean = torch.where(unstable, taylor, mean)
        of_value = torch.clamp((mean - 0.5) / 0.99, -0.5, 0.5)
        return velocity, of_value, presence > 0


def export_graphs(model, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(Core(model).eval(), (torch.zeros(1, SEGMENT_FRAMES, 229, 6),),
                          os.path.join(out_dir, "transkun_core.onnx"),
                          input_names=["features"], output_names=["score", "ctx"],
                          opset_version=17, dynamo=False)
        torch.onnx.export(Heads(model).eval(), (torch.zeros(4, 768),),
                          os.path.join(out_dir, "transkun_heads.onnx"),
                          input_names=["attr"], output_names=["velocity", "of_value", "of_presence"],
                          dynamic_axes={"attr": {0: "n"}, "velocity": {0: "n"},
                                        "of_value": {0: "n"}, "of_presence": {0: "n"}},
                          opset_version=17, dynamo=False)
    meta = {"fs": 44100, "hop": 1024, "window": 4096, "frames": SEGMENT_FRAMES,
            "segment_seconds": 16, "step_seconds": 8, "tracks": TRACKS}
    with open(os.path.join(out_dir, "transkun_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)


def write_int8(core_path: str, out_path: str) -> None:
    """Dynamic int8 core: MatMul/Gemm weights quantised per output channel, activations
    quantised on the fly, INT8_FP32_NODES kept fp32.

    Measured (12 ASAP recordings x 60 s, bake-off scoring): onset F 0.989 / offset F
    0.909 against fp32's 0.991 / 0.923; one core run on the phone at 4 threads takes
    2.82 s against 5.0 s. Ablations on the same set (onset / offset F): per-tensor
    weights 0.809 / 0.623; per-channel with every MatMul quantised 0.951 / 0.797;
    scorer kept fp32 0.988 / 0.901; scorer and attention position embeddings kept
    fp32 (this recipe) 0.989 / 0.909.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(core_path, out_path, per_channel=True, weight_type=QuantType.QInt8,
                     op_types_to_quantize=["MatMul", "Gemm"], nodes_to_exclude=INT8_FP32_NODES)


def main() -> int:
    os.makedirs(paths.OUT, exist_ok=True)
    model = load_model()
    write_frontend(os.path.join(paths.OUT, "transkun_frontend.bin"), model)
    export_graphs(model, paths.OUT)
    write_int8(os.path.join(paths.OUT, "transkun_core.onnx"),
               os.path.join(paths.OUT, "transkun_core_int8.onnx"))
    print("wrote", paths.OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
