"""Our notation model (`notation.model.NotationModel`, e.g. bd1-synth) as two graphs: an encoder, and one decoder step.

`notation.predict.decode_fixed` re-runs the whole decoder over the whole prefix for every slot it writes, which is
fine on a GPU and far too slow on a phone. With a KV cache one decoder step is a fixed sequence of the model's own
layers, so the wrappers below walk `nn.Transformer`'s modules and call their weights directly: pre-norm self
attention over the cache, pre-norm cross attention over the encoder's K/V (computed once per chunk), pre-norm ReLU
feed-forward, the decoder's final norm, and the nine heads. The decode loop moves to Kotlin (and to
`export.bd1_ref`, which checks these graphs against `notation.predict` and writes the fixtures Kotlin is tested on).

The weights are local only, like every other model under android/models/ (gitignored).

Run in the repo's .venv:  python -m export.bd1_export [--checkpoint PATH]
"""

import argparse
import os

import torch
import torch.nn.functional as F

from export import paths

IN_STREAMS = ["pitch", "onset", "duration", "velocity"]
OUT_STREAMS = ["pitch", "onset", "duration", "bar", "hand", "voice", "accidental", "metre", "key"]

CHECKPOINT = os.environ.get("BD1_CHECKPOINT", os.path.join(paths.DATA, "runs", "notation", "bd1-synth", "best.pt"))


def load_model(checkpoint: str = CHECKPOINT):
    from notation.predict import load

    model = load(checkpoint)
    if not model.config.key_stream or model.config.targets != 2 or model.config.input_encoding != 2:
        raise ValueError("only aligned, gap-encoded checkpoints with a key stream are exported")
    return model


def _heads(x, heads):
    b, t, d = x.shape
    return x.view(b, t, heads, d // heads).transpose(1, 2)


def _merge(x):
    b, h, t, d = x.shape
    return x.transpose(1, 2).reshape(b, t, h * d)


def _project(attention, x, which):
    """Rows `which` (0 q, 1 k, 2 v) of a MultiheadAttention's packed input projection."""
    d = attention.embed_dim
    return F.linear(x, attention.in_proj_weight[which * d:(which + 1) * d],
                    attention.in_proj_bias[which * d:(which + 1) * d])


class Encoder(torch.nn.Module):
    """Input indices [1, S] x 4 -> each decoder layer's cross-attention K and V, [1, heads, S, d/heads]."""

    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, pitch, onset, duration, velocity):
        m = self.m
        n = pitch.shape[1]
        x = sum(m.source_embeddings[name](value) for name, value in
                zip(IN_STREAMS, (pitch, onset, duration, velocity))) + m.positions[:n]
        heads = m.config.heads
        # Walked by hand: in eval, nn.TransformerEncoderLayer takes a fused fast path ONNX cannot export.
        for layer in m.transformer.encoder.layers:
            att = layer.self_attn
            y = layer.norm1(x)
            q, k, v = (_heads(_project(att, y, j), heads) for j in range(3))
            x = x + att.out_proj(_merge(F.scaled_dot_product_attention(q, k, v)))
            x = x + layer.linear2(F.relu(layer.linear1(layer.norm2(x))))
        memory = m.transformer.encoder.norm(x)
        out = []
        for layer in m.transformer.decoder.layers:
            att = layer.multihead_attn
            out += [_heads(_project(att, memory, 1), heads), _heads(_project(att, memory, 2), heads)]
        return tuple(out)


class DecoderStep(torch.nn.Module):
    """One cached decoder step: the previous slot's nine stream indices at position `pos` -> nine logit vectors."""

    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, tokens, pos, *caches):
        m = self.m
        layers = len(m.transformer.decoder.layers)
        heads = m.config.heads
        past = caches[:2 * layers]
        cross = caches[2 * layers:]
        h = sum(m.target_embeddings[name](tokens[i:i + 1]) for i, name in enumerate(OUT_STREAMS))
        h = (h + torch.index_select(m.positions, 0, pos)).view(1, 1, -1)
        present = []
        for i, layer in enumerate(m.transformer.decoder.layers):
            att = layer.self_attn
            y = layer.norm1(h)
            q = _heads(_project(att, y, 0), heads)
            k = torch.cat([past[2 * i], _heads(_project(att, y, 1), heads)], dim=2)
            v = torch.cat([past[2 * i + 1], _heads(_project(att, y, 2), heads)], dim=2)
            h = h + att.out_proj(_merge(F.scaled_dot_product_attention(q, k, v)))
            present += [k, v]
            att = layer.multihead_attn
            q = _heads(_project(att, layer.norm2(h), 0), heads)
            h = h + att.out_proj(_merge(F.scaled_dot_product_attention(q, cross[2 * i], cross[2 * i + 1])))
            h = h + layer.linear2(F.relu(layer.linear1(layer.norm3(h))))
        h = m.transformer.decoder.norm(h)
        return tuple(m.heads[name](h)[0, 0] for name in OUT_STREAMS) + tuple(present)


def names(layers: int):
    past = [f"past_{kv}{i}" for i in range(layers) for kv in "kv"]
    cross = [f"cross_{kv}{i}" for i in range(layers) for kv in "kv"]
    present = [f"present_{kv}{i}" for i in range(layers) for kv in "kv"]
    return past, cross, present


def export_graphs(model, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    layers = len(model.transformer.decoder.layers)
    heads = model.config.heads
    width = model.config.d_model // heads
    past_names, cross_names, present_names = names(layers)
    n = 16
    example = tuple(torch.zeros(1, n, dtype=torch.long) for _ in IN_STREAMS)
    with torch.no_grad():
        torch.onnx.export(Encoder(model).eval(), example, os.path.join(out_dir, "bd1_encoder.onnx"),
                          input_names=IN_STREAMS, output_names=cross_names,
                          dynamic_axes={**{k: {1: "S"} for k in IN_STREAMS}, **{c: {2: "S"} for c in cross_names}},
                          opset_version=17, dynamo=False)
        cross = Encoder(model)(*example)
        past = tuple(torch.zeros(1, heads, 3, width) for _ in range(2 * layers))
        torch.onnx.export(
            DecoderStep(model).eval(),
            (torch.zeros(len(OUT_STREAMS), dtype=torch.long), torch.tensor([3])) + past + tuple(cross),
            os.path.join(out_dir, "bd1_step.onnx"),
            input_names=["tokens", "pos"] + past_names + cross_names,
            output_names=[f"logits_{k}" for k in OUT_STREAMS] + present_names,
            dynamic_axes={**{p: {2: "P"} for p in past_names}, **{c: {2: "S"} for c in cross_names},
                          **{p: {2: "P1"} for p in present_names}},
            opset_version=17, dynamo=False)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m export.bd1_export")
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    args = parser.parse_args(argv)
    export_graphs(load_model(args.checkpoint), paths.OUT)
    print("wrote bd1_encoder.onnx and bd1_step.onnx to", paths.OUT, "-- gitignored; never commit or publish them")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
