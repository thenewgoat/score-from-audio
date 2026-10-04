"""The notation model: performance tokens in, score tokens out.

One embedding per input stream, summed, so a note enters as a single vector
rather than four positions. One linear head per output stream, so each decision
is its own small classification. The backbone is an ordinary transformer
encoder-decoder -- the representation is where the work is, not the architecture.

Every vocabulary size except pitch is read from the live tables in
`notation.vocab` and `notation.tokens`, never written down here. Those tables
were widened twice during this stage (durations 96 -> 160 slots, onsets
145 -> 241, and a `metre` stream added), and a head sized from a copied number
is a head that silently stops matching its encoder. Pitch is sized from
`Config.pitch_vocab`, which `notation.predict.load` reads off a checkpoint's
own pitch embedding: 132 for run4 and recipe16, 134 (`PITCH_VOCAB`, with the
space and unplayed slots) for aligned checkpoints -- one live table cannot
describe both generations.
"""

import math
from dataclasses import dataclass

import torch
from torch import nn

from notation.tokens import (ACCIDENTALS, ALIGNED_TARGETS, HANDS, INPUT_ENCODING, INPUT_STREAMS,
                             KEY_VOCAB, OUTPUT_STREAMS, PITCH_PAD, PITCH_SPACE, PITCH_VOCAB,
                             SECONDS_BUCKETS, VOICES)
from notation.vocab import BAR_LENGTHS, DURATIONS, ONSETS, TIME_SIGNATURES, VELOCITY_BUCKETS

# The vocabulary a freshly built model uses. `pitch` is a ceiling, not what
# every model actually has: growing `PITCH_VOCAB` (two reserved slots, task 2)
# would resize every existing checkpoint's pitch tables out from under it, so
# `NotationModel` sizes its pitch embeddings and head from `Config.pitch_vocab`
# instead of this constant -- see `Config.pitch_vocab` below.
INPUT_VOCAB = {
    "pitch": PITCH_VOCAB,
    "onset": SECONDS_BUCKETS,
    "duration": SECONDS_BUCKETS,
    "velocity": VELOCITY_BUCKETS,
}

OUTPUT_VOCAB = {
    "pitch": PITCH_VOCAB,
    "onset": len(ONSETS),
    "duration": len(DURATIONS),
    "bar": len(BAR_LENGTHS),
    "hand": HANDS,
    "voice": VOICES,
    "accidental": len(ACCIDENTALS),
    # Added after the rest: a bar LENGTH cannot name a metre -- 6/8 and 3/4 are
    # both three quarters. It was added to explain the Barcarolle's 54.5 and
    # changed it not at all (54.513 before and after); the cause was accidental
    # spelling. Kept on its own merits. See `notation.vocab.metre_index`.
    "metre": len(TIME_SIGNATURES),
    # The key signature in force at each slot, "none" plus 7 flats .. 7
    # sharps (`notation.tokens.key_index`). Added last, and optional per model:
    # a checkpoint from before it has no key head (`Config.key_stream`).
    "key": KEY_VOCAB,
}

assert set(INPUT_VOCAB) == set(INPUT_STREAMS)
assert set(OUTPUT_VOCAB) == set(OUTPUT_STREAMS)


@dataclass
class Config:
    """The architecture, and the facts about a checkpoint's data that its shape
    or its decoding depends on.

    The default size is Beyer & Dai's: d_model 512, 8 heads, 4 + 4 layers,
    feedforward 1536. Measured on this corpus's windows: 26.3 M parameters,
    4.0 GB peak at batch 32 x 512 positions in bf16 autocast, 0.34 s a step on
    the RTX 4060 (8.6 GB). run4, the stage 4 model it replaces, was 128 wide
    and 2 + 2 layers (1.2 M) and records its own size, so it still loads.
    """
    d_model: int = 512
    heads: int = 8
    encoder_layers: int = 4
    decoder_layers: int = 4
    feedforward: int = 1536
    dropout: float = 0.1
    max_length: int = 512
    # How the input was encoded in training -- see `notation.tokens`. A
    # checkpoint that records none predates the field and is the legacy
    # encoding; `notation.predict.load` says so rather than this default.
    input_encoding: int = INPUT_ENCODING
    # The fraction of decoder-input positions replaced by a learned "hidden"
    # embedding in TRAINING. 0 builds no such embedding at all, so a model
    # that does not use it has exactly the legacy parameters and run4's state
    # dict still loads strictly. See `NotationModel.decoder_inputs`.
    decoder_dropout: float = 0.0
    # The size of the PITCH stream's embeddings and head, input and output
    # alike -- a played note's pitch and a written note's pitch share the same
    # 128 MIDI values plus the same reserved tokens. A module constant cannot
    # record this: `PITCH_VOCAB` has already grown once (task 2, 132 -> 134,
    # reserving `PITCH_SPACE`/`PITCH_UNPLAYED`) and every growth reshapes the
    # pitch tables, so an old checkpoint's tables are narrower than whatever
    # a fresh `Config()` would build. `notation.predict.load` reads the actual
    # width off the checkpoint's own state dict and sets this before building
    # the model, the same way it already handles `input_encoding`.
    pitch_vocab: int = PITCH_VOCAB
    # Which layout the OUTPUT streams follow -- see
    # `notation.tokens.LEGACY_TARGETS`/`ALIGNED_TARGETS`. A checkpoint that
    # records none predates the aligned representation; `notation.predict.load`
    # says so rather than trust this default, again as `input_encoding` does.
    targets: int = ALIGNED_TARGETS
    # Whether the encoder reads each note's MIDI velocity (bucketed) or the
    # reserved "unknown" bucket. False for everything through aligned1, whose
    # notes came from `notes_from_midi` without velocity; a checkpoint that
    # does not record it loads as False.
    velocity_input: bool = False
    # Whether this model writes a key signature per slot (`notation.tokens`,
    # KEY_*). True for every model built from now on; a checkpoint from before
    # the stream existed has no key head, and `notation.predict.load` reads
    # that off its state dict and builds it without one.
    key_stream: bool = True


def _positions(length: int, width: int) -> torch.Tensor:
    """Sinusoidal positions, as in the original transformer."""
    position = torch.arange(length).unsqueeze(1).float()
    step = torch.exp(torch.arange(0, width, 2).float() * (-math.log(10000.0) / width))
    encoded = torch.zeros(length, width)
    encoded[:, 0::2] = torch.sin(position * step)
    encoded[:, 1::2] = torch.cos(position * step)
    return encoded


class NotationModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        # The output streams this model reads and writes: every stream, less
        # `key` for a model without a key stream (`Config.key_stream`). A
        # keyless model has no key embedding, so the key a target carries is
        # never read, and no key head, so decoders fill the key it cannot
        # write with 0 ("none") -- see `notation.predict`.
        self.streams = tuple(name for name in OUTPUT_STREAMS
                             if config.key_stream or name != "key")
        # `pitch` is sized from the checkpoint-specific `config.pitch_vocab`,
        # not the module ceiling `INPUT_VOCAB`/`OUTPUT_VOCAB["pitch"]` -- see
        # `Config.pitch_vocab`. Every other stream's vocabulary is fixed by
        # `notation.vocab`'s tables, which every checkpoint shares.
        self.source_embeddings = nn.ModuleDict(
            {name: nn.Embedding(config.pitch_vocab if name == "pitch" else INPUT_VOCAB[name],
                                config.d_model)
             for name in INPUT_STREAMS})
        self.target_embeddings = nn.ModuleDict(
            {name: nn.Embedding(config.pitch_vocab if name == "pitch" else OUTPUT_VOCAB[name],
                                config.d_model)
             for name in self.streams})
        encoder_layer = nn.TransformerEncoderLayer(
            config.d_model, config.heads, config.feedforward, dropout=config.dropout,
            batch_first=True, norm_first=True)
        self.transformer = nn.Transformer(
            d_model=config.d_model, nhead=config.heads,
            num_encoder_layers=config.encoder_layers,
            num_decoder_layers=config.decoder_layers,
            dim_feedforward=config.feedforward, dropout=config.dropout,
            batch_first=True, norm_first=True,
            # Spelt out only to turn nested tensors off: they are incompatible
            # with `norm_first`, and left on PyTorch warns once per model built.
            custom_encoder=nn.TransformerEncoder(
                encoder_layer, config.encoder_layers, norm=nn.LayerNorm(config.d_model),
                enable_nested_tensor=False))
        self.heads = nn.ModuleDict(
            {name: nn.Linear(config.d_model,
                             config.pitch_vocab if name == "pitch" else OUTPUT_VOCAB[name])
             for name in self.streams})
        self.dropout = nn.Dropout(config.dropout)
        if config.decoder_dropout > 0:
            # Scaled like the sum it stands in for: n N(0, 1) embeddings add
            # to a vector of standard deviation sqrt(n) per component, n being
            # the streams this model embeds (eight, or nine with a key).
            self.hidden = nn.Parameter(
                torch.randn(config.d_model) * math.sqrt(len(self.streams)))
        # Computed once and carried on the module so it follows `.to(device)`.
        # Not persisted: it is a pure function of `max_length` and `d_model`, and
        # a checkpoint that stored it could not be loaded into a longer model.
        self.register_buffer("positions", _positions(config.max_length, config.d_model),
                             persistent=False)

    def _embed(self, stacked: torch.Tensor) -> torch.Tensor:
        length = stacked.shape[1]
        if length > self.config.max_length:
            raise ValueError(
                f"sequence of {length} exceeds max_length {self.config.max_length}; "
                "raise Config.max_length or window the input more tightly")
        return self.dropout(stacked + self.positions[:length])

    def decoder_inputs(self, target: dict, hide: bool = True) -> torch.Tensor:
        """The decoder's input embeddings, before positions, with some hidden.

        Measured on run4: given the true previous tokens it wrote onsets 0.969,
        bars 0.992 and metre 0.993 right; free-running, order-free, 0.760,
        0.916 and 0.738. The decoder leans on its own history, and at
        inference that history is its own mistakes. So in TRAINING a random
        `decoder_dropout` of positions have their whole embedding replaced by
        one learned `hidden` vector, and the decoder has to find those tokens
        in the encoder instead. Beyer & Dai's checkpoint records
        `input_dropout: 0.75`, but their training code is unreleased, so what
        the number means is inferred and the rate is a setting, not a copy.

        Never position 0: that is BOS, the one input every decode starts from.
        Never the labels: only what the decoder READS is hidden, and it is
        still scored on every position. Off in eval (`hide` or no), so decoding
        is deterministic.
        """
        stacked = sum(self.target_embeddings[name](target[name]) for name in self.streams)
        rate = self.config.decoder_dropout
        if not (hide and self.training and rate > 0):
            return stacked
        chosen = torch.rand(stacked.shape[:2], device=stacked.device) < rate
        chosen[:, 0] = False
        return torch.where(chosen.unsqueeze(-1), self.hidden.to(stacked.dtype), stacked)

    def forward(self, source: dict, target: dict, source_pad_mask=None) -> dict[str, torch.Tensor]:
        """Logits per output stream, shaped (batch, target length, vocabulary).

        Only this model's own `streams`: a model without a key stream returns
        no `key` logits, and reads no `key` from `target` either.

        `source_pad_mask` is true where a source position is PADDING. It defaults
        to the positions whose pitch is `PITCH_PAD`, which is what the loaders
        pad a batch with, so a short window in a long batch predicts what it
        would have predicted alone -- without it the encoder attends to padding
        and the batching silently changes the answer.

        The TARGET needs no such mask: padding is only ever appended, and the
        causal mask already stops a real position from seeing anything after it.
        """
        encoded = self._embed(sum(self.source_embeddings[name](source[name])
                                  for name in INPUT_STREAMS))
        decoded = self._embed(self.decoder_inputs(target))
        if source_pad_mask is None:
            source_pad_mask = source["pitch"] == PITCH_PAD
        # Attention over an entirely-masked row is a softmax over nothing, which
        # is NaN and poisons the whole batch. A row with no real note cannot say
        # anything useful anyway, so it keeps its first position instead.
        source_pad_mask = source_pad_mask.clone()
        source_pad_mask[source_pad_mask.all(dim=1), 0] = False
        # Bool throughout: mixing a float attention mask with a bool padding mask
        # makes PyTorch promote one of them, which has been a source of silent
        # dtype churn in other projects.
        causal = torch.triu(torch.ones(decoded.shape[1], decoded.shape[1],
                                       dtype=torch.bool, device=decoded.device), diagonal=1)
        hidden = self.transformer(encoded, decoded, tgt_mask=causal, tgt_is_causal=True,
                                  src_key_padding_mask=source_pad_mask,
                                  memory_key_padding_mask=source_pad_mask)
        return {name: head(hidden) for name, head in self.heads.items()}


def loss(predicted: dict, target: dict, pad_mask: torch.Tensor,
         weights: dict | None = None, row_masks: dict | None = None,
         space_only_pitch: bool = False, reduction: str = "mean",
         streams=OUTPUT_STREAMS) -> torch.Tensor:
    """Weighted cross entropy over every stream, counting real positions only.

    Streams are summed rather than averaged so that a stream with a large
    vocabulary is not quietly worth less than `hand`, which has two values.

    `weights` maps a stream to its multiplier; a stream it does not name
    weighs 1.0, so no weights is the plain sum every earlier run trained on.
    Beyer & Dai weight pitch 1.0, onset 1.0, accidental 0.5, bar 0.4, metre
    0.4, duration 0.3, voice 0.3, hand 0.25; `notation.train` passes whatever
    its config says, and nothing here writes those numbers down. A name that
    is not a stream raises: a typo would otherwise train at the default.

    `row_masks` maps a stream to a `(batch,)` or `(batch, length)` bool
    tensor: a False row, or position, contributes nothing to THAT stream, and
    still counts in every other. It exists for transposition: a transposed
    row's pitches are right and its accidentals are the untransposed score's
    spelling, which is wrong for the pitches it now carries, so
    `notation.train` masks `accidental` on those rows -- or, when the batches
    re-spell them (`notation.dataset.respell`), on just the notes the new key
    cannot write. A per-position mask is aligned with `pad_mask` (the labels,
    BOS already dropped). The stream's mean is over the positions it keeps,
    and a stream that keeps none adds 0 rather than a NaN.

    `space_only_pitch` is Beyer & Dai's eq. 6-8: their aligned layout has a
    "space" slot between two simultaneous notes so the decoder can insert one
    without shifting everything after it, and at that slot only the pitch
    stream (which carries the space token) is scored -- every other stream's
    label there is a don't-care, not a real answer. `reduction` picks how
    positions combine within a stream: "mean" (default, every run before this
    one) averages them; "sum" (the paper) sums a row's kept positions, then
    averages over rows -- so a longer row is not quietly worth less than a
    short one, but a position is not quietly worth less as a row grows either.

    It iterates `streams` (every output stream unless told otherwise) rather
    than whatever `predicted` happens to hold, so a head that is missing raises
    instead of being trained on nothing. A model without a key head
    (`Config.key_stream` False) passes its own `NotationModel.streams`, which
    is how `notation.train` calls it.
    """
    if reduction not in ("mean", "sum"):
        raise ValueError(f"unknown reduction {reduction!r}")
    weights = weights or {}
    row_masks = row_masks or {}
    unknown = (set(weights) | set(row_masks)) - set(OUTPUT_STREAMS)
    if unknown:
        raise ValueError(f"loss weights or row masks for no such stream: {sorted(unknown)}")
    real = pad_mask.reshape(-1)
    # Zero, but attached to the graph: a loader that emits an all-padding batch
    # should cost nothing, not raise on `backward()`.
    total = predicted["pitch"].sum() * 0.0
    if not bool(real.any()):
        return total
    if space_only_pitch:
        not_space = (target["pitch"] != PITCH_SPACE).reshape(-1)
    rows = max(int(pad_mask.shape[0]), 1)
    for name in streams:
        kept = real
        if name in row_masks:
            keep = row_masks[name].to(device=pad_mask.device, dtype=torch.bool)
            if keep.dim() == 1:
                keep = keep.reshape(-1, 1).expand_as(pad_mask)
            kept = pad_mask & keep[:, :pad_mask.shape[1]]
            kept = kept.reshape(-1)
        if space_only_pitch and name != "pitch":
            kept = kept & not_space
        if not bool(kept.any()):
            continue
        logits = predicted[name]
        flat = logits.reshape(-1, logits.shape[-1])[kept]
        wanted = target[name].reshape(-1)[kept]
        value = nn.functional.cross_entropy(flat.float(), wanted,
                                            reduction="sum" if reduction == "sum" else "mean")
        if reduction == "sum":
            value = value / rows
        total = total + float(weights.get(name, 1.0)) * value
    return total
