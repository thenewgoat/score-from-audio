"""A trained model's score for a performance.

Greedy decoding: at each step every stream takes its argmax, and the pitch
stream decides when to stop. Beam search would trade a lot of complexity for a
little quality, and there is no evidence yet that it is the binding constraint.

A whole performance does not fit in one sequence -- `Config.max_length` is 512
positions and a piece is tens of thousands -- so it is decoded window by window,
the same `bars`-bar windows training was cut into, and the windows are
concatenated. Concatenation is sound because the bar stream carries the bar
lines: each window opens with a bar marker, so joining two windows joins two
bars rather than running them together.

A checkpoint is decoded the way it was trained, which it records: the input
encoding (`Config.input_encoding`) and the windowing (`TrainConfig.windowing`,
see `notation.dataset.WINDOWING`). One that records neither -- run4 and
everything before it -- is the legacy pair, and decodes exactly as it did.

An ALIGNED checkpoint (`Config.targets` = `ALIGNED_TARGETS`) is not decoded in
bar windows at all: it writes exactly one output slot per played note, over
the whole performance, in rolling overlapping chunks -- see `aligned_greedy`.
`predict` dispatches on the checkpoint's `targets`, so every caller keeps one
entry point.
"""

import os

import numpy as np
import torch

from notation.tokens import (ALIGNED_TARGETS, INPUT_ENCODING, INPUT_STREAMS, LEGACY_ENCODING,
                             LEGACY_TARGETS, OUTPUT_STREAMS, PITCH_BOS, PITCH_EOS, PITCH_PAD,
                             PITCH_REST, PITCH_SPACE, PITCH_UNPLAYED, SECONDS_LIMIT,
                             encode_aligned, encode_performance)

# How many written slots each aligned chunk after the first is given as decoder
# context, and re-decodes as its own first positions: Beyer & Dai's overlap.
OVERLAP = 64

# Pitch tokens an aligned decode must never write. PAD, BOS and EOS are the
# decoder's own framing, and UNPLAYED is an INPUT marker (a score note nobody
# played) -- an input slot at inference is always a played note, and no
# target ever carries it. An aligned decode writes a fixed number of slots, so
# EOS in particular would mean nothing; left unmasked, an untrained or
# confused model could still pick any of them.
NEVER_WRITTEN = (PITCH_PAD, PITCH_BOS, PITCH_EOS, PITCH_UNPLAYED)


def _fill_headless(chosen: dict) -> None:
    """Give `chosen` a 0 for every output stream the model has no head for.

    A checkpoint from before the key stream (`Config.key_stream` False) writes
    no key, and 0 is `KEY_NONE`: the builder then writes the score exactly as
    it did before the stream existed. Filling here, where a step's choices are
    made, lets everything after -- the decoder's next input, the written slots,
    the returned streams -- carry every stream whichever model decoded.
    """
    for name in OUTPUT_STREAMS:
        chosen.setdefault(name, torch.zeros_like(chosen["pitch"]))


def greedy_batch(model, source: dict, max_length: int = 512,
                 constrain_pitch: bool = False,
                 counts: dict | None = None) -> list[dict[str, np.ndarray]]:
    """Greedy decoding for a whole batch of chunks at once.

    One chunk at a time is correct and unusably slow: a forward pass at batch 1
    is almost all kernel-launch overhead, and one test performance measured 40
    seconds for 23 windows -- the full test split would have been hours. Rows
    are decoded together and each stops at its own EOS.

    The source must be padded with `PITCH_PAD` in its pitch stream, which is
    where `NotationModel.forward` derives its padding mask from; a row is then
    decoded exactly as it would have been alone.

    `constrain_pitch` makes every written pitch one of the input's, each used at
    most as often as it was played, and ends the row exactly when they are used
    up. Measured on training windows, the free-running model already writes the
    right pitch MULTISET (0.816 order-free, against 0.811 teacher-forced) but
    loses its place while doing it; this tests how much of that is recovered by
    forbidding it to invent, repeat or drop a note, without retraining. It is
    only meaningful on clean input: on decoder output the played notes are not
    the score's.

    Measured, and it does not help: order-free pitch rises (0.816 to 0.875 on
    training windows) but onset and duration fall slightly, and test MeanER gets
    WORSE -- 31.76 to 34.26 on seen composers, 35.48 to 37.32 on unseen. The
    drift is in the rhythm, not in which pitch comes next. Kept, off by
    default, so the negative result can be reproduced.

    `counts`, if given, gains `truncated`: rows that reached the step limit
    without writing EOS. Such a row is kept as far as it got -- what it wrote
    is still the model's answer -- but its window may end mid-bar, so a run
    has to be able to say how often that happened rather than find out from a
    bar count.
    """
    device = next(model.parameters()).device
    source = {name: value.to(device) for name, value in source.items()}
    rows = source["pitch"].shape[0]
    target = {name: torch.zeros(rows, 1, dtype=torch.long, device=device)
              for name in OUTPUT_STREAMS}
    # Only the pitch stream carries control tokens; the others have no slot for
    # one, and take index 0 -- which is a real, harmless value in each of them
    # ("no bar starts here", "no accidental", hand 0, voice 0). The trainer
    # writes the same thing, so BOS looks the same at training and inference.
    target["pitch"][:, 0] = PITCH_BOS

    limit = min(max_length, model.config.max_length - 1)
    if constrain_pitch:
        # The model's own pitch head, not the module ceiling `OUTPUT_VOCAB`:
        # an older checkpoint's head can be narrower than a freshly built
        # model's (`Config.pitch_vocab`), and `last["pitch"]` below is exactly
        # that head's width.
        vocab = model.config.pitch_vocab
        real_input = source["pitch"].clamp(max=vocab - 1)
        remaining = torch.zeros(rows, vocab, dtype=torch.long, device=device)
        remaining.scatter_add_(1, real_input, (source["pitch"] < 128).long())
    finished = torch.zeros(rows, dtype=torch.bool, device=device)
    lengths = torch.full((rows,), limit, dtype=torch.long, device=device)
    with torch.no_grad():
        for step in range(limit):
            predicted = model(source, target)
            last = {name: logits[:, -1] for name, logits in predicted.items()}
            if constrain_pitch:
                spent = remaining[:, :128].sum(dim=1) == 0
                allowed = remaining > 0
                allowed[:, PITCH_EOS] = spent
                # An empty bar is a position of its own carrying PITCH_REST; it
                # is not an input note, so it stays available until the notes
                # are used up -- after that only EOS is.
                allowed[:, PITCH_REST] = ~spent
                last["pitch"] = last["pitch"].masked_fill(~allowed, float("-inf"))
            chosen = {name: logits.argmax(dim=-1) for name, logits in last.items()}
            _fill_headless(chosen)
            if constrain_pitch:
                used = (chosen["pitch"] < 128) & ~finished
                remaining.scatter_add_(1, chosen["pitch"].clamp(max=remaining.shape[1] - 1).unsqueeze(1),
                                       -used.long().unsqueeze(1))
            stopping = (chosen["pitch"] == PITCH_EOS) & ~finished
            lengths = torch.where(stopping, torch.full_like(lengths, step), lengths)
            finished = finished | stopping
            if bool(finished.all()):
                break
            for name in OUTPUT_STREAMS:
                # A finished row is padded out rather than cut: its later
                # positions are discarded by `lengths`, and keeping the batch
                # rectangular is what makes this worth doing at all.
                column = torch.where(finished, torch.full_like(chosen[name], PITCH_PAD)
                                     if name == "pitch" else torch.zeros_like(chosen[name]),
                                     chosen[name])
                target[name] = torch.cat([target[name], column.unsqueeze(1)], dim=1)

    if counts is not None:
        counts["truncated"] = counts.get("truncated", 0) + int((~finished).sum())

    # Drop the leading BOS column; nothing after it is a control token.
    out = []
    for row, length in enumerate(lengths.tolist()):
        out.append({name: target[name][row, 1:1 + length].cpu().numpy()
                    for name in OUTPUT_STREAMS})
    return out


def greedy(model, source: dict, max_length: int = 512) -> dict[str, np.ndarray]:
    """Output-stream arrays for one performance chunk, control tokens removed.

    The cap is whichever is smaller of `max_length` and what the model can
    embed. The decoder grows by one position per step and `Config.max_length`
    is enforced in `NotationModel._embed`, so an unbounded caller would raise
    from inside the forward pass rather than simply stop.
    """
    return greedy_batch(model, source, max_length)[0]


def batch_of(sources: list[dict]) -> dict:
    """Encoded chunks stacked into one padded batch of tensors.

    Padded with `PITCH_PAD` in the pitch stream, where `NotationModel.forward`
    reads its padding mask from, as the trainer pads (`Batches._collate`).
    """
    width = max(len(one["pitch"]) for one in sources)
    batch = {}
    for name in INPUT_STREAMS:
        block = np.full((len(sources), width), PITCH_PAD if name == "pitch" else 0,
                        dtype=np.int64)
        for row, one in enumerate(sources):
            block[row, :len(one[name])] = one[name]
        batch[name] = torch.from_numpy(block)
    return batch


def decode_fixed(model, source: dict, prefix=(), steps: int | None = None) -> list[dict]:
    """Exactly `steps` output slots for a one-row `source`, `prefix` first.

    The decoder reads `[BOS] + prefix` and then writes greedily until `steps`
    slots exist, the prefix counted among them. It never stops early: an
    aligned target has one slot per input position and no end token, so the
    length is decided by the input, not the model. The pitch head's control
    and input-only tokens (`NEVER_WRITTEN`) are masked out of every choice.

    Returns every slot, prefix included, as `{stream: int}`.
    """
    steps = int(source["pitch"].shape[1]) if steps is None else int(steps)
    prefix = list(prefix)
    if len(prefix) > steps:
        raise ValueError(f"a prefix of {len(prefix)} slots is longer than the {steps} "
                         "slots to write")
    device = next(model.parameters()).device
    source = {name: value.to(device) for name, value in source.items()}
    target = {name: torch.zeros(1, 1 + len(prefix), dtype=torch.long, device=device)
              for name in OUTPUT_STREAMS}
    # BOS in the pitch stream only, the others at index 0 -- what the trainer
    # writes (`Batches._collate`).
    target["pitch"][0, 0] = PITCH_BOS
    for at, slot in enumerate(prefix, start=1):
        for name in OUTPUT_STREAMS:
            target[name][0, at] = int(slot[name])
    # Sized from THIS model's head, not the module constant: checkpoint
    # generations differ in pitch width (`Config.pitch_vocab`).
    blocked = torch.zeros(model.config.pitch_vocab, dtype=torch.bool, device=device)
    blocked[[token for token in NEVER_WRITTEN if token < model.config.pitch_vocab]] = True
    # A model trained with `space_loss="pitch"` had only its pitch head scored
    # at a space slot, where every other stream's target was 0
    # (`aligned_targets`). What those heads write at a space is untrained, and
    # fed back it would sum into the next decoder input a combination training
    # never showed -- and a stray bar value would be carried onto the next
    # kept slot by `aligned_greedy` as a bar line. So they are written as the
    # 0 training held there. Every other model (aligned1 and before, `load`'s
    # "all") keeps what its heads chose, exactly as before.
    pitch_only_space = getattr(model, "space_loss", "all") == "pitch"
    written = [dict(slot) for slot in prefix]
    with torch.no_grad():
        for _ in range(steps - len(prefix)):
            predicted = model(source, target)
            last = {name: logits[:, -1] for name, logits in predicted.items()}
            last["pitch"] = last["pitch"].masked_fill(blocked, float("-inf"))
            chosen = {name: logits.argmax(dim=-1) for name, logits in last.items()}
            _fill_headless(chosen)
            if pitch_only_space:
                space = chosen["pitch"] == PITCH_SPACE
                chosen = {name: value if name == "pitch" else value.masked_fill(space, 0)
                          for name, value in chosen.items()}
            written.append({name: int(chosen[name][0]) for name in OUTPUT_STREAMS})
            target = {name: torch.cat([target[name], chosen[name].view(1, 1)], dim=1)
                      for name in OUTPUT_STREAMS}
    return written


def performance_notes(notes) -> list[tuple]:
    """A performance as aligned training holds it: float32-rounded, then sorted.

    `notation.dataset.aligned_examples` rounds to float32 BEFORE sorting, and
    the slot order and every gap follow from that; doing the same here makes
    a chunk's input exactly what training would have built from the same MIDI.
    Each note's tail beyond `(onset, offset, pitch)` -- velocity, when `notes`
    carries it -- is kept, and never itself a sort key, exactly as
    `aligned_examples` orders it.
    """
    rounded = np.asarray([tuple(note[:3]) for note in notes],
                        dtype=np.float32).reshape(-1, 3)
    order = sorted(range(len(notes)), key=lambda k: (float(rounded[k][0]),
                   float(rounded[k][1]), int(rounded[k][2])))
    return [(float(rounded[k][0]), float(rounded[k][1]), int(rounded[k][2]),
            *tuple(int(v) for v in notes[k][3:])) for k in order]


def chunk_starts(count: int, chunk: int, overlap: int) -> list[tuple[int, int]]:
    """`(start, stop)` of each rolling chunk over `count` notes.

    A chunk holds `chunk - 1` notes: the decoder spends a position on BOS and
    reads at most `chunk`, and training crops are `max_length - 1` slots for
    the same reason. Consecutive chunks share exactly `overlap` notes, so the
    step is `chunk - 1 - overlap`. (The plan's loop stepped by `chunk -
    overlap`, which with `chunk - 1`-note chunks skipped one note per chunk
    boundary and fed each later chunk context one slot out of place.) The last
    chunk is the first that reaches the end, so every chunk after the first
    has at least one new note.
    """
    width = chunk - 1
    if count == 0:
        return []
    if width <= overlap:
        raise ValueError(f"chunk {chunk} leaves no new notes after an overlap of {overlap}")
    step = width - overlap
    starts = list(range(0, max(count - overlap, 1), step))
    return [(start, min(start + width, count)) for start in starts]


def aligned_chunks(notes, chunk: int = 512, overlap: int = OVERLAP,
                   encoding: int = INPUT_ENCODING, counts: dict | None = None,
                   velocity: bool = False):
    """`(start, stop, source)` per chunk of `performance_notes(notes)`.

    Each chunk is encoded by `notation.dataset.crop_inputs` with every note
    played -- the function training crops go through -- so its first gap is
    measured from the performance note before it (Ruling B), not from itself.
    `counts` gains `clipped` from `encode_aligned`. `velocity` is passed
    straight to `encode_aligned`: a velocity checkpoint reads it from the
    notes, and refuses notes that carry none.
    """
    from notation.dataset import crop_inputs

    every = list(range(len(notes)))
    for start, stop in chunk_starts(len(notes), chunk, overlap):
        played, local, begin = crop_inputs(notes, every, start, stop)
        yield start, stop, encode_aligned(played, local, start=begin, encoding=encoding,
                                          counts=counts, velocity=velocity)


def aligned_greedy(model, notes, chunk: int = 512, overlap: int = OVERLAP,
                   encoding: int = INPUT_ENCODING,
                   counts: dict | None = None) -> dict[str, np.ndarray]:
    """One output slot per played note, decoded in rolling chunks.

    Beyer & Dai's inference: chunks of `chunk - 1` input notes overlapping by
    `overlap`; each chunk after the first is decoded with the previous chunk's
    last `overlap` written slots as decoder context, and only its new slots are
    kept. There is no end token: the decoder writes exactly one slot per input
    position, so it can neither overrun nor stop short.

    Returns the output streams in slot order -- `performance_notes` order --
    with the spaces (`PITCH_SPACE`, a played note the score does not write)
    removed. `counts`, if given, gains:

      chunks      chunks decoded
      slots       slots written, one per played note, before space removal
      spaces      slots removed as spaces
      space_bars  bar markers a removed space carried, moved to the next kept
                  slot: the builder reads a bar line positionally, so dropping
                  one would merge two bars and move every later bar. Always
                  0 for a `space_loss="pitch"` model, whose spaces are written
                  with no bar (`decode_fixed`)
      bars_lost   of those, markers that could not move because the next kept
                  slot opened a bar of its own, or none followed
      clipped     gaps and durations the gap encoding put on its top slot
    """
    counts = counts if counts is not None else {}
    for name in ("chunks", "slots", "spaces", "space_bars", "bars_lost", "clipped"):
        counts.setdefault(name, 0)
    notes = performance_notes(notes)
    written: list[dict] = []
    # `getattr(model, "config", None)`, not `model.config`: several tests here
    # exercise the chunking contract alone, with `decode_fixed` monkeypatched
    # and `model` left `None` -- reading velocity off a model that is not
    # really there must default to off, not raise.
    velocity = getattr(getattr(model, "config", None), "velocity_input", False)
    for start, stop, source in aligned_chunks(notes, chunk, overlap, encoding, counts,
                                              velocity=velocity):
        # The slots already written for this chunk's first notes: `overlap` of
        # them for every chunk after the first, which `chunk_starts` lines up.
        context = written[start:]
        slots = decode_fixed(model, batch_of([source]), prefix=context, steps=stop - start)
        written.extend(slots[len(context):])
        counts["chunks"] += 1
    counts["slots"] += len(written)

    out = {name: [] for name in OUTPUT_STREAMS}
    pending = 0                          # a bar marker a removed space carried
    for slot in written:
        if slot["pitch"] == PITCH_SPACE:
            counts["spaces"] += 1
            if slot["bar"]:
                counts["space_bars"] += 1
                if pending:
                    counts["bars_lost"] += 1
                pending = slot["bar"]
            continue
        bar = slot["bar"]
        if pending:
            if bar:
                counts["bars_lost"] += 1
            else:
                bar = pending
            pending = 0
        for name in OUTPUT_STREAMS:
            out[name].append(bar if name == "bar" else slot[name])
    if pending:
        counts["bars_lost"] += 1
    return {name: np.asarray(values, dtype=np.int64) for name, values in out.items()}


def performance_windows(annotations: str, bars: int) -> list[tuple[float, float]]:
    """`(start, end)` for each whole `bars`-bar span of a performance.

    Used when there is no score to pair against -- transcribing audio, which is
    what this stage is ultimately for. `notation.dataset.bar_windows` is the
    paired version and is what training and the evaluation use. This is the
    LEGACY windowing; `decode_windows` picks the one a checkpoint needs.
    """
    from notation.dataset import bar_lines, window_spans

    lines = bar_lines(annotations)
    return [(float(lines[first]), float(lines[first + count]))
            for first, count in window_spans(len(lines) - 1, bars, tail=False)]


def decode_windows(model, annotations: str, score_bars=None, offset: int = 0,
                   bars: int | None = None) -> list[tuple]:
    """The windows to decode a performance with, cut the way `model` was trained.

    Legacy windowing: `(start, end)` for each whole `model.bars`-bar window,
    the trailing part-window dropped -- exactly what run4 was always decoded
    with. `WINDOWING` 2: the trailing part-window kept, and each window given
    as ALL its bar-line times, `(start, line, ..., end)`, so `predict` can
    halve one on a bar line when it is too dense to embed.

    With `score_bars` (and the score's `offset`) the windows stop where the
    score does, as `notation.dataset.bar_windows` does; without, they follow
    the performance's downbeats. `bars` defaults to the checkpoint's own; see
    `window_bars` for why a disagreeing one is refused before this is reached.
    """
    from notation.dataset import LEGACY_WINDOWING, bar_lines, window_spans

    bars = int(bars if bars is not None else model.bars)

    lines = bar_lines(annotations, score_bars, offset)
    legacy = getattr(model, "windowing", LEGACY_WINDOWING) == LEGACY_WINDOWING
    spans = window_spans(len(lines) - 1, bars, tail=not legacy)
    if legacy:
        return [(float(lines[first]), float(lines[first + count])) for first, count in spans]
    return [tuple(float(line) for line in lines[first:first + count + 1])
            for first, count in spans]


def predict(model, notes, windows, max_length: int = 512,
            batch: int = 32, positions: int = 2048,
            constrain_pitch: bool = False,
            counts: dict | None = None) -> dict[str, np.ndarray]:
    """One score's streams for a whole performance, decoded window by window.

    An aligned checkpoint ignores `windows`, `max_length`, `batch` and
    `positions`, refuses `constrain_pitch`, and decodes the whole of `notes`
    with `aligned_greedy`; everything below is the bar-window path.

    `windows` is each window's bar-line times in seconds: `(start, end)`, or
    `(start, line, ..., end)` when its inner bar lines are known (see
    `decode_windows`). Anything beyond `bar_windows`' reach is simply not
    notated, which is the honest outcome when the score side and the
    performance side stop agreeing.

    `counts`, if given, accumulates what happened to the windows, because
    several things here lose or distort bars without failing:

      windows     windows asked for
      empty       skipped: no note starts inside
      split       halved on a bar line because it held more notes than the
                  model can embed (only a window given with its inner bar
                  lines can be; the target is unknown here, so the input
                  alone decides). Each half is decoded, or halved again.
      too_dense   skipped: more notes than the model can embed, and no bar
                  line left to halve on
      too_long    DECODED although longer than `SECONDS_LIMIT` -- legacy
                  encoding only. Training drops such windows, so the model
                  has never seen one, and the legacy encoding piles every
                  note past the limit onto the last onset bucket. Measured on
                  the 16 test pieces: 105 of 1,190 windows, 33 of Ravel's 35
                  and 19 of Glinka's 37. The gap encoding has no such limit.
      clipped     gaps and durations the gap encoding put on its top slot
      truncated   decoded to the step limit without an EOS (`greedy_batch`)

    A skipped window contributes no bars, and the builder joins windows end
    to end, so every later bar comes out earlier by that window's bars. These
    are counted rather than changed: changing what is decoded would move
    every figure already published from this code. The review of the test
    split found no skipped window on the 16 test pieces; how many decodes
    were truncated there was not counted, since this counter came later.
    """
    from notation.dataset import fit_windows

    if getattr(model.config, "targets", LEGACY_TARGETS) == ALIGNED_TARGETS:
        # An aligned model writes one slot per played note and was trained on
        # whole performances cropped anywhere, not on bar windows: it is given
        # every note, and `windows` does not apply. Its counts are
        # `aligned_greedy`'s.
        if constrain_pitch:
            # Refused rather than ignored: `notation_eval --constrain-pitch`
            # labels its rows constrained, and an aligned row decoded without
            # the constraint would carry that label falsely.
            raise ValueError("constrain_pitch is implemented for bar-window (legacy "
                             "targets) checkpoints only, not for an aligned checkpoint")
        return aligned_greedy(model, notes, chunk=model.config.max_length,
                              encoding=getattr(model.config, "input_encoding", INPUT_ENCODING),
                              counts=counts)
    counts = counts if counts is not None else {}
    for name in ("windows", "empty", "split", "too_dense", "too_long", "clipped",
                 "truncated"):
        counts.setdefault(name, 0)
    encoding = getattr(model.config, "input_encoding", LEGACY_ENCODING)
    onsets = [note[0] for note in notes]
    ordered = all(a <= b for a, b in zip(onsets, onsets[1:]))

    def inside_of(start: float, end: float) -> list:
        if ordered:
            low = int(np.searchsorted(onsets, start, side="left"))
            return list(notes[low:int(np.searchsorted(onsets, end, side="left"))])
        return [note for note in notes if start <= note[0] < end]

    encoded = []
    for window in windows:
        lines = [float(line) for line in window]
        counts["windows"] += 1
        if not inside_of(lines[0], lines[-1]):
            counts["empty"] += 1
            continue

        def fits(first: int, count: int) -> bool:
            return len(inside_of(lines[first], lines[first + count])) <= model.config.max_length

        # `fit_windows` counts a single bar it cannot fit as `too_long`; here
        # that bar is too DENSE, so it is counted under its own name.
        fitted: dict = {}
        pieces = fit_windows(0, len(lines) - 1, fits, fitted)
        counts["split"] += fitted["split"]
        # Too dense to embed. Dropping it loses those bars; the alternative,
        # truncating, writes a bar that stops mid-way and shifts everything
        # after it.
        counts["too_dense"] += fitted["too_long"]
        for first, count in pieces:
            start, end = lines[first], lines[first + count]
            inside = inside_of(start, end)
            if not inside:                     # a half of a window can be empty
                counts["empty"] += 1
                continue
            one = encode_performance(inside, start=start, encoding=encoding, counts=counts)
            if encoding == LEGACY_ENCODING and end - start > SECONDS_LIMIT:
                counts["too_long"] += 1
            encoded.append(one)

    # Grouped by TOTAL positions, not by count. Window lengths range from a
    # handful of notes to the 512-position cap, and a fixed count of the long
    # ones filled the 8 GB card to 7.8 GB beside a training run; the attention
    # cost is batch x width squared, so bounding batch x width bounds it.
    groups, current, width = [], [], 0
    for one in encoded:
        wider = max(width, len(one["pitch"]))
        if current and (len(current) >= batch or wider * (len(current) + 1) > positions):
            groups.append(current)
            current, width = [], 0
            wider = len(one["pitch"])
        current.append(one)
        width = wider
    if current:
        groups.append(current)

    pieces = []
    for group in groups:
        width = max(len(one["pitch"]) for one in group)
        source = {}
        for name in group[0]:
            block = np.full((len(group), width), PITCH_PAD if name == "pitch" else 0,
                            dtype=np.int64)
            for row, one in enumerate(group):
                block[row, :len(one[name])] = one[name]
            source[name] = torch.from_numpy(block)
        pieces.extend(greedy_batch(model, source, max_length, constrain_pitch, counts))
    if not pieces:
        return {name: np.zeros(0, dtype=np.int64) for name in OUTPUT_STREAMS}
    return {name: np.concatenate([piece[name] for piece in pieces]).astype(np.int64)
            for name in OUTPUT_STREAMS}


def load(checkpoint: str, device: str = "cpu"):
    """The model a training run saved, in evaluation mode.

    The architecture is rebuilt from the checkpoint's own config when it
    carries one, so a run trained at a different width still loads.
    """
    from notation.model import Config, NotationModel

    from notation.dataset import LEGACY_WINDOWING

    state = torch.load(checkpoint, map_location=device, weights_only=False)
    stored = dict(state.get("model_config") or {})
    # A checkpoint from before the input encoding was recorded was trained on
    # the legacy one -- run4 above all -- and must be fed it. Left to the
    # `Config` default it would silently be fed gaps, and still decode.
    stored.setdefault("input_encoding", LEGACY_ENCODING)
    # `PITCH_VOCAB` has grown before and may again, so `model_config` (or the
    # `Config` default, for a checkpoint that predates this field entirely)
    # cannot be trusted to name the width this checkpoint was actually built
    # with. The state dict is the one place that width cannot lie: it is set
    # from the checkpoint's own pitch embedding before the model is built, so
    # `load_state_dict(strict=True)` below always sees matching shapes.
    stored["pitch_vocab"] = state["model"]["source_embeddings.pitch.weight"].shape[0]
    # A checkpoint from before the aligned representation trained on the old
    # one bar-window-per-slot layout -- run4 and recipe16 included -- and must
    # be scored and decoded as that, not as the new default.
    stored.setdefault("targets", LEGACY_TARGETS)
    # A checkpoint from before velocity was an input read no velocity at all,
    # which is `Config`'s own default -- unlike `input_encoding` and
    # `targets` above, this `setdefault` is not load-bearing, only explicit.
    stored.setdefault("velocity_input", False)
    # A checkpoint from before the key stream has no key head. Read off the
    # state dict, like the pitch width, because a config can omit a field but
    # the weights cannot lie about which heads exist.
    stored["key_stream"] = "heads.key.weight" in state["model"]
    model = NotationModel(Config(**stored)).to(device)
    model.load_state_dict(state["model"])
    trained = state.get("config") or {}
    # The window size it was trained on, which a caller must decode with -- see
    # `window_bars`. None for a checkpoint that did not record one.
    model.bars = trained.get("bars")
    # And how its windows were cut; legacy when it does not say.
    model.windowing = trained.get("windowing", LEGACY_WINDOWING)
    # Which streams it was scored on at a space slot: "pitch" alone (Beyer &
    # Dai's loss) or "all", every run through aligned1 and every checkpoint
    # that predates the setting. `decode_fixed` reads it.
    model.space_loss = trained.get("space_loss", "all")
    return model.eval()


def window_bars(model, requested: int | None) -> int:
    """How many bars per window to decode `model` with.

    A model has only ever seen windows of the size it was trained on, and the
    checkpoint records that size. `--bars` used to default to 4 while every
    checkpoint was trained at 2, so the default decoded windows twice the size
    the model knew, and nothing said so. The recorded size is the default; an
    explicit request that disagrees with it raises; a checkpoint that recorded
    none needs an explicit request.
    """
    recorded = getattr(model, "bars", None)
    if requested is None:
        if recorded is None:
            raise ValueError("the checkpoint records no bars per window; pass --bars")
        return int(recorded)
    if recorded is not None and int(requested) != int(recorded):
        raise ValueError(f"--bars {requested} disagrees with the {recorded} bars "
                         "per window the checkpoint was trained on")
    return int(requested)


def main(argv=None) -> int:
    import argparse

    from evaluation.notation_eval import notes_from_midi
    from notation.build import write_musicxml
    from notation.dataset import bar_offset, score_bar_numbers

    parser = argparse.ArgumentParser(
        prog="python -m notation.predict",
        description="Notate a performance with a trained notation model.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--midi", required=True, help="the performance to notate")
    parser.add_argument("--annotations", required=True, help="its beat annotations")
    parser.add_argument("--score", default="",
                        help="the reference MusicXML, to pair windows against its bars; "
                             "without one the windows follow the performance's downbeats")
    parser.add_argument("--bars", type=int, default=None,
                        help="bars per window; defaults to what the checkpoint was "
                             "trained on, and must agree with it if given")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    model = load(args.checkpoint, args.device)
    bars = window_bars(model, args.bars)
    # With velocity exactly when the checkpoint reads it: `encode_aligned`
    # refuses a velocity model notes that carry none.
    notes = notes_from_midi(args.midi,
                            velocity=getattr(model.config, "velocity_input", False))
    if args.score:
        # `bar_offset` off the score's OWN annotation, beside the score: an
        # anacrusis means the first downbeat is the second bar, and cutting
        # without it notates every window one bar late.
        windows = decode_windows(model, args.annotations, score_bar_numbers(args.score),
                                 bar_offset(os.path.join(os.path.dirname(args.score),
                                                         "midi_score_annotations.txt")),
                                 bars)
    else:
        windows = decode_windows(model, args.annotations, bars=bars)
    streams = predict(model, notes, windows, model.config.max_length)
    write_musicxml(streams, args.out)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
