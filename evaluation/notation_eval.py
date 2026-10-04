"""Phase 1's table: what each way of notating the same notes actually scores.

A row here is anything that turns a note list into a MusicXML file. The
hand-written planner is one; a trained model is another. Both are scored the same
way, against the same reference score, so the numbers mean the same thing.
"""

import os

# Bumped whenever `normalised_reference` writes a different file for the same
# source, so that a cache shared between runs -- which is the point of it -- is
# not silently serving the previous version's references. Version 2 renumbers
# voices so that MUSTER cannot read the two staves of one part as one voice
# (`bb10df2`); version 3 writes `<chord/>` without the space music21 puts before
# the slash, which MUSTER's tag match missed -- 34.8% of note elements had been
# read as separate notes (`5b44742`). See `notation.musicxml` for both. The
# output also depends on the installed music21, which this number does not
# track: upgrade music21 and bump it, or clear the cache.
REFERENCE_VERSION = 3


def notes_from_midi(path: str, velocity: bool = False) -> list[tuple]:
    """(onset seconds, offset seconds, MIDI pitch) from a MIDI file.

    With `velocity`, each note carries its note-on velocity as a fourth
    element. The default output is byte-identical to before this option
    existed: every other caller of this function reads 3-tuples.

    Note-offs are matched to the *oldest* still-sounding note of that pitch, so a
    repeated note stays two notes rather than collapsing into one. A note left
    sounding at the end of the file is closed there, and dropped if that leaves
    it with no length: a file cut off mid-note ends with an unmatched note-on,
    and a note that starts and stops at the same instant is not a note. Three
    later tasks take this list as their input contract.
    """
    import mido

    midi = mido.MidiFile(path)
    sounding: dict[int, list[tuple[float, int]]] = {}
    notes: list[tuple] = []
    now = 0.0
    for message in midi:
        now += message.time
        if message.type == "note_on" and message.velocity > 0:
            sounding.setdefault(message.note, []).append((now, message.velocity))
        elif message.type in ("note_off", "note_on"):
            started = sounding.get(message.note)
            if started:
                onset, vel = started.pop(0)
                notes.append((onset, now, int(message.note), vel))
    for pitch, times in sounding.items():
        for onset, vel in times:
            notes.append((onset, now, int(pitch), vel))
    notes = [n if velocity else n[:3] for n in notes if n[1] > n[0]]
    notes.sort(key=lambda note: (note[0], note[2]))
    return notes


def write_planner_score(notes, bpm: float, beats_per_bar: int, out_path: str) -> str:
    """The hand-written planner's MusicXML for a note list.

    This is the baseline row: an eighth-note grid, a fitted phase and a fixed
    pitch split between the hands. It is what the project notates with today.
    """
    from notation.musicxml import write
    from transcribe.audio import Event, build_score, plan

    # `*_`: the planner reads no velocity, but is handed a velocity
    # checkpoint's 4-tuples by `test_rows`, beside that model's row.
    events = sorted((Event(onset, offset, pitch) for onset, offset, pitch, *_ in notes
                     if offset > onset), key=lambda event: (event.onset, event.pitch))
    if not events:
        raise ValueError("no notes to notate")
    # Through `notation.musicxml`, like every other writer whose output MUSTER
    # reads. The planner writes one voice per staff and so escaped the voice
    # collision by construction -- but only until a measure where one staff is
    # empty and music21 leaves the other's notes with no <voice> element at
    # all, which is voice 1 to MUSTER on whichever staff it lands.
    return write(build_score(plan(events, bpm), bpm, beats_per_bar), out_path)


def normalised_reference(source_xml: str, out_path: str,
                         cache: str | None = None) -> str:
    """The reference score, re-serialised by music21 -- the only honest baseline.

    MUSTER is not indifferent to how a file is laid out. Plain music21
    re-serialisation of an ASAP source, with nothing of this project involved,
    costs a mean of 25.42 MeanER across the sixteen composers (median 24.26),
    so an estimate scored against the RAW source reads as about 25 even when it
    is perfect. Everything this project builds is written by music21, so every
    estimate is compared against a reference written the same way and the layout
    term cancels. The residual is then the representation's, which is mean 1.37
    and median 0.68 over the same sixteen.

    Three composers -- Balakirev, Mozart, Ravel -- have sources music21 refuses
    to write back at all ("inexpressible durations"). Those pieces have no
    normalised reference, and this raises rather than quietly falling back to
    the raw source, which would put a ~25-point layout penalty on one row of a
    table and nowhere else.

    The result is byte-stable across calls and across processes, which it was
    not before. music21 stamps a fresh random `<score-instrument id>` on every
    export, and once -- on Schumann's Kreisleriana -- it also resolved a
    one-note opening measure as an anacrusis on one pass and not on another,
    writing 1,475 notes in 77 measures where another pass wrote 1,513 in 78.
    The identity floor followed it from 3.22 to 9.76. A floor that moves
    between runs cannot calibrate anything, and two runs scored against
    different references cannot be compared at all. So the random id is
    replaced by one derived from the source, and the whole file is cached
    under a key taken from the source's CONTENTS: a corrected score gets a new
    reference, and two copies of the same music share one.

    `cache` is a directory. Without one the output is still deterministic --
    the id is still stripped -- but the work is repeated.

    The cache key carries `REFERENCE_VERSION` as well as the source's digest.
    The key names what went IN, so on its own it cannot tell that what comes
    OUT has changed: the first time this function's serialisation was
    corrected, every reference already in a shared cache would have stayed at
    the old one, and a run would have scored half its rows against a file the
    code no longer writes. Bump the version whenever the output changes.
    """
    import hashlib
    import re
    import shutil

    import music21

    from notation.musicxml import write

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(source_xml, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()

    if cache:
        os.makedirs(cache, exist_ok=True)
        entry = os.path.join(cache, f"v{REFERENCE_VERSION}-{digest}.musicxml")
        if os.path.exists(entry):
            shutil.copyfile(entry, out_path)
            return out_path

    write(music21.converter.parse(source_xml), out_path)
    with open(out_path, encoding="utf-8") as handle:
        text = handle.read()
    # Every random id is rewritten, not removed: each is referenced elsewhere in
    # the file -- <midi-instrument> names its <score-instrument>, <part> names
    # its <score-part> -- so they must stay valid XML names and stay consistent
    # with each other. Each distinct id is replaced by a deterministic one in
    # order of first appearance, which keeps those cross-references intact.
    # Both prefixes occur: instruments are always random, and a part is too
    # unless the source named it.
    seen: dict[str, str] = {}
    for found in re.findall(r"\b([IP][0-9a-f]{32})\b", text):
        if found not in seen:
            seen[found] = f"{found[0]}{digest[:28]}{len(seen):04d}"
    for original, stable in seen.items():
        text = text.replace(original, stable)
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(text)

    if cache:
        shutil.copyfile(out_path, entry)
    return out_path


def write_roundtrip_score(source_xml: str, out_path: str) -> str:
    """The reference score through this stage's own tokens and back.

    This is the representation's floor: the best any model could do, because a
    perfect model would emit exactly these streams. Reported beside the model's
    number so a reader can tell model error from representation error.
    """
    import music21

    from notation.build import write_musicxml
    from notation.tokens import encode_score

    return write_musicxml(encode_score(music21.converter.parse(source_xml)), out_path)


def write_model_score(model, notes, windows, out_path: str,
                      constrain_pitch: bool = False, counts: dict | None = None) -> str:
    """A trained notation model's MusicXML for a note list.

    `counts` receives `notation.predict.predict`'s window counts -- skipped,
    over-long and truncated windows -- so the row can report them. An aligned
    checkpoint ignores `windows`, decodes every note of the performance, and
    reports `aligned_greedy`'s counts (chunks, slots, spaces, ...) instead.
    """
    from notation.build import write_musicxml
    from notation.predict import predict

    return write_musicxml(predict(model, notes, windows, model.config.max_length,
                                  constrain_pitch=constrain_pitch, counts=counts),
                          out_path)


def score_row(reference_xml: str, estimate_xml: str, root: str | None = None) -> dict[str, float]:
    """One row: an estimated score against the reference, through MUSTER."""
    from evaluation.muster import compare

    return compare(reference_xml, estimate_xml, root)


def tempo_and_bar(annotations_path: str) -> tuple[float, int]:
    """Tempo in bpm and beats per bar, from ASAP's annotated beats.

    Phase 1 measures notation, so the beats come from the annotations rather than
    from a tracker: a tracker's octave error would land on the notation step's
    score and be indistinguishable from a notation failure.
    """
    import numpy as np

    from evaluation.pairs import beat_map, downbeat_map

    beats = beat_map(annotations_path)
    if len(beats) < 2:
        raise ValueError(f"{annotations_path} holds fewer than two beats")
    bpm = 60.0 / float(np.median(np.diff(beats)))

    downbeats = downbeat_map(annotations_path)
    beats_per_bar = 4
    if len(downbeats) > 1:
        spacing = np.median(np.diff(downbeats)) / np.median(np.diff(beats))
        rounded = int(round(float(spacing)))
        if 2 <= rounded <= 12:
            beats_per_bar = rounded
    return bpm, beats_per_bar


def benchmark(rows: list[dict], out_dir: str, root: str | None = None,
              skipped: list[dict] | None = None) -> list[dict]:
    """Score every row's estimate against its reference, averaged per label.

    Each row is `{label, stem, reference, estimate}`. Averaging happens per
    recording so that one long piece cannot dominate a label's number -- the same
    rule the note-level harness uses.

    A label's line says what it lost as well as what it measured. `failed` is
    MUSTER aborting on a row; `skipped` is rows that never reached MUSTER,
    from `skipped`'s records -- `test_rows` writes one per estimate music21
    refused to export, each naming the rows it cost under `lost`. Before this
    the second count lived only in `skipped.jsonl`: on the training split the
    model line read n=138, failed=4, with 15 more model rows lost to export
    failures in the other file, so the line did not add up to the 150
    recordings the other rows had. A row carrying `decode` (the model's
    window counts, see `notation.predict.predict`) has them summed into the
    line under the same name.
    """
    import json
    from collections import defaultdict

    from evaluation.muster import default_root

    # Resolved once, before any scoring. A missing MUSTER_DIR is a broken setup,
    # not an unscoreable recording, and must not arrive row by row disguised as
    # data.
    root = root or default_root()

    os.makedirs(out_dir, exist_ok=True)
    per_recording, failures = [], []
    for row in rows:
        try:
            measured = score_row(row["reference"], row["estimate"], root)
        except RuntimeError as failure:
            # MUSTER is research C++ from 2017-2022 and aborts outright on some
            # real scores; its wrapper raises RuntimeError for exactly that. One
            # unusable recording must cost that recording, not the whole run --
            # and the count is reported, because a row averaged over whichever
            # recordings happened to survive is not a row. Anything else, a
            # ValueError from a misconfigured archive above all, is a bug and is
            # left to propagate.
            failures.append({"label": row["label"], "stem": row["stem"],
                             "error": str(failure)[:500]})
            continue
        per_recording.append({"label": row["label"], "stem": row["stem"], **measured,
                              **({"decode": row["decode"]} if "decode" in row else {})})

    grouped = defaultdict(list)
    for record in per_recording:
        grouped[record["label"]].append(record)

    # Every label in `rows` gets a line, including one where nothing survived.
    # A label that simply vanishes is the worst outcome for a harness whose job
    # is a gate number: it reads as a clean run that measured nothing.
    # A label every one of whose rows was skipped before scoring is a label
    # too, for the same reason.
    lost = defaultdict(int)
    for record in skipped or ():
        for label, count in (record.get("lost") or {}).items():
            lost[label] += count
    summary = []
    for label in dict.fromkeys([row["label"] for row in rows] + list(lost)):
        records = grouped.get(label, [])
        line = {"label": label, "recordings": len(records),
                "failed": sum(1 for f in failures if f["label"] == label),
                "skipped": lost.get(label, 0)}
        for name in (key for key in (records[0] if records else {})
                     if key not in ("label", "stem", "decode")):
            line[name] = sum(record[name] for record in records) / len(records)
        decoded = [row["decode"] for row in rows if row["label"] == label and "decode" in row]
        if decoded:
            line["decode"] = {name: sum(one.get(name, 0) for one in decoded)
                              for name in dict.fromkeys(k for one in decoded for k in one)}
        summary.append(line)

    with open(os.path.join(out_dir, "per-recording.jsonl"), "w") as handle:
        for record in per_recording:
            handle.write(json.dumps(record) + "\n")
    with open(os.path.join(out_dir, "failures.jsonl"), "w") as handle:
        for record in failures:
            handle.write(json.dumps(record) + "\n")
    with open(os.path.join(out_dir, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=1)
    return summary


def select_entries(entries: list[dict], limit: int, seed: int = 0) -> list[dict]:
    """Which recordings to score, sampled rather than sliced.

    Taking an alphabetical prefix takes one composer: ASAP's first entries are
    all Bach, whose rhythmic regularity flatters a grid-based planner. A seeded
    permutation samples the corpus, and the seed is an argument so a run is
    reproducible.
    """
    import numpy as np

    ordered = sorted(entries, key=lambda entry: entry["midi_performance"])
    if not limit or limit >= len(ordered):
        return ordered
    chosen = np.random.default_rng(seed).permutation(len(ordered))[:limit]
    return [ordered[index] for index in sorted(chosen)]


def asap_rows(dataset: str, limit: int, out_dir: str, candidates: str = "",
              seed: int = 0) -> list[dict]:
    """One row per recording for each input the planner can be given.

    `clean` is the performance MIDI as recorded. `decoded` is stage 3's kept
    notes, read from a candidates directory written by `corrector.classify
    predict`; without one, only the clean row is produced.
    """
    import csv

    import numpy as np

    from evaluation.folds import stem_for

    rows = []
    with open(os.path.join(dataset, "metadata.csv")) as handle:
        entries = [entry for entry in csv.DictReader(handle) if entry["midi_performance"]]
    for entry in select_entries(entries, limit, seed):
        stem = stem_for(entry["audio_performance"]) if entry["audio_performance"] else \
            os.path.splitext(os.path.basename(entry["midi_performance"]))[0]
        reference = os.path.join(dataset, os.path.dirname(entry["midi_performance"]),
                                 "xml_score.musicxml")
        annotations = os.path.join(dataset, entry["performance_annotations"])
        if not (os.path.exists(reference) and os.path.exists(annotations)):
            continue
        bpm, beats_per_bar = tempo_and_bar(annotations)

        clean = notes_from_midi(os.path.join(dataset, entry["midi_performance"]))
        estimate = os.path.join(out_dir, "scores", f"{stem}.clean.musicxml")
        write_planner_score(clean, bpm, beats_per_bar, estimate)
        rows.append({"label": "planner, clean MIDI", "stem": stem,
                     "reference": reference, "estimate": estimate})

        stored = os.path.join(candidates, f"{stem}.candidates.npz") if candidates else ""
        if stored and os.path.exists(stored):
            from corrector.classify import notes_from
            from evaluation.workbench import maps_for
            maps = maps_for(os.path.join(dataset, entry["audio_performance"]),
                            os.environ["MAPS_CACHE"])
            with np.load(stored) as held:
                cand = {key: held[key] for key in ("frame", "key", "time", "source")}
                probability = held["pass2"]
            decoded = notes_from(maps, cand, probability, 0.4, 0.3, 3)
            estimate = os.path.join(out_dir, "scores", f"{stem}.decoded.musicxml")
            write_planner_score(decoded, bpm, beats_per_bar, estimate)
            rows.append({"label": "planner, decoded notes", "stem": stem,
                         "reference": reference, "estimate": estimate})
    return rows


def test_rows(dataset: str, checkpoint: str, out_dir: str, bars: int | None = None,
              device: str = "cpu", seed: int = 0, corrupt_input: bool = False,
              per_piece: int = 0, reference_cache: str | None = None,
              split: str = "test", constrain_pitch: bool = False) -> list[dict]:
    """The held-out test split's table: two floors, the planner, and the model.

    Every estimate is scored against the NORMALISED reference (see
    `normalised_reference`), and every label carries `| seen` or `| unseen`
    according to whether the composer has any piece in training at all. Seven
    of ASAP's sixteen composers do not -- 27 of the 60 test performances and 7
    of the 16 test pieces -- so an average over the whole test set is an
    average of two different questions. The spec requires both rows.

    The two floor rows are per PIECE, not per performance: they do not depend
    on which performance of it was played.

      identity      the normalised reference against itself. MUSTER's own floor
                    is not zero -- the reference side converts through
                    MusicXMLToHMM and the estimate side through MusicXMLToTrx.
      round trip    the reference through this stage's own tokens and back. No
                    model can beat this; the gap between it and the model row
                    is the model's error, and the rest is the representation's.

    Every model row carries `decode`: `notation.predict.predict`'s window
    counts, plus `mismatched`, 1 when the performance fails the pairing rule
    training throws performances out by (`notation.dataset.examples`) and 0
    otherwise. Such a performance is still notated -- this table has always
    included them, and changing that would move every published figure -- but
    `bar_windows` stops at the score's last bar, so the model is never shown
    the rest: measured, 1,080 of Mozart K. 331/iii's 2,821 notes and 367 of
    Kreisleriana 1's 1,423 (an aligned checkpoint is given every note, so
    this applies to bar-window checkpoints only). The planner notates the
    whole performance. 3 of
    the 16 performances in the published test table (`eval-1b-plain`) are
    such, which is why it is counted per row.

    Every record in `skipped.jsonl` that cost rows names them under `lost`,
    `{label: rows}`, which `benchmark` folds into each line's `skipped`.
    """
    import csv

    import music21

    from evaluation.pairs import downbeat_map
    from notation.dataset import bar_numbers_of, bar_offset, splits, untrained_composers
    from notation.predict import decode_windows, load, window_bars

    assignment = splits(dataset, seed)
    untrained = untrained_composers(assignment)
    # A path, or a model already in hand. The second form is what lets the
    # seen/unseen labelling -- which the spec requires of every report of this
    # table -- be tested without a trained checkpoint.
    model = load(checkpoint, device) if isinstance(checkpoint, str) else checkpoint
    # The window size the checkpoint was trained on, unless told otherwise --
    # and told otherwise only in agreement with it. See `window_bars`.
    bars = window_bars(model, bars)

    with open(os.path.join(dataset, "metadata.csv")) as handle:
        entries = [entry for entry in csv.DictReader(handle) if entry["midi_performance"]]
    grouped: dict[str, list[dict]] = {}
    for entry in entries:
        key = f"{entry['composer']}/{entry['title']}"
        # `split` is normally "test". Scoring "train" answers a different
        # question: whether the model learned anything at all. A model that
        # cannot reproduce the data it was fitted on has not learned; one that
        # does well there and badly on test has memorised. Both sides go
        # through this same function so the numbers are comparable.
        if assignment.get(key) == split:
            grouped.setdefault(key, []).append(entry)

    if per_piece:
        # At most `per_piece` performances of each piece, sampled with `seed`.
        # Two reasons, both reported rather than hidden. Balakirev's Islamey
        # alone is 10 of the 27 unseen-composer performances and its estimate
        # takes nine minutes to decode, so the full 60 is hours and the
        # unseen row would be mostly one piece. Capping per piece weights the
        # pieces roughly equally and keeps every one of the sixteen. It is a
        # seeded permutation, never the first N: an alphabetical prefix of a
        # piece's performances is one pianist.
        import numpy as np

        for key, group in grouped.items():
            if len(group) > per_piece:
                order = np.random.default_rng(
                    seed + len(key)).permutation(len(group))[:per_piece]
                grouped[key] = [group[index] for index in sorted(order)]

    def whole_piece(side: str, performances: int) -> dict:
        # What losing the piece before any row was written costs: one row of
        # each floor, and one planner and one model row per performance.
        return {f"identity floor | {side}": 1, f"round trip floor | {side}": 1,
                f"planner | {side}": performances, f"model | {side}": performances}

    rows, skipped = [], []
    for key, group in sorted(grouped.items()):
        side = "unseen" if key.split("/", 1)[0] in untrained else "seen"
        stem = key.replace("/", "__")
        folder = os.path.join(dataset, os.path.dirname(group[0]["midi_performance"]))
        source = os.path.join(folder, "xml_score.musicxml")
        if not os.path.exists(source):
            skipped.append({"piece": key, "why": "no score",
                            "lost": whole_piece(side, len(group))})
            continue
        reference = os.path.join(out_dir, "scores", f"{stem}.reference.musicxml")
        try:
            score = music21.converter.parse(source)
            score_bars = bar_numbers_of(score)
        except Exception as failure:              # noqa: BLE001
            skipped.append({"piece": key, "why": str(failure)[:300],
                            "lost": whole_piece(side, len(group))})
            continue
        finally:
            score = None
        offset = bar_offset(os.path.join(folder, "midi_score_annotations.txt"))
        try:
            normalised_reference(source, reference, cache=reference_cache)
        except Exception as failure:              # noqa: BLE001
            # music21 can PARSE every ASAP source and cannot WRITE all of them
            # back: measured over the sixteen test pieces, Balakirev's Islamey,
            # Rachmaninoff's op. 32/10 and Ravel's Pavane refuse. Averaging a
            # raw-source row in with normalised ones would put a ~25-point
            # layout penalty on those three and nowhere else, so they keep the
            # raw source as their reference and are labelled `raw-ref`, in
            # their own block. Dropping them instead would cost 14 of the 60
            # test performances -- 10 of them Balakirev, the largest single
            # contributor to the unseen-composer row.
            skipped.append({"piece": key, "why": f"no normalised reference: {failure}"[:300]})
            reference = source
            side = f"{side} raw-ref"
        rows.append({"label": f"identity floor | {side}", "stem": stem,
                     "reference": reference, "estimate": reference})
        try:
            roundtrip = write_roundtrip_score(source, os.path.join(
                out_dir, "scores", f"{stem}.roundtrip.musicxml"))
        except Exception as failure:              # noqa: BLE001
            # The reference and model paths already record an unwritable score
            # and carry on; this one did not, so a single piece whose rebuild
            # music21 refuses to serialise killed the whole table. None of the
            # 16 test pieces happens to contain one; the 179 training pieces do,
            # so it only surfaced when the training split was first scored.
            # A missing floor loses one row, not the run.
            skipped.append({"piece": key, "row": "round trip floor",
                            "why": f"unwritable round trip: {failure}"[:300],
                            "lost": {f"round trip floor | {side}": 1}})
        else:
            rows.append({"label": f"round trip floor | {side}", "stem": stem,
                         "reference": reference, "estimate": roundtrip})

        for entry in group:
            performance = os.path.splitext(os.path.basename(entry["midi_performance"]))[0]
            name = f"{stem}.{performance}"
            annotations = os.path.join(dataset, entry["performance_annotations"])
            midi = os.path.join(dataset, entry["midi_performance"])
            if not (os.path.exists(annotations) and os.path.exists(midi)):
                skipped.append({"piece": key, "why": "no annotations or MIDI",
                                "lost": {f"planner | {side}": 1, f"model | {side}": 1}})
                continue
            # With velocity exactly when the checkpoint reads it:
            # `encode_aligned` refuses a velocity model notes that carry none.
            # `corrupt` and the planner both take the 4-tuples as they come.
            notes = notes_from_midi(midi, velocity=getattr(model.config, "velocity_input",
                                                           False))
            if corrupt_input:
                import hashlib

                import numpy as np

                from notation.dataset import corrupt
                # The condition the model was TRAINED for, and the condition it
                # will meet in the pipeline: stage 3's decoder loses a quarter
                # of the notes. Seeded per performance by a STABLE hash, so the
                # row is the same noise every run -- `hash()` of a string is
                # salted per process and would make this unreproducible.
                notes = corrupt(notes, np.random.default_rng(int(
                    hashlib.md5(name.encode()).hexdigest()[:8], 16)))

            bpm, beats_per_bar = tempo_and_bar(annotations)
            # The same anacrusis offset training cuts its windows with: a
            # score whose first downbeat is not its first bar has to skip that
            # bar on BOTH sides or the windows name the wrong bars. Cut the
            # way the checkpoint was trained: a legacy one (run4) gets exactly
            # the fixed windows it was always scored with; a `WINDOWING` 2 one
            # keeps the trailing part-window and can be halved on bar lines.
            windows = decode_windows(model, annotations, score_bars, offset, bars)
            # The rule `notation.dataset.examples` drops a performance by: its
            # downbeats and the score's bars disagree by more than the closing
            # downbeat, i.e. it plays a repeat the score does not write out.
            decode = {"mismatched": int(abs(len(downbeat_map(annotations))
                                            - (len(score_bars) - offset)) > 1)}
            written = {
                "planner": lambda path: write_planner_score(notes, bpm, beats_per_bar, path),
                "model": lambda path: write_model_score(model, notes, windows, path,
                                                         constrain_pitch, decode),
            }
            for label, write in written.items():
                path = os.path.join(out_dir, "scores", f"{name}.{label}.musicxml")
                try:
                    row = {"label": f"{label} | {side}", "stem": name,
                           "reference": reference, "estimate": write(path)}
                    if label == "model":
                        row["decode"] = decode
                    rows.append(row)
                except Exception as failure:      # noqa: BLE001
                    # music21's exporter refuses some streams outright
                    # ("inexpressible durations"). One unwritable estimate must
                    # cost that estimate, not the run -- and it is recorded,
                    # because a row averaged over whichever performances
                    # happened to survive is not a row.
                    skipped.append({"piece": key, "performance": name, "row": label,
                                    "why": str(failure)[:300],
                                    "lost": {f"{label} | {side}": 1}})

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "skipped.jsonl"), "w") as handle:
        import json
        for record in skipped:
            handle.write(json.dumps(record) + "\n")
    return rows


def main(argv=None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        prog="python -m evaluation.notation_eval",
        description="Phase 1: what the notation step scores today, by MUSTER.")
    parser.add_argument("--dataset", required=True, help="the asap-dataset checkout")
    parser.add_argument("--candidates", default="",
                        help="a candidates directory from corrector.classify predict; "
                             "without one only the clean-MIDI row is produced")
    parser.add_argument("--limit", type=int, default=20,
                        help="recordings to score; MUSTER is slow, so start small")
    parser.add_argument("--seed", type=int, default=0,
                        help="which sample of the corpus to score")
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkpoint", default="",
                        help="a notation model; with one, the table is the held-out "
                             "TEST split with its floors, not the phase-1 sample")
    parser.add_argument("--bars", type=int, default=None,
                        help="window size; defaults to the one the checkpoint was "
                             "trained with, and must agree with it if given")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--constrain-pitch", action="store_true",
                        help="decode only pitches the input contains, each at most as often")
    parser.add_argument("--corrupt-input", action="store_true",
                        help="notate the decoder's notes rather than the clean MIDI")
    parser.add_argument("--split", default="test", choices=("train", "validation", "test"),
                        help="which split to score; 'train' measures whether the "
                             "model learned its own training data at all")
    parser.add_argument("--reference-cache", default="",
                        help="directory of normalised references, shared between "
                             "runs; defaults to a sibling of --out so two runs "
                             "score against the same reference. Numbers from runs "
                             "that did not share one cannot be compared.")
    parser.add_argument("--per-piece", type=int, default=0,
                        help="score at most N performances of each test piece, sampled "
                             "with --seed; 0 scores every one")
    args = parser.parse_args(argv)

    if args.checkpoint:
        # A sibling of --out by default, so eval-run2 and eval-run3 share one
        # cache without being told to. A cache under --out would be per-run,
        # which is exactly the situation that made two runs incomparable.
        cache = args.reference_cache or os.path.join(
            os.path.dirname(os.path.abspath(args.out)), "reference-cache")
        rows = test_rows(args.dataset, args.checkpoint, args.out, args.bars,
                         args.device, args.seed, args.corrupt_input, args.per_piece,
                         reference_cache=cache, split=args.split,
                         constrain_pitch=args.constrain_pitch)
    else:
        rows = asap_rows(args.dataset, args.limit, args.out, args.candidates, args.seed)
    if not rows:
        raise ValueError("no recordings had both a score and annotations")
    skipped = None
    if args.checkpoint:
        # Written by `test_rows` just above, into this run's own --out.
        with open(os.path.join(args.out, "skipped.jsonl")) as handle:
            skipped = [json.loads(line) for line in handle if line.strip()]
    summary = benchmark(rows, args.out, skipped=skipped)
    if not any(line["recordings"] for line in summary):
        raise ValueError(f"every comparison failed; see {args.out}/failures.jsonl")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
