"""Stage 6: finished pieces as an ASAP-shaped dataset. INTERNAL USE ONLY.

A piece is finished when its owner has saved it in MuseScore: the .mscz that
carries its id is the score, corrected by hand. The notation model trains on
ASAP's layout (`notation.dataset.aligned_examples`), so each piece is written
in exactly that shape and the training code is not touched:

    <out>/ytpipe/<id>/xml_score.musicxml            the MuseScore save, as MusicXML
                      performance.mid                Transkun's notes, with velocity
                      performance_annotations.txt    beats on the recording's clock
                      midi_score_annotations.txt     beats on the score's own clock
    <out>/metadata.csv                               ASAP's columns, one row per piece

The performance is Transkun's transcription of the recording, not the notes
the editor kept: it is what the notation model is given in use, so it is what
it should learn from. Transkun runs in its own environment, so its MIDI is an
input here (`--performances`, one `<id>.mid` per piece), not a step.

The beats come from placing the score on the recording, the way stage 5 does
(`ytpipe.align`: chroma DTW, the ends pinned, refined through matched notes),
here against Transkun's notes. Each bar line and beat is read off that map.
ASAP's conventions are kept: a `db` at every bar line that starts a FULL bar,
so a score opening with an anacrusis lists that bar's beats as `b` first
(`notation.dataset.bar_offset` reads that), and a closing `db` where the last
bar ends, so the last bar has both of its lines.

`combine` makes the corpus a run trains on: ASAP's folders linked in, these
pieces beside them, one metadata.csv over both, and a split file that is the
given one plus these pieces.
"""

import csv
import json
import os
import re
import shutil
import subprocess

import numpy as np

from ytpipe import layout

ASAP_COLUMNS = ("composer", "title", "folder", "xml_score", "midi_score", "midi_performance",
                "performance_annotations", "midi_score_annotations",
                "maestro_midi_performance", "maestro_audio_performance", "start", "end",
                "audio_performance")
COMPOSER = "ytpipe"
SCORE_SECONDS_PER_QUARTER = 0.5     # the score-side clock: a nominal 120 bpm
EXTRAPOLATE_QUARTERS = 8.0          # tempo past the map's ends: its last 8 quarters


def finished(root: str) -> list[tuple[str, str]]:
    """`(video_id, MuseScore save)` for every piece saved in MuseScore."""
    from ytpipe.edit import saved_in_musescore

    out = []
    for video_id in sorted(os.listdir(root)):
        folder = layout.item_dir(root, video_id)
        if not os.path.exists(os.path.join(folder, "meta.json")):
            continue
        source = saved_in_musescore(video_id, 0.0, layout.piece_title(folder))
        if source is not None and source.endswith(".mscz"):
            out.append((video_id, source))
    return out


def fixed_rests(text: str) -> str:
    """`<rest measure="yes">` as a plain rest: see `layout.parse_score` for why.

    The training code parses the file with plain music21, so the fix is
    written into the file rather than applied on reading.
    """
    text = re.sub(r'<rest\s+measure="yes"\s*/>', "<rest/>", text)
    return re.sub(r'<rest\s+measure="yes"\s*>', "<rest>", text)


def musescore_to_musicxml(source: str, target: str) -> None:
    from ytpipe.edit import MUSESCORE
    from ytpipe.omr import to_windows

    staged = target + ".musescore.musicxml"
    subprocess.run([MUSESCORE, "-o", to_windows(staged), to_windows(source)],
                   capture_output=True, check=True)
    with open(staged, encoding="utf-8") as handle:
        text = fixed_rests(handle.read())
    os.remove(staged)
    with open(target, "w", encoding="utf-8") as handle:
        handle.write(text)


def bar_grid(score) -> list[dict]:
    """One entry per bar of the top staff: where it starts, how long, its beats.

    Positions are quarters from the start of the part, as `align.omr_notes`
    measures them. A beat is the time signature's own (`beatDuration`: a
    dotted quarter in 6/8). In an anacrusis the beats are counted back from
    the bar line, so they fall where the full bar's would.
    """
    import music21

    part = score.parts[0]
    bars, signature = [], music21.meter.TimeSignature("4/4")
    for measure in part.getElementsByClass(music21.stream.Measure):
        signature = measure.timeSignature or signature
        start = float(measure.getOffsetBySite(part))
        nominal = float(signature.barDuration.quarterLength)
        length = float(measure.duration.quarterLength) or nominal
        beat = float(signature.beatDuration.quarterLength)
        pickup = not bars and length < nominal - 1e-6
        if pickup:
            beats = sorted(start + length - k * beat
                           for k in range(1, int(np.ceil(length / beat - 1e-6)) + 1)
                           if length - k * beat >= -1e-6)
        else:
            beats = [start + k * beat for k in range(int(np.ceil(length / beat - 1e-6)))]
        bars.append({"start": start, "length": length, "pickup": pickup, "beats": beats,
                     "signature": f"{signature.numerator}/{signature.denominator}"})
    return bars


def clock(quarters, seconds):
    """Quarters -> seconds along the map, extended linearly past both ends.

    `np.interp` clamps, which would put everything before the first matched
    chord at the same instant; past each end the map keeps its tempo over
    the `EXTRAPOLATE_QUARTERS` nearest it.
    """
    quarters, seconds = np.asarray(quarters, float), np.asarray(seconds, float)

    def slope(at_start: bool) -> float:
        if at_start:
            far = min(np.searchsorted(quarters, quarters[0] + EXTRAPOLATE_QUARTERS),
                      len(quarters) - 1)
            a, b = 0, far
        else:
            far = max(np.searchsorted(quarters, quarters[-1] - EXTRAPOLATE_QUARTERS) - 1, 0)
            a, b = far, len(quarters) - 1
        if b <= a or quarters[b] <= quarters[a]:
            return SCORE_SECONDS_PER_QUARTER
        return float((seconds[b] - seconds[a]) / (quarters[b] - quarters[a]))

    early, late = slope(True), slope(False)

    def to_seconds(q: float) -> float:
        if q < quarters[0]:
            return float(seconds[0] - (quarters[0] - q) * early)
        if q > quarters[-1]:
            return float(seconds[-1] + (q - quarters[-1]) * late)
        return float(np.interp(q, quarters, seconds))
    return to_seconds


def annotation_lines(bars: list[dict], to_seconds) -> list[str]:
    """ASAP's annotation format: `time<TAB>time<TAB>label`, one line per beat."""
    lines, signature = [], None
    for bar in bars:
        for index, beat in enumerate(bar["beats"]):
            downbeat = index == 0 and not bar["pickup"]
            label = "b"
            if downbeat:
                label = "db" if bar["signature"] == signature else f"db,{bar['signature']}"
                signature = bar["signature"]
            t = round(to_seconds(beat), 6)
            lines.append(f"{t}\t{t}\t{label}")
    if bars:
        end = round(to_seconds(bars[-1]["start"] + bars[-1]["length"]), 6)
        lines.append(f"{end}\t{end}\tdb")
    return lines


def time_map(score_path: str, played: list[tuple]) -> tuple[np.ndarray, np.ndarray, dict]:
    """The score placed on the recording, as stage 5 places it, and how well."""
    from ytpipe.align import anchor_ends, coarse_map, match, omr_notes, refine

    written = omr_notes(score_path)
    heard = [{"pitch": int(p), "t_on": float(on), "t_off": float(off)}
             for on, off, p, *_ in played]
    quarters, seconds = coarse_map(written, heard, None)
    quarters, seconds = anchor_ends(written, heard, quarters, seconds)
    quarters, seconds = refine(written, heard, quarters, seconds)
    matched = len(match(written, heard))
    return quarters, seconds, {"written": len(written), "heard": len(heard),
                               "matched": matched}


def export_piece(video_id: str, source: str, performance: str, root: str, out: str) -> dict:
    from evaluation.notation_eval import notes_from_midi

    folder = os.path.join(out, COMPOSER, video_id)
    os.makedirs(folder, exist_ok=True)
    score_path = os.path.join(folder, "xml_score.musicxml")
    musescore_to_musicxml(source, score_path)
    shutil.copyfile(performance, os.path.join(folder, "performance.mid"))

    played = notes_from_midi(performance)
    quarters, seconds, stats = time_map(score_path, played)
    bars = bar_grid(layout.parse_score(score_path))
    with open(os.path.join(folder, "performance_annotations.txt"), "w") as handle:
        handle.write("\n".join(annotation_lines(bars, clock(quarters, seconds))) + "\n")
    with open(os.path.join(folder, "midi_score_annotations.txt"), "w") as handle:
        handle.write("\n".join(annotation_lines(
            bars, lambda q: q * SCORE_SECONDS_PER_QUARTER)) + "\n")

    title = re.sub(r"[^A-Za-z0-9]+", "_", layout.piece_title(layout.item_dir(root, video_id)))
    relative = f"{COMPOSER}/{video_id}"
    row = {name: "" for name in ASAP_COLUMNS}
    row.update(composer=COMPOSER, title=f"{title.strip('_')}_{video_id}", folder=relative,
               xml_score=f"{relative}/xml_score.musicxml",
               midi_performance=f"{relative}/performance.mid",
               performance_annotations=f"{relative}/performance_annotations.txt",
               midi_score_annotations=f"{relative}/midi_score_annotations.txt")
    return {"row": row, "source": source, "bars": len(bars),
            "pickup": bool(bars and bars[0]["pickup"]), **stats}


def export(root: str, out: str, performances: str, only=(), log=print) -> list[dict]:
    """Every finished piece (or `only` these ids) into `out`; returns their reports."""
    pieces = [(v, s) for v, s in finished(root) if not only or v in only]
    reports = []
    for video_id, source in pieces:
        performance = os.path.join(performances, f"{video_id}.mid")
        if not os.path.exists(performance):
            log(f"{video_id}: no {performance}; run Transkun on its audio.wav first")
            continue
        report = export_piece(video_id, source, performance, root, out)
        reports.append(report)
        log(f"{video_id}: {report['bars']} bars, {report['matched']} of "
            f"{report['written']} written notes matched to {report['heard']} heard")
    with open(os.path.join(out, "metadata.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ASAP_COLUMNS)
        writer.writeheader()
        writer.writerows(r["row"] for r in reports)
    layout.write_json(os.path.join(out, "export.json"),
                      {"pieces": reports, **layout.PROVENANCE})
    return reports


def combine(asap: str, extra: str, out: str, split_file: str, test=(),
            side: str = "train") -> str:
    """ASAP and `extra` as one dataset in `out`; returns the combined split file.

    ASAP's top-level entries are symlinked, not copied, and `extra`'s pieces
    are linked in beside them. Each extra piece goes to `side`, or to `test`
    if its id is listed there. Nothing is written inside either source.
    """
    os.makedirs(out, exist_ok=True)
    for name in os.listdir(asap):
        if name == "metadata.csv":
            continue
        link = os.path.join(out, name)
        if not os.path.lexists(link):
            os.symlink(os.path.join(os.path.abspath(asap), name), link)
    link = os.path.join(out, COMPOSER)
    if os.path.lexists(link):
        os.remove(link)
    os.symlink(os.path.join(os.path.abspath(extra), COMPOSER), link)

    with open(os.path.join(asap, "metadata.csv"), newline="") as handle:
        reader = csv.DictReader(handle)
        columns, rows = reader.fieldnames, list(reader)
    with open(os.path.join(extra, "metadata.csv"), newline="") as handle:
        added = list(csv.DictReader(handle))
    with open(os.path.join(out, "metadata.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows + [{c: r.get(c, "") for c in columns} for r in added])

    with open(split_file) as handle:
        split = json.load(handle)
    for row in added:
        video_id = row["folder"].split("/")[-1]
        split[row["midi_performance"]] = "test" if video_id in test else side
    target = os.path.join(out, "split.json")
    with open(target, "w") as handle:
        json.dump(split, handle, indent=1, sort_keys=True)
        handle.write("\n")
    return target
