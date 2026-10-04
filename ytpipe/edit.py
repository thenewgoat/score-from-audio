"""The editor: listen to both candidates against the recording, decide per note.

A local page served from Python, like `evaluation/workbench.py`. It reads
alignment.json (stage 5) and keeps the owner's decisions in decisions.json,
written on every change, so closing the page loses nothing.

A note both sources have is kept unless dropped. A disagreement starts
undecided, and an undecided one follows the OMR score, which is the skeleton
the result is built on: an OMR-only note stays, an audio-only note stays out.

The result is written as two files:

  score.musicxml   the OMR score, with dropped notes removed and kept
                   audio-only notes written into the bar they sound in
  perf.mid         the kept notes on the recording's clock, at the times they
                   were actually played where the audio heard them

Anything that is not one note -- a wrong clef, a bar to rewrite -- is fixed in
MuseScore: "Open in MuseScore" hands it the result, "Reload" takes back what
was saved there. Reload replaces score.musicxml, and a later export from the
decisions would overwrite that, so the page asks first.
"""

import datetime
import json
import os
import re
import shutil
import subprocess
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from ytpipe import layout

MUSESCORE = os.environ.get("YTPIPE_MUSESCORE",
                           "/mnt/c/Program Files/MuseScore 4/bin/MuseScore4.exe")
DISAGREE = ("omr_only", "audio_only")
PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "editor.html")
GONE = (ConnectionResetError, BrokenPipeError, ConnectionAbortedError)


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


# --- decisions ---------------------------------------------------------------

def load_decisions(folder: str) -> dict:
    path = os.path.join(folder, "decisions.json")
    if os.path.exists(path):
        return layout.read_json(path)
    return {"decisions": {}, "edited_in_musescore": None}


def save_decisions(folder: str, state: dict) -> None:
    layout.write_json(os.path.join(folder, "decisions.json"),
                      {**state, "saved": _now(), **layout.PROVENANCE})


def kept(row: dict, decisions: dict) -> bool:
    """Whether a note is in the result. Undecided follows the OMR score."""
    choice = decisions.get(str(row["id"]))
    if choice in ("keep", "drop", "grace"):
        return choice != "drop"
    return row["status"] not in ("audio_only", "grace_candidate")


def heard(row: dict) -> tuple[float, float]:
    """When the note sounds: as played where the audio heard it, else as placed."""
    if "heard" in row:
        return row["heard"][0], row["heard"][1]
    return row["t_on"], row["t_off"]


# --- the result --------------------------------------------------------------

def _elements_at(measure, at: float):
    """Notes and chords starting `at` quarters into a measure, voices included."""
    for element in measure.recurse().notes:
        if abs(float(element.getOffsetInHierarchy(measure)) - at) < 1e-4:
            yield element


def drop_note(score, hand: str, number: int, at: float, pitch: int) -> bool:
    """Remove one pitch from the score, with any notes tied on from it."""
    from music21 import note

    part = score.parts[0 if hand == "RH" else 1]
    measures = {m.number: m for m in part.getElementsByClass("Measure")}
    measure = measures.get(number)
    if measure is None:
        return False
    for element in _elements_at(measure, at):
        if pitch not in [p.midi for p in element.pitches]:
            continue
        target = element
        while target is not None:
            follow = None
            if target.isChord:
                inner = next(n for n in target.notes if n.pitch.midi == pitch)
                tied = inner.tie is not None and inner.tie.type in ("start", "continue")
                if len(target.notes) > 1:
                    target.remove(inner)
                else:
                    target.activeSite.replace(target, note.Rest(quarterLength=target.quarterLength))
            else:
                tied = target.tie is not None and target.tie.type in ("start", "continue")
                target.activeSite.replace(target, note.Rest(quarterLength=target.quarterLength))
            if tied:
                follow = _next_with_pitch(part, target, pitch)
            target = follow
        return True
    return False


def _next_with_pitch(part, element, pitch: int):
    after = float(element.getOffsetInHierarchy(part)) + float(element.quarterLength)
    for candidate in part.recurse().notes:
        if (abs(float(candidate.getOffsetInHierarchy(part)) - after) < 1e-4
                and pitch in [p.midi for p in candidate.pitches]):
            return candidate
    return None


def _writable(quarters: float) -> float:
    from ytpipe.omr import PLAIN

    return next((w for w in PLAIN if w <= quarters + 1e-6), PLAIN[-1])


def _locate(part, quarters: float):
    """The measure `quarters` falls in, with its start and length."""
    for measure in part.getElementsByClass("Measure"):
        start = float(measure.getOffsetBySite(part))
        span = float(measure.duration.quarterLength)
        if start - 1e-6 <= quarters < start + span - 1e-6:
            return measure, start, span
    return None, 0.0, 0.0


def _added_voice(measure, at: float, length: float):
    """A voice of added notes with room for [at, at + length), made if need be.

    Existing notes are first moved into a voice of their own, so the measure
    holds voices only. Added notes go to the first added voice free over their
    span; one voice cannot hold two overlapping notes.
    """
    from music21 import stream

    if not measure.voices:
        first = stream.Voice(id="1")
        # Offsets are read before removal: a removed element's offset resets.
        for offset, element in [(e.offset, e) for e in measure.notesAndRests]:
            measure.remove(element)
            first.insert(offset, element)
        measure.insert(0, first)
    added = [v for v in measure.voices if str(v.id).startswith("yt")]
    for voice in added:
        if all(float(n.offset) + float(n.quarterLength) <= at + 1e-6
               or float(n.offset) >= at + length - 1e-6 for n in voice.notes):
            return voice
    voice = stream.Voice(id=f"yt{len(added) + 1}")
    measure.insert(0, voice)
    return voice


def insert_notes(score, items) -> dict:
    """Write notes into the bars they fall in, on a 16th grid.

    `items` are (key, pitch, quarters, length). A note joins a chord starting at
    the same place with the same length; otherwise it goes into an added voice,
    so the bar's own rhythm is untouched. Returns key -> the written note.
    """
    from music21 import chord, note

    placed = {}
    for key, pitch, quarters, length in sorted(items, key=lambda i: (i[2], i[1])):
        part = score.parts[0 if pitch >= 60 else 1]
        quarters = round(quarters * 4) / 4
        measure, start, span = _locate(part, quarters)
        if measure is None:
            continue
        at = quarters - start
        length = _writable(max(0.25, min(length, span - at)))
        written = None
        for element in _elements_at(measure, at):
            if abs(float(element.quarterLength) - length) > 1e-6:
                continue
            if pitch in [p.midi for p in element.pitches]:
                break
            if element.isChord:
                element.add(note.Note(pitch))
                written = next(n for n in element.notes if n.pitch.midi == pitch)
            else:
                old = note.Note(element.pitch, quarterLength=element.quarterLength)
                old.id = element.id
                merged = chord.Chord([old, note.Note(pitch)], quarterLength=element.quarterLength)
                element.activeSite.replace(element, merged)
                written = next(n for n in merged.notes if n.pitch.midi == pitch)
            break
        if written is None and not any(pitch in [p.midi for p in e.pitches]
                                       for e in _elements_at(measure, at)):
            written = note.Note(pitch, quarterLength=length)
            _added_voice(measure, at, length).insert(at, written)
        if written is not None:
            placed[key] = written
    return placed


def _audio_items(alignment: dict, rows) -> list[tuple]:
    quarters = np.array(alignment["map"]["quarters"])
    seconds = np.array(alignment["map"]["seconds"])
    items = []
    for row in rows:
        start = float(np.interp(row["t_on"], seconds, quarters))
        end = float(np.interp(row["t_off"], seconds, quarters))
        items.append((row["id"], row["pitch"], start, end - start))
    return items


def tag_note(score, row: dict, name: str) -> bool:
    """Give the OMR note a row describes an id, so its engraving can be found."""
    part = score.parts[0 if row["hand"] == "RH" else 1]
    measure = next((m for m in part.getElementsByClass("Measure")
                    if m.number == row["measure"]), None)
    if measure is None:
        return False
    for element in _elements_at(measure, row.get("at", 0.0)):
        if str(getattr(element.activeSite, "id", "")).startswith("yt"):
            continue
        for member in (element.notes if element.isChord else [element]):
            if member.pitch.midi == row["pitch"] and not str(member.id).startswith("row"):
                member.id = name
                return True
    return False


def build_review(folder: str, alignment: dict) -> str:
    """The OMR score for reading, each note's id naming its alignment row.

    Only the OMR score is engraved: it is the main text, and audio-only notes
    are flagged beside it by the page rather than written into it.
    """
    import music21

    out = os.path.join(folder, "review.musicxml")
    source = os.path.join(folder, "alignment.json")
    if (os.path.exists(out) and os.path.getmtime(out) >= os.path.getmtime(source)
            and os.path.getmtime(out) >= os.path.getmtime(__file__)
            and os.path.getmtime(out) >= os.path.getmtime(layout.main_score(folder))):
        return out
    score = layout.parse_score(layout.main_score(folder))
    layout.set_title(score, folder)
    for row in alignment["notes"]:
        if "omr" in row:
            tag_note(score, row, f"row{row['id']}")
    score.write("musicxml", fp=out, makeNotation=False)
    return out


def engraving(folder: str, alignment: dict) -> dict:
    """The review score engraved, every page, cached beside it."""
    review = build_review(folder, alignment)
    cache = os.path.join(folder, "review.svg.json")
    if os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(review):
        return layout.read_json(cache)
    import sys

    done = subprocess.run([sys.executable, "-m", "ytpipe.engrave", review],
                          capture_output=True, text=True, cwd=layout.REPO)
    if done.returncode != 0:
        detail = (done.stderr or "").strip().splitlines()
        raise ValueError(detail[-1] if detail else "engraving failed")
    result = json.loads(done.stdout)
    layout.write_json(cache, result)
    return result


def build_result(folder: str, alignment: dict, decisions: dict) -> dict:
    """score.musicxml and perf.mid from the main score and the decisions.

    The main score is the MuseScore edit once there is one, so an export
    builds on it rather than replacing it.
    """
    score = layout.parse_score(layout.main_score(folder))
    layout.set_title(score, folder)
    dropped = 0
    for row in alignment["notes"]:
        if "omr" in row and not kept(row, decisions):
            dropped += drop_note(score, row["hand"], row["measure"], row.get("at", 0.0),
                                 row["pitch"])
    wanted = [r for r in alignment["notes"] if r["status"] == "audio_only" and kept(r, decisions)]
    added = len(insert_notes(score, _audio_items(alignment, wanted)))
    by_id = {r["id"]: r for r in alignment["notes"]}
    graces = 0
    for row in alignment["notes"]:
        if row["status"] == "grace_candidate" and kept(row, decisions):
            graces += add_grace(score, by_id[row["target"]], row["pitch"])
    score.write("musicxml", fp=os.path.join(folder, "score.musicxml"), makeNotation=False)
    notes = [(*heard(row), row["pitch"]) for row in alignment["notes"] if kept(row, decisions)]
    write_midi(notes, os.path.join(folder, "perf.mid"))
    return {"dropped": dropped, "added": added, "graces": graces, "notes": len(notes)}


def add_grace(score, target: dict, pitch: int) -> bool:
    """Write a grace note before the written note `target` describes."""
    from music21 import note

    part = score.parts[0 if target["hand"] == "RH" else 1]
    measure = next((m for m in part.getElementsByClass("Measure")
                    if m.number == target["measure"]), None)
    if measure is None:
        return False
    for element in _elements_at(measure, target.get("at", 0.0)):
        if target["pitch"] in [p.midi for p in element.pitches] and not element.duration.isGrace:
            grace = note.Note(pitch, type="eighth").getGrace()
            grace.duration.slash = True
            site = element.activeSite
            site.insert(element.getOffsetBySite(site), grace)
            return True
    return False


def write_midi(notes: list[tuple], path: str, ticks_per_second: int = 960) -> None:
    """(onset s, offset s, pitch) as a MIDI file at 120 bpm, 480 ticks a beat."""
    import mido

    events = []
    for onset, offset, pitch in notes:
        events.append((round(onset * ticks_per_second), 1, pitch))
        events.append((round(max(offset, onset + 0.03) * ticks_per_second), 0, pitch))
    events.sort()
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))
    clock = 0
    for tick, on, pitch in events:
        track.append(mido.Message("note_on" if on else "note_off", note=int(pitch),
                                  velocity=80 if on else 0, time=tick - clock))
        clock = tick
    midi = mido.MidiFile(ticks_per_beat=480)
    midi.tracks.append(track)
    midi.save(path)


# --- MuseScore ---------------------------------------------------------------

def _windows_env(name: str) -> str:
    done = subprocess.run(["cmd.exe", "/c", f"echo %{name}%"], capture_output=True, text=True,
                          cwd="/mnt/c")
    value = done.stdout.strip().splitlines()[-1]
    return subprocess.run(["wslpath", "-u", value], capture_output=True, text=True).stdout.strip()


def scores_dir() -> str:
    """Where scores are handed to MuseScore and saved from it: the folder the
    sheet PDFs go to, Documents\\ytpipe sheets, unless YTPIPE_SCORES says else.
    Not the Windows temp folder, which Windows may clear."""
    return os.environ.get("YTPIPE_SCORES") or os.path.join(
        _windows_env("USERPROFILE"), "Documents", "ytpipe sheets")


def musescore_stage(video_id: str) -> str:
    """Scratch space for converting a save back to MusicXML; nothing kept."""
    from ytpipe.omr import windows_temp

    return os.path.join(windows_temp(), "ytpipe-edit", video_id)


def _safe(title: str) -> str:
    return re.sub(r'[\\/:*?"<>|]', "", title)


def open_in_musescore(folder: str, video_id: str) -> str:
    from ytpipe.omr import to_windows

    target_dir = scores_dir()
    os.makedirs(target_dir, exist_ok=True)
    # Named for the piece, with the id in brackets: MuseScore offers the
    # piece's name when saving, and either one traces a save to its video.
    target = os.path.join(target_dir, f"{_safe(layout.piece_title(folder))} [{video_id}].musicxml")
    shutil.copyfile(os.path.join(folder, "score.musicxml"), target)
    subprocess.Popen([MUSESCORE, to_windows(target)], stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, start_new_session=True)
    return target


def saved_in_musescore(video_id: str, since: float, title: str = "") -> str | None:
    """The newest file MuseScore saved for this item after `since`.

    Looked for in the scores folder (see `scores_dir`) and in MuseScore's own
    Scores folder, where it may offer to save. Both hold every piece, so a file
    counts only if it carries the id in brackets, as the handed-over name does,
    or is named after the piece, as MuseScore's suggested name is.
    """
    wanted = (".mscz", ".musicxml", ".mxl")
    places = []
    for place in (lambda: scores_dir(),
                  lambda: os.path.join(_windows_env("USERPROFILE"), "Documents",
                                       "MuseScore4", "Scores")):
        try:
            places.append(place())
        except Exception:                                       # noqa: BLE001
            pass
    found = []
    for place in places:
        if not os.path.isdir(place):
            continue
        for name in os.listdir(place):
            stem, extension = os.path.splitext(name)
            ours = (f"[{video_id}]" in stem or stem == video_id
                    or (title and stem in (title, _safe(title))))
            path = os.path.join(place, name)
            if ours and extension in wanted and os.path.getmtime(path) > since + 1:
                found.append(path)
    return max(found, key=os.path.getmtime) if found else None


def reload_from_musescore(folder: str, video_id: str, since: float) -> str:
    from ytpipe.omr import to_windows

    source = saved_in_musescore(video_id, since, layout.piece_title(folder))
    if source is None:
        raise FileNotFoundError("nothing saved in MuseScore since it was opened; save the "
                                "score there (or export it as MusicXML) and reload again")
    # Kept as edited.musicxml, which becomes the piece's main score: the
    # editor engraves it and the piece is re-aligned against it.
    edited = os.path.join(folder, "edited.musicxml")
    if source.endswith(".musicxml"):
        shutil.copyfile(source, edited)
        return source
    os.makedirs(musescore_stage(video_id), exist_ok=True)
    converted = os.path.join(musescore_stage(video_id), f"{video_id}.reload.musicxml")
    subprocess.run([MUSESCORE, "-o", to_windows(converted), to_windows(source)],
                   capture_output=True, check=True)
    shutil.copyfile(converted, edited)
    return source


# --- the server --------------------------------------------------------------

def items(root: str) -> list[dict]:
    out = []
    for name in sorted(os.listdir(root)):
        path = layout.item_path(root, name, "alignment.json")
        if os.path.exists(path):
            meta_path = layout.item_path(root, name, "meta.json")
            title = layout.read_json(meta_path).get("title") if os.path.exists(meta_path) else name
            out.append({"id": name, "title": title})
    return out


class Handler(BaseHTTPRequestHandler):
    root = layout.DEFAULT_ROOT

    def log_message(self, *args):
        pass

    def _send(self, body: bytes, content_type: str, status: int = 200, extra=()):
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for key, value in extra:
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)
        except GONE:
            pass

    def _json(self, payload, status: int = 200):
        self._send(json.dumps(payload).encode(), "application/json", status)

    def _item(self, query) -> tuple[str, str]:
        video_id = os.path.basename(query.get("id", [""])[0])
        folder = layout.item_dir(self.root, video_id)
        if not video_id or not os.path.exists(os.path.join(folder, "alignment.json")):
            raise FileNotFoundError(f"no aligned item {video_id!r}")
        return video_id, folder

    def _file(self, path: str, content_type: str):
        """A file, honouring a single byte range so audio can be seeked."""
        size = os.path.getsize(path)
        header = self.headers.get("Range")
        start, end = 0, size - 1
        if header and header.startswith("bytes="):
            first, _, last = header[6:].partition("-")
            start = int(first) if first else size - int(last)
            end = int(last) if first and last else size - 1
        length = end - start + 1
        try:
            self.send_response(206 if header else 200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(length))
            self.send_header("Accept-Ranges", "bytes")
            if header:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with open(path, "rb") as handle:
                handle.seek(start)
                while length > 0:
                    chunk = handle.read(min(256 * 1024, length))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    length -= len(chunk)
        except GONE:
            pass

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed.query)
        try:
            if parsed.path == "/":
                with open(PAGE, "rb") as handle:
                    return self._send(handle.read(), "text/html; charset=utf-8")
            if parsed.path == "/api/items":
                return self._json(items(self.root))
            if parsed.path == "/api/item":
                video_id, folder = self._item(query)
                alignment = layout.read_json(os.path.join(folder, "alignment.json"))
                systems_path = os.path.join(folder, "omr", "systems.json")
                systems = (layout.read_json(systems_path)["systems"]
                           if os.path.exists(systems_path) else [])
                meta = layout.read_json(os.path.join(folder, "meta.json"))
                return self._json({"id": video_id, "title": meta.get("title"),
                                   "duration": meta.get("duration"),
                                   "counts": alignment["counts"], "notes": alignment["notes"],
                                   "systems": systems, **load_decisions(folder)})
            if parsed.path == "/api/engraving":
                _, folder = self._item(query)
                alignment = layout.read_json(os.path.join(folder, "alignment.json"))
                try:
                    return self._json(engraving(folder, alignment))
                except ValueError as error:
                    return self._json({"error": str(error)}, 500)
            if parsed.path == "/audio":
                _, folder = self._item(query)
                return self._file(os.path.join(folder, "audio.wav"), "audio/wav")
            if parsed.path == "/sheet.pdf":
                from ytpipe.sheet import build

                _, folder = self._item(query)
                return self._file(build(folder), "application/pdf")
            if parsed.path == "/system":
                _, folder = self._item(query)
                name = os.path.basename(query.get("file", [""])[0])
                return self._file(os.path.join(folder, "omr", "systems", name), "image/png")
            return self._send(b"not found", "text/plain", 404)
        except FileNotFoundError as error:
            return self._json({"error": str(error)}, 404)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed.query)
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            body = {}
        try:
            video_id, folder = self._item(query)
            state = load_decisions(folder)
            if parsed.path == "/api/decide":
                for key, choice in (body.get("decisions") or {}).items():
                    if choice in ("keep", "drop", "grace"):
                        state["decisions"][str(key)] = choice
                    else:
                        state["decisions"].pop(str(key), None)
                save_decisions(folder, state)
                return self._json({"ok": True, "decided": len(state["decisions"])})
            if parsed.path == "/api/export":
                result = build_result(folder, layout.read_json(
                    os.path.join(folder, "alignment.json")), state["decisions"])
                return self._json({"ok": True, **result})
            if parsed.path == "/api/musescore/open":
                if not os.path.exists(os.path.join(folder, "score.musicxml")):
                    build_result(folder, layout.read_json(
                        os.path.join(folder, "alignment.json")), state["decisions"])
                target = open_in_musescore(folder, video_id)
                state["opened_in_musescore"] = os.path.getmtime(target)
                save_decisions(folder, state)
                return self._json({"ok": True, "path": target})
            if parsed.path == "/api/musescore/reload":
                since = state.get("opened_in_musescore")
                if since is None:
                    return self._json({"error": "open it in MuseScore from this page first"}, 409)
                source = reload_from_musescore(folder, video_id, since)
                state["edited_in_musescore"] = _now()
                save_decisions(folder, state)
                from ytpipe import align
                aligned = align.run(video_id, self.root, force=True, log=lambda *_: None)
                return self._json({"ok": True, "from": source, "counts": aligned["counts"]})
            return self._json({"error": "unknown"}, 404)
        except FileNotFoundError as error:
            return self._json({"error": str(error)}, 404)
        except Exception as error:                              # noqa: BLE001
            return self._json({"error": f"{type(error).__name__}: {error}"}, 500)


def serve(root: str, video_id: str | None = None, port: int = 8765) -> None:
    Handler.root = root
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    where = f"http://localhost:{port}/" + (f"?id={video_id}" if video_id else "")
    print(f"editor at {where}  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
