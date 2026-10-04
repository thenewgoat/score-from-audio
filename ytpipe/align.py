"""Stage 5: both candidate scores on the recording's clock, note against note.

The audio side needs no alignment. `transcribe` keeps the detector's notes in
seconds -- what the transcription heard, before it was quantised onto one
global tempo -- so they are already on the recording's clock. The quantised
MusicXML is kept as the written audio-side score, but it drifts against any
performance with rubato, so it is not what is compared or played.

The OMR side is in quarters and has to be placed. It is aligned to the
detected notes, not to the audio: both are symbolic, so their chroma is clean,
and the detected notes are precise to a frame. Dynamic time warping does the
coarse placement, constrained by when each page's systems were on screen --
a bar cannot sound while a different page is shown -- and then the matched
notes themselves refine the map, which chroma cannot do to the note.

The output, alignment.json, is one table of notes from either source, each
marked `agree`, `omr_only`, `audio_only` or `duration_differs`.
"""

import os

import numpy as np

from ytpipe import layout

HOP = 0.02                   # seconds per audio-side chroma column
TOLERANCE = 0.08             # seconds between onsets for two notes to be one
ROLL = 0.30                  # extra seconds a chord's notes may arrive late: a rolled chord
SCREEN_MARGIN = 3.0          # seconds a page's bars may sound outside its display
BLOCKED = 1e3                # DTW cost outside the display constraint


# --- the audio side ----------------------------------------------------------

def transcribe(video_id: str, root: str, force: bool = False, log=print) -> dict:
    """score-from-audio's pipeline, keeping the notes in seconds as well."""
    from transcribe.audio import build_score, detect_events, estimate_beats, plan
    from transcribe.detect import frames_for_tempo

    folder = layout.item_dir(root, video_id)
    out_json = os.path.join(folder, "transcribed.json")
    if os.path.exists(out_json) and not force:
        return layout.read_json(out_json)
    wav = os.path.join(folder, "audio.wav")
    tracked = estimate_beats(wav)
    bpm = float(tracked.tempo)
    if not np.isfinite(bpm):
        raise ValueError(f"beat tracking found too few beats in {wav}")
    beats_per_bar = tracked.beats_per_bar or 4
    events = detect_events(wav, min_note_frames=frames_for_tempo(bpm))
    if not events:
        raise ValueError(f"no notes detected in {wav}")
    build_score(plan(events, bpm), bpm, beats_per_bar).write(
        "musicxml", fp=os.path.join(folder, "transcribed.musicxml"))
    result = {"bpm": round(bpm, 2), "beats_per_bar": beats_per_bar,
              "beats": [round(float(b), 3) for b in tracked.beats],
              "notes": [{"pitch": int(e.pitch), "t_on": round(float(e.onset), 3),
                         "t_off": round(float(e.offset), 3)} for e in events],
              **layout.PROVENANCE}
    layout.write_json(out_json, result)
    log(f"{video_id}: transcribed {len(events)} notes at {bpm:.0f} bpm")
    return result


# --- the OMR side ------------------------------------------------------------

def omr_notes(path: str) -> list[dict]:
    """Every sounding note of the OMR score: one per pitch, ties joined.

    Position is measured in the part, so a hand padded to its bar's length (see
    `omr.join`) stays on the shared grid. Grace notes have no length and are
    left out; they would all land on the note they ornament.
    """
    import music21

    score = layout.parse_score(path)
    out = []
    for staff, part in enumerate(score.parts):
        hand = "RH" if staff == 0 else "LH"
        open_ties: dict[int, dict] = {}
        for element in part.recurse().notes:
            if element.duration.isGrace or element.quarterLength == 0:
                continue
            measure = element.getContextByClass("Measure")
            onset = float(element.getOffsetInHierarchy(part))
            length = float(element.quarterLength)
            for pitch in element.pitches:
                midi = int(pitch.midi)
                tie = element.tie if not element.isChord else next(
                    (n.tie for n in element.notes if n.pitch.midi == midi), None)
                kind = tie.type if tie is not None else None
                if kind in ("stop", "continue") and midi in open_ties:
                    open_ties[midi]["quarters_len"] += length
                    if kind == "stop":
                        del open_ties[midi]
                    continue
                at = float(element.getOffsetInHierarchy(measure)) if measure is not None else onset
                note = {"pitch": midi, "quarters": round(onset, 4),
                        "quarters_len": length, "hand": hand,
                        "measure": int(measure.number) if measure is not None else 0,
                        "at": round(at, 4)}
                out.append(note)
                if kind == "start":
                    open_ties[midi] = note
    for index, note in enumerate(sorted(out, key=lambda n: (n["quarters"], n["pitch"]))):
        note["ref"] = f"o{index}"
    return sorted(out, key=lambda n: (n["quarters"], n["pitch"]))


# --- chroma and DTW ----------------------------------------------------------

def _chroma(onsets, offsets, pitches, columns: int) -> np.ndarray:
    chroma = np.zeros((12, columns))
    for start, stop, pitch in zip(onsets, offsets, pitches):
        a = int(np.clip(np.floor(start), 0, columns - 1))
        b = int(np.clip(np.ceil(stop), a + 1, columns))
        pitch = int(pitch) % 12
        chroma[pitch, a:b] += 1.0
        chroma[pitch, a] += 2.0               # attacks weigh more than sustain
    norms = np.linalg.norm(chroma, axis=0, keepdims=True)
    out = np.divide(chroma, norms, out=np.zeros_like(chroma), where=norms > 0)
    out[:, (norms <= 0).ravel()] = 1.0 / np.sqrt(12.0)
    return out


def coarse_map(omr: list[dict], audio: list[dict], windows=None,
               step_penalty: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """(quarters, seconds) pairs placing the OMR score on the recording.

    `windows` is a list of (quarter_start, quarter_end, t_start, t_end): bars in
    that quarter range may only be placed in that time range. Cells outside
    are made prohibitively expensive rather than removed, so an impossible
    constraint degrades the path instead of breaking it.
    """
    import librosa

    end_seconds = max(n["t_off"] for n in audio) + 1.0
    columns = int(np.ceil(end_seconds / HOP))
    audio_chroma = _chroma([n["t_on"] / HOP for n in audio], [n["t_off"] / HOP for n in audio],
                           [n["pitch"] for n in audio], columns)
    span = max(n["quarters"] + n["quarters_len"] for n in omr)
    per_quarter = max(1.0, min(64.0, columns / span))
    rows = int(np.ceil(span * per_quarter))
    omr_chroma = _chroma([n["quarters"] * per_quarter for n in omr],
                         [(n["quarters"] + n["quarters_len"]) * per_quarter for n in omr],
                         [n["pitch"] for n in omr], rows)

    cost = 1.0 - omr_chroma.T @ audio_chroma                 # cosine distance
    for q0, q1, t0, t1 in windows or []:
        r0, r1 = int(q0 * per_quarter), int(np.ceil(q1 * per_quarter))
        c0 = max(0, int((t0 - SCREEN_MARGIN) / HOP))
        c1 = min(columns, int(np.ceil((t1 + SCREEN_MARGIN) / HOP)))
        cost[r0:r1, :c0] += BLOCKED
        cost[r0:r1, c1:] += BLOCKED
    _, path = librosa.sequence.dtw(C=cost, subseq=True, backtrack=True,
                                   weights_add=np.array([0.0, step_penalty, step_penalty]))
    path = path[::-1]
    frames = np.unique(path[:, 0])
    seconds = np.array([np.median(path[path[:, 0] == f, 1]) for f in frames]) * HOP
    return frames / per_quarter, seconds


def monotone(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop points that would make the map go backwards."""
    order = np.argsort(x)
    x, y = x[order], y[order]
    keep = [0]
    for i in range(1, len(x)):
        if x[i] > x[keep[-1]] and y[i] >= y[keep[-1]]:
            keep.append(i)
    return x[keep], y[keep]


# --- matching ----------------------------------------------------------------

def match(omr: list[dict], audio: list[dict], tolerance: float = TOLERANCE,
          roll: float = ROLL) -> list[tuple]:
    """(omr index, audio index) pairs: same pitch, onsets within tolerance.

    A note of a written chord (three or more notes struck together) may also
    sound up to `roll` seconds late: a rolled chord is written as one attack
    and played as a spread. Per pitch, an optimal assignment on onset
    distance, so two close repeated notes are not both claimed by one partner.
    """
    from collections import Counter

    from scipy.optimize import linear_sum_assignment

    struck = Counter(n["t_on"] for n in omr)
    pairs = []
    for pitch in {n["pitch"] for n in omr} & {n["pitch"] for n in audio}:
        a = [i for i, n in enumerate(omr) if n["pitch"] == pitch]
        b = [j for j, n in enumerate(audio) if n["pitch"] == pitch]
        late = np.array([[audio[j]["t_on"] - omr[i]["t_on"] for j in b] for i in a])
        reach = np.array([[tolerance + (roll if struck[omr[i]["t_on"]] >= 3 else 0.0)]
                          for i in a])
        allowed = (late >= -tolerance) & (late <= reach)
        gap = np.abs(late)
        rows, cols = linear_sum_assignment(np.where(allowed, gap, 1e6))
        pairs += [(a[r], b[c]) for r, c in zip(rows, cols) if allowed[r, c]]
    return pairs


def _pitch_classes(notes) -> set:
    return {n["pitch"] % 12 for n in notes}


def anchor_ends(omr: list[dict], audio: list[dict], quarters, seconds,
                window: float = 0.5, reach: float = 5.0):
    """Pin the score's first chord to the first note heard, and its last to the last.

    These videos play the sheet they show, so the music starts on its first
    chord. DTW does not know that: a subsequence match may start the score
    anywhere, and on one piece it put the opening chord 0.8 s after it was
    played, stranding everything before it outside the score. Each end is
    pinned only when the pitch classes heard there contain at least half the
    chord's, and when DTW had it within `reach` seconds anyway.
    """
    quarters, seconds = np.asarray(quarters, float), np.asarray(seconds, float)
    onsets = sorted({n["quarters"] for n in omr})
    heard = sorted(audio, key=lambda n: n["t_on"])
    if not onsets or not heard:
        return quarters, seconds

    def agrees(chord_q, around):
        written = _pitch_classes(n for n in omr if n["quarters"] == chord_q)
        played = _pitch_classes(n for n in heard if abs(n["t_on"] - around) <= window)
        return written and len(written & played) >= 0.5 * len(written)

    first_q, first_t = onsets[0], heard[0]["t_on"]
    if (agrees(first_q, first_t + window / 2)
            and abs(np.interp(first_q, quarters, seconds) - first_t) <= reach):
        keep = (quarters > first_q) & (seconds > first_t)
        quarters = np.concatenate([[first_q], quarters[keep]])
        seconds = np.concatenate([[first_t], seconds[keep]])

    last_q = onsets[-1]
    final = [n for n in heard if n["t_on"] >= heard[-1]["t_on"] - window]
    last_t = min(n["t_on"] for n in final)
    if (agrees(last_q, last_t + window / 2)
            and abs(np.interp(last_q, quarters, seconds) - last_t) <= reach):
        keep = (quarters < last_q) & (seconds < last_t)
        quarters = np.concatenate([quarters[keep], [last_q]])
        seconds = np.concatenate([seconds[keep], [last_t]])
    return quarters, seconds


def place(omr: list[dict], quarters: np.ndarray, seconds: np.ndarray) -> None:
    for note in omr:
        note["t_on"] = round(float(np.interp(note["quarters"], quarters, seconds)), 3)
        note["t_off"] = round(float(np.interp(note["quarters"] + note["quarters_len"],
                                              quarters, seconds)), 3)


def refine(omr: list[dict], audio: list[dict], quarters, seconds,
           rounds: int = 2, tolerance: float = TOLERANCE):
    """Re-fit the map through matched notes, widening the net first.

    The coarse map is only as good as a chroma frame; matched onsets are
    exact. Each round matches with a generous tolerance, fits the map through
    the matches (a monotone piecewise-linear map through the median time of
    each matched OMR onset), and places the OMR notes again.
    """
    for _ in range(rounds):
        place(omr, quarters, seconds)
        pairs = match(omr, audio, tolerance * 3)
        if len(pairs) < 8:
            break
        by_onset: dict[float, list[float]] = {}
        for i, j in pairs:
            by_onset.setdefault(omr[i]["quarters"], []).append(audio[j]["t_on"])
        # A chord's time is when it starts to sound -- the earliest of its
        # matches, not the middle of a roll.
        q = np.array(sorted(by_onset))
        t = np.array([min(by_onset[k]) if len(by_onset[k]) >= 3 else np.median(by_onset[k])
                      for k in q])
        # The pinned ends survive a refit that happens to match nothing there.
        ends = [(quarters[0], seconds[0]), (quarters[-1], seconds[-1])]
        for end_q, end_t in ends:
            if end_q not in by_onset:
                q, t = np.append(q, end_q), np.append(t, end_t)
        quarters, seconds = monotone(q, t)
    place(omr, quarters, seconds)
    return quarters, seconds


def table(omr: list[dict], audio: list[dict], tolerance: float = TOLERANCE) -> list[dict]:
    """One row per note heard or written, marked by which sources have it."""
    pairs = match(omr, audio, tolerance)
    paired_audio = {j: i for i, j in pairs}
    paired_omr = {i: j for i, j in pairs}
    rows = []
    for i, note in enumerate(omr):
        row = {"pitch": note["pitch"], "t_on": note["t_on"], "t_off": note["t_off"],
               "omr": note["ref"], "hand": note["hand"], "measure": note["measure"],
               "at": note.get("at", 0.0)}
        if i in paired_omr:
            heard = audio[paired_omr[i]]
            row["audio"] = f"a{paired_omr[i]}"
            written, played = note["t_off"] - note["t_on"], heard["t_off"] - heard["t_on"]
            row["status"] = ("duration_differs"
                             if abs(written - played) > max(0.25, 0.5 * written) else "agree")
            row["heard"] = [heard["t_on"], heard["t_off"]]
        else:
            row["status"] = "omr_only"
        rows.append(row)
    for j, note in enumerate(audio):
        if j not in paired_audio:
            rows.append({"pitch": note["pitch"], "t_on": note["t_on"], "t_off": note["t_off"],
                         "audio": f"a{j}", "status": "audio_only"})
    rows.sort(key=lambda r: (r["t_on"], r["pitch"]))
    for index, row in enumerate(rows):
        row["id"] = index
    mark_artefacts(rows)
    return rows


OVERTONES = (12, 19, 24, 28, 31)     # octave, twelfth, two octaves, and above


def mark_artefacts(rows: list[dict], slack: float = 0.05) -> None:
    """Label audio-only notes the transcription probably invented.

    `overtone`: an octave, twelfth or higher partial of a written note that is
    sounding at the time -- a piano string's overtones read as a note.
    `restrike`: the pitch of a written note still held -- one long note heard
    as two. Both could be real playing; with the OMR score as the main text
    they are left out unless someone keeps them, so the editor hides them.
    """
    written = [r for r in rows if "omr" in r]
    written.sort(key=lambda r: r.get("heard", [r["t_on"]])[0])
    for row in rows:
        if row["status"] != "audio_only":
            continue
        t = row["t_on"]
        live = [w for w in written
                if w.get("heard", [w["t_on"], w["t_off"]])[0] - slack <= t
                <= w.get("heard", [w["t_on"], w["t_off"]])[1] + slack]
        if any(row["pitch"] - w["pitch"] in OVERTONES for w in live):
            row["likely"] = "overtone"
        elif any(row["pitch"] == w["pitch"] for w in live):
            row["likely"] = "restrike"


# --- the stage ---------------------------------------------------------------

def screen_windows(folder: str, omr: list[dict]) -> list[tuple]:
    """(quarter range, time range) for each system, from when it was shown.

    A bar may only sound while its system is on screen (within
    SCREEN_MARGIN). Per system, not per page: a page covers half a minute,
    and within that span a piece whose harmony repeats -- a bossa's two
    chords -- offers DTW several equally good places, and it took the wrong
    one. Where Audiveris found a different number of systems on a page than
    the video showed, its system numbering cannot be trusted, and the page's
    whole span is used instead.
    """
    report_path = os.path.join(folder, "omr", "report.json")
    systems_path = os.path.join(folder, "omr", "systems.json")
    if not (os.path.exists(report_path) and os.path.exists(systems_path)):
        return []
    report = layout.read_json(report_path)
    systems = layout.read_json(systems_path)["systems"]
    measure_page = report.get("measure_page") or []
    measure_system = report.get("measure_system") or [None] * len(measure_page)
    groups = report.get("page_systems") or []
    starts: dict[int, float] = {}
    ends: dict[int, float] = {}
    for note in omr:
        starts[note["measure"]] = min(starts.get(note["measure"], 1e9), note["quarters"])
        ends[note["measure"]] = max(ends.get(note["measure"], 0.0),
                                    note["quarters"] + note["quarters_len"])
    windows = []
    for page, group in enumerate(groups, 1):
        bars = [m for m, p in enumerate(measure_page, 1) if p == page and m in starts]
        if not bars or not group:
            continue
        found = {measure_system[m - 1] for m in bars}
        if None not in found and len(found) == len(group) and max(found) == len(group) - 1:
            for k, index in enumerate(group):
                mine = [m for m in bars if measure_system[m - 1] == k]
                windows.append((min(starts[m] for m in mine), max(ends[m] for m in mine),
                                systems[index]["t_on"], systems[index]["t_off"]))
        else:
            windows.append((min(starts[m] for m in bars), max(ends[m] for m in bars),
                            systems[group[0]]["t_on"], systems[group[-1]]["t_off"]))
    return windows


GRACE_FLOOR = 3              # frames (~35 ms): short enough to keep a grace note
GRACE_LONGEST = 0.128        # seconds: longer notes are the main transcription's
GRACE_WINDOW = (-0.03, 0.12)  # seconds from the grace note to the note it ornaments
GRACE_STEP = 2               # semitones between them, at most


def grace_candidates(folder: str, rows: list[dict], start: int = 0) -> list[dict]:
    """Short notes heard just before a written note: possible grace notes.

    The main transcription drops notes shorter than about 128 ms, which is
    where grace notes live, so the detector's cached maps are decoded again
    with a ~35 ms floor and only the short notes kept. One is a candidate
    when it starts within GRACE_WINDOW of a written note no more than
    GRACE_STEP semitones away. Measured against 66 grace notes the owner wrote
    into one piece: about a quarter found, about a third of candidates real --
    a hint to check by ear, never applied unasked.
    """
    from evaluation.workbench import maps_for
    from transcribe.detect import decode_to_notes

    cache = os.path.join(folder, "maps")
    os.makedirs(cache, exist_ok=True)
    maps = maps_for(os.path.join(folder, "audio.wav"), cache)
    shorts = [d for d in decode_to_notes(maps, min_note_frames=GRACE_FLOOR)
              if d.offset - d.onset < GRACE_LONGEST]
    written = [r for r in rows if "omr" in r]
    lo, hi = GRACE_WINDOW
    out, taken = [], set()
    for short in sorted(shorts, key=lambda d: d.onset):
        targets = [w for w in written
                   if lo <= w.get("heard", [w["t_on"]])[0] - short.onset <= hi
                   and 0 < abs(w["pitch"] - int(short.pitch)) <= GRACE_STEP]
        if not targets:
            continue
        target = min(targets, key=lambda w: w.get("heard", [w["t_on"]])[0])
        if (target["id"], int(short.pitch)) in taken:
            continue
        taken.add((target["id"], int(short.pitch)))
        out.append({"id": start + len(out), "status": "grace_candidate", "pitch": int(short.pitch),
                    "t_on": round(float(short.onset), 3), "t_off": round(float(short.offset), 3),
                    "target": target["id"]})
    return out


def note_key(row: dict) -> tuple:
    """What a row is, independent of its position in the table.

    Decisions are stored by row id, and ids are positions: re-aligning
    renumbers them, which would silently move every decision onto another
    note. A written note is the pitch at a place in the OMR score; a heard
    note is its index in the transcription.
    """
    if "omr" in row:
        return ("omr", row.get("hand"), row.get("measure"), row.get("at"), row["pitch"])
    if row.get("status") == "grace_candidate":
        return ("grace", round(row["t_on"], 2), row["pitch"])
    return ("audio", row.get("audio"))


def carry_decisions(folder: str, old_path: str, rows: list[dict]):
    """Re-key decisions.json from the old table's ids to the new table's.

    Returns (carried, lost), or None when there were no decisions.
    """
    path = os.path.join(folder, "decisions.json")
    if not (os.path.exists(path) and os.path.exists(old_path)):
        return None
    state = layout.read_json(path)
    decided = state.get("decisions") or {}
    if not decided:
        return None
    old = {str(r["id"]): note_key(r) for r in layout.read_json(old_path)["notes"]}
    new = {note_key(r): str(r["id"]) for r in rows}
    moved = {new[old[k]]: v for k, v in decided.items() if k in old and old[k] in new}
    state["decisions"] = moved
    layout.write_json(path, state)
    return len(moved), len(decided) - len(moved)


def run(video_id: str, root: str, force: bool = False, log=print) -> dict:
    folder = layout.item_dir(root, video_id)
    out = os.path.join(folder, "alignment.json")
    if os.path.exists(out) and not force:
        return layout.read_json(out)
    score = layout.main_score(folder)
    if not os.path.exists(score):
        raise FileNotFoundError(f"{score} missing; run `ytpipe omr {video_id}` first")
    audio = transcribe(video_id, root, log=log)["notes"]
    omr = omr_notes(score)
    # The display windows are keyed to the OMR score's bars; an edited score
    # may number its bars differently, so it is aligned without them.
    edited = not score.endswith(os.path.join("omr", "score.musicxml"))
    quarters, seconds = coarse_map(omr, audio, None if edited else screen_windows(folder, omr))
    quarters, seconds = anchor_ends(omr, audio, quarters, seconds)
    quarters, seconds = refine(omr, audio, quarters, seconds)
    rows = table(omr, audio)
    try:
        rows += grace_candidates(folder, rows, start=len(rows))
    except Exception as error:                                  # noqa: BLE001
        log(f"{video_id}: no grace candidates ({type(error).__name__}: {error})")
    counts = {status: sum(1 for r in rows if r["status"] == status)
              for status in ("agree", "duration_differs", "omr_only", "audio_only", "grace_candidate")}
    result = {"tolerance": TOLERANCE, "counts": counts, "score": os.path.relpath(score, folder),
              "map": {"quarters": [round(float(q), 4) for q in quarters],
                      "seconds": [round(float(s), 3) for s in seconds]},
              "notes": rows, **layout.PROVENANCE}
    carried = carry_decisions(folder, out, rows)
    layout.write_json(out, result)
    if carried is not None:
        log(f"{video_id}: carried {carried[0]} decisions over, {carried[1]} lost (note gone)")
    agreed = counts["agree"] + counts["duration_differs"]
    log(f"{video_id}: {agreed} notes agree, {counts['omr_only']} OMR only, "
        f"{counts['audio_only']} audio only -> alignment.json")
    return result
