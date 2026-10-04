"""The YouTube pipeline's offline parts: manifests, meta, roi.json, resumability.

Nothing here touches the network. yt-dlp and ffmpeg are replaced by a fake
runner that writes the files they would have written and records the calls,
so skipping is checked by counting what was run.
"""

import json
import os

import pytest

from ytpipe import fetch, layout, roi

FLAT = {
    "_type": "playlist", "id": "PLabc", "title": "Piano covers",
    "webpage_url": "https://www.youtube.com/playlist?list=PLabc",
    "entries": [
        {"id": "vid00000001", "title": "One", "url": "https://www.youtube.com/watch?v=vid00000001",
         "duration": 120.0, "channel": "Someone"},
        None,
        {"id": None, "title": "[Private video]"},
        {"id": "vid00000002", "title": "Two", "duration": 95.5, "uploader": "Else"},
    ],
}

INFO = {"id": "vid00000001", "title": "One", "duration": 120.0, "channel": "Someone",
        "formats": [{"format_id": "137"}] * 50, "width": 1920, "height": 1080}


class FakeRunner:
    """Stands in for `fetch.run`: writes what yt-dlp/ffmpeg would, records the call."""

    def __init__(self):
        self.calls = []

    def __call__(self, command):
        self.calls.append(command)
        if command[0] == "ffmpeg":
            open(command[-1], "wb").write(b"RIFF")
            return ""
        args = command[3:]
        if "--flat-playlist" in args:
            return json.dumps(FLAT)
        if "--skip-download" in args:
            return json.dumps(INFO)
        out = args[args.index("-o") + 1]
        open(out.replace("%(ext)s", "webm"), "wb").write(b"media")
        return ""

    def count(self, word):
        return sum(1 for c in self.calls if word in " ".join(c))


def test_flat_playlist_drops_entries_without_ids_and_fills_urls():
    manifest = fetch.parse_flat_playlist(FLAT, "https://x/playlist")
    assert [e["id"] for e in manifest["entries"]] == ["vid00000001", "vid00000002"]
    assert manifest["entries"][1]["url"] == "https://www.youtube.com/watch?v=vid00000002"
    assert manifest["entries"][1]["channel"] == "Else"
    assert manifest["entries"][1]["index"] == 4
    assert manifest["source"] == "youtube" and manifest["internal_only"] is True


def test_a_single_video_becomes_a_one_entry_playlist():
    manifest = fetch.parse_flat_playlist({"id": "vid00000009", "title": "Solo",
                                          "webpage_url": "https://y/watch?v=vid00000009"})
    assert [e["id"] for e in manifest["entries"]] == ["vid00000009"]


def test_meta_keeps_the_useful_fields_and_is_stamped_internal():
    meta = fetch.build_meta(INFO)
    assert meta["source"] == "youtube" and meta["internal_only"] is True
    assert meta["duration"] == 120.0 and meta["width"] == 1920
    assert "formats" not in meta


def test_fetch_writes_the_layout_and_a_second_run_downloads_nothing(tmp_path):
    root = str(tmp_path)
    runner = FakeRunner()
    result = fetch.fetch("https://x/playlist", root, sleep=(0, 0), runner=runner, log=lambda *_: None)
    assert result["done"] == ["vid00000001", "vid00000002"] and not result["failed"]

    manifest = layout.read_json(layout.playlist_path(root, "PLabc"))
    assert manifest["internal_only"] is True
    for video in result["done"]:
        folder = tmp_path / video
        assert sorted(os.listdir(folder)) == ["audio.wav", "meta.json", "video.mp4"]
        assert json.load(open(folder / "meta.json"))["internal_only"] is True
    assert runner.count("-f bestaudio") == 2 and runner.count("ffmpeg") == 2

    again = FakeRunner()
    fetch.fetch("https://x/playlist", root, sleep=(0, 0), runner=again, log=lambda *_: None)
    assert [c for c in again.calls if "--flat-playlist" not in c] == []


def test_fetch_passes_pacing_and_separate_archives(tmp_path):
    runner = FakeRunner()
    fetch.fetch("https://x/playlist", str(tmp_path), limit=1, sleep=(2, 5), runner=runner,
                log=lambda *_: None)
    audio = next(c for c in runner.calls if "bestaudio" in c)
    video = next(c for c in runner.calls if fetch.VIDEO_FORMAT in c)
    assert audio[audio.index("--sleep-interval") + 1] == "2"
    assert audio[audio.index("--download-archive") + 1].endswith("archive-audio.txt")
    assert video[video.index("--download-archive") + 1].endswith("archive-video.txt")


def test_a_leftover_audio_download_is_converted_without_downloading_again(tmp_path):
    folder = tmp_path / "vid00000001"
    folder.mkdir()
    (folder / "audio.src.m4a").write_bytes(b"media")
    runner = FakeRunner()
    entry = {"id": "vid00000001", "url": "u"}
    assert fetch.fetch_audio(entry, str(tmp_path), (0, 0), runner=runner)
    assert runner.count("bestaudio") == 0 and runner.count("ffmpeg") == 1
    assert sorted(os.listdir(folder)) == ["audio.wav"]


def test_a_download_that_produces_nothing_names_the_archive(tmp_path):
    entry = {"id": "vid00000001", "url": "u"}
    with pytest.raises(RuntimeError, match="archive-video.txt"):
        fetch.fetch_video(entry, str(tmp_path), (0, 0), runner=lambda c: "")


def test_one_failing_video_does_not_stop_the_rest(tmp_path):
    runner = FakeRunner()

    def flaky(command):
        if "vid00000001" in " ".join(command) and "-f" in command:
            raise RuntimeError("private video")
        return runner(command)

    result = fetch.fetch("https://x/p", str(tmp_path), sleep=(0, 0), runner=flaky,
                         log=lambda *_: None)
    assert result["done"] == ["vid00000002"]
    assert result["failed"][0][0] == "vid00000001"


# --- roi ---------------------------------------------------------------------

def test_parse_region_forms():
    assert roi.parse_region("3.2-181:10,20,300,400") == {
        "t_start": 3.2, "t_end": 181.0, "box": [10, 20, 300, 400]}
    assert roi.parse_region("1:05-2:00.5 0,0,10,10")["t_end"] == 120.5
    assert roi.parse_region("0-end:0,0,1,1", duration=99.0)["t_end"] == 99.0
    with pytest.raises(ValueError):
        roi.parse_region("0-end:0,0,1,1")
    with pytest.raises(ValueError):
        roi.parse_region("0-10:1,2,3")


def _roi(*regions, kind="pages"):
    return {"type": kind, "regions": list(regions)}


def _region(start, end, box=(0, 0, 100, 100)):
    return {"t_start": start, "t_end": end, "box": list(box)}


def test_validate_accepts_the_documented_shape():
    good = _roi(_region(3.2, 181.0, (40, 120, 1840, 700)), _region(181.0, 200.0))
    assert roi.validate(good, duration=200.0, frame_size=(1920, 1080)) is good


@pytest.mark.parametrize("bad, why", [
    (_roi(_region(0, 10), kind="sheet_music"), "type"),
    (_roi(), "at least one"),
    (_roi(_region(10, 5)), "t_start < t_end"),
    (_roi(_region(0, 10), _region(5, 20)), "disjoint"),
    (_roi(_region(0, 10, (0, 0, 0, 10))), "no area"),
    (_roi(_region(0, 10, (1900, 0, 100, 10))), "outside"),
    (_roi(_region(0, 500)), "past the video"),
    (_roi({"t_start": 0, "t_end": 1, "box": [0.5, 0, 1, 1]}), "whole pixels"),
])
def test_validate_refuses(bad, why):
    with pytest.raises(ValueError, match=why):
        roi.validate(bad, duration=200.0, frame_size=(1920, 1080))


def test_build_sorts_regions_and_stamps_provenance():
    built = roi.build("scroll", [_region(50, 60), _region(0, 10)], (1280, 720), "flags")
    assert [r["t_start"] for r in built["regions"]] == [0, 50]
    assert built["frame_size"] == [1280, 720]
    assert built["internal_only"] is True and built["source"] == "youtube"


def _fake_video(tmp_path, count=4, size=(64, 48)):
    """An item with meta.json and a video.mp4 placeholder, and an ffmpeg stand-in
    that writes `count` numbered frames where the thumbnail step asks."""
    import cv2
    import numpy as np

    folder = tmp_path / "vid00000001"
    folder.mkdir()
    (folder / "video.mp4").write_bytes(b"")
    layout.write_json(str(folder / "meta.json"), {"duration": 20.0})
    calls = []

    def ffmpeg(command):
        calls.append(command)
        pattern = command[-1]
        for n in range(1, count + 1):
            cv2.imwrite(pattern % n, np.full((size[1], size[0], 3), n * 40, np.uint8))
    return folder, ffmpeg, calls


def test_thumbnails_are_named_by_time_and_kept(tmp_path):
    folder, ffmpeg, calls = _fake_video(tmp_path)
    thumbs = roi.thumbnails(str(folder / "video.mp4"), str(folder / "frames" / "thumbs"),
                            every=5.0, runner=ffmpeg)
    assert [os.path.basename(p) for p in thumbs] == [
        "t0000.0.jpg", "t0005.0.jpg", "t0010.0.jpg", "t0015.0.jpg"]
    assert roi.thumbnail_time("t0015.0.jpg") == 15.0
    roi.thumbnails(str(folder / "video.mp4"), str(folder / "frames" / "thumbs"), runner=ffmpeg)
    assert len(calls) == 1


def test_mark_with_flags_writes_roi_and_does_not_overwrite(tmp_path, monkeypatch):
    folder, ffmpeg, _ = _fake_video(tmp_path)
    real = roi.thumbnails
    monkeypatch.setattr(roi, "thumbnails", lambda v, f, e=5.0: real(v, f, e, runner=ffmpeg))
    written = roi.mark("vid00000001", str(tmp_path), "pages", ["0-end:2,2,50,40"],
                       log=lambda *_: None)
    stored = json.load(open(folder / "roi.json"))
    assert stored == written
    assert stored["regions"] == [{"t_start": 0.0, "t_end": 20.0, "box": [2, 2, 50, 40]}]
    assert stored["frame_size"] == [64, 48] and stored["method"] == "flags"
    assert roi.mark("vid00000001", str(tmp_path), "scroll", ["0-1:0,0,1,1"],
                    log=lambda *_: None) is None
    assert json.load(open(folder / "roi.json"))["type"] == "pages"


def test_mark_with_the_sheet_prompts_and_draws_the_sheet(tmp_path, monkeypatch):
    folder, ffmpeg, _ = _fake_video(tmp_path)
    real = roi.thumbnails
    monkeypatch.setattr(roi, "thumbnails", lambda v, f, e=5.0: real(v, f, e, runner=ffmpeg))
    answers = iter(["falling", "falling_notes", "0-9 0,0,64,48", "5-10 0,0,1,1",
                    "9-end 0,24,64,24", ""])
    written = roi.mark("vid00000001", str(tmp_path), ui="sheet",
                       ask=lambda _: next(answers), log=lambda *_: None)
    assert written["type"] == "falling_notes" and written["method"] == "sheet"
    assert [r["t_start"] for r in written["regions"]] == [0.0, 9.0]
    assert (folder / "frames" / "contact_sheet.png").exists()
    assert (folder / "frames" / "reference_grid.png").exists()


def test_cli_later_stages_say_they_are_not_built(capsys):
    from ytpipe.cli import main

    assert main(["report"]) == 2
    assert "stage 6" in capsys.readouterr().err


# --- omr ---------------------------------------------------------------------

def _frame(pattern: int, size=(120, 80)):
    """A white page with a black bar whose position encodes which system it is."""
    import numpy as np

    frame = np.full((size[1], size[0]), 255, np.uint8)
    frame[20:30, 10 + 20 * pattern: 30 + 20 * pattern] = 0
    return frame


def test_segment_finds_each_displayed_system_with_its_times():
    from ytpipe import omr

    # 2 fps: system 0 for 3 s, a 0.5 s flicker, system 1 for 2 s, system 0 again for 2 s
    shown = [0] * 6 + [3] + [1] * 4 + [0] * 4
    stream = [(i / 2, _frame(p)) for i, p in enumerate(shown)]
    region = [{"t_start": 0, "t_end": 100, "box": [0, 0, 120, 80]}]
    systems = omr.segment(stream, region)
    assert [(s["t_on"], s["t_off"]) for s in systems] == [(0.0, 3.0), (3.5, 5.5), (5.5, 7.5)]


def test_segment_merges_a_system_split_by_a_dropped_flicker():
    from ytpipe import omr

    shown = [0] * 4 + [3] + [0] * 4
    stream = [(i / 2, _frame(p)) for i, p in enumerate(shown)]
    systems = omr.segment(stream, [{"t_start": 0, "t_end": 100, "box": [0, 0, 120, 80]}])
    assert [(s["t_on"], s["t_off"]) for s in systems] == [(0.0, 4.5)]


def test_segment_ignores_frames_outside_the_regions():
    from ytpipe import omr

    stream = [(i / 2, _frame(i // 4 % 2)) for i in range(16)]
    systems = omr.segment(stream, [{"t_start": 4.0, "t_end": 7.5, "box": [0, 0, 120, 80]}])
    assert systems and all(s["t_on"] >= 4.0 for s in systems)


def test_trim_drops_letterboxing_and_stack_blackens_ink():
    import numpy as np

    from ytpipe import omr

    frame = np.zeros((100, 200), np.uint8)
    frame[30:70, 20:180] = 250
    frame[50, 40:160] = 180                      # a faint grey staff line
    trimmed = omr.trim(frame, margin=2)
    assert trimmed.shape == (36, 156)
    page = omr.stack([trimmed, trimmed], gap=10, pad=5)
    assert page.shape == (5 + 36 + 10 + 36 + 5, 5 + 156 + 5)
    assert set(np.unique(page)) == {0, 255}
    assert (page == 0).sum() == 2 * 120


def test_page_groups():
    from ytpipe import omr

    assert omr.page_groups(7, 3) == [[0, 1, 2], [3, 4, 5], [6]]


def _audiveris_like():
    """Bar 1 from a two-staff piano part, bar 2 from two single-staff parts --
    the split Audiveris makes when it misses a system's brace."""
    from music21 import chord, clef, meter, note, stream

    def measure(number, *notes, sign=None):
        m = stream.Measure(number=number)
        if sign:
            m.insert(0, clef.BassClef() if sign == "F" else clef.TrebleClef())
        for n in notes:
            m.append(n)
        return m

    def rest(number):
        return measure(number, note.Rest(quarterLength=4))

    upper, lower = stream.Part(id="P1"), stream.Part(id="P2")
    upper.append([rest(1), measure(2, note.Note("E5", quarterLength=4), sign="G")])
    lower.append([rest(1), measure(2, note.Note("C3", quarterLength=4), sign="F")])
    right, left = stream.PartStaff(id="P3-Staff1"), stream.PartStaff(id="P3-Staff2")
    first = measure(1, note.Note("G4", quarterLength=4), sign="G")
    first.insert(0, meter.TimeSignature("4/4"))
    right.append([first, rest(2)])
    left.append([measure(1, chord.Chord(["C2", "G2"], quarterLength=2), sign="F"), rest(2)])
    return stream.Score([upper, lower, right, left])


def test_hands_takes_the_staves_that_carry_notes_bar_by_bar():
    from ytpipe import omr

    bars = omr.hands(_audiveris_like())
    assert [m.recurse().notes[0].pitches[0].nameWithOctave for m, _ in bars] == ["G4", "E5"]
    assert [m.recurse().notes[0].pitches[0].nameWithOctave for _, m in bars] == ["C2", "C3"]


def test_join_writes_two_hands_and_reports_irregular_bars(tmp_path):
    from evaluation.reference import read_score
    from ytpipe import omr

    out = str(tmp_path / "score.musicxml")
    report = omr.join([_audiveris_like(), _audiveris_like()], out)
    assert report["measures"] == 4 and report["time_signature"] == "4/4"
    assert report["measure_page"] == [1, 1, 2, 2]
    assert report["irregular"] == []
    assert sorted(n.pitch for n in read_score(out))[:2] == [36, 36]


# --- align -------------------------------------------------------------------

def _omr(*notes):
    """(quarters, length, pitch) -> OMR note dicts as `align.omr_notes` makes them."""
    return [{"pitch": p, "quarters": q, "quarters_len": l, "hand": "RH", "measure": 1 + int(q // 4),
             "ref": f"o{i}"} for i, (q, l, p) in enumerate(notes)]


def _heard(*notes):
    return [{"pitch": p, "t_on": t, "t_off": t + d} for t, d, p in notes]


def test_match_pairs_same_pitch_within_tolerance_one_to_one():
    from ytpipe import align

    omr = [{"pitch": 60, "t_on": 1.00}, {"pitch": 60, "t_on": 1.10}, {"pitch": 62, "t_on": 2.0}]
    audio = [{"pitch": 60, "t_on": 1.08}, {"pitch": 64, "t_on": 2.0}]
    assert align.match(omr, audio, tolerance=0.08) == [(1, 0)]


def test_monotone_drops_backward_points():
    import numpy as np

    from ytpipe import align

    x, y = align.monotone(np.array([0, 1, 2, 3.0]), np.array([0, 2, 1, 3.0]))
    assert list(x) == [0, 1, 3] and list(y) == [0, 2, 3]


def test_table_marks_every_note_by_source():
    from ytpipe import align

    omr = _omr((0, 1, 60), (1, 1, 62), (2, 1, 64))
    align.place(omr, [0, 4], [1.0, 3.0])                  # 2 quarters a second, from 1 s
    audio = _heard((1.01, 0.5, 60), (1.5, 2.0, 62), (2.2, 0.3, 67))
    rows = align.table(omr, audio)
    assert [(r["pitch"], r["status"]) for r in rows] == [
        (60, "agree"), (62, "duration_differs"), (64, "omr_only"), (67, "audio_only")]
    assert [r["id"] for r in rows] == [0, 1, 2, 3]


def test_alignment_recovers_a_tempo_and_offset():
    import numpy as np

    from ytpipe import align

    # A two-bar figure repeated with changing pitches, played from 2 s at 120 bpm.
    written = [(q, 0.5, 60 + (q * 2 + q // 4) % 12) for q in np.arange(0, 32, 0.5)]
    omr = _omr(*written)
    audio = _heard(*[(2.0 + q * 0.5, 0.25, p) for q, _, p in written])
    quarters, seconds = align.coarse_map(omr, audio)
    quarters, seconds = align.refine(omr, audio, quarters, seconds)
    rows = align.table(omr, audio)
    assert sum(r["status"] == "agree" for r in rows) >= 0.95 * len(written)
    assert abs(np.interp(16.0, quarters, seconds) - 10.0) < 0.05


def test_screen_windows_restrict_each_page(tmp_path):
    from ytpipe import align

    folder = tmp_path / "v"
    (folder / "omr").mkdir(parents=True)
    layout.write_json(str(folder / "omr" / "systems.json"), {"systems": [
        {"t_on": 0.0, "t_off": 10.0}, {"t_on": 10.0, "t_off": 20.0}, {"t_on": 20.0, "t_off": 30.0}]})
    layout.write_json(str(folder / "omr" / "report.json"),
                      {"measure_page": [1, 1, 2], "page_systems": [[0, 1], [2]]})
    omr = _omr((0, 4, 60), (4, 4, 62), (8, 4, 64))
    assert align.screen_windows(str(folder), omr) == [(0, 8, 0.0, 20.0), (8, 12, 20.0, 30.0)]


def test_omr_notes_joins_ties_and_splits_chords(tmp_path):
    from music21 import chord, meter, note, stream, tie

    from ytpipe import align

    right, left = stream.PartStaff(), stream.PartStaff()
    first, second = stream.Measure(number=1), stream.Measure(number=2)
    first.insert(0, meter.TimeSignature("4/4"))
    held = note.Note("C5", quarterLength=4)
    held.tie = tie.Tie("start")
    first.append(held)
    rest_of_it = note.Note("C5", quarterLength=2)
    rest_of_it.tie = tie.Tie("stop")
    second.append([rest_of_it, chord.Chord(["E4", "G4"], quarterLength=2)])
    right.append([first, second])
    low = stream.Measure(number=1)
    low.append(note.Note("C3", quarterLength=4))
    left.append([low, stream.Measure([note.Rest(quarterLength=4)], number=2)])
    path = str(tmp_path / "s.musicxml")
    stream.Score([right, left]).write("musicxml", fp=path)

    notes = align.omr_notes(path)
    assert [(n["pitch"], n["quarters"], n["quarters_len"], n["hand"]) for n in notes] == [
        (48, 0.0, 4.0, "LH"), (72, 0.0, 6.0, "RH"), (64, 6.0, 2.0, "RH"), (67, 6.0, 2.0, "RH")]


def test_rests_fill_awkward_lengths_with_writable_ones():
    from ytpipe import omr

    assert [r.quarterLength for r in omr.rests(1.5)] == [1.5]
    for quarters in (4.5, 11 / 3, 0.625, 4.0):
        parts = omr.rests(quarters)
        assert abs(sum(r.quarterLength for r in parts) - quarters) < 1e-6
        assert all(float(r.quarterLength) in [round(w, 6) for w in omr.WRITABLE]
                   or any(abs(float(r.quarterLength) - w) < 1e-6 for w in omr.WRITABLE)
                   for r in parts)


# --- edit --------------------------------------------------------------------

def test_undecided_disagreements_follow_the_omr_score():
    from ytpipe import edit

    rows = [{"id": 0, "status": "agree"}, {"id": 1, "status": "omr_only"},
            {"id": 2, "status": "audio_only"}, {"id": 3, "status": "audio_only"}]
    decisions = {"0": "drop", "3": "keep"}
    assert [edit.kept(r, decisions) for r in rows] == [False, True, False, True]


def _small_score(path):
    from music21 import chord, meter, note, stream

    right, left = stream.PartStaff(), stream.PartStaff()
    bar = stream.Measure(number=1)
    bar.insert(0, meter.TimeSignature("4/4"))
    bar.append([chord.Chord(["C5", "E5"], quarterLength=2), note.Note("G5", quarterLength=2)])
    right.append(bar)
    low = stream.Measure(number=1)
    low.append(note.Note("C3", quarterLength=4))
    left.append(low)
    stream.Score([right, left]).write("musicxml", fp=path)


def test_build_result_drops_and_adds_notes(tmp_path):
    import mido
    import music21

    from evaluation.reference import read_score
    from ytpipe import edit

    (tmp_path / "omr").mkdir()
    _small_score(str(tmp_path / "omr" / "score.musicxml"))
    alignment = {
        "map": {"quarters": [0, 4], "seconds": [0.0, 2.0]},       # 2 quarters a second
        "notes": [
            {"id": 0, "status": "agree", "pitch": 72, "t_on": 0, "t_off": 1, "heard": [0.02, 0.9],
             "omr": "o0", "hand": "RH", "measure": 1, "at": 0.0},
            {"id": 1, "status": "omr_only", "pitch": 76, "t_on": 0, "t_off": 1,
             "omr": "o1", "hand": "RH", "measure": 1, "at": 0.0},
            {"id": 2, "status": "omr_only", "pitch": 79, "t_on": 1, "t_off": 2,
             "omr": "o2", "hand": "RH", "measure": 1, "at": 2.0},
            {"id": 3, "status": "agree", "pitch": 48, "t_on": 0, "t_off": 2,
             "omr": "o3", "hand": "LH", "measure": 1, "at": 0.0},
            {"id": 4, "status": "audio_only", "pitch": 74, "t_on": 0.5, "t_off": 1.0},
            {"id": 5, "status": "audio_only", "pitch": 50, "t_on": 1.0, "t_off": 1.5},
        ]}
    decisions = {"1": "drop", "4": "keep"}          # 2 stays (undecided OMR), 5 stays out
    result = edit.build_result(str(tmp_path), alignment, decisions)
    assert result == {"dropped": 1, "added": 1, "graces": 0, "notes": 4}

    written = sorted((n.onset, n.pitch) for n in read_score(str(tmp_path / "score.musicxml")))
    assert written == [(0.0, 48), (0.0, 72), (1.0, 74), (2.0, 79)]
    music21.converter.parse(str(tmp_path / "score.musicxml"))

    onsets = []
    clock = 0.0
    for message in mido.MidiFile(str(tmp_path / "perf.mid")):
        clock += message.time
        if message.type == "note_on" and message.velocity:
            onsets.append((round(clock, 2), message.note))
    assert sorted(onsets) == [(0.0, 48), (0.02, 72), (0.5, 74), (1.0, 79)]


def test_add_note_joins_a_chord_of_the_same_length(tmp_path):
    import music21

    from ytpipe import edit

    path = str(tmp_path / "s.musicxml")
    _small_score(path)
    score = music21.converter.parse(path)
    placed = edit.insert_notes(score, [("a", 67, 0.0, 2.0)])
    assert placed["a"].pitch.midi == 67
    first = score.parts[0].getElementsByClass("Measure")[0].notes[0]
    assert sorted(p.midi for p in first.pitches) == [67, 72, 76]


def test_overlapping_added_notes_go_to_separate_voices(tmp_path):
    import music21

    from ytpipe import edit

    path = str(tmp_path / "s.musicxml")
    _small_score(path)
    score = music21.converter.parse(path)
    placed = edit.insert_notes(score, [("a", 62, 0.0, 1.5), ("b", 65, 1.0, 1.0),
                                       ("c", 69, 3.0, 1.0)])
    assert set(placed) == {"a", "b", "c"}
    bar = score.parts[0].getElementsByClass("Measure")[0]
    added = [v for v in bar.voices if str(v.id).startswith("yt")]
    for voice in added:
        spans = sorted((float(n.offset), float(n.offset + n.quarterLength)) for n in voice.notes)
        assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))
    assert len(added) == 2
    out = str(tmp_path / "out.musicxml")
    score.write("musicxml", fp=out, makeNotation=False)
    music21.converter.parse(out)


def test_review_score_names_every_omr_note_after_its_row(tmp_path):
    import re

    from ytpipe import edit, engrave

    (tmp_path / "omr").mkdir()
    _small_score(str(tmp_path / "omr" / "score.musicxml"))
    alignment = {"map": {"quarters": [0, 4], "seconds": [0.0, 2.0]}, "notes": [
        {"id": 0, "status": "agree", "pitch": 72, "t_on": 0, "t_off": 1,
         "omr": "o0", "hand": "RH", "measure": 1, "at": 0.0},
        {"id": 1, "status": "omr_only", "pitch": 79, "t_on": 1, "t_off": 2,
         "omr": "o1", "hand": "RH", "measure": 1, "at": 2.0},
        {"id": 2, "status": "agree", "pitch": 48, "t_on": 0, "t_off": 2,
         "omr": "o2", "hand": "LH", "measure": 1, "at": 0.0},
        {"id": 3, "status": "audio_only", "pitch": 74, "t_on": 0.5, "t_off": 1.0}]}
    layout.write_json(str(tmp_path / "alignment.json"), alignment)
    review = edit.build_review(str(tmp_path), alignment)
    ids = set(re.findall(r'<note[^>]* id="(row\d+)"', open(review).read()))
    assert ids == {"row0", "row1", "row2"}           # audio-only notes are flagged, not engraved
    engraved = engrave.render(review)
    assert set(engraved["page_of"]) == ids and engraved["pages"][0].startswith("<svg")


def test_the_server_serves_an_item_and_saves_decisions(tmp_path):
    import threading
    import urllib.request
    from http.server import ThreadingHTTPServer

    from ytpipe import edit

    folder = tmp_path / "vid00000001"
    folder.mkdir()
    layout.write_json(str(folder / "meta.json"), {"title": "One", "duration": 2.0})
    layout.write_json(str(folder / "alignment.json"), {
        "counts": {}, "map": {"quarters": [0, 1], "seconds": [0, 1]},
        "notes": [{"id": 0, "status": "audio_only", "pitch": 60, "t_on": 0, "t_off": 1}]})
    edit.Handler.root = str(tmp_path)
    server = ThreadingHTTPServer(("127.0.0.1", 0), edit.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        assert json.load(urllib.request.urlopen(f"{base}/api/items"))[0]["id"] == "vid00000001"
        request = urllib.request.Request(f"{base}/api/decide?id=vid00000001", method="POST",
                                         data=json.dumps({"decisions": {"0": "keep"}}).encode())
        assert json.load(urllib.request.urlopen(request))["decided"] == 1
        item = json.load(urllib.request.urlopen(f"{base}/api/item?id=vid00000001"))
        assert item["decisions"] == {"0": "keep"} and item["title"] == "One"
        assert urllib.request.urlopen(f"{base}/").read().startswith(b"<!doctype html>")
    finally:
        server.shutdown()


def test_a_rolled_chord_matches_its_written_attack():
    from ytpipe import align

    written = [{"pitch": p, "t_on": 1.0} for p in (48, 55, 64)]
    played = [{"pitch": 48, "t_on": 1.0}, {"pitch": 55, "t_on": 1.12}, {"pitch": 64, "t_on": 1.25}]
    assert len(align.match(written, played)) == 3
    lone = [{"pitch": 64, "t_on": 1.0}]
    assert align.match(lone, [{"pitch": 64, "t_on": 1.25}]) == []


def test_the_score_is_pinned_to_the_first_and_last_notes_heard():
    import numpy as np

    from ytpipe import align

    omr = _omr((0, 1, 51), (0, 1, 58), (0, 1, 67), (1, 1, 72), (8, 4, 63), (8, 4, 70))
    audio = _heard((0.58, 0.5, 51), (0.62, 0.5, 58), (0.80, 0.5, 67), (1.8, 0.3, 72),
                   (6.0, 3.0, 63), (6.02, 3.0, 70))
    # DTW started the score 0.8 s late and ended it 0.5 s early.
    quarters, seconds = align.anchor_ends(omr, audio, [0, 4, 8], [1.38, 3.5, 5.5])
    assert np.interp(0, quarters, seconds) == 0.58
    assert np.interp(8, quarters, seconds) == 6.0
    # A first chord the audio does not play is not pinned.
    wrong = _heard((0.58, 0.5, 49), (0.62, 0.5, 54), (6.0, 3.0, 63))
    quarters, seconds = align.anchor_ends(omr, wrong, [0, 4, 8], [1.38, 3.5, 5.5])
    assert np.interp(0, quarters, seconds) == 1.38


def test_overtones_and_restrikes_of_written_notes_are_marked():
    from ytpipe import align

    rows = [
        {"id": 0, "status": "agree", "pitch": 48, "t_on": 0.0, "t_off": 2.0, "heard": [0.0, 2.0], "omr": "o0"},
        {"id": 1, "status": "audio_only", "pitch": 60, "t_on": 0.5, "t_off": 0.8},   # octave above, while held
        {"id": 2, "status": "audio_only", "pitch": 48, "t_on": 1.0, "t_off": 1.5},   # same pitch, while held
        {"id": 3, "status": "audio_only", "pitch": 62, "t_on": 1.0, "t_off": 1.5},   # a real extra note
        {"id": 4, "status": "audio_only", "pitch": 60, "t_on": 3.0, "t_off": 3.5},   # after it ends
    ]
    align.mark_artefacts(rows)
    assert [r.get("likely") for r in rows] == [None, "overtone", "restrike", None, None]


def test_a_page_with_a_clef_on_an_impossible_line_still_loads(tmp_path):
    import zipfile

    from music21 import clef

    from ytpipe import omr

    xml = ('<?xml version="1.0"?><score-partwise version="4.0"><part-list><score-part id="P1">'
           '<part-name>x</part-name></score-part></part-list><part id="P1"><measure number="1">'
           '<attributes><divisions>1</divisions><clef><sign>G</sign><line>0</line></clef></attributes>'
           '<note><pitch><step>C</step><octave>5</octave></pitch><duration>4</duration></note>'
           '</measure></part></score-partwise>')
    path = tmp_path / "p01.mxl"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("p01.xml", xml)
    score = omr.load_page(str(path))
    assert isinstance(score.recurse().getElementsByClass(clef.Clef)[0], clef.TrebleClef)


def test_join_keeps_page_numbers_across_an_unread_page(tmp_path):
    from ytpipe import omr

    report = omr.join([_audiveris_like(), _audiveris_like()], str(tmp_path / "s.musicxml"), [1, 3])
    assert report["measure_page"] == [1, 1, 3, 3]


def test_audiveris_movements_are_collected_as_their_page(tmp_path):
    from ytpipe import omr

    out = tmp_path / "pages"
    out.mkdir()
    for name in ("p01.mvt2.mxl", "p01.mvt1.mxl", "p02.mxl"):
        (out / name).write_bytes(b"x")
    pages = [str(out / "p01.png"), str(out / "p02.png")]
    found = omr.audiveris(pages, str(out), runner=lambda c: None, log=lambda *_: None)
    assert [p.split("/")[-1] for p in found] == ["p01.mvt1.mxl", "p01.mvt2.mxl", "p02.mxl"]


def test_screen_windows_narrow_to_each_system_when_the_counts_agree(tmp_path):
    from ytpipe import align

    folder = tmp_path / "v"
    (folder / "omr").mkdir(parents=True)
    layout.write_json(str(folder / "omr" / "systems.json"), {"systems": [
        {"t_on": 0.0, "t_off": 10.0}, {"t_on": 10.0, "t_off": 20.0}, {"t_on": 20.0, "t_off": 30.0}]})
    omr = _omr((0, 4, 60), (4, 4, 62), (8, 4, 64))
    report = {"measure_page": [1, 1, 2], "measure_system": [0, 1, 0], "page_systems": [[0, 1], [2]]}
    layout.write_json(str(folder / "omr" / "report.json"), report)
    assert align.screen_windows(str(folder), omr) == [
        (0, 4, 0.0, 10.0), (4, 8, 10.0, 20.0), (8, 12, 20.0, 30.0)]
    # Audiveris saw one system where the video showed two: fall back to the page.
    report["measure_system"] = [0, 0, 0]
    layout.write_json(str(folder / "omr" / "report.json"), report)
    assert align.screen_windows(str(folder), omr)[0] == (0, 8, 0.0, 20.0)


def test_piece_titles_drop_the_channel_tags(tmp_path):
    for raw, clean in [("[Ballad Jazz Piano] Misty (sheet music)", "Misty"),
                       ("[Jazz Standard] 'Autumn Leaves' for solo piano (sheet music)", "Autumn Leaves"),
                       ("'It Could Happen To You' | vocal & piano duo | with Jugrace",
                        "It Could Happen To You"),
                       ("[Ballad Jazz Piano] When you wish upon a star (Pinoccio OST)",
                        "When you wish upon a star")]:
        layout.write_json(str(tmp_path / "meta.json"), {"title": raw})
        assert layout.piece_title(str(tmp_path)) == clean


def test_realigning_carries_decisions_to_the_same_notes(tmp_path):
    from ytpipe import align

    old = [{"id": 0, "status": "omr_only", "pitch": 60, "omr": "o0", "hand": "RH", "measure": 1, "at": 0.0},
           {"id": 1, "status": "audio_only", "pitch": 62, "audio": "a5"},
           {"id": 2, "status": "audio_only", "pitch": 64, "audio": "a6"}]
    layout.write_json(str(tmp_path / "alignment.json"), {"notes": old})
    layout.write_json(str(tmp_path / "decisions.json"),
                      {"decisions": {"0": "keep", "1": "drop", "2": "keep"}})
    # Re-aligned: a new note first shifts every id, and a6 is no longer audio-only.
    new = [{"id": 0, "status": "audio_only", "pitch": 50, "audio": "a1"},
           {"id": 1, "status": "omr_only", "pitch": 60, "omr": "o0", "hand": "RH", "measure": 1, "at": 0.0},
           {"id": 2, "status": "audio_only", "pitch": 62, "audio": "a5"}]
    assert align.carry_decisions(str(tmp_path), str(tmp_path / "alignment.json"), new) == (2, 1)
    assert layout.read_json(str(tmp_path / "decisions.json"))["decisions"] == {"1": "keep", "2": "drop"}


def test_a_musescore_save_is_found_by_piece_name_or_id(tmp_path, monkeypatch):
    import os
    import time

    from ytpipe import edit

    shared = tmp_path / "ytpipe sheets"
    shared.mkdir()
    monkeypatch.setenv("YTPIPE_SCORES", str(shared))
    monkeypatch.setattr(edit, "_windows_env", lambda name: str(tmp_path / "profile"))
    since = time.time() - 100
    handed = shared / "New York State of Mind [vid00000001].musicxml"
    handed.write_text("x")
    os.utime(handed, (since, since))
    other = shared / "Misty.mscz"                         # another piece's save
    other.write_text("x")
    assert edit.saved_in_musescore("vid00000001", since, "New York State of Mind") is None
    saved = shared / "New York State of Mind.mscz"
    saved.write_text("x")
    assert edit.saved_in_musescore("vid00000001", since, "New York State of Mind") == str(saved)

SHORT_BAR = ('<?xml version="1.0"?><score-partwise version="4.0"><part-list>'
             '<score-part id="P1"><part-name>R</part-name></score-part>'
             '<score-part id="P2"><part-name>L</part-name></score-part></part-list>'
             '<part id="P1"><measure number="1"><attributes><divisions>2</divisions>'
             '<time><beats>4</beats><beat-type>4</beat-type></time></attributes>'
             '<note><pitch><step>C</step><octave>5</octave></pitch><duration>7</duration></note></measure>'
             '<measure number="2"><note><pitch><step>D</step><octave>5</octave></pitch><duration>8</duration></note></measure></part>'
             '<part id="P2"><measure number="1"><attributes><divisions>2</divisions>'
             '<time><beats>4</beats><beat-type>4</beat-type></time></attributes>'
             '<note><rest measure="yes"/><duration>7</duration></note></measure>'
             '<measure number="2"><note><pitch><step>C</step><octave>3</octave></pitch><duration>8</duration></note></measure></part>'
             '</score-partwise>')


def test_a_whole_bar_rest_in_a_short_bar_keeps_the_hands_together(tmp_path):
    from ytpipe import align

    path = tmp_path / "s.musicxml"
    path.write_text(SHORT_BAR)
    score = layout.parse_score(str(path))
    second = [float(p.getElementsByClass("Measure")[1].getOffsetBySite(p)) for p in score.parts]
    assert second == [3.5, 3.5]
    notes = {n["pitch"]: n["quarters"] for n in align.omr_notes(str(path))}
    assert notes[74] == notes[48] == 3.5


def test_audiveris_pages_keep_whole_bar_rests_at_their_written_length(tmp_path):
    import zipfile

    from ytpipe import omr

    path = tmp_path / "p01.mxl"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("p01.xml", SHORT_BAR)
    score = omr.load_page(str(path))
    assert [float(p.getElementsByClass("Measure")[1].getOffsetBySite(p)) for p in score.parts] == [3.5, 3.5]


def test_a_whole_bar_rest_of_awkward_length_is_split_on_join(tmp_path):
    import zipfile

    from ytpipe import omr

    # Bar 1 read as 4.5 quarters: a right-hand note of that length would be
    # written tied; the left hand's whole-bar rest keeps 4.5 and must be split.
    xml = (SHORT_BAR.replace('<duration>7</duration></note></measure>\n', '')
           .replace("<step>C</step><octave>5</octave></pitch><duration>7</duration>",
                    "<step>C</step><octave>5</octave></pitch><duration>8</duration></note>"
                    "<note><pitch><step>E</step><octave>5</octave></pitch><duration>1</duration>")
           .replace('<rest measure="yes"/><duration>7</duration>', '<rest measure="yes"/><duration>9</duration>'))
    path = tmp_path / "p01.mxl"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("p01.xml", xml)
    out = str(tmp_path / "s.musicxml")
    report = omr.join([omr.load_page(str(path))], out)
    assert report["measures"] == 2
    score = layout.parse_score(out)
    assert [float(p.getElementsByClass("Measure")[1].getOffsetBySite(p)) for p in score.parts] == [4.5, 4.5]


def test_a_musescore_edit_becomes_the_main_score(tmp_path):
    (tmp_path / "omr").mkdir()
    omr = tmp_path / "omr" / "score.musicxml"
    omr.write_text("x")
    assert layout.main_score(str(tmp_path)) == str(omr)
    (tmp_path / "edited.musicxml").write_text("y")
    assert layout.main_score(str(tmp_path)) == str(tmp_path / "edited.musicxml")


def test_the_sheet_pdf_stacks_every_system_onto_pages(tmp_path):
    import numpy as np
    import cv2

    from ytpipe import sheet

    (tmp_path / "omr" / "systems").mkdir(parents=True)
    listing = []
    for i in range(7):
        name = f"{i + 1:03d}.png"
        cv2.imwrite(str(tmp_path / "omr" / "systems" / name), np.full((300, 1000), 255, np.uint8))
        listing.append({"file": name, "t_on": 5.0 * i, "t_off": 5.0 * i + 5})
    layout.write_json(str(tmp_path / "omr" / "systems.json"), {"systems": listing})
    layout.write_json(str(tmp_path / "meta.json"), {"title": "[Jazz Piano] Misty (sheet music)"})
    laid = sheet.pages(str(tmp_path))
    assert len(laid) == 2                                  # about 4 systems fit an A4 page here
    assert abs(laid[0].height / laid[0].width - 297 / 210) < 0.01
    out = sheet.build(str(tmp_path))
    assert open(out, "rb").read(5) == b"%PDF-"


def test_join_keeps_slurs(tmp_path):
    from music21 import meter, note, spanner, stream

    from ytpipe import omr

    right, left = stream.PartStaff(id="P1-Staff1"), stream.PartStaff(id="P1-Staff2")
    first, second = stream.Measure(number=1), stream.Measure(number=2)
    first.insert(0, meter.TimeSignature("4/4"))
    a, b = note.Note("C5", quarterLength=4), note.Note("D5", quarterLength=4)
    first.append(a)
    second.append(b)
    right.append([first, second])
    right.insert(0, spanner.Slur(a, b))
    left.append([stream.Measure([note.Note("C3", quarterLength=4)], number=1),
                 stream.Measure([note.Note("D3", quarterLength=4)], number=2)])
    out = str(tmp_path / "s.musicxml")
    omr.join([stream.Score([right, left])], out)
    assert open(out).read().count('<slur ') == 2            # start and stop


def test_an_accepted_grace_candidate_is_written_before_its_note(tmp_path):
    import music21

    from ytpipe import edit

    (tmp_path / "omr").mkdir()
    _small_score(str(tmp_path / "omr" / "score.musicxml"))
    alignment = {"map": {"quarters": [0, 4], "seconds": [0.0, 2.0]}, "notes": [
        {"id": 0, "status": "agree", "pitch": 79, "t_on": 1.0, "t_off": 2.0, "heard": [1.0, 2.0],
         "omr": "o0", "hand": "RH", "measure": 1, "at": 2.0},
        {"id": 1, "status": "grace_candidate", "pitch": 77, "t_on": 0.95, "t_off": 1.0, "target": 0},
        {"id": 2, "status": "grace_candidate", "pitch": 81, "t_on": 0.96, "t_off": 1.0, "target": 0}]}
    assert not edit.kept(alignment["notes"][1], {})              # never applied unasked
    result = edit.build_result(str(tmp_path), alignment, {"1": "grace"})
    assert result["graces"] == 1
    bar = music21.converter.parse(str(tmp_path / "score.musicxml")).parts[0].getElementsByClass("Measure")[0]
    sequence = [(n.duration.isGrace, n.pitch.midi) for n in bar.recurse().notes if not n.isChord]
    assert sequence[-2:] == [(True, 77), (False, 79)]
