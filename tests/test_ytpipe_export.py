"""Stage 6: the ASAP-shaped export. MuseScore and Transkun are not run here."""

import csv
import json

import music21
import pytest

from notation.dataset import bar_offset, bar_lines, repeat_mismatch, score_bar_numbers
from evaluation.pairs import beat_map, downbeat_map
from ytpipe import export


def _score(signature: str, bars: list[float]):
    """Two staves; each bar filled with one note of the given length."""
    score = music21.stream.Score()
    for _ in range(2):
        part = music21.stream.Part()
        for index, length in enumerate(bars):
            measure = music21.stream.Measure(number=index + (0 if len(bars) > 1 and length < bars[1] else 1))
            if index == 0:
                measure.insert(0, music21.meter.TimeSignature(signature))
            measure.append(music21.note.Note("C4", quarterLength=length))
            part.append(measure)
        score.insert(0, part)
    return score


def test_full_bars_have_a_downbeat_each_and_a_closing_line():
    bars = export.bar_grid(_score("3/4", [3.0, 3.0]))
    assert [b["beats"] for b in bars] == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
    lines = export.annotation_lines(bars, lambda q: q)
    assert lines == ["0.0\t0.0\tdb,3/4", "1.0\t1.0\tb", "2.0\t2.0\tb",
                     "3.0\t3.0\tdb", "4.0\t4.0\tb", "5.0\t5.0\tb", "6.0\t6.0\tdb"]


def test_an_anacrusis_counts_its_beats_back_from_the_bar_line(tmp_path):
    bars = export.bar_grid(_score("4/4", [1.0, 4.0, 4.0]))
    assert bars[0]["pickup"] and bars[0]["beats"] == [0.0]
    assert bars[1]["beats"][0] == 1.0
    path = tmp_path / "score_annotations.txt"
    path.write_text("\n".join(export.annotation_lines(bars, lambda q: q * 0.5)) + "\n")
    # The notation code's reading: one bar before the first downbeat, and the
    # downbeats (two bar starts and the closing line) pair with the score's bars.
    assert bar_offset(str(path)) == 1
    assert list(downbeat_map(str(path))) == [0.5, 2.5, 4.5]
    assert list(beat_map(str(path)))[:2] == [0.0, 0.5]


def test_compound_meter_beats_on_the_dotted_quarter():
    bars = export.bar_grid(_score("6/8", [3.0]))
    assert bars[0]["beats"] == [0.0, 1.5]


def test_clock_extends_the_map_at_its_own_tempo():
    to_seconds = export.clock([4.0, 8.0, 12.0], [10.0, 12.0, 14.0])
    assert to_seconds(8.0) == 12.0
    assert to_seconds(0.0) == pytest.approx(8.0)       # 0.5 s per quarter before the map
    assert to_seconds(16.0) == pytest.approx(16.0)     # and after it


def test_fixed_rests_drops_the_whole_bar_flag():
    assert export.fixed_rests('<rest measure="yes"/><rest measure="yes">x</rest>') == \
        "<rest/><rest>x</rest>"


def test_combine_links_asap_and_adds_the_pieces_to_the_split(tmp_path):
    asap, extra, out = tmp_path / "asap", tmp_path / "extra", tmp_path / "both"
    (asap / "Bach").mkdir(parents=True)
    with open(asap / "metadata.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=export.ASAP_COLUMNS)
        writer.writeheader()
        writer.writerow({**{c: "" for c in export.ASAP_COLUMNS}, "composer": "Bach",
                         "title": "Fugue", "midi_performance": "Bach/a.mid"})
    (extra / "ytpipe" / "vid1").mkdir(parents=True)
    rows = [{**{c: "" for c in export.ASAP_COLUMNS}, "composer": "ytpipe", "title": f"T_{v}",
             "folder": f"ytpipe/{v}", "midi_performance": f"ytpipe/{v}/performance.mid"}
            for v in ("vid1", "vid2")]
    with open(extra / "metadata.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=export.ASAP_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    split_in = tmp_path / "split.json"
    split_in.write_text(json.dumps({"Bach/a.mid": "validation"}))

    split = export.combine(str(asap), str(extra), str(out), str(split_in), test=("vid2",))

    assert (out / "Bach").is_symlink() and (out / "ytpipe" / "vid1").is_dir()
    with open(out / "metadata.csv") as handle:
        assert [r["composer"] for r in csv.DictReader(handle)] == ["Bach", "ytpipe", "ytpipe"]
    assert json.load(open(split)) == {"Bach/a.mid": "validation",
                                      "ytpipe/vid1/performance.mid": "train",
                                      "ytpipe/vid2/performance.mid": "test"}
    assert json.load(open(split_in)) == {"Bach/a.mid": "validation"}   # source untouched
