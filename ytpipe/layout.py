"""Where everything lives, and how it is written.

Every path the pipeline touches is built here, so the layout is described once.
JSON is written atomically -- to a temporary file, then renamed -- so an
interrupted run never leaves a half-written file that a resumed run would
mistake for a finished one.
"""

import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ROOT = os.path.join(REPO, "data", "yt")

# Stamped on every item. Downloaded media is for internal research only.
PROVENANCE = {"source": "youtube", "internal_only": True}


def data_root(override: str | None = None) -> str:
    return override or os.environ.get("YTPIPE_DATA") or DEFAULT_ROOT


def playlist_path(root: str, playlist_id: str) -> str:
    return os.path.join(root, "playlists", f"{playlist_id}.json")


def item_dir(root: str, video_id: str) -> str:
    return os.path.join(root, video_id)


def item_path(root: str, video_id: str, name: str) -> str:
    """A file inside one video's directory: meta.json, audio.wav, video.mp4, roi.json..."""
    return os.path.join(root, video_id, name)


def archive_path(root: str, stream: str) -> str:
    """yt-dlp's download archive, one per stream.

    Audio and video are separate downloads of the same id, so one shared
    archive would record the id after the first and skip the second.
    """
    return os.path.join(root, f"archive-{stream}.txt")


def write_json(path: str, data) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w") as handle:
        json.dump(data, handle, indent=1, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)


def read_json(path: str):
    with open(path) as handle:
        return json.load(handle)


def piece_title(folder: str) -> str:
    """The piece's name from its video title, without the channel's tags.

    "[Ballad Jazz Piano] Misty (sheet music)" -> "Misty";
    "[Jazz Standard] 'Autumn Leaves' for solo piano (sheet music)" -> "Autumn Leaves".
    """
    import re

    path = os.path.join(folder, "meta.json")
    title = read_json(path).get("title", "") if os.path.exists(path) else ""
    title = re.sub(r"^\s*\[[^\]]*\]\s*", "", title)
    title = title.split("|")[0]
    title = re.sub(r"\((sheet music|[^)]*OST)\)", "", title, flags=re.I)
    title = re.sub(r"\bfor (solo piano|piano solo)\b", "", title, flags=re.I)
    title = re.sub(r"\s+", " ", title).strip(" .'\"")
    return title or os.path.basename(folder)


def set_title(score, folder: str) -> None:
    """Name a music21 score after its piece, instead of music21's placeholder."""
    from music21 import metadata

    if score.metadata is None:
        score.metadata = metadata.Metadata()
    score.metadata.title = piece_title(folder)
    score.metadata.movementName = piece_title(folder)
    score.metadata.composer = "transcribed from video " + os.path.basename(folder)


def parse_score(path: str):
    """A MusicXML file as a music21 score, with each rest its written length.

    MuseScore writes a staff's empty bar as <rest measure="yes"/> with the
    bar's real duration. music21 gives such a rest the time signature's full
    bar instead, so in a bar shorter than the signature (3.5 quarters in 4/4)
    that hand gains the difference, and every later bar of it drifts against
    the other hand: on one piece the left hand ran a beat late from bar 23 to
    the end. The flag is dropped so the written duration is used.
    """
    import re

    import music21

    if path.endswith(".mxl"):
        return music21.converter.parse(path)
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    text = re.sub(r'<rest\s+measure="yes"\s*/>', "<rest/>", text)
    text = re.sub(r'<rest\s+measure="yes"\s*>', "<rest>", text)
    return music21.converter.parse(text, format="musicxml")


def main_score(folder: str) -> str:
    """The score a piece is read from: the owner's MuseScore edit once there
    is one (edited.musicxml), the OMR score until then."""
    edited = os.path.join(folder, "edited.musicxml")
    return edited if os.path.exists(edited) else os.path.join(folder, "omr", "score.musicxml")
