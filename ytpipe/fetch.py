"""Stages 1-2: list a playlist, then download each video's audio and video.

yt-dlp is run as `python -m yt_dlp` from this interpreter, so the version
installed in the project's environment is the one used, not whatever is first
on PATH. Every step checks for its output first and skips if it is there, so a
run that stops -- a network error, a rate limit, Ctrl-C -- is resumed by
running the same command again.

Audio is downloaded as whatever the best audio stream is and converted with
ffmpeg directly, rather than through yt-dlp's `-x`: that post-processor needs
ffprobe as well, and ffmpeg alone is enough.
"""

import datetime
import json
import os
import subprocess
import sys

from ytpipe import layout

# The fields of yt-dlp's info JSON worth keeping. The full JSON is mostly the
# list of every format and runs to hundreds of kilobytes per video.
META_FIELDS = (
    "id", "title", "fulltitle", "description", "channel", "channel_id", "uploader",
    "uploader_id", "upload_date", "duration", "webpage_url", "license", "tags",
    "categories", "width", "height", "fps", "playlist_id", "playlist_title",
    "playlist_index",
)

# H.264 first: YouTube's default mp4 at 1080p is often AV1, which ffmpeg
# decodes but many OpenCV builds cannot, and later stages may read frames
# through OpenCV. AV1 (or whatever mp4 there is) remains the fallback.
VIDEO_FORMAT = "bestvideo[height<=1080][ext=mp4][vcodec^=avc1]/bestvideo[height<=1080][ext=mp4]"
AUDIO_RATE = 44100


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def run(command: list[str]) -> str:
    """Run a command, returning stdout; raise with its stderr if it fails."""
    done = subprocess.run(command, capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(f"{command[0]} {' '.join(command[1:3])}... failed "
                           f"({done.returncode}): {done.stderr.strip()[-800:]}")
    return done.stdout


def ytdlp(*args: str) -> list[str]:
    return [sys.executable, "-m", "yt_dlp", "--no-warnings", "--no-progress", *args]


def parse_flat_playlist(info: dict, url: str = "") -> dict:
    """The manifest for a playlist from `yt-dlp --flat-playlist -J`.

    A single video's URL gives an info dict with no entries; it becomes a
    one-entry playlist named after the video, so `fetch` takes either.
    Entries YouTube reports without an id (deleted or private) are dropped.
    """
    raw = info.get("entries")
    if raw is None:
        raw = [info]
    entries = []
    for index, entry in enumerate(raw, 1):
        if not entry or not entry.get("id"):
            continue
        entries.append({
            "index": index,
            "id": entry["id"],
            "title": entry.get("title"),
            "url": entry.get("url") or entry.get("webpage_url")
                   or f"https://www.youtube.com/watch?v={entry['id']}",
            "duration": entry.get("duration"),
            "channel": entry.get("channel") or entry.get("uploader"),
        })
    return {
        "id": info.get("id"),
        "title": info.get("title"),
        "url": url or info.get("webpage_url"),
        "fetched": _now(),
        **layout.PROVENANCE,
        "entries": entries,
    }


def build_meta(info: dict) -> dict:
    """meta.json for one video: the useful part of yt-dlp's info JSON, stamped."""
    meta = {key: info[key] for key in META_FIELDS if info.get(key) is not None}
    return {**meta, **layout.PROVENANCE, "fetched": _now()}


def list_playlist(url: str, root: str, runner=run) -> dict:
    manifest = parse_flat_playlist(json.loads(runner(ytdlp("--flat-playlist", "-J", url))), url)
    name = manifest["id"] or "unnamed"
    layout.write_json(layout.playlist_path(root, name), manifest)
    return manifest


def fetch_meta(entry: dict, root: str, runner=run) -> bool:
    """Write meta.json unless it exists. True if it was written."""
    path = layout.item_path(root, entry["id"], "meta.json")
    if os.path.exists(path):
        return False
    info = json.loads(runner(ytdlp("-J", "--skip-download", entry["url"])))
    layout.write_json(path, build_meta(info))
    return True


def fetch_audio(entry: dict, root: str, sleep: tuple[float, float], archive: bool = True,
                runner=run) -> bool:
    """Write audio.wav, 44.1 kHz mono, unless it exists. True if it was written."""
    folder = layout.item_dir(root, entry["id"])
    wav = os.path.join(folder, "audio.wav")
    if os.path.exists(wav):
        return False
    os.makedirs(folder, exist_ok=True)
    source = _downloaded(folder, "audio.src")
    if source is None:
        runner(ytdlp("-f", "bestaudio", "-o", os.path.join(folder, "audio.src.%(ext)s"),
                     *_pacing(sleep), *_archive(root, "audio", archive), entry["url"]))
        source = _downloaded(folder, "audio.src")
        if source is None:
            raise RuntimeError(_missing("audio", entry["id"], archive))
    temporary = os.path.join(folder, "audio.tmp.wav")
    runner(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", source,
            "-vn", "-ac", "1", "-ar", str(AUDIO_RATE), temporary])
    os.replace(temporary, wav)
    os.remove(source)
    return True


def fetch_video(entry: dict, root: str, sleep: tuple[float, float], archive: bool = True,
                runner=run) -> bool:
    """Write video.mp4 -- video only, at most 1080p -- unless it exists.

    yt-dlp downloads to a `.part` file and renames on completion, so a
    video.mp4 that exists is a whole one.
    """
    folder = layout.item_dir(root, entry["id"])
    path = os.path.join(folder, "video.mp4")
    if os.path.exists(path):
        return False
    os.makedirs(folder, exist_ok=True)
    runner(ytdlp("-f", VIDEO_FORMAT, "-o", path, *_pacing(sleep),
                 *_archive(root, "video", archive), entry["url"]))
    if not os.path.exists(path):
        raise RuntimeError(_missing("video", entry["id"], archive))
    return True


def _downloaded(folder: str, stem: str) -> str | None:
    for name in sorted(os.listdir(folder)):
        if name.startswith(stem + ".") and not name.endswith((".part", ".ytdl")):
            return os.path.join(folder, name)
    return None


def _pacing(sleep: tuple[float, float]) -> list[str]:
    low, high = sleep
    if high <= 0:
        return []
    return ["--sleep-requests", "1", "--sleep-interval", str(low),
            "--max-sleep-interval", str(max(low, high))]


def _archive(root: str, stream: str, enabled: bool) -> list[str]:
    return ["--download-archive", layout.archive_path(root, stream)] if enabled else []


def _missing(stream: str, video_id: str, archive: bool) -> str:
    why = f"yt-dlp produced no {stream} for {video_id}"
    if archive:
        why += (f"; if it is listed in archive-{stream}.txt from an earlier run, "
                "remove that line or pass --no-archive")
    return why


def fetch(url: str, root: str, streams=("audio", "video"), limit: int | None = None,
          sleep: tuple[float, float] = (2.0, 6.0), archive: bool = True,
          runner=run, log=print) -> dict:
    """List the playlist at `url`, then fetch every entry. Failures are
    collected rather than raised, so one private video does not stop the rest."""
    manifest = list_playlist(url, root, runner)
    entries = manifest["entries"][:limit]
    log(f"{manifest['title'] or manifest['id']}: {len(entries)} video(s) -> {root}")
    done, failed = [], []
    for number, entry in enumerate(entries, 1):
        label = f"[{number}/{len(entries)}] {entry['id']} {(entry['title'] or '')[:50]}"
        try:
            steps = [("meta", fetch_meta(entry, root, runner))]
            if "audio" in streams:
                steps.append(("audio", fetch_audio(entry, root, sleep, archive, runner)))
            if "video" in streams:
                steps.append(("video", fetch_video(entry, root, sleep, archive, runner)))
            log(f"{label}: " + ", ".join(f"{name} {'new' if new else 'kept'}"
                                         for name, new in steps))
            done.append(entry["id"])
        except Exception as error:                              # noqa: BLE001
            log(f"{label}: FAILED {error}")
            failed.append((entry["id"], str(error)))
    return {"playlist": manifest["id"], "done": done, "failed": failed}
