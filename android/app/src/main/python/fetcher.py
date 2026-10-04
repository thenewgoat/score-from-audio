"""Audio from a YouTube or Instagram link, with the yt-dlp bundled in the app.

Kotlin calls fetch() through Chaquopy and gets JSON back; no Python exception
crosses into Kotlin. The caller decides retries and turns errors into words
(dev.scorefromaudio.pipeline.Links).
"""

import json
import os

import yt_dlp


class _Cancelled(Exception):
    pass


class _Quiet:
    """Collects yt-dlp's errors instead of printing them, and stops it once cancelled.

    yt-dlp sends every step's screen message (to_screen: "Downloading webpage",
    "Downloading player API JSON", ...) to logger.debug whenever a logger is set,
    whatever "quiet" and "verbose" say, so raising here stops extraction at its
    next step instead of only once the download's progress hook runs.
    """

    def __init__(self, cancelled):
        self.errors = []
        self._cancelled = cancelled

    def _check(self):
        if self._cancelled.get():
            raise _Cancelled()

    def debug(self, msg):
        self._check()

    def info(self, msg):
        self._check()

    def warning(self, msg):
        self._check()

    def error(self, msg):
        self.errors.append(msg)


def fetch(url, out_dir, cookie_file, qjs_path, use_js, progress, cancelled):
    log = _Quiet(cancelled)

    def hook(d):
        if cancelled.get():
            raise _Cancelled()
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or -1
            progress.onProgress(int(d.get("downloaded_bytes") or 0), int(total))

    opts = {
        "format": "bestaudio[ext=m4a]/bestaudio",
        "outtmpl": os.path.join(out_dir, "%(extractor)s-%(id)s.%(ext)s"),
        "noplaylist": True,
        "socket_timeout": 30,
        "retries": 3,
        "overwrites": True,
        # A .part left by an attempt in another format must not be resumed into this one.
        "continuedl": False,
        # Only the extractors for the links Links.site() lets through; never the generic one.
        "allowed_extractors": ["youtube.*", "instagram.*"],
        "verbose": False,
        "progress_hooks": [hook],
        "logger": log,
        "cachedir": os.path.join(out_dir, ".cache"),
        "js_runtimes": {"quickjs": {"path": qjs_path}} if use_js and qjs_path else {},
    }
    if cookie_file:
        opts["cookiefile"] = cookie_file
    try:
        os.makedirs(out_dir, exist_ok=True)
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
        if info is None:
            return json.dumps({"ok": False, "error": "\n".join(log.errors) or "yt-dlp returned no info"})
        download = (info.get("requested_downloads") or [{}])[0]
        return json.dumps({
            "ok": True,
            "path": download.get("filepath"),
            "title": info.get("title"),
            "extractor": info.get("extractor_key"),
            "duration": info.get("duration"),
            "ext": info.get("ext"),
        })
    except _Cancelled:
        return json.dumps({"ok": False, "cancelled": True})
    except Exception as e:  # noqa: BLE001 - everything becomes an error message for Kotlin
        if cancelled.get() or isinstance(getattr(e, "exc_info", [None, None])[1], _Cancelled):
            return json.dumps({"ok": False, "cancelled": True})
        text = "\n".join(log.errors) or str(e)
        return json.dumps({"ok": False, "error": text})
