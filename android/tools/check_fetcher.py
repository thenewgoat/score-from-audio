"""Manual check of fetcher.fetch on the desktop, with the app's pinned packages. Needs the network.

    python3.13 -m venv /tmp/fetchcheck && /tmp/fetchcheck/bin/pip install yt-dlp==2026.8.19 yt-dlp-ejs==0.8.0 certifi
    /tmp/fetchcheck/bin/python android/tools/check_fetcher.py [--qjs /path/to/qjs] [--cancel-after N] [url ...]

--cancel-after N turns the cancel flag on after N checks of it, which lands in
extraction for small N: the result should be {"cancelled": true} with no file.
"""

import argparse
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app", "src", "main", "python"))
import fetcher  # noqa: E402

DEFAULT_URLS = ["https://www.youtube.com/watch?v=jNQXAC9IVRw", "https://www.instagram.com/reel/CDUMkliABpa/"]


class Progress:
    def onProgress(self, done, total):
        print(f"  {done}/{total}", end="\r")


class CancelAfter:
    def __init__(self, checks):
        self.checks = checks
        self.seen = 0

    def get(self):
        self.seen += 1
        return self.checks is not None and self.seen > self.checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qjs")
    parser.add_argument("--cancel-after", type=int)
    parser.add_argument("urls", nargs="*", default=DEFAULT_URLS)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as out:
        for url in args.urls:
            for use_js in ([False, True] if args.qjs else [False]):
                result = json.loads(fetcher.fetch(url, out, None, args.qjs, use_js, Progress(), CancelAfter(args.cancel_after)))
                files = [f for f in os.listdir(out) if not f.startswith(".")]
                print(url, "js" if use_js else "no-js", "->", result, "files:", files)
                for f in files:
                    os.remove(os.path.join(out, f))


if __name__ == "__main__":
    main()
