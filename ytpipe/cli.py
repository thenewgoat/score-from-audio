"""ytpipe: YouTube playlist -> audio, video and notation regions; downloads are for your own use only."""

import argparse
import sys

from ytpipe import layout

LATER = {"report": 6}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="ytpipe", description=__doc__)
    parser.add_argument("--data", default=None,
                        help=f"data root (default $YTPIPE_DATA or {layout.DEFAULT_ROOT})")
    commands = parser.add_subparsers(dest="command", required=True)

    fetch = commands.add_parser("fetch", help="stages 1-2: list a playlist, download audio + video")
    fetch.add_argument("url", help="playlist URL (a single video URL also works)")
    fetch.add_argument("--limit", type=int, default=None, help="only the first N entries")
    fetch.add_argument("--only", choices=("audio", "video"), default=None,
                       help="download one stream only (meta.json is always written)")
    fetch.add_argument("--sleep", type=float, nargs=2, default=(2.0, 6.0),
                       metavar=("MIN", "MAX"),
                       help="seconds yt-dlp waits before each download (0 0 for none)")
    fetch.add_argument("--no-archive", action="store_true",
                       help="do not use or update the yt-dlp download archives")

    roi = commands.add_parser("roi", help="stage 3: mark where the notation is, and when")
    roi.add_argument("video_id")
    roi.add_argument("--type", dest="kind", choices=("pages", "scroll", "falling_notes"))
    roi.add_argument("--region", action="append", default=[],
                     help="T0-T1:x,y,w,h, repeatable; skips the interactive step")
    roi.add_argument("--ui", choices=("auto", "window", "sheet"), default="auto",
                     help="auto uses a window when OpenCV can open one, else the sheet")
    roi.add_argument("--every", type=float, default=5.0, help="seconds between thumbnails")
    roi.add_argument("--force", action="store_true", help="replace an existing roi.json")

    omr = commands.add_parser("omr", help="stage 4: read the sheet music off the video (Audiveris)")
    omr.add_argument("video_ids", nargs="*", help="default: every item with a pages roi.json")
    omr.add_argument("--force", action="store_true", help="redo from the systems onwards")

    aligned = commands.add_parser(
        "align", help="stage 5: transcribe the audio, put both scores on its clock, compare notes")
    aligned.add_argument("video_ids", nargs="*", help="default: every item with an OMR score")
    aligned.add_argument("--force", action="store_true", help="redo alignment.json")

    edit = commands.add_parser("edit", help="the editor: compare OMR and audio per note, in a browser")
    edit.add_argument("video_id", nargs="?", help="piece to open first (all aligned ones are listed)")
    edit.add_argument("--port", type=int, default=8765)

    sheet = commands.add_parser("sheet", help="the video's sheet music compiled into a PDF, to read beside MuseScore")
    sheet.add_argument("video_ids", nargs="*", help="default: every item with OMR systems")
    sheet.add_argument("--copy-to", default=None, metavar="DIR",
                       help="also copy each PDF here, named '<piece> [<id>].pdf'")

    export = commands.add_parser(
        "export", help="stage 6: pieces saved in MuseScore as an ASAP-shaped dataset")
    export.add_argument("--format", choices=("asap",), default="asap")
    export.add_argument("--out", required=True, help="dataset folder to write")
    export.add_argument("--performances", required=True, metavar="DIR",
                        help="Transkun's MIDI of each piece, named <video_id>.mid")
    export.add_argument("--only", nargs="*", default=(), metavar="ID", help="only these pieces")
    export.add_argument("--asap", default=None, metavar="DIR",
                        help="also write <out>-with-asap: this ASAP checkout plus the export")
    export.add_argument("--split-file", default=None,
                        help="with --asap: the split the export's pieces are added to")
    export.add_argument("--test", nargs="*", default=(), metavar="ID",
                        help="with --asap: pieces put on the test side (the rest train)")

    for name, stage in LATER.items():
        commands.add_parser(name, help=f"stage {stage} (not built yet)")

    args = parser.parse_args(argv)
    root = layout.data_root(args.data)

    if args.command == "fetch":
        from ytpipe.fetch import fetch as run_fetch

        streams = (args.only,) if args.only else ("audio", "video")
        result = run_fetch(args.url, root, streams, args.limit, tuple(args.sleep),
                           not args.no_archive)
        print(f"{len(result['done'])} done, {len(result['failed'])} failed")
        return 1 if result["failed"] else 0
    if args.command == "roi":
        from ytpipe.roi import mark

        mark(args.video_id, root, args.kind, args.region, args.ui, args.every, args.force)
        return 0
    if args.command in ("omr", "align"):
        import os

        if args.command == "omr":
            from ytpipe.omr import run as stage

            def ready(name):
                path = layout.item_path(root, name, "roi.json")
                return os.path.exists(path) and layout.read_json(path)["type"] == "pages"
        else:
            from ytpipe.align import run as stage

            def ready(name):
                return os.path.exists(layout.item_path(root, name, "omr/score.musicxml"))

        ids = args.video_ids or sorted(name for name in os.listdir(root) if ready(name))
        failed = 0
        for video_id in ids:
            try:
                stage(video_id, root, args.force)
            except Exception as error:                          # noqa: BLE001
                failed += 1
                print(f"{video_id}: FAILED {type(error).__name__}: {error}", file=sys.stderr)
        return 1 if failed else 0
    if args.command == "sheet":
        import os
        import re
        import shutil

        from ytpipe.sheet import build

        ids = args.video_ids or sorted(
            name for name in os.listdir(root)
            if os.path.exists(layout.item_path(root, name, "omr/systems.json")))
        for video_id in ids:
            folder = layout.item_dir(root, video_id)
            path = build(folder)
            if args.copy_to:
                os.makedirs(args.copy_to, exist_ok=True)
                name = re.sub(r'[\\/:*?"<>|]', "", layout.piece_title(folder))
                path = shutil.copyfile(path, os.path.join(args.copy_to, f"{name} [{video_id}].pdf"))
            print(path)
        return 0
    if args.command == "export":
        from ytpipe.export import combine, export as run_export

        reports = run_export(root, args.out, args.performances, tuple(args.only))
        if args.asap:
            if not args.split_file:
                parser.error("--asap needs --split-file")
            split = combine(args.asap, args.out, args.out.rstrip("/") + "-with-asap",
                            args.split_file, tuple(args.test))
            print(split)
        return 0 if reports else 1
    if args.command == "edit":
        from ytpipe.edit import serve

        serve(root, args.video_id, args.port)
        return 0
    print(f"ytpipe {args.command}: stage {LATER[args.command]} is not built yet", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
