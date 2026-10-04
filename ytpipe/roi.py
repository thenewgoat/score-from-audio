"""Stage 3: where on screen the notation is, and when it is there.

A person marks it; nothing here guesses. The output is roi.json:

    {"type": "pages",
     "regions": [{"t_start": 3.2, "t_end": 181.0, "box": [x, y, w, h]}],
     "frame_size": [1920, 1080], ...}

`type` says how the notation moves -- `pages` (a static page turned now and
then), `scroll` (a strip that slides past) or `falling_notes` (a piano-roll
visualiser, no notation to read) -- because the later stages read each
differently. `box` is in the video's own pixels.

Three ways to give it, the first that applies winning:

  flags     `--type pages --region 3.2-181:40,120,1840,700`, repeatable. No
            window and no prompt, so it can be scripted or run remotely.
  window    an OpenCV window to drag the box in, when OpenCV has a GUI
            backend and a display. The environment's OpenCV is headless, so
            this is the exception rather than the rule.
  sheet     a contact sheet of thumbnails every few seconds, labelled with
            their times, and a full-size reference frame with a pixel grid, to
            read the times and coordinates from; then a prompt in the terminal.
            Works anywhere, WSL included.
"""

import datetime
import os
import re
import subprocess

from ytpipe import layout

TYPES = ("pages", "scroll", "falling_notes")
THUMB_EVERY = 5.0


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


# --- roi.json ---------------------------------------------------------------

def parse_region(text: str, duration: float | None = None) -> dict:
    """`T0-T1:x,y,w,h` (or `T0-T1 x,y,w,h`) to a region. `end` stands for the
    video's duration. Times may be seconds or m:ss."""
    match = re.fullmatch(r"\s*([\d.:]+)\s*-\s*([\d.:]+|end)\s*[: ]\s*"
                         r"(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*", text)
    if not match:
        raise ValueError(f"region {text!r} is not T0-T1:x,y,w,h")
    start, end, *box = match.groups()
    if end == "end":
        if duration is None:
            raise ValueError("'end' needs the video's duration, and meta.json has none")
        t_end = float(duration)
    else:
        t_end = _seconds(end)
    return {"t_start": _seconds(start), "t_end": t_end, "box": [int(v) for v in box]}


def _seconds(text: str) -> float:
    seconds = 0.0
    for part in text.split(":"):
        seconds = seconds * 60 + float(part)
    return round(seconds, 3)


def validate(roi: dict, duration: float | None = None,
             frame_size: tuple[int, int] | None = None) -> dict:
    """Refuse an roi that later stages could not use. Returns it unchanged."""
    if roi.get("type") not in TYPES:
        raise ValueError(f"type must be one of {', '.join(TYPES)}, not {roi.get('type')!r}")
    regions = roi.get("regions")
    if not regions:
        raise ValueError("an roi needs at least one region")
    previous_end = None
    for index, region in enumerate(regions):
        where = f"region {index}"
        start, end, box = region.get("t_start"), region.get("t_end"), region.get("box")
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            raise ValueError(f"{where}: t_start and t_end must be numbers")
        if not 0 <= start < end:
            raise ValueError(f"{where}: needs 0 <= t_start < t_end, got {start}-{end}")
        if duration is not None and end > duration + 1.0:
            raise ValueError(f"{where}: ends at {end}s, past the video's {duration}s")
        if previous_end is not None and start < previous_end:
            raise ValueError(f"{where}: starts at {start}s, before the previous region "
                             f"ends ({previous_end}s); regions are in order and disjoint")
        previous_end = end
        if (not isinstance(box, list) or len(box) != 4
                or not all(isinstance(v, int) for v in box)):
            raise ValueError(f"{where}: box must be [x, y, w, h] in whole pixels")
        x, y, w, h = box
        if x < 0 or y < 0 or w <= 0 or h <= 0:
            raise ValueError(f"{where}: box {box} has a negative origin or no area")
        if frame_size is not None and (x + w > frame_size[0] or y + h > frame_size[1]):
            raise ValueError(f"{where}: box {box} runs outside the "
                             f"{frame_size[0]}x{frame_size[1]} frame")
    return roi


def build(kind: str, regions: list[dict], frame_size=None, method: str = "") -> dict:
    roi = {"type": kind, "regions": sorted(regions, key=lambda r: r["t_start"])}
    if frame_size:
        roi["frame_size"] = list(frame_size)
    return {**roi, "method": method, "created": _now(), **layout.PROVENANCE}


# --- thumbnails and the contact sheet ----------------------------------------

def thumbnail_time(name: str) -> float:
    """t0012.5.jpg -> 12.5"""
    return float(name[1:-4])


def thumbnails(video: str, folder: str, every: float = THUMB_EVERY, runner=None) -> list[str]:
    """One full-size frame every `every` seconds, named by its time. Kept once made.

    Extracted into a scratch directory and renamed on success, so a directory
    of thumbnails that exists is a complete one.
    """
    if os.path.isdir(folder) and os.listdir(folder):
        return sorted(os.path.join(folder, n) for n in os.listdir(folder))
    scratch = folder + ".tmp"
    os.makedirs(scratch, exist_ok=True)
    for name in os.listdir(scratch):
        os.remove(os.path.join(scratch, name))
    command = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", video,
               "-vf", f"fps=1/{every}", "-q:v", "3", os.path.join(scratch, "%05d.jpg")]
    (runner or (lambda c: subprocess.run(c, check=True)))(command)
    # The fps filter emits its n-th frame (from 1) at (n - 1) * every seconds.
    for name in sorted(os.listdir(scratch)):
        time = (int(name[:-4]) - 1) * every
        os.replace(os.path.join(scratch, name), os.path.join(scratch, f"t{time:06.1f}.jpg"))
    if os.path.isdir(folder):
        os.rmdir(folder)
    os.replace(scratch, folder)
    return sorted(os.path.join(folder, n) for n in os.listdir(folder))


def contact_sheet(paths: list[str], out: str, columns: int = 6, width: int = 320) -> str:
    """Tile the thumbnails with their times written on them."""
    import cv2
    import numpy as np

    tiles = []
    for path in paths:
        image = cv2.imread(path)
        height = round(image.shape[0] * width / image.shape[1])
        tile = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        label = _clock(thumbnail_time(os.path.basename(path)))
        cv2.rectangle(tile, (0, 0), (8 + 11 * len(label), 24), (0, 0, 0), -1)
        cv2.putText(tile, label, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    height = max(tile.shape[0] for tile in tiles)
    blank = np.zeros((height, width, 3), np.uint8)
    tiles = [cv2.copyMakeBorder(t, 0, height - t.shape[0], 0, 0, cv2.BORDER_CONSTANT)
             for t in tiles]
    tiles += [blank] * (-len(tiles) % columns)
    rows = [np.hstack(tiles[i:i + columns]) for i in range(0, len(tiles), columns)]
    cv2.imwrite(out, np.vstack(rows))
    return out


def reference_frame(path: str, out: str, step: int = 100) -> str:
    """A full-size frame with a labelled pixel grid, to read box coordinates off."""
    import cv2

    image = cv2.imread(path)
    height, width = image.shape[:2]
    for x in range(0, width, step):
        cv2.line(image, (x, 0), (x, height), (0, 255, 255) if x % 500 else (0, 0, 255), 1)
        cv2.putText(image, str(x), (x + 2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (0, 0, 255), 1, cv2.LINE_AA)
    for y in range(0, height, step):
        cv2.line(image, (0, y), (width, y), (0, 255, 255) if y % 500 else (0, 0, 255), 1)
        cv2.putText(image, str(y), (2, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imwrite(out, image)
    return out


def _clock(seconds: float) -> str:
    return f"{int(seconds // 60)}:{seconds % 60:04.1f}"


def frame_size(path: str) -> tuple[int, int]:
    import cv2

    height, width = cv2.imread(path).shape[:2]
    return width, height


def nearest(paths: list[str], time: float) -> str:
    return min(paths, key=lambda p: abs(thumbnail_time(os.path.basename(p)) - time))


# --- the ways of giving it ----------------------------------------------------

def gui_available() -> bool:
    """True when OpenCV can open a window: it has a GUI backend and a display."""
    import cv2

    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return False
    try:
        cv2.namedWindow("ytpipe-probe")
        cv2.destroyWindow("ytpipe-probe")
        return True
    except cv2.error:
        return False


def ask_type(ask=input) -> str:
    while True:
        answer = ask(f"video type [{'/'.join(TYPES)}]: ").strip()
        if answer in TYPES:
            return answer
        print(f"  one of {', '.join(TYPES)}")


def ask_regions(duration, size, ask=input) -> list[dict]:
    """Region lines from the terminal until a blank one."""
    print("regions as T0-T1 x,y,w,h (times in s or m:ss, 'end' for the end); blank to finish")
    regions = []
    while True:
        line = ask(f"region {len(regions) + 1}: ").strip()
        if not line:
            if regions:
                return regions
            print("  at least one region")
            continue
        try:
            region = parse_region(line, duration)
            validate({"type": TYPES[0], "regions": regions + [region]}, duration, size)
            regions.append(region)
        except ValueError as error:
            print(f"  {error}")


def window_regions(thumbs: list[str], duration, size, ask=input) -> list[dict]:
    """Times from the terminal, the box dragged in a window on the frame at t_start."""
    import cv2

    regions = []
    while True:
        line = ask(f"region {len(regions) + 1} times T0-T1 (blank to finish): ").strip()
        if not line:
            if regions:
                return regions
            continue
        try:
            times = parse_region(f"{line}:0,0,1,1", duration)
        except ValueError as error:
            print(f"  {error}")
            continue
        image = cv2.imread(nearest(thumbs, times["t_start"]))
        x, y, w, h = cv2.selectROI("drag the notation box, then Enter", image, False)
        cv2.destroyAllWindows()
        region = {**times, "box": [int(x), int(y), int(w), int(h)]}
        try:
            validate({"type": TYPES[0], "regions": regions + [region]}, duration, size)
            regions.append(region)
        except ValueError as error:
            print(f"  {error}")


def mark(video_id: str, root: str, kind: str | None = None, regions: list[str] = (),
         ui: str = "auto", every: float = THUMB_EVERY, force: bool = False,
         ask=input, log=print) -> dict | None:
    """Stage 3 for one video. Returns the roi written, or None if one existed."""
    out = layout.item_path(root, video_id, "roi.json")
    if os.path.exists(out) and not force:
        log(f"{out} exists; --force to redo it")
        return None
    video = layout.item_path(root, video_id, "video.mp4")
    if not os.path.exists(video):
        raise FileNotFoundError(f"{video} missing; run `ytpipe fetch` first")
    meta_path = layout.item_path(root, video_id, "meta.json")
    duration = layout.read_json(meta_path).get("duration") if os.path.exists(meta_path) else None

    frames = layout.item_path(root, video_id, "frames")
    thumbs = thumbnails(video, os.path.join(frames, "thumbs"), every)
    if not thumbs:
        raise RuntimeError(f"ffmpeg extracted no frames from {video}")
    size = frame_size(thumbs[0])

    if regions:
        if kind is None:
            raise ValueError("--region needs --type")
        parsed, method = [parse_region(r, duration) for r in regions], "flags"
    else:
        if ui == "window" or (ui == "auto" and gui_available()):
            method = "window"
            kind = kind or ask_type(ask)
            parsed = window_regions(thumbs, duration, size, ask)
        else:
            method = "sheet"
            sheet = contact_sheet(thumbs, os.path.join(frames, "contact_sheet.png"))
            grid = reference_frame(thumbs[len(thumbs) // 2],
                                   os.path.join(frames, "reference_grid.png"))
            log(f"contact sheet:   {sheet}\nreference frame: {grid}  "
                f"({size[0]}x{size[1]}, grid every 100px)\n"
                f"full-size frames: {os.path.dirname(thumbs[0])}/t<seconds>.jpg")
            kind = kind or ask_type(ask)
            parsed = ask_regions(duration, size, ask)

    roi = build(kind, parsed, size, method)
    validate(roi, duration, size)
    layout.write_json(out, roi)
    log(f"wrote {out}")
    return roi
