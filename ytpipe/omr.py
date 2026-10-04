"""Stage 4: read the sheet music off the video with Audiveris.

Four steps, each written to disk under `omr/` and skipped when already there:

  systems   The video is sampled inside roi.json's box and split wherever the
            picture changes. Each stable stretch is one displayed system --
            one line of the score -- kept as a lossless crop with the times it
            was on screen. Those times are the alignment's best prior: the
            bars of a system sound while it is shown.
  pages     Audiveris is built for pages, not strips. On a single system it
            found one staff of the two and no barlines; stacked five to a page
            it found the time signature and most barlines. So systems are
            stacked into page images, with the video's faint grey staff lines
            thresholded to black.
  audiveris `Audiveris.exe -batch -export`, the Windows install driven from
            WSL. It is handed Windows paths into a staging folder under the
            Windows temp directory, and its outputs are copied back.
  join      The pages' measures, concatenated into one two-hand score. Audiveris
            names parts per system -- a system whose brace it saw becomes one
            two-staff piano part, one whose brace it missed becomes two
            single-staff parts -- so hands are re-assigned bar by bar from
            whichever staves carry notes.
"""

import os
import shutil
import subprocess

import numpy as np

from ytpipe import layout

AUDIVERIS = os.environ.get("YTPIPE_AUDIVERIS", "/mnt/c/Program Files/Audiveris/Audiveris.exe")
SAMPLE_FPS = 2.0
CHANGE = 0.01          # fraction of the (downsampled, binarised) crop that must differ
MIN_SHOWN = 1.0        # seconds; shorter stretches are transitions, not systems
SYSTEMS_PER_PAGE = 5
INK = 210              # grey level below which a pixel is ink, after cropping


# --- systems -----------------------------------------------------------------

def frames(video: str, fps: float = SAMPLE_FPS):
    """(time, grey frame) at `fps`, decoded by ffmpeg, full resolution."""
    import cv2

    capture = cv2.VideoCapture(video)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if not width or not height:
        raise RuntimeError(f"cannot read the frame size of {video}")
    process = subprocess.Popen(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", video,
         "-vf", f"fps={fps},format=gray", "-f", "rawvideo", "-"],
        stdout=subprocess.PIPE)
    size, index = width * height, 0
    try:
        while True:
            raw = process.stdout.read(size)
            if len(raw) < size:
                break
            yield index / fps, np.frombuffer(raw, np.uint8).reshape(height, width)
            index += 1
    finally:
        process.stdout.close()
        process.wait()


def _signature(crop: np.ndarray) -> np.ndarray:
    return crop[::4, ::4] < 160


def _inside(time: float, regions: list[dict]):
    for region in regions:
        if region["t_start"] <= time <= region["t_end"]:
            return region
    return None


def segment(stream, regions: list[dict], fps: float = SAMPLE_FPS,
            change: float = CHANGE, min_shown: float = MIN_SHOWN) -> list[dict]:
    """Split sampled frames into stretches of one unchanging picture.

    Each stretch keeps its LAST frame -- by then any fade-in has finished --
    and its times. Stretches shorter than `min_shown` are transitions and are
    dropped; neighbours left showing the same picture are merged, which also
    folds a system shown twice in a row into one.
    """
    stretches, current = [], None
    for time, frame in stream:
        region = _inside(time, regions)
        if region is None:
            current = None
            continue
        x, y, w, h = region["box"]
        crop = frame[y:y + h, x:x + w]
        signature = _signature(crop)
        if (current is not None and current["signature"].shape == signature.shape
                and np.mean(current["signature"] != signature) <= change):
            current.update(t_off=time + 1 / fps, image=crop)
            continue
        current = {"t_on": time, "t_off": time + 1 / fps, "image": crop,
                   "signature": signature}
        stretches.append(current)

    kept = [s for s in stretches if s["t_off"] - s["t_on"] >= min_shown]
    merged = []
    for stretch in kept:
        last = merged[-1] if merged else None
        if (last is not None and last["signature"].shape == stretch["signature"].shape
                and np.mean(last["signature"] != stretch["signature"]) <= change):
            last.update(t_off=stretch["t_off"], image=stretch["image"])
        else:
            merged.append(dict(stretch))
    return [{"t_on": round(s["t_on"], 2), "t_off": round(s["t_off"], 2),
             "image": trim(s["image"])} for s in merged]


def trim(image: np.ndarray, margin: int = 6) -> np.ndarray:
    """Cut a crop down to its white paper, dropping black letterboxing."""
    rows = np.where(image.mean(1) > 100)[0]
    if len(rows) == 0:
        return image
    # Columns are judged within the paper's rows only; over the full height a
    # short strip in a tall letterboxed frame averages below the threshold.
    columns = np.where(image[rows.min():rows.max() + 1].mean(0) > 100)[0]
    if len(columns) == 0:
        return image
    top, bottom = rows.min() + margin, rows.max() - margin
    left, right = columns.min() + margin, columns.max() - margin
    if bottom <= top or right <= left:
        return image
    return image[top:bottom + 1, left:right + 1]


def write_systems(systems: list[dict], folder: str) -> list[dict]:
    import cv2

    os.makedirs(folder, exist_ok=True)
    listing = []
    for number, system in enumerate(systems, 1):
        name = f"{number:03d}.png"
        cv2.imwrite(os.path.join(folder, name), system["image"])
        listing.append({"file": name, "t_on": system["t_on"], "t_off": system["t_off"],
                        "height": int(system["image"].shape[0]),
                        "width": int(system["image"].shape[1])})
    return listing


# --- pages -------------------------------------------------------------------

def stack(images: list[np.ndarray], gap: int = 60, pad: int = 100,
          ink: int = INK, scale: float = 1.0) -> np.ndarray:
    """Systems one above another on white paper, ink forced to black.

    `scale` enlarges the systems first (cubic interpolation, before the ink
    threshold), for symbols too small to read at the video's resolution.
    """
    if scale != 1.0:
        import cv2

        images = [cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                  for image in images]
        gap, pad = int(gap * scale), int(pad * scale)
    width = max(image.shape[1] for image in images)
    rows = []
    for image in images:
        right = np.full((image.shape[0], width - image.shape[1]), 255, np.uint8)
        rows += [np.hstack([image, right]), np.full((gap, width), 255, np.uint8)]
    page = np.vstack(rows[:-1])
    page = np.where(page < ink, 0, 255).astype(np.uint8)
    return np.pad(page, pad, constant_values=255)


def page_groups(count: int, per_page: int = SYSTEMS_PER_PAGE) -> list[list[int]]:
    return [list(range(i, min(i + per_page, count))) for i in range(0, count, per_page)]


# --- Audiveris ---------------------------------------------------------------

def windows_temp() -> str:
    """The Windows temp directory as a WSL path."""
    override = os.environ.get("YTPIPE_WIN_STAGE")
    if override:
        return override
    done = subprocess.run(["cmd.exe", "/c", "echo %TEMP%"], capture_output=True, text=True,
                          cwd="/mnt/c")
    windows = done.stdout.strip().splitlines()[-1]
    return subprocess.run(["wslpath", "-u", windows], capture_output=True,
                          text=True).stdout.strip()


def to_windows(path: str) -> str:
    return subprocess.run(["wslpath", "-w", path], capture_output=True, text=True,
                          check=True).stdout.strip()


def audiveris(pages: list[str], out_dir: str, stage_root: str | None = None,
              runner=None, log=print, constants: dict | None = None) -> list[str]:
    """Run Audiveris over page images; returns the .mxl paths, in page order.

    Audiveris writes `pNN.mxl`, or -- when it decides a page starts a new
    movement, as an indented first system can make it -- `pNN.mvt1.mxl`,
    `pNN.mvt2.mxl`, ...; the movements are that page's music in order. A page
    that already has its output in `out_dir` is not run again. All the rest go
    in one invocation, so the JVM starts once.
    """
    def outputs(folder: str, stem: str) -> list[str]:
        if not os.path.isdir(folder):
            return []
        whole = [n for n in os.listdir(folder) if n == stem + ".mxl"]
        parts = sorted((n for n in os.listdir(folder)
                        if n.startswith(stem + ".mvt") and n.endswith(".mxl")),
                       key=lambda n: int(n[len(stem) + 4:-4]))
        return whole + parts

    stems = [os.path.splitext(os.path.basename(p))[0] for p in pages]
    todo = [p for p, stem in zip(pages, stems) if not outputs(out_dir, stem)]
    if todo:
        stage = os.path.join(stage_root or windows_temp(), "ytpipe",
                             os.path.basename(os.path.dirname(os.path.dirname(out_dir)))
                             or "item")
        shutil.rmtree(stage, ignore_errors=True)
        os.makedirs(os.path.join(stage, "in"))
        staged = []
        for page in todo:
            target = os.path.join(stage, "in", os.path.basename(page))
            shutil.copyfile(page, target)
            staged.append(to_windows(target))
        options = [a for key, value in (constants or {}).items()
                   for a in ("-constant", f"{key}={value}")]
        command = [AUDIVERIS, "-batch", "-export", *options, "-output",
                   to_windows(os.path.join(stage, "out")), "--", *staged]
        log(f"audiveris: {len(todo)} page(s)")
        (runner or (lambda c: subprocess.run(c, capture_output=True, text=True)))(command)
        for page in todo:
            stem = os.path.splitext(os.path.basename(page))[0]
            for name in outputs(os.path.join(stage, "out"), stem):
                shutil.copyfile(os.path.join(stage, "out", name), os.path.join(out_dir, name))
        shutil.rmtree(stage, ignore_errors=True)
    return [os.path.join(out_dir, name) for stem in stems for name in outputs(out_dir, stem)]


# --- join --------------------------------------------------------------------

def staves(score) -> list:
    """Every staff of an Audiveris score, top to bottom (part-list order)."""
    return list(score.parts)


def _has_notes(measure) -> bool:
    return measure is not None and len(measure.recurse().notes) > 0


def _clef_of(measure, staff) -> str:
    from music21 import clef

    found = measure.recurse().getElementsByClass(clef.Clef) if measure is not None else []
    if not found:
        found = staff.recurse().getElementsByClass(clef.Clef)
    return found[0].sign if found else "G"


def bar_numbers(score) -> list[int]:
    """The bar numbers of a page, in the order `hands` returns them."""
    return sorted({m.number for part in staves(score) for m in part.getElementsByClass("Measure")})


def system_starts(score) -> set[int]:
    """Bars that begin a new system, from Audiveris's system breaks."""
    from music21 import layout as music21_layout

    out = set()
    for part in staves(score):
        for measure in part.getElementsByClass("Measure"):
            if any(getattr(item, "isNew", False)
                   for item in measure.getElementsByClass(music21_layout.SystemLayout)):
                out.add(measure.number)
    return out


def hands(score) -> list[tuple]:
    """(right hand measure or None, left hand measure or None), per bar.

    The top two staves that carry notes in a bar are the right and left hand.
    A bar with notes on one staff only is placed by that staff's clef.
    """
    parts = staves(score)
    by_part = [{m.number: m for m in part.getElementsByClass("Measure")} for part in parts]
    numbers = sorted({n for measures in by_part for n in measures})
    out = []
    for number in numbers:
        active = [(measures[number], part) for measures, part in zip(by_part, parts)
                  if _has_notes(measures.get(number))]
        if len(active) >= 2:
            out.append((active[0][0], active[1][0]))
        elif len(active) == 1:
            measure, part = active[0]
            out.append((None, measure) if _clef_of(measure, part) == "F" else (measure, None))
        else:
            out.append((None, None))
    return out


PLAIN = (4.0, 3.0, 2.0, 1.5, 1.0, 0.75, 0.5, 0.375, 0.25, 0.125)
WRITABLE = tuple(sorted(PLAIN + (4 / 3, 2 / 3, 1 / 3, 1 / 6), reverse=True))


def rests(quarters: float) -> list:
    """Rests filling `quarters`, each one a length MusicXML can write alone.

    A bar Audiveris read as 4.5 or 11/3 quarters is kept at that length, but a
    single rest of it has no written form, and the export refuses it. Plain
    lengths are tried first; triplet ones only when plain ones cannot fill it.
    """
    from music21 import note

    for sizes in (PLAIN, WRITABLE):
        out, left = [], float(quarters)
        while left > 1e-6:
            size = next((w for w in sizes if w <= left + 1e-6), None)
            if size is None:
                break
            out.append(size)
            left -= size
        if left <= 1e-6:
            break
    return [note.Rest(quarterLength=size) for size in out]


SPANNER_ATTRIBUTES = ("type", "placement", "lineType", "number")


def carry_spanners(source, memo: dict) -> int:
    """Rebuild a page's spanners (slurs, ottavas...) on the copies of its notes.

    Copying bars one by one leaves every spanner behind, because music21
    keeps them beside the bars rather than in them: joined scores had no
    slurs at all. A spanner whose notes were all copied is recreated on the
    copies, in the part the first of them landed in. Returns how many.
    """
    carried = 0
    for spanner in source.recurse().getElementsByClass("Spanner"):
        spanned = spanner.getSpannedElements()
        copies = [memo.get(id(element)) for element in spanned]
        if not copies or any(c is None for c in copies):
            continue
        part = copies[0].getContextByClass("Part") or copies[0].getContextByClass("PartStaff")
        if part is None:
            continue
        fresh = type(spanner)()
        for name in SPANNER_ATTRIBUTES:
            if hasattr(spanner, name):
                try:
                    setattr(fresh, name, getattr(spanner, name))
                except Exception:                              # noqa: BLE001
                    pass
        fresh.addSpannedElements(copies)
        part.insert(0, fresh)
        carried += 1
    return carried


def split_rests(measure) -> None:
    """Replace each rest no single MusicXML rest can write with ones that can.

    A whole-bar rest keeps its written length (see layout.parse_score), and in
    a bar Audiveris read as 4.5 quarters that length has no one written form.
    """
    for rest in list(measure.recurse().getElementsByClass("Rest")):
        if rest.duration.type != "complex":
            continue
        site, at = rest.activeSite, float(rest.offset)
        site.remove(rest)
        for piece in rests(float(rest.quarterLength)):
            site.insert(at, piece)
            at += float(piece.quarterLength)


def join(page_scores: list, out_path: str, page_numbers: list[int] | None = None,
         folder: str | None = None) -> dict:
    """Concatenate pages into one score with a right-hand and a left-hand part."""
    import copy

    from music21 import clef, meter, note, stream

    right, left = stream.PartStaff(id="RH"), stream.PartStaff(id="LH")
    right.partName, left.partName = "Right hand", "Left hand"
    number, lengths, signature, page_of, system_of = 0, [], None, [], []
    previous_page, system = None, -1
    # A page Audiveris could not read is absent from `page_scores`; the pages
    # after it keep their own numbers, which say when their systems were shown.
    for page, score in zip(page_numbers or range(1, len(page_scores) + 1), page_scores):
        found = score.recurse().getElementsByClass(meter.TimeSignature)
        if signature is None and found:
            signature = found[0].ratioString
        starts = system_starts(score)
        # One memo per page: every copied note is recorded against its
        # original, so the page's slurs (and other spanners, which music21
        # keeps outside the bars) can be rebuilt on the copies below.
        memo: dict = {}
        for index, (bar, (upper, lower)) in enumerate(zip(bar_numbers(score), hands(score))):
            number += 1
            page_of.append(page)
            # Which system of its page the bar is on: counted from the page's
            # first bar, and on at every system break Audiveris marked (a
            # new movement file of the same page starts one too).
            if page != previous_page:
                system, previous_page = 0, page
            elif index == 0 or bar in starts:
                system += 1
            system_of.append(system)
            length = max((m.duration.quarterLength for m in (upper, lower) if m is not None),
                         default=0.0)
            for part, measure, sign in ((right, upper, "G"), (left, lower, "F")):
                if measure is None:
                    fresh = stream.Measure(number=number)
                    fresh.append(rests(length or 4.0))
                else:
                    fresh = copy.deepcopy(measure, memo)
                    fresh.number = number
                    split_rests(fresh)
                    # A hand Audiveris read short is padded to the longer
                    # one, so the two hands keep one bar grid; otherwise
                    # every later bar of one hand drifts against the other.
                    short = length - fresh.duration.quarterLength
                    at = fresh.duration.quarterLength
                    for rest in rests(short) if short > 1e-6 else []:
                        fresh.insert(at, rest)
                        at += rest.quarterLength
                if number == 1:
                    if not fresh.recurse().getElementsByClass(clef.Clef):
                        fresh.insert(0, clef.TrebleClef() if sign == "G" else clef.BassClef())
                    if signature and not fresh.recurse().getElementsByClass(meter.TimeSignature):
                        fresh.insert(0, meter.TimeSignature(signature))
                part.append(fresh)
            lengths.append(float(length))
        carry_spanners(score, memo)
    score = stream.Score([right, left])
    if folder:
        layout.set_title(score, folder)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    # Audiveris bars are already notated; re-notating them trips on overfull
    # bars with voices, which are exactly the ones the editor must show as-is.
    score.write("musicxml", fp=out_path, makeNotation=False)
    # Irregular against the commonest bar length, not the first time
    # signature: Audiveris often misses a change of metre (4/4 intro, then
    # 12/8), and then every bar after it would be flagged.
    from collections import Counter

    usual = Counter(lengths).most_common(1)[0][0] if lengths else 0.0
    irregular = [{"measure": i + 1, "page": page_of[i], "quarters": q}
                 for i, q in enumerate(lengths) if abs(q - usual) > 1e-6]
    # Which page each measure came from: with the page's systems' display
    # times, a bound on when that measure can sound.
    return {"measures": number, "time_signature": signature, "usual_quarters": usual,
            "irregular": irregular,
            "measure_page": page_of, "measure_system": system_of}


CLEF_LINES = {"G": "2", "F": "4", "C": "3"}


def load_page(path: str):
    """Parse one Audiveris page, repairing clefs placed on impossible lines.

    Audiveris sometimes writes a clef on staff line 0 or 6, which music21
    refuses outright; the page is then lost whole. Such a clef is put back on
    its sign's usual line. Whole-bar rests keep their written length.
    """
    import re
    import zipfile

    import music21

    with zipfile.ZipFile(path) as archive:
        name = next(n for n in archive.namelist()
                    if n.endswith(".xml") and not n.startswith("META-INF"))
        text = archive.read(name).decode("utf-8")

    def repair(match):
        sign, line = match.group(2), match.group(4)
        if sign in CLEF_LINES and not 1 <= int(line) <= 5:
            line = CLEF_LINES[sign]
        return f"{match.group(1)}{sign}{match.group(3)}{line}{match.group(5)}"

    text = re.sub(r"(<clef[^>]*>\s*<sign>)(\w+)(</sign>\s*<line>)(-?\d+)(</line>)", repair, text)
    # A whole-bar rest keeps its written length (see layout.parse_score).
    text = re.sub(r'<rest\s+measure="yes"\s*/?>', lambda m: "<rest/>" if m.group(0).endswith("/>")
                  else "<rest>", text)
    return music21.converter.parse(text, format="musicxml")


# --- the stage ---------------------------------------------------------------

def run(video_id: str, root: str, force: bool = False, log=print) -> dict:
    import cv2
    import music21

    folder = layout.item_dir(root, video_id)
    omr = os.path.join(folder, "omr")
    roi_path = os.path.join(folder, "roi.json")
    if not os.path.exists(roi_path):
        raise FileNotFoundError(f"{roi_path} missing; run `ytpipe roi {video_id}` first")
    roi = layout.read_json(roi_path)
    if roi["type"] != "pages":
        raise NotImplementedError(f"omr reads `pages` videos only, not {roi['type']!r}")
    if force:
        shutil.rmtree(omr, ignore_errors=True)

    listing_path = os.path.join(omr, "systems.json")
    if os.path.exists(listing_path):
        listing = layout.read_json(listing_path)["systems"]
    else:
        systems = segment(frames(os.path.join(folder, "video.mp4")), roi["regions"])
        listing = write_systems(systems, os.path.join(omr, "systems"))
        layout.write_json(listing_path, {"systems": listing, **layout.PROVENANCE})
    log(f"{video_id}: {len(listing)} systems")

    pages_dir = os.path.join(omr, "pages")
    os.makedirs(pages_dir, exist_ok=True)
    pages = []
    for index, group in enumerate(page_groups(len(listing)), 1):
        path = os.path.join(pages_dir, f"p{index:02d}.png")
        if not os.path.exists(path):
            images = [cv2.imread(os.path.join(omr, "systems", listing[i]["file"]), 0)
                      for i in group]
            cv2.imwrite(path, stack(images))
        pages.append(path)

    results = audiveris(pages, pages_dir, log=log)
    read = {os.path.basename(r)[:3] for r in results}
    if len(read) < len(pages):
        missing = sorted(set(os.path.basename(p)[:3] for p in pages) - read)
        log(f"{video_id}: audiveris produced nothing for {', '.join(missing)}")
    scores = [load_page(path) for path in results]
    numbers = [int(os.path.basename(path)[1:3]) for path in results]
    report = join(scores, os.path.join(omr, "score.musicxml"), numbers, folder)
    report.update(pages=len(pages), pages_read=len(read), systems=len(listing),
                  page_systems=page_groups(len(listing)))
    layout.write_json(os.path.join(omr, "report.json"), {**report, **layout.PROVENANCE})
    log(f"{video_id}: {report['measures']} measures, {len(report['irregular'])} "
        f"of irregular length -> omr/score.musicxml")
    return report
