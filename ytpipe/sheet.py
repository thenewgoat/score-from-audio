"""The video's own sheet music, compiled into pages: a PDF to read beside MuseScore.

The systems stage 4 cut from the video (omr/systems/, one per line of music
shown) are stacked onto A4-proportioned pages in the order they were shown,
unthresholded, each labelled with when it is on screen, so a bar found in the
editor can be found here by its time.
"""

import os

from ytpipe import layout

A4 = 297 / 210                 # height / width
MARGIN = 0.05                  # of the page width
GAP = 0.025                    # between systems, of the page width


def _clock(seconds: float) -> str:
    return f"{int(seconds // 60)}:{int(seconds % 60):02d}"


def build(folder: str, out: str | None = None) -> str:
    """Write sheet.pdf for one item; returns its path. Rebuilt only when stale."""
    systems_path = os.path.join(folder, "omr", "systems.json")
    if not os.path.exists(systems_path):
        raise FileNotFoundError(f"{systems_path} missing; run `ytpipe omr` first")
    out = out or os.path.join(folder, "sheet.pdf")
    if os.path.exists(out) and os.path.getmtime(out) >= os.path.getmtime(systems_path):
        return out
    laid = pages(folder)
    laid[0].save(out, "PDF", resolution=200.0, save_all=True, append_images=laid[1:])
    return out


def pages(folder: str) -> list:
    """The PDF's pages as greyscale images."""
    from PIL import Image, ImageDraw, ImageFont

    systems_path = os.path.join(folder, "omr", "systems.json")
    systems = layout.read_json(systems_path)["systems"]
    images = [Image.open(os.path.join(folder, "omr", "systems", s["file"])).convert("L")
              for s in systems]
    width = max(image.width for image in images)
    margin, gap = int(width * MARGIN), int(width * GAP)
    page_w = width + 2 * margin
    page_h = int(page_w * A4)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", max(14, width // 70))
        big = ImageFont.truetype("DejaVuSans-Bold.ttf", max(20, width // 40))
    except OSError:
        font = big = ImageFont.load_default()
    title = layout.piece_title(folder)

    laid, page, y = [], None, 0
    label_h = int(font.size * 1.6) if hasattr(font, "size") else 16

    def new_page():
        nonlocal page, y
        page = Image.new("L", (page_w, page_h), 255)
        draw = ImageDraw.Draw(page)
        draw.text((margin, margin // 2), f"{title}", font=big, fill=0)
        draw.text((page_w - margin, margin // 2), f"page {len(laid) + 1}", font=font,
                  fill=90, anchor="ra")
        y = margin // 2 + int(big.size * 1.8 if hasattr(big, "size") else 30)
        laid.append(page)

    new_page()
    for system, image in zip(systems, images):
        # A system too tall for any page (a whole page shown at once) is
        # scaled to fit rather than cut.
        room = page_h - margin - (margin // 2 + label_h * 3)
        if image.height + label_h > room:
            scale = (room - label_h) / image.height
            image = image.resize((int(image.width * scale), int(image.height * scale)))
        if y + label_h + image.height > page_h - margin:
            new_page()
        draw = ImageDraw.Draw(page)
        draw.text((margin, y), f"{_clock(system['t_on'])}–{_clock(system['t_off'])}",
                  font=font, fill=110)
        y += label_h
        page.paste(image, (margin, y))
        y += image.height + gap

    return laid
