"""Engraving MusicXML to SVG, in a process of its own.

Verovio only works on a process's MAIN thread. Not "one thread at a time", and
not "the first thread to claim it" -- measured, a render on any background
thread returns zero pages, with no concurrency involved and whether or not the
main thread has ever rendered. It fails by logging to stderr and returning a
document that is complete except for its glyph outlines, so the page arrives
blank and every structural check still passes.

That is fatal for a threaded HTTP server, which hands each request to a worker
thread. Rather than restructure the server around one library's constraint,
rendering happens in a subprocess, where it gets a main thread to itself.

The cost is 0.15s against 0.02s in-process, and it parallelises: four
concurrent renders finish in the time of one. For something asked for
explicitly, that is the right trade.
"""

import json
import os
import re
import subprocess
import sys

DEFAULT_OPTIONS = {
    "scale": 40, "pageWidth": 2200, "pageHeight": 2970,
    "adjustPageHeight": True, "footer": "none", "header": "none",
    "breaks": "auto", "spacingStaff": 8, "spacingSystem": 10,
}


def require_glyphs(svg: str) -> None:
    """Refuse a page whose glyphs are referenced but never defined.

    This is what a fontless verovio produces: correct structure, nothing drawn.
    Served as-is it reaches the browser as a blank page with no clue why.
    """
    referenced = set(re.findall(r'<use[^>]*?(?:xlink:)?href="#([^"]+)"', svg))
    if not referenced:
        return
    defined = set(re.findall(r'id="([^"]+)"', svg))
    missing = referenced - defined
    if missing:
        raise ValueError(
            f"verovio drew {len(missing)} glyphs it never defined "
            f"(e.g. {sorted(missing)[0]}) -- its fonts did not load")


def render_here(path: str, page: int = 1, scale: int = 40) -> tuple[str, int]:
    """Render in THIS process. Only valid on the main thread -- see the module
    docstring. `render` is the one to call from anywhere else."""
    import verovio

    bundled = os.path.join(os.path.dirname(verovio.__file__), "data")
    if os.path.isdir(bundled):
        verovio.setDefaultResourcePath(bundled)

    toolkit = verovio.toolkit()
    toolkit.setOptions({**DEFAULT_OPTIONS, "scale": scale})
    if not toolkit.loadFile(path):
        raise ValueError("verovio could not read the score")
    pages = toolkit.getPageCount()
    if pages < 1:
        raise ValueError("verovio produced no pages -- its fonts did not load")
    svg = toolkit.renderToSVG(max(1, min(page, pages)))
    require_glyphs(svg)
    return svg, pages


def render(path: str, page: int = 1, scale: int = 40) -> tuple[str, int]:
    """Render in a subprocess. Safe to call from any thread."""
    finished = subprocess.run(
        [sys.executable, "-m", "evaluation.engrave", path, str(page), str(scale)],
        capture_output=True, text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if finished.returncode != 0:
        detail = (finished.stderr or "").strip().splitlines()
        raise ValueError(detail[-1] if detail else "engraving failed")
    payload = json.loads(finished.stdout)
    return payload["svg"], payload["pages"]


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print("usage: python -m evaluation.engrave SCORE [page] [scale]", file=sys.stderr)
        return 2
    path = argv[0]
    page = int(argv[1]) if len(argv) > 1 else 1
    scale = int(argv[2]) if len(argv) > 2 else 40
    try:
        svg, pages = render_here(path, page, scale)
    except Exception as error:                                 # noqa: BLE001
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1
    json.dump({"svg": svg, "pages": pages}, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
