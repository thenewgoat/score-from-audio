"""Engrave a score to SVG, every page, in a process of its own.

Verovio renders only on a process's main thread (see evaluation/engrave.py),
and the editor's server answers on worker threads; so it runs here instead.
Prints {"pages": [svg, ...], "page_of": {note id: page}} as JSON.
"""

import json
import re
import sys


def render(path: str, scale: int = 45) -> dict:
    import os

    import verovio

    from evaluation.engrave import DEFAULT_OPTIONS, require_glyphs

    bundled = os.path.join(os.path.dirname(verovio.__file__), "data")
    if os.path.isdir(bundled):
        verovio.setDefaultResourcePath(bundled)
    toolkit = verovio.toolkit()
    toolkit.setOptions({**DEFAULT_OPTIONS, "scale": scale, "pageWidth": 2400})
    if not toolkit.loadFile(path):
        raise ValueError("verovio could not read the score")
    pages, page_of = [], {}
    for number in range(1, toolkit.getPageCount() + 1):
        svg = toolkit.renderToSVG(number)
        require_glyphs(svg)
        pages.append(svg)
        for name in re.findall(r'<g id="(row\d+)" class="note', svg):
            page_of[name] = number
    if not pages:
        raise ValueError("verovio produced no pages")
    return {"pages": pages, "page_of": page_of}


if __name__ == "__main__":
    print(json.dumps(render(sys.argv[1])))
