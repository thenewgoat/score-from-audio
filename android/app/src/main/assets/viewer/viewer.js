// The score page. Verovio engraves the MusicXML to SVG and renders it to MIDI; Kotlin plays either the MIDI or
// the original recording and calls Viewer.at (MIDI time) or Viewer.atAudio (recording time) with the position.
// Everything is placed by written-tempo time ("score ms"), which the timemap ties to quarter notes, and the
// recording's anchors (sync.json) tie quarter notes to recording time.
let toolkit = null;
let xml = "";
let timemap = [];
let bars = [];       // [{id, ms, end, q, qEnd, label, box}]
let points = [];     // per bar: [{ms, x}] for the cursor
let anchors = [];
let lit = new Set();
let speed = 1.0;
let shownBar = -1;
const SCALE = 38;

verovio.module.onRuntimeInitialized = () => {
  toolkit = new verovio.toolkit();
  Android.onViewerReady();
};

function score() { return document.getElementById("score"); }

/** An element's box relative to the score, in CSS px. */
function box(element) {
  const s = score().getBoundingClientRect();
  const r = element.getBoundingClientRect();
  return { left: r.left - s.left, top: r.top - s.top, right: r.right - s.left, bottom: r.bottom - s.top };
}

function measureGeometry() {
  for (const bar of bars) {
    const element = document.getElementById(bar.id);
    bar.box = element ? box(element) : null;
  }
  // The cursor runs through each bar's note onsets: from the bar's left edge to its right, via every chord.
  points = bars.map((bar) => (bar.box ? [{ ms: bar.ms, x: bar.box.left + 6 }] : []));
  for (const event of timemap) {
    if (!event.on || !event.on.length) continue;
    const b = Timemap.barAt(bars, event.tstamp);
    if (b < 0 || !bars[b].box) continue;
    const element = document.getElementById(event.on[0]);
    if (!element) continue;
    const r = box(element);
    const list = points[b];
    if (list.length && list[list.length - 1].ms === event.tstamp) list[list.length - 1].x = r.left;
    else list.push({ ms: event.tstamp, x: r.left });
  }
  bars.forEach((bar, i) => { if (bar.box) points[i].push({ ms: bar.end, x: bar.box.right - 4 }); });
}

function showCursor(ms) {
  const cursor = document.getElementById("cursor");
  const b = Timemap.barAt(bars, ms);
  if (b < 0 || !bars[b].box || ms < 0) { cursor.style.display = "none"; return null; }
  const bar = bars[b];
  const x = Timemap.interpolate(points[b], ms, "ms", "x");
  Object.assign(cursor.style, {
    display: "block", left: x + "px", top: bar.box.top - 6 + "px", height: bar.box.bottom - bar.box.top + 12 + "px",
  });
  if (b !== shownBar) {
    shownBar = b;
    Android.onBarShown(b, bar.label);
  }
  return bar;
}

function show(ms) {
  const now = ms < 0 ? new Set() : Timemap.activeAt(timemap, ms);
  for (const id of lit) if (!now.has(id)) document.getElementById(id)?.classList.remove("playing");
  for (const id of now) if (!lit.has(id)) document.getElementById(id)?.classList.add("playing");
  lit = now;
  const bar = showCursor(ms);
  if (bar) {
    // Keep the playing system in view, turning the "page" a little before it reaches the bottom.
    const top = bar.box.top + score().getBoundingClientRect().top;
    const bottom = bar.box.bottom + score().getBoundingClientRect().top;
    if (top < 8 || bottom > window.innerHeight - 24) window.scrollBy({ top: top - window.innerHeight * 0.2, behavior: "smooth" });
  }
}

function barInfo(i) {
  const bar = bars[i];
  return [i, bar.ms, bar.end, Timemap.qToAudio(anchors, bar.q) * 1000, Timemap.qToAudio(anchors, bar.qEnd) * 1000];
}

window.Viewer = {
  /** Engraves `text` to fit `widthPx` and scroll vertically; `sync` is the recording's anchors, or []. */
  load(text, widthPx, sync) {
    xml = text;
    anchors = sync || [];
    toolkit.setOptions({
      scale: SCALE, pageWidth: Math.round((widthPx * 100) / SCALE), adjustPageHeight: true,
      breaks: "auto", header: "none", footer: "none", pageMarginLeft: 30, pageMarginRight: 30,
      pageMarginTop: 60, midiTempoAdjustment: speed,
    });
    if (!toolkit.loadData(xml)) {
      Android.onError("This score could not be displayed");
      return;
    }
    const s = score();
    s.innerHTML = '<div id="cursor"></div>';
    for (let page = 1; page <= toolkit.getPageCount(); page++) {
      const div = document.createElement("div");
      div.className = "page";
      div.innerHTML = toolkit.renderToSVG(page);
      s.appendChild(div);
    }
    // The timemap is taken at the written tempo; a speed change only re-renders the MIDI.
    timemap = toolkit.renderToTimemap({ includeMeasures: true });
    bars = Timemap.measures(timemap);
    for (const bar of bars) {
      let n = "";
      try { n = toolkit.getElementAttr(bar.id).n || ""; } catch (e) { /* keep the index */ }
      bar.label = n;
    }
    lit = new Set();
    shownBar = -1;
    requestAnimationFrame(() => {
      measureGeometry();
      Android.onMidi(toolkit.renderToMIDI(), Timemap.durationOf(timemap) / speed);
      Android.onRendered(bars.length);
    });
  },

  setSpeed(factor) {
    speed = factor;
    toolkit.setOptions({ midiTempoAdjustment: factor });
    Android.onMidi(toolkit.renderToMIDI(), Timemap.durationOf(timemap) / factor);
  },

  /** The MIDI player's position (its time runs at the speed chosen). */
  at(ms) { show(ms < 0 ? -1 : ms * speed); },

  /** The recording player's position. */
  atAudio(ms) { show(ms < 0 ? -1 : Timemap.qToMs(timemap, Timemap.audioToQ(anchors, ms / 1000))); },

  /** Converters for switching between the two players at the same place. */
  midiToAudio(ms) { return Timemap.qToAudio(anchors, Timemap.msToQ(timemap, ms * speed)) * 1000; },
  audioToMidi(ms) { return Timemap.qToMs(timemap, Timemap.audioToQ(anchors, ms / 1000)) / speed; },

  /** Shades bars `first` to `last` (indices), or clears the shading when first < 0. */
  setLoop(first, last) {
    score().querySelectorAll(".looped").forEach((e) => e.remove());
    if (first < 0) return;
    for (let i = first; i <= last && i < bars.length; i++) {
      const b = bars[i].box;
      if (!b) continue;
      const shade = document.createElement("div");
      shade.className = "looped";
      Object.assign(shade.style, { left: b.left + "px", top: b.top - 8 + "px", width: b.right - b.left + "px", height: b.bottom - b.top + 16 + "px" });
      score().appendChild(shade);
    }
  },

  /** MIDI at the written tempo, for sharing. */
  exportMidi() {
    toolkit.setOptions({ midiTempoAdjustment: 1.0 });
    const midi = toolkit.renderToMIDI();
    toolkit.setOptions({ midiTempoAdjustment: speed });
    Android.onExport("mid", midi);
  },

  /** A4 pages with the title, drawn at 200 dpi and handed over one by one for Kotlin to bind into a PDF. */
  async exportPdf() {
    try {
      const paper = new verovio.toolkit();
      paper.setOptions({ scale: 40, pageWidth: 2100, pageHeight: 2970, pageMarginLeft: 100, pageMarginRight: 100,
                         pageMarginTop: 100, pageMarginBottom: 100, breaks: "auto", header: "auto", footer: "none" });
      paper.loadData(xml);
      const count = paper.getPageCount();
      for (let page = 1; page <= count; page++) {
        const svg = paper.renderToSVG(page);
        const image = new Image();
        const url = URL.createObjectURL(new Blob([svg], { type: "image/svg+xml" }));
        await new Promise((resolve, reject) => { image.onload = resolve; image.onerror = reject; image.src = url; });
        const canvas = document.createElement("canvas");
        canvas.width = 1654; canvas.height = 2339;
        const context = canvas.getContext("2d");
        context.fillStyle = "#fff";
        context.fillRect(0, 0, canvas.width, canvas.height);
        context.drawImage(image, 0, 0, canvas.width, canvas.height);
        URL.revokeObjectURL(url);
        Android.onPdfPage(page, count, canvas.toDataURL("image/png").split(",")[1]);
      }
      paper.destroy?.();
    } catch (e) {
      Android.onError("The PDF could not be made");
    }
  },
};

score().addEventListener("click", (event) => {
  const s = score().getBoundingClientRect();
  const x = event.clientX - s.left, y = event.clientY - s.top;
  const i = bars.findIndex((bar) => bar.box && x >= bar.box.left && x <= bar.box.right && y >= bar.box.top - 12 && y <= bar.box.bottom + 12);
  if (i >= 0) Android.onBar(...barInfo(i));
});
