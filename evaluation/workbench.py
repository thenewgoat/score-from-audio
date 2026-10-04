"""A local workbench for looking at what the transcriber did, and why.

Accuracy work needs a fast loop. Every experiment so far has been a script
written by hand, run once, and thrown away -- which is why nothing could be
compared against anything else. This puts the whole pipeline behind one page:
load a source, see the notes, move a parameter, see the notes again.

The design turns on one measurement. Running the model costs seconds; planning
and notation cost milliseconds. So the model runs ONCE per source and its
probability maps are cached to disk, and every knob after that -- thresholds
included, because the decoder reads the cached maps rather than the audio -- is
instant. Only loading a new source is slow.

Sources may be a local file or a URL. A URL is fetched with yt-dlp, so a video
can play in one window while the tool works on its audio in another.
"""

import hashlib
import json
import mimetypes
import os
import posixpath
import re
import subprocess
import urllib.parse
from fractions import Fraction
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

GONE = (ConnectionResetError, BrokenPipeError, ConnectionAbortedError)
CHUNK = 256 * 1024


def cache_key(source: str) -> str:
    return hashlib.sha1(source.encode()).hexdigest()[:16]


def fetch(source: str, cache_dir: str) -> tuple[str, str]:
    """Get a local WAV for `source`, downloading it first if it is a URL.

    Returns (wav path, title). Both the download and the decode are cached, so
    re-opening a source the tool has seen costs nothing.
    """
    os.makedirs(cache_dir, exist_ok=True)
    key = cache_key(source)
    wav = os.path.join(cache_dir, f"{key}.wav")
    meta_path = os.path.join(cache_dir, f"{key}.json")
    if os.path.exists(wav) and os.path.exists(meta_path):
        return wav, json.load(open(meta_path)).get("title", source)

    title = os.path.basename(source)
    local = source
    if source.startswith(("http://", "https://")):
        pattern = os.path.join(cache_dir, f"{key}.%(ext)s")
        subprocess.run(
            ["yt-dlp", "-q", "--no-warnings", "-f", "bestaudio/best",
             "-o", pattern, "--print-to-file", "%(title)s",
             os.path.join(cache_dir, f"{key}.title"), source],
            check=True)
        found = [f for f in os.listdir(cache_dir)
                 if f.startswith(key) and not f.endswith((".wav", ".json", ".title"))]
        if not found:
            raise RuntimeError("yt-dlp produced no audio")
        local = os.path.join(cache_dir, found[0])
        title_file = os.path.join(cache_dir, f"{key}.title")
        if os.path.exists(title_file):
            title = open(title_file).read().strip() or source

    subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", local,
         "-ac", "1", "-ar", "22050", wav],
        check=True)
    json.dump({"title": title, "source": source}, open(meta_path, "w"))
    return wav, title


def maps_for(wav: str, cache_dir: str) -> dict:
    """The model's probability maps, computed once and cached.

    This is the only slow step, and separating it is the whole point: with the
    maps on disk every threshold becomes a re-decode of a few milliseconds
    rather than another pass of the network.
    """
    key = cache_key(os.path.abspath(wav))
    path = os.path.join(cache_dir, f"{key}.maps.npz")
    if os.path.exists(path):
        stored = np.load(path)
        return {"note": stored["note"], "onset": stored["onset"]}

    from transcribe.detect import load_audio, posteriorgrams

    maps = posteriorgrams(load_audio(wav))
    # Uncompressed on purpose. These are a few megabytes, and compressing them
    # traded 3 seconds of decompression on every reopen against disk that is
    # not scarce -- measured, after wondering why a cached source was slow.
    np.savez(path, note=maps["note"], onset=maps["onset"])
    return {"note": maps["note"], "onset": maps["onset"]}


def transcribe_from_maps(maps: dict, params: dict) -> dict:
    """Decode, plan, and notate only if asked.

    Notation is skipped by default because it costs about fifteen times what
    the rest does -- 1.5s against 0.4s, measured -- and while a threshold is
    being moved it is the piano roll and the counts that are being read, not
    the engraving. Asking for a score is a deliberate act.
    """
    from transcribe.audio import GRIDS, Event, build_score, plan
    from transcribe.detect import decode_to_notes
    from transcribe.validate import validate

    from transcribe.detect import frames_for_tempo

    bpm_for_floor = float(params.get("bpm") or 120.0)
    detections = decode_to_notes(
        maps,
        onset_threshold=float(params.get("onset_threshold", 0.5)),
        frame_threshold=float(params.get("frame_threshold", 0.3)),
        min_note_frames=int(params.get("min_note_frames")
                            or frames_for_tempo(bpm_for_floor)),
    )
    events = [Event(d.onset, d.offset, d.pitch) for d in detections]
    bpm = float(params.get("bpm") or 120.0)
    grid = GRIDS.get(params.get("grid", "eighth"), Fraction(1, 2))

    placements = plan(
        events, bpm, grid=grid,
        max_fill=Fraction(str(params.get("max_fill", 2))).limit_denominator(16),
        window=float(params.get("cluster_window", 0.06)),
        split_pitch=int(params.get("split_pitch", 60)),
    )

    result = {
        "notes": [{"on": round(d.onset, 4), "off": round(d.offset, 4),
                   "pitch": d.pitch, "amp": round(d.amplitude, 3)} for d in detections],
        "placements": len(placements),
        "chords": sum(1 for p in placements if len(p.pitches) > 1),
        "bpm": bpm,
    }
    if placements and params.get("notate"):
        try:
            score = build_score(placements, bpm)
            path = os.path.join(params["_out"], "current.musicxml")
            score.write("musicxml", fp=path)
            result["musicxml"] = os.path.basename(path)
            result["problems"] = [str(p) for p in validate(path)]
            result["measures"] = len(score.recurse().getElementsByClass("Measure"))
            try:
                _, result["pages"] = engrave(path)
            except Exception:                                  # noqa: BLE001
                result["pages"] = 0
        except Exception as error:                             # noqa: BLE001 - shown in the UI
            result["problems"] = [f"{type(error).__name__}: {error}"]
    return result


def engrave(musicxml_path: str, page: int = 1, scale: int = 40) -> tuple[str, int]:
    """Render a page of a score, and say how many there are.

    Delegates to `evaluation.engrave`, which runs verovio in a subprocess
    because verovio only works on a process's main thread and this server hands
    every request to a worker thread. SVG rather than PDF: it draws in
    hundredths of a second, scales without blurring, and stays addressable, so
    a later version can colour a note the transcriber was unsure of.
    """
    from evaluation.engrave import render

    return render(musicxml_path, page, scale)


def estimate_tempo(wav: str) -> float:
    from transcribe.audio import estimate_tempo as tracked_tempo

    return tracked_tempo(wav)


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Transcription workbench</title>
<style>
:root{--bg:#fbfbfa;--fg:#1a1a19;--dim:#6b6b66;--line:#e2e2dd;--card:#fff;--sunk:#f3f3ef;
      --accent:#2f5fd0;--good:#1a7f4b;--bad:#b3261e;--ref:#2f5fd0;--det:#d0342f}
@media(prefers-color-scheme:dark){:root{--bg:#141418;--fg:#e8e8e4;--dim:#9a9a94;
  --line:#2c2c33;--card:#1c1c21;--sunk:#232329;--accent:#7aa2f7;--good:#4ac57e;
  --bad:#ef6b62;--ref:#7aa2f7;--det:#ef6b62}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 ui-sans-serif,system-ui,sans-serif}
header{display:flex;gap:.5rem;padding:.6rem .9rem;border-bottom:1px solid var(--line);
  position:sticky;top:0;background:var(--bg);z-index:3;align-items:center}
header input{flex:1;min-width:0}
main{padding:.9rem;display:grid;gap:.9rem}
input,select,button{font:inherit;padding:.35rem .5rem;border-radius:6px;
  border:1px solid var(--line);background:var(--card);color:var(--fg)}
button{cursor:pointer;font-weight:600}
button:hover{border-color:var(--accent)}
button:disabled{opacity:.5;cursor:default}
.card{background:var(--card);border:1px solid var(--line);border-radius:9px;padding:.8rem .9rem}
.title{font-size:.75rem;text-transform:uppercase;letter-spacing:.06em;color:var(--dim);
  margin-bottom:.5rem}
.knobs{display:grid;grid-template-columns:9rem 1fr 3.4rem;gap:.35rem .6rem;align-items:center}
.knobs label{color:var(--dim);font-size:.85rem}
.knobs output{font-variant-numeric:tabular-nums;text-align:right;font-size:.85rem}
input[type=range]{width:100%;padding:0}
.stats{display:flex;gap:1.1rem;flex-wrap:wrap;font-variant-numeric:tabular-nums}
.stats b{font-size:1.15rem;display:block}
.stats span{color:var(--dim);font-size:.78rem}
audio{width:100%}
svg{display:block;width:100%;height:auto;background:var(--sunk);border-radius:6px}
.problems{color:var(--bad);font-size:.85rem;max-height:7rem;overflow:auto}
.ok{color:var(--good)}
.busy{color:var(--dim)}
.legend{font-size:.78rem;color:var(--dim);margin-top:.35rem}
#score{background:#fff;border-radius:6px;overflow:auto;max-height:78vh}
#score svg{width:100%;height:auto;background:#fff}
#pager button{padding:.1rem .45rem;font-size:.8rem}
.sw{display:inline-block;width:.7rem;height:.7rem;vertical-align:-1px;border-radius:2px}
@media(min-width:1100px){main{grid-template-columns:20rem 1fr;align-items:start}
  .full{grid-column:1/-1}}
</style></head><body>
<header>
  <input id="src" placeholder="YouTube / Instagram URL, or a local file path" spellcheck="false">
  <button id="go" onclick="load()">Load</button>
  <span id="status" class="busy"></span>
</header>
<main>
  <div>
    <div class="card">
      <div class="title">source</div>
      <div id="title" style="margin-bottom:.5rem">nothing loaded</div>
      <audio id="au" controls preload="metadata"></audio>
    </div>
    <div class="card" style="margin-top:.9rem">
      <div class="title">detection &mdash; re-decodes instantly</div>
      <div class="knobs" id="detKnobs"></div>
      <div class="title" style="margin-top:.9rem">notation</div>
      <div class="knobs" id="planKnobs"></div>
    </div>
  </div>
  <div>
    <div class="card">
      <div class="stats" id="stats"><span class="busy">load a source to begin</span></div>
      <div id="problems" class="problems" style="margin-top:.5rem"></div>
    </div>
    <div class="card" style="margin-top:.9rem">
      <div class="title">piano roll</div>
      <div id="roll"></div>
      <div class="legend"><span class="sw" style="background:var(--det)"></span> detected</div>
    </div>
    <div class="card" style="margin-top:.9rem">
      <div class="title">score
        <span id="pager" style="float:right;font-weight:400;text-transform:none"></span>
      </div>
      <div id="score" class="busy">press <b>notate</b> to engrave</div>
    </div>
  </div>
</main>
<script>
const DET=[["onset_threshold","onset threshold",0.05,0.95,0.05,0.5],
           ["frame_threshold","frame threshold",0.05,0.95,0.05,0.3],
           ["min_note_frames","min length (frames)",1,40,1,11]];
const PLAN=[["bpm","tempo (bpm)",40,240,0.1,120],
            ["cluster_window","cluster window (s)",0.01,0.2,0.005,0.06],
            ["max_fill","max fill (quarters)",0.5,8,0.5,2],
            ["split_pitch","staff split (midi)",40,84,1,60]];
let params={}, loaded=false, pending=false;

function knob(box,[key,label,min,max,step,def]){
  params[key]=def;
  const id="k_"+key;
  box.insertAdjacentHTML("beforeend",
    `<label for="${id}">${label}</label>
     <input type="range" id="${id}" min="${min}" max="${max}" step="${step}" value="${def}">
     <output id="o_${key}">${def}</output>`);
  const el=document.getElementById(id);
  el.addEventListener("input",()=>{
    params[key]=parseFloat(el.value);
    document.getElementById("o_"+key).textContent=el.value;
  });
  el.addEventListener("change",run);
}
DET.forEach(k=>knob(document.getElementById("detKnobs"),k));
PLAN.forEach(k=>knob(document.getElementById("planKnobs"),k));

// grid is a choice, not a slider
document.getElementById("planKnobs").insertAdjacentHTML("beforeend",
  `<label for="k_grid">grid</label>
   <select id="k_grid"><option value="sixteenth">sixteenth</option>
   <option value="eighth" selected>eighth</option><option value="quarter">quarter</option></select>
   <span></span>`);
document.getElementById("k_grid").addEventListener("change",e=>{
  params.grid=e.target.value; run();
});
params.grid="eighth";

const status=(t,cls)=>{const s=document.getElementById("status");
  s.textContent=t; s.className=cls||"busy"};

async function load(){
  const src=document.getElementById("src").value.trim();
  if(!src)return;
  document.getElementById("go").disabled=true;
  status("loading…");
  try{
    const r=await fetch("/load",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({source:src})});
    const d=await r.json();
    if(d.error){status(d.error,"problems");return}
    document.getElementById("title").textContent=d.title;
    document.getElementById("au").src="/audio?k="+encodeURIComponent(d.key);
    if(d.bpm){params.bpm=Math.round(d.bpm*10)/10;
      const b=document.getElementById("k_bpm");
      if(b){b.value=params.bpm;document.getElementById("o_bpm").textContent=params.bpm}}
    // the shortest note that can be real depends on the tempo, so the floor
    // follows it rather than sitting at a fixed 128ms
    if(d.min_note_frames){params.min_note_frames=d.min_note_frames;
      const m=document.getElementById("k_min_note_frames");
      if(m){m.value=d.min_note_frames;
        document.getElementById("o_min_note_frames").textContent=d.min_note_frames}}
    loaded=true; status(""); await run();
  }catch(e){status(String(e),"problems")}
  finally{document.getElementById("go").disabled=false}
}

function notate(){ run(true) }

async function run(notate){
  if(!loaded||pending)return;
  pending=true; status(notate?"engraving…":"…");
  try{
    const r=await fetch("/run",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify(Object.assign({notate:!!notate}, params))});
    const d=await r.json();
    if(d.error){status(d.error,"problems");return}
    render(d); status("");
  }catch(e){status(String(e),"problems")}
  finally{pending=false}
}

function render(d){
  const p=d.problems||[];
  document.getElementById("stats").innerHTML=
    `<div><b>${d.notes.length}</b><span>notes</span></div>
     <div><b>${d.placements}</b><span>placements</span></div>
     <div><b>${d.chords}</b><span>chords</span></div>
     <div><b>${d.measures??"–"}</b><span>bars</span></div>
     <div><b class="${p.length?"":"ok"}">${d.measures==null?"–":p.length}</b><span>problems</span></div>
     <div><button onclick="notate()" id="nb">notate</button>
       ${d.musicxml?` <a href="/out/${d.musicxml}?t=${Date.now()}" download>download</a>`:""}</div>`;
  document.getElementById("problems").innerHTML=p.slice(0,8).map(x=>`<div>${x}</div>`).join("");
  drawRoll(d.notes);
  if(d.pages!==undefined){ pages=d.pages; page=Math.min(page,pages||1); showScore() }
}

let page=1, pages=0;

function showScore(){
  const box=document.getElementById("score");
  if(!pages){box.innerHTML='<span class="busy">press <b>notate</b> to engrave</span>';
    document.getElementById("pager").innerHTML="";return}
  page=Math.max(1,Math.min(page,pages));
  box.innerHTML=`<img src="/score.svg?page=${page}&t=${Date.now()}" alt="score">`;
  document.getElementById("pager").innerHTML=
    `<button onclick="turn(-1)">&lsaquo;</button>
     page ${page} of ${pages}
     <button onclick="turn(1)">&rsaquo;</button>
     <button onclick="window.open('/print','_blank')">print / PDF</button>`;
}
function turn(d){ page+=d; showScore() }

function drawRoll(notes){
  const box=document.getElementById("roll");
  if(!notes.length){box.innerHTML="";return}
  const W=1000,H=300,PAD=26;
  const t1=Math.max(...notes.map(n=>n.off));
  const lo=Math.min(...notes.map(n=>n.pitch))-2, hi=Math.max(...notes.map(n=>n.pitch))+2;
  const x=t=>PAD+(W-PAD-6)*t/t1, y=p=>H-PAD-(H-PAD-6)*(p-lo)/(hi-lo);
  let g="";
  for(let p=Math.ceil(lo/12)*12;p<=hi;p+=12){
    g+=`<line x1="${PAD}" y1="${y(p)}" x2="${W}" y2="${y(p)}" stroke="currentColor"
         opacity=".12"/><text x="2" y="${y(p)+4}" font-size="10" fill="currentColor"
         opacity=".5">C${Math.floor(p/12)-1}</text>`;
  }
  for(let s=0;s<=t1;s+=Math.max(1,Math.round(t1/10))){
    g+=`<line x1="${x(s)}" y1="6" x2="${x(s)}" y2="${H-PAD}" stroke="currentColor"
         opacity=".08"/><text x="${x(s)}" y="${H-8}" font-size="10" fill="currentColor"
         opacity=".5">${s}s</text>`;
  }
  const h=Math.max(2,(H-PAD-6)/(hi-lo)*0.8);
  for(const n of notes){
    g+=`<rect x="${x(n.on).toFixed(1)}" y="${(y(n.pitch)-h/2).toFixed(1)}"
         width="${Math.max(1.5,x(n.off)-x(n.on)).toFixed(1)}" height="${h.toFixed(1)}"
         fill="var(--det)" opacity="${(0.35+0.65*n.amp).toFixed(2)}" rx="1"/>`;
  }
  box.innerHTML=`<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none"
    style="height:300px">${g}</svg>`;
}
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    cache_dir = ".workbench"
    out_dir = ".workbench/out"
    state: dict = {}

    def log_message(self, *args):
        pass

    def _headers(self, status, content_type, length, extra=()):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        if not content_type.startswith(("audio", "video")):
            self.send_header("Cache-Control", "no-store, must-revalidate")
        for key, value in extra:
            self.send_header(key, value)
        self.end_headers()

    def _send(self, body: bytes, content_type: str, status: int = 200, extra=()):
        try:
            self._headers(status, content_type, len(body), extra)
            self.wfile.write(body)
        except GONE:
            pass

    def _json(self, payload, status: int = 200):
        self._send(json.dumps(payload).encode(), "application/json", status)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        try:
            return json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return {}

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/":
            return self._send(PAGE.encode(), "text/html; charset=utf-8")
        if parsed.path == "/audio":
            wav = self.state.get("wav")
            return self._send_file(wav) if wav else self._send(b"", "text/plain", 404)
        if parsed.path == "/score.svg":
            query = urllib.parse.parse_qs(parsed.query)
            path = os.path.join(self.out_dir, "current.musicxml")
            if not os.path.exists(path):
                return self._send(b"", "image/svg+xml", 404)
            try:
                svg, _ = engrave(path, int(query.get("page", ["1"])[0]))
            except Exception as error:                         # noqa: BLE001
                return self._send(str(error).encode(), "text/plain", 500)
            return self._send(svg.encode(), "image/svg+xml")
        if parsed.path == "/print":
            path = os.path.join(self.out_dir, "current.musicxml")
            if not os.path.exists(path):
                return self._send(b"nothing engraved yet", "text/plain", 404)
            try:
                first, pages = engrave(path)
                body = [first] + [engrave(path, n)[0] for n in range(2, pages + 1)]
            except Exception as error:                         # noqa: BLE001
                return self._send(str(error).encode(), "text/plain", 500)
            html = ("<!doctype html><meta charset=utf-8><title>score</title>"
                    "<style>body{margin:0;background:#fff}"
                    "svg{display:block;margin:0 auto;page-break-after:always}"
                    "@media print{@page{margin:10mm}}</style>"
                    + "".join(body))
            return self._send(html.encode(), "text/html; charset=utf-8")
        if parsed.path.startswith("/out/"):
            name = posixpath.basename(urllib.parse.unquote(parsed.path))
            path = os.path.join(self.out_dir, name)
            return self._send_file(path) if os.path.isfile(path) else self._send(
                b"not found", "text/plain", 404)
        self._send(b"not found", "text/plain", 404)

    def _send_file(self, path: str):
        kind = mimetypes.guess_type(path)[0] or "application/octet-stream"
        size = os.path.getsize(path)
        match = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        start, end, partial = 0, size - 1, False
        if match and (match.group(1) or match.group(2)):
            partial = True
            if match.group(1):
                start = int(match.group(1))
                if match.group(2):
                    end = min(int(match.group(2)), size - 1)
            else:
                start = max(0, size - int(match.group(2)))
        if start >= size:
            return self._send(b"", kind, 416, [("Content-Range", f"bytes */{size}")])
        extra = [("Accept-Ranges", "bytes")]
        if partial:
            extra.append(("Content-Range", f"bytes {start}-{end}/{size}"))
        try:
            self._headers(206 if partial else 200, kind, end - start + 1, extra)
            with open(path, "rb") as handle:
                handle.seek(start)
                remaining = end - start + 1
                while remaining > 0:
                    piece = handle.read(min(CHUNK, remaining))
                    if not piece:
                        break
                    self.wfile.write(piece)
                    remaining -= len(piece)
        except GONE:
            pass

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        body = self._body()
        try:
            if path == "/load":
                source = (body.get("source") or "").strip()
                if not source:
                    return self._json({"error": "no source"}, 400)
                wav, title = fetch(source, self.cache_dir)
                self.state["wav"] = wav
                self.state["maps"] = maps_for(wav, self.cache_dir)
                try:
                    bpm = estimate_tempo(wav)
                except Exception:                              # noqa: BLE001
                    bpm = None
                from transcribe.detect import frames_for_tempo

                return self._json({"key": cache_key(source), "title": title,
                                   "bpm": bpm,
                                   "min_note_frames": frames_for_tempo(bpm or 120.0)})

            if path == "/run":
                if "maps" not in self.state:
                    return self._json({"error": "nothing loaded"}, 400)
                os.makedirs(self.out_dir, exist_ok=True)
                body["_out"] = self.out_dir
                return self._json(transcribe_from_maps(self.state["maps"], body))
        except subprocess.CalledProcessError as error:
            return self._json({"error": f"fetch failed ({error.returncode})"}, 500)
        except Exception as error:                             # noqa: BLE001 - shown in the UI
            return self._json({"error": f"{type(error).__name__}: {error}"}, 500)
        self._send(b"not found", "text/plain", 404)


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        import sys

        if issubclass(sys.exc_info()[0] or Exception, GONE):
            return
        super().handle_error(request, client_address)


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m evaluation.workbench",
        description="A local page for tuning the transcriber against real audio.")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--cache", default=".workbench")
    args = parser.parse_args(argv)

    Handler.cache_dir = args.cache
    Handler.out_dir = os.path.join(args.cache, "out")
    os.makedirs(Handler.out_dir, exist_ok=True)
    print(f"cache: {args.cache}")
    print(f"\n  open  http://localhost:{args.port}\n")
    Server(("127.0.0.1", args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
