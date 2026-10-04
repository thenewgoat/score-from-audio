"""Every path outside this repository that the export step touches.

Each can be overridden by an environment variable, so the step runs on another
machine without editing code. Nothing written to OUT is committed: it is
gitignored, and Beyer & Dai's graphs may not be redistributed.
"""

import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The shared workspace: repos, external/ checkouts and data/ (raw, processed, runs, eval, logs, models).
ROOT = os.environ.get("SCORE_ROOT", os.path.expanduser("~/projects/score-reading"))
DATA = os.path.join(ROOT, "data")
OUT = os.environ.get("EXPORT_OUT", os.path.join(REPO, "android", "models"))
FIXTURES = os.path.join(REPO, "android", "pipeline", "src", "test", "resources", "fixtures")

M2S_REPO = os.environ.get("M2S_REPO", os.path.join(ROOT, "external", "MIDI2ScoreTransformer"))
M2S_CKPT = os.path.join(M2S_REPO, "MIDI2ScoreTF.ckpt")
# ASAP with MAESTRO's audio cropped in, at 44.1 kHz stereo (evaluation.asap_audio --canonical).
ASAP_AUDIO = os.environ.get("ASAP_AUDIO", os.path.join(DATA, "processed", "asap-audio"))
# ASAP scores and performance MIDI.
ASAP = os.environ.get("ASAP", os.path.join(DATA, "raw", "asap"))

# Five MAESTRO-test recordings from the engine bake-off, first 40 s of each:
# long enough to cross four segment boundaries.
PARITY_CLIPS = [
    "Bach/Fugue/bwv_858/VuV01M.wav",
    "Beethoven/Piano_Sonatas/3-1/Lin05M.wav",
    "Liszt/Ballade_2/Jin07M.wav",
    "Liszt/Ballade_2/Min06M.wav",
    "Liszt/Mephisto_Waltz/Avdeeva07M.wav",
]
PARITY_SECONDS = 40

# The three test pieces neither Beyer & Dai nor our model trained on.
HELDOUT = [
    ("Brahms__Six_Pieces_op_118_2", "Brahms/Six_Pieces_op_118/2", "Shilyaev03"),
    ("Prokofiev__Toccata", "Prokofiev/Toccata", "Dossin07"),
    ("Scriabin__Etudes_op_8_11", "Scriabin/Etudes_op_8/11", "YeF09"),
]
