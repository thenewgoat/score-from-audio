"""Beat-tracking backends. Importing this module registers them.

Each backend imports its own library lazily inside `track()`, so registering
them all costs nothing and a missing optional dependency surfaces at use rather
than at import.
"""

from transcribe.trackers import beat_this_beats  # noqa: F401
from transcribe.trackers import librosa_beats  # noqa: F401
