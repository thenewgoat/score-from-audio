"""A pipeline from YouTube piano videos to aligned score/performance pairs.

Everything it downloads stays under a gitignored data directory and every item
is marked `internal_only` in its meta.json: the downloads are for your own use
and are not to be committed or redistributed.

One directory per video, under `data/yt/` unless `--data` or `YTPIPE_DATA`
says otherwise:

    data/yt/playlists/<playlist_id>.json    the playlist as fetched
    data/yt/<video_id>/meta.json            from yt-dlp's info JSON
                       audio.wav            44.1 kHz mono
                       video.mp4            video only, at most 1080p
                       roi.json             where the notation is, and when
                       frames/              thumbnails and the contact sheet
                       omr/  score.musicxml  perf.mid  alignment.json   (later stages)

Stages, each idempotent -- an output that exists is never redone unless asked:

    1-2  ytpipe fetch <playlist_url>    list the playlist, download audio + video
    3    ytpipe roi <video_id>          mark the notation box and its time ranges
    4    ytpipe omr                     read the sheet music off the video (Audiveris)
    5    ytpipe align                   transcribe the audio, put both scores on its clock
         ytpipe edit / ytpipe sheet     compare OMR and audio per note; the sheet music as a PDF
    6    ytpipe export                  pieces finished in MuseScore as an ASAP-shaped dataset
"""
