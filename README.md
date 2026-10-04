# Score from audio: free piano-to-sheet-music app for Android

**Turn piano audio into sheet music on your phone.** Record yourself playing,
import an audio or video file, or share a YouTube or Instagram link, and get a
readable score you can play back, slow down, loop and export to MusicXML, MIDI
or PDF.

It's free, open source, needs no account, and transcribes entirely on your
phone. Your recordings are never uploaded.

**[Download the APK](https://github.com/thenewgoat/score-from-audio/releases/latest)**
(Android 10 or later, 64-bit ARM phones, which is almost every phone from the
last several years).

## What you can use it for

- **Write down what you just played.** Improvised something you like? Record
  it and get it on paper before you forget it.
- **Transcribe a song from YouTube.** Found a piano cover, a tutorial or a
  performance you want to learn? Share the link to the app and get the notes.
- **Learn a piece by ear, faster.** Play the score back with a piano sound or
  with the original recording, slow it down without changing the pitch, and
  loop the bars you're stuck on.
- **Turn Instagram and short-form piano clips into sheet music.** Share the
  post or reel straight from Instagram.
- **Get practice recordings into notation software.** Export MusicXML and
  open it in MuseScore, Sibelius, Finale, Dorico or Noteflight to edit and
  print. Export MIDI for a DAW or a piano-learning app.
- **Teach and share.** Make a score for a student from a
  recording, export a PDF, and send it.
- **Transcribe audio or video files you already have:** voice memos, concert
  recordings, lesson videos, MP3, M4A, WAV, MP4 and more.
- **Work offline.** Recording, importing and transcribing need no internet
  connection. Only fetching a link does.

## Features

- Audio to sheet music: a grand staff with bars, time signature, key, note
  values, hands and voices.
- Three ways in: record with the microphone, import a file, or share or paste
  a YouTube or Instagram link.
- Playback in sync with the score: hear a piano sound or your own recording,
  with the current notes highlighted.
- Practice tools: slow-down without pitch change, and looping between any two
  bars.
- Fix the basics: change beats per bar, key or tempo, and the whole score is
  redrawn.
- Export and share as MusicXML, MIDI or PDF.
- Private by design: no account, no analytics, no ads, nothing uploaded.
- Free and open source.

## Good to know

- **Piano only, for now.** It's trained on solo piano. Other instruments,
  singing, or piano buried in a full band mix will give poor results.
- **Clean recordings work best.** Put the phone near the piano in a quiet
  room.
- **It's automatic, so check the result.** Expect some wrong notes, rhythms
  or hand splits, especially in fast or heavily pedalled passages. Exporting
  to MusicXML and tidying it in MuseScore is the quickest way to a finished
  score.
- **Transcribing takes a while** on a phone, longer for longer pieces. It
  keeps going in the background and notifies you when the score is ready.
- **Only transcribe music you have the right to use.** The terms in the app
  explain this.

## Install

1. Open the [latest release](https://github.com/thenewgoat/score-from-audio/releases/latest)
   on your phone and download the `.apk` file.
2. Open it. Android will ask you to allow installs from your browser or file
   manager. Allow it, then tap Install.
3. Open **Score from audio**, agree to the terms, and record or import
   something.

The app isn't on Google Play. Each release lists the APK's SHA-256 checksum
and signing certificate, so you can check your download.

## How it works

```
audio ──► Transkun ──► notes ──► bd1 ──► score ──► display, playback, MusicXML / MIDI / PDF
          (piano transcription)  (notation model)
```

- **[Transkun](https://github.com/Yujia-Yan/Skipping-The-Frame-Level)**, one of the
  most accurate piano transcription models, by Yujia Yan, Frank Cwitkowitz
  and Zhiyao Duan, hears the notes: pitch, onset, release and loudness.
- **bd1**, our notation model, turns those notes into a score: bars, metre,
  key, note values, hands and voices. It's a transformer in the style of
  Beyer and Dai's MIDI-to-score model.
- **[Verovio](https://www.verovio.org)** engraves the score.

Both models run on the phone through ONNX Runtime.

## For developers

| path | what |
|---|---|
| `android/` | the app: Kotlin pipeline, Compose UI, Verovio viewer. Build with `android/README.md`, release with `android/RELEASE.md` |
| `export/` | turns the Transkun and bd1 checkpoints into the ONNX graphs the app runs |
| `notation/` | bd1's model, tokens and checkpoint loading, used by `export/` |

Bug reports and ideas are welcome in
[Issues](https://github.com/thenewgoat/score-from-audio/issues).

## Licence, models and data

The code is Apache-2.0. The models in the released APK are not covered by it:

- Transkun's weights (Yujia Yan, MIT) were trained on MAESTRO, which is
  CC BY-NC-SA 4.0.
- bd1 was trained on ASAP (CC BY-NC-SA 4.0), public-domain and CC0 scores from
  PDMX, and a few of our own jazz transcriptions. Its weights are released
  under CC BY-NC-SA 4.0.

So the app, and anything built on those weights, is for non-commercial use
only. Full credits and licence texts are in
`android/app/src/main/assets/legal/` and in the app under ⋮ → Settings. A few
seconds of a MAESTRO recording are kept as a test fixture
(`android/pipeline/src/test/resources/fixtures/frontend_audio.npy`) under the
same licence.

<sub>Keywords: piano transcription app, audio to sheet music, music to notes,
transcribe piano by ear, convert MP3 to sheet music, YouTube to sheet music,
song to notes, MIDI from audio, automatic music transcription, AMT, MusicXML
export, free, open source, offline, Android.</sub>
