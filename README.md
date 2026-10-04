# score-from-audio

Record piano on your Android phone and get sheet music, made on the phone.

Record, import an audio or video file, or share a YouTube or Instagram link.
The app finds the notes, writes them as a score, shows it and plays it back in
sync with your recording, and exports MusicXML, MIDI and PDF. Nothing is
uploaded.

**Download:** the APK is on
[Releases](https://github.com/thenewgoat/score-from-audio/releases). It needs
an arm64 phone with Android 10 or later. Free, and for non-commercial use only;
see [Models and data](#models-and-data).

## How it works

```
audio ──► Transkun ──► notes ──► bd1 ──► score ──► Verovio (display) / MusicXML, MIDI, PDF
          (piano transcription)  (notation model)
```

- **Transkun** (Yan, Cwitkowitz and Duan) turns the audio into notes with
  onsets, offsets and velocities. Its frontend, semi-CRF decoding and Viterbi
  are ported to Kotlin; the network runs under ONNX Runtime with an int8 core.
- **bd1**, our notation model, turns those notes into a score: bars, metre,
  key, note values, hands and voices. It is a transformer in the style of Beyer
  and Dai's MIDI-to-score model, trained on aligned performances and scores,
  exported to ONNX as an encoder and a cached decoder step.
- **Verovio** engraves the score in a WebView; Android's MIDI synthesiser plays
  it.

Everything runs on the phone. The only network use is fetching a link you
share.

## Layout

| path | what |
|---|---|
| `android/` | the app: Kotlin pipeline, Compose UI, Verovio viewer. See `android/README.md` to build, `android/RELEASE.md` to release |
| `export/` | turns the Transkun and bd1 checkpoints into the ONNX graphs the app runs, and writes the test fixtures |
| `notation/` | bd1's model, tokens and checkpoint loading, which `export/` needs |

## Models and data

The code is Apache-2.0. The models in the released APK are not covered by it:

- Transkun's weights (Yujia Yan, MIT) were trained on MAESTRO, which is CC
  BY-NC-SA 4.0.
- bd1 was trained on ASAP (CC BY-NC-SA 4.0), public-domain and CC0 scores from
  PDMX, and a few of our own jazz transcriptions. Its weights are released
  under CC BY-NC-SA 4.0.

So the app, and anything built on those weights, is for non-commercial use
only. The full credits and licence texts are in
`android/app/src/main/assets/legal/`, and in the app under ⋮ → Settings. A few
seconds of a MAESTRO recording are kept as a test fixture
(`android/pipeline/src/test/resources/fixtures/frontend_audio.npy`) under the
same CC BY-NC-SA 4.0 licence.

## Licence

Apache-2.0 for the code; see above for the models.
