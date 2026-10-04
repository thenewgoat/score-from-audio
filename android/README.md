# The phone app

An Android app that turns a recording into a score on the phone: Transkun finds
the notes, our notation model bd1 writes them as notation, both through ONNX
Runtime; Verovio shows the score and Android's MIDI synthesiser plays it. Your
recordings and scores are never uploaded. The app goes online only to fetch a
YouTube or Instagram link you share or paste, and to show Instagram's login
page if you log in. The privacy policy
(`app/src/main/assets/legal/`) relies on that.
Releasing it is covered in `RELEASE.md`.

## Build

1. `android/tools/setup.sh` -- JDK 17, Android SDK, Gradle wrapper, Verovio.
   It prints `JAVA_HOME`/`ANDROID_HOME` to export; set those before running
   `./gradlew`. It also needs a host Python 3.13 to build Chaquopy against
   (it looks for `python3.13` on `PATH`, then miniconda's). Gradle will not
   configure the build without `chaquopy.buildPython=<path to python3.13>` in
   `android/local.properties`. It also downloads QuickJS-ng, which yt-dlp
   uses for YouTube's JavaScript challenges.
2. Export the models (see `export/README.md`) into `android/models/`.
3. `cd android && ./gradlew :app:assembleRelease` -- not debuggable, so faster
   on the phone. Signed with the upload key when `android/keystore.properties`
   names one (see `RELEASE.md`), otherwise with the debug key.
4. `adb install -r app/build/outputs/apk/release/app-release.apk`

## Screens (v1)

Library → Record → Making your score → Score, with "Fix the basics" as a sheet on the score. Each job keeps
its recording (`audio.*` in its folder), so "My recording" plays it in sync with the score (through `sync.json`,
which ties score positions to the notes' onsets) and a failed job can be tried again. Recordings and files queue
behind a running job. A debug build: `./gradlew :app:assembleDebug`.

## Which models may ship

bd1 is ours (trained on ASAP, PDMX and a few of our own jazz transcriptions), and replaces Beyer &
Dai's checkpoint, whose repository carries no licence and which must never go
into a published APK. Transkun's weights are MIT, trained on MAESTRO. ASAP and
MAESTRO are CC BY-NC-SA 4.0, so a published app stays free and non-commercial;
see `RELEASE.md`. The exported graphs stay gitignored in `models/`.

## Layout

| module | what |
|---|---|
| `pipeline` | pure Kotlin: frontend, Viterbi, notation loop, MusicXML writer; tested on the JVM |
| `desktop` | a CLI over `pipeline`, to run it on a computer |
| `app` | Compose UI, foreground service, MediaCodec decoding, Verovio viewer |

Tests: `./gradlew :pipeline:test :app:testDebugUnitTest` (model-gated tests skip
without `android/models/`), `node --test viewer/timemap.test.cjs`.
