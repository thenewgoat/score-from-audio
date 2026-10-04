# Releasing the app

The app is published as an APK on GitHub Releases, not on Google Play.

## What a release is

- Free: no ads, no purchases. Transkun's training data (MAESTRO) and bd1's
  (ASAP) are CC BY-NC-SA 4.0, so the app stays non-commercial.
- Transkun 2.0 (stock, int8 core) + bd1-synth. The V1 reranker is still
  Python (boosted trees) and is not in the app.
- YouTube and Instagram links through a bundled yt-dlp, with the optional
  Instagram login. The sites' own terms may forbid downloading; the in-app
  terms leave that to the user. This is also why the app is not on Google
  Play, which removes downloaders.
- On first launch, a welcome screen asks people to agree to the terms. The
  terms, privacy policy, credits and full licence texts are in
  `app/src/main/assets/legal/` and appear under ⋮ → Settings, terms and
  privacy.

## The signing key

`~/.android-keys/score-from-audio-upload.jks` (alias `upload`). Its passwords
are in `~/.android-keys/score-from-audio.properties`. Copy that file to
`android/keystore.properties` (gitignored) to sign with it.

**Back up both files off this machine.** Every later APK must be signed with
this key, or Android refuses to install it over the previous one. A lost key
means users must uninstall, which loses their scores.

Certificate SHA-256: `2b:72:27:44:27:4a:20:c6:db:ea:b5:e9:b9:e9:9e:05:1c:64:c3:0f:60:df:bf:82:86:5d:56:83:d7:e8:77:2c`

## Cutting a release

1. Bump `versionCode` and `versionName` in `app/build.gradle.kts`.
2. If the terms or privacy policy changed in substance, bump
   `Consent.TERMS_VERSION` so people are asked again.
3. Build:
   ```
   cd android
   ./gradlew :app:testDebugUnitTest :pipeline:test
   ./gradlew :app:assembleRelease
   ```
4. Publish:
   ```
   cp app/build/outputs/apk/release/app-release.apk score-from-audio-<version>.apk
   sha256sum score-from-audio-<version>.apk
   gh release create v<version> score-from-audio-<version>.apk --title "<version>" --notes "..."
   ```
   Put the SHA-256 and the signing certificate fingerprint in the notes.

yt-dlp is pinned in `app/build.gradle.kts`. When YouTube breaks it, bump it
and cut a new release; the app can't update it on its own.

## Size

The APK is about 187 MB, mostly bd1 (105 MB, fp32), Transkun (32 MB) and
Python with yt-dlp. GitHub allows release assets up to 2 GB. Quantising bd1 to
int8 would roughly halve the download.
