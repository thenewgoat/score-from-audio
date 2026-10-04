#!/usr/bin/env bash
# One-time toolchain for android/: JDK 17, the Android SDK, the Gradle wrapper, Verovio.
# Everything lands outside the repo except local.properties (gitignored), the
# wrapper, and viewer/node_modules (gitignored).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TOOLS="${SCOREAPP_TOOLS:-$HOME/.local/share/score-from-audio}"
SDK="${ANDROID_HOME:-$HOME/Android/Sdk}"
mkdir -p "$TOOLS" "$SDK"

if [ ! -x "$TOOLS/jdk17/bin/java" ]; then
  curl -fsSL -o "$TOOLS/jdk17.tar.gz" \
    "https://api.adoptium.net/v3/binary/latest/17/ga/linux/x64/jdk/hotspot/normal/eclipse"
  mkdir -p "$TOOLS/jdk17"
  tar -xzf "$TOOLS/jdk17.tar.gz" -C "$TOOLS/jdk17" --strip-components=1
fi
export JAVA_HOME="$TOOLS/jdk17"
export PATH="$JAVA_HOME/bin:$PATH"

if [ ! -x "$SDK/cmdline-tools/latest/bin/sdkmanager" ]; then
  curl -fsSL -o "$TOOLS/cmdline-tools.zip" \
    "https://dl.google.com/android/repository/commandlinetools-linux-11076708_latest.zip"
  rm -rf "$SDK/cmdline-tools/latest" "$SDK/cmdline-tools/cmdline-tools"
  mkdir -p "$SDK/cmdline-tools"
  unzip -q "$TOOLS/cmdline-tools.zip" -d "$SDK/cmdline-tools"
  mv "$SDK/cmdline-tools/cmdline-tools" "$SDK/cmdline-tools/latest"
fi
yes | "$SDK/cmdline-tools/latest/bin/sdkmanager" --licenses >/dev/null || true
"$SDK/cmdline-tools/latest/bin/sdkmanager" "platform-tools" "platforms;android-35" "build-tools;35.0.0"
echo "sdk.dir=$SDK" > "$ROOT/local.properties"

if [ ! -x "$ROOT/gradlew" ]; then
  curl -fsSL -o "$TOOLS/gradle.zip" "https://services.gradle.org/distributions/gradle-8.10.2-bin.zip"
  unzip -qo "$TOOLS/gradle.zip" -d "$TOOLS"
  (cd "$ROOT" && "$TOOLS/gradle-8.10.2/bin/gradle" wrapper --gradle-version 8.10.2)
fi

(cd "$ROOT/viewer" && npm install --no-audit --no-fund)

# Chaquopy builds with a host Python of the same minor version as the app's (3.13).
PY313="$(command -v python3.13 || true)"
[ -z "$PY313" ] && [ -x "$HOME/miniconda3/bin/python3.13" ] && PY313="$HOME/miniconda3/bin/python3.13"
if [ -z "$PY313" ]; then
  echo "Python 3.13 not found: install it (e.g. conda install python=3.13) and re-run" >&2; exit 1
fi
grep -v '^chaquopy.buildPython=' "$ROOT/local.properties" > "$ROOT/local.properties.tmp" || true
echo "chaquopy.buildPython=$PY313" >> "$ROOT/local.properties.tmp"
mv "$ROOT/local.properties.tmp" "$ROOT/local.properties"

# QuickJS-ng, run by yt-dlp to solve YouTube's JavaScript challenges. Shipped as a
# "native library" because nativeLibraryDir is the only place Android lets an app execute a file.
# Pinned: the SHA-256 GitHub lists for the v0.17.0 release's qjs-linux-aarch64 asset.
QJS="$ROOT/app/src/main/jniLibs/arm64-v8a/libqjs.so"
QJS_SHA256=3372133484edf50a69f3c67903af41206d22a061e930e3cfb63269272ef56d2e
if [ ! -f "$QJS" ]; then
  mkdir -p "$(dirname "$QJS")"
  curl -fsSL -o "$QJS" "https://github.com/quickjs-ng/quickjs/releases/download/v0.17.0/qjs-linux-aarch64"
fi
if ! echo "$QJS_SHA256  $QJS" | sha256sum -c --status; then
  rm -f "$QJS"
  echo "QuickJS-ng download does not match its pinned SHA-256; deleted $QJS" >&2; exit 1
fi

cat <<EOF
Done. Add to your shell profile:
  export JAVA_HOME=$JAVA_HOME
  export ANDROID_HOME=$SDK
  export PATH=\$JAVA_HOME/bin:\$ANDROID_HOME/platform-tools:\$PATH
EOF
