import java.util.Properties

plugins {
    id("com.android.application")
    kotlin("android")
    kotlin("plugin.compose")
    kotlin("plugin.serialization")
    id("com.chaquo.python")
}

val localProperties = Properties().apply {
    rootProject.file("local.properties").takeIf { it.isFile }?.inputStream()?.use { load(it) }
}
// Named apart from the DSL's own `buildPython` property (below), which would otherwise
// shadow this val inside the `defaultConfig { }` block.
val hostPython = localProperties.getProperty("chaquopy.buildPython")
    ?: throw GradleException("chaquopy.buildPython missing from local.properties: run android/tools/setup.sh")
val qjs = file("src/main/jniLibs/arm64-v8a/libqjs.so")
val checkQuickJs by tasks.registering {
    doLast { if (!qjs.isFile) throw GradleException("QuickJS missing ($qjs): run android/tools/setup.sh") }
}
// The upload key lives outside the repo; keystore.properties (gitignored) points at it. Without it, release
// builds fall back to the debug key, which installs on a phone but cannot be uploaded to a store.
val keystoreProperties = Properties().apply {
    rootProject.file("keystore.properties").takeIf { it.isFile }?.inputStream()?.use { load(it) }
}

val requiredModels = listOf("transkun_frontend.bin", "transkun_core_int8.onnx", "transkun_heads.onnx", "bd1_encoder.onnx", "bd1_step.onnx")
val modelsDir = rootProject.file("models")
val generatedAssets = layout.buildDirectory.dir("generated/appAssets")

val checkModels by tasks.registering {
    doLast {
        val missing = requiredModels.filter { !File(modelsDir, it).isFile }
        if (missing.isNotEmpty()) throw GradleException(
            "Missing models in $modelsDir: $missing\n" +
            "Export them first (see export/README.md):\n" +
            "  transkun-env/bin/python -m export.transkun_export   (writes transkun_core_int8.onnx too)\n" +
            "  .venv/bin/python -m export.bd1_export")
    }
}
// Sync, not Copy: a model dropped from requiredModels must leave the APK too.
val copyModels by tasks.registering(Sync::class) {
    dependsOn(checkModels)
    from(modelsDir) { include(requiredModels) }
    into(generatedAssets.map { it.dir("models") })
}
val verovioSource = rootProject.file("viewer/node_modules/verovio/dist/verovio-toolkit-wasm.js")
// A separate check: a Copy whose source is missing is skipped as NO-SOURCE, so its own doFirst never runs.
val checkVerovio by tasks.registering {
    doLast {
        if (!verovioSource.isFile) throw GradleException("Verovio missing: run android/tools/setup.sh (npm install in android/viewer)")
    }
}
val copyVerovio by tasks.registering(Copy::class) {
    dependsOn(checkVerovio)
    from(verovioSource)
    into(generatedAssets.map { it.dir("viewer") })
}
tasks.named("preBuild") { dependsOn(copyModels, copyVerovio, checkQuickJs) }

// The store upload must not carry the legal drafts' {{...}} placeholders (developer name, contact, governing law).
// A sideloaded APK may, for testing.
val checkLegalPlaceholders by tasks.registering {
    val legal = file("src/main/assets/legal")
    inputs.dir(legal)
    doLast {
        val left = legal.listFiles().orEmpty().filter { it.isFile }.flatMap { f ->
            Regex("\\{\\{[A-Z_]+}}").findAll(f.readText()).map { "${f.name}: ${it.value}" }.toList()
        }
        if (left.isNotEmpty()) throw GradleException("Fill in the legal placeholders before a store build:\n  " + left.distinct().joinToString("\n  "))
    }
}
tasks.matching { it.name == "bundleRelease" }.configureEach { dependsOn(checkLegalPlaceholders) }

android {
    namespace = "dev.scorefromaudio.app"
    compileSdk = 35
    defaultConfig {
        applicationId = "dev.scorefromaudio.app"
        minSdk = 29
        targetSdk = 35
        versionCode = 1
        versionName = "1.0.0"
        ndk { abiFilters += listOf("arm64-v8a") }
    }
    packaging { jniLibs { useLegacyPackaging = true } }
    signingConfigs {
        if (keystoreProperties.getProperty("storeFile") != null) create("upload") {
            storeFile = file(keystoreProperties.getProperty("storeFile"))
            storePassword = keystoreProperties.getProperty("storePassword")
            keyAlias = keystoreProperties.getProperty("keyAlias")
            keyPassword = keystoreProperties.getProperty("keyPassword")
        }
    }
    buildTypes {
        // Not debuggable, so ART compiles the pipeline's Kotlin fully. No shrinking: ONNX Runtime and
        // kotlinx.serialization reach classes by reflection.
        release {
            isMinifyEnabled = false
            isShrinkResources = false
            signingConfig = signingConfigs.findByName("upload") ?: signingConfigs.getByName("debug")
        }
    }
    buildFeatures { compose = true; buildConfig = true }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    sourceSets["main"].assets.srcDir(generatedAssets)
    androidResources { noCompress += listOf("onnx", "bin") }
    testOptions { unitTests.isReturnDefaultValues = true }
}

chaquopy {
    defaultConfig {
        version = "3.13"
        buildPython(hostPython)
        pip {
            install("yt-dlp==2026.8.19")
            install("yt-dlp-ejs==0.8.0")
            install("certifi")
        }
    }
}

dependencies {
    implementation(project(":pipeline"))
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.20.0")
    implementation(platform("androidx.compose:compose-bom:2024.12.01"))
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.activity:activity-compose:1.9.3")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.8.7")
    implementation("androidx.core:core-ktx:1.15.0")
    implementation("androidx.webkit:webkit:1.12.1")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.9.0")
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.7.3")
    testImplementation(kotlin("test-junit"))
    testImplementation("junit:junit:4.13.2")
}
