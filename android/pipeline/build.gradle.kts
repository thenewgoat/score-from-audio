plugins {
    kotlin("jvm")
    kotlin("plugin.serialization")
}

kotlin { jvmToolchain(17) }

dependencies {
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.7.3")
    // Provided at run time by the desktop module (JVM build) or the app (Android build).
    compileOnly("com.microsoft.onnxruntime:onnxruntime:1.20.0")
    testImplementation("com.microsoft.onnxruntime:onnxruntime:1.20.0")
    testImplementation(kotlin("test"))
    testImplementation("org.junit.jupiter:junit-jupiter:5.11.3")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}

tasks.test {
    useJUnitPlatform()
    maxHeapSize = "6g"
    systemProperty("models.dir", rootProject.file("models").absolutePath)
    systemProperty("asap.audio", System.getenv("ASAP_AUDIO") ?: (System.getenv("SCORE_ROOT") ?: "${System.getProperty("user.home")}/projects/score-reading") + "/data/processed/asap-audio")
    systemProperty("writer.out", layout.buildDirectory.dir("writer-out").get().asFile.absolutePath)
    testLogging { events("passed", "skipped", "failed"); showStandardStreams = true }
}
