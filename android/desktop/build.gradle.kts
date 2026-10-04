plugins {
    kotlin("jvm")
    application
}

kotlin { jvmToolchain(17) }

dependencies {
    implementation(project(":pipeline"))
    implementation("com.microsoft.onnxruntime:onnxruntime:1.20.0")
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.7.3")
}

application {
    mainClass.set("dev.scorefromaudio.desktop.CliKt")
    applicationDefaultJvmArgs = listOf("-Xmx8g")
}

tasks.named<JavaExec>("run") {
    environment("MODELS_DIR", System.getenv("MODELS_DIR") ?: rootProject.file("models").absolutePath)
}
