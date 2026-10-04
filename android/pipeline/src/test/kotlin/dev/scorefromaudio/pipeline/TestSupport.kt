package dev.scorefromaudio.pipeline

import org.junit.jupiter.api.Assumptions.assumeTrue
import java.io.File
import java.io.InputStream

object Fixtures {
    fun stream(name: String): InputStream =
        requireNotNull(javaClass.getResourceAsStream("/fixtures/$name")) { "missing fixture $name" }

    fun text(name: String): String = stream(name).use { it.readBytes().decodeToString() }

    val modelsDir = File(System.getProperty("models.dir", "models"))
    val asapAudio = File(System.getProperty("asap.audio", ""))
    val writerOut = File(System.getProperty("writer.out", "build/writer-out"))

    /** Skip the calling test unless every named model file has been exported. */
    fun requireModels(vararg names: String) {
        for (name in names) assumeTrue(File(modelsDir, name).isFile, "model $name not exported; see export/README.md")
    }
}
