package dev.scorefromaudio.pipeline

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import org.junit.jupiter.api.Assumptions.assumeTrue
import java.io.File
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class TranskunParityTest {
    @Serializable
    private class Clip(val path: String, val seconds: Int, val notes: List<List<Double>>)
    @Serializable
    private class Clips(val clips: List<Clip>)

    @Test
    fun matchesTorchTranscribeOnFiveRecordings() {
        Fixtures.requireModels("transkun_core.onnx", "transkun_heads.onnx")
        val clips = Json.decodeFromString(Clips.serializer(), Fixtures.text("transkun_notes.json")).clips
        val env = OrtEnvironment.getEnvironment()
        val options = OrtSession.SessionOptions()
        OrtTranskun(env, File(Fixtures.modelsDir, "transkun_core.onnx").path,
                    File(Fixtures.modelsDir, "transkun_heads.onnx").path, options).use { model ->
            val transcriber = Transcriber(model, Frontend(Fixtures.stream("transkun_frontend.bin").use { FrontendConstants.read(it) }))
            for (clip in clips) {
                val file = File(Fixtures.asapAudio, clip.path)
                assumeTrue(file.isFile, "ASAP audio not found: $file")
                val started = System.nanoTime()
                val notes = transcriber.transcribe(Wav.read(file, clip.seconds.toDouble()))
                println("${clip.path}: ${notes.size} notes in ${(System.nanoTime() - started) / 1e9} s")
                assertEquals(clip.notes.size, notes.size, clip.path)
                for ((i, expected) in clip.notes.withIndex()) {
                    val n = notes[i]
                    assertEquals(expected[2].toInt(), n.pitch, "${clip.path} note $i pitch")
                    assertEquals(expected[3].toInt(), n.velocity, "${clip.path} note $i velocity")
                    assertTrue(abs(expected[0] - n.start) <= 1e-3, "${clip.path} note $i start ${n.start} vs ${expected[0]}")
                    assertTrue(abs(expected[1] - n.end) <= 1e-3, "${clip.path} note $i end ${n.end} vs ${expected[1]}")
                }
            }
        }
    }
}
