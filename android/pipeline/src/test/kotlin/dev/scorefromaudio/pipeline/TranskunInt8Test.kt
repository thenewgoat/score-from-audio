package dev.scorefromaudio.pipeline

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import org.junit.jupiter.api.Assumptions.assumeTrue
import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * The app's configuration -- the int8 core, 12 s steps -- against torch's fp32
 * notes at 8 s steps on the parity clips. It cannot match note for note, so
 * notes are matched by onset: same pitch, onsets within 50 ms, one to one.
 */
class TranskunInt8Test {
    @Serializable
    private class Clip(val path: String, val seconds: Int, val notes: List<List<Double>>)
    @Serializable
    private class Clips(val clips: List<Clip>)

    @Test
    fun int8CoreAtTwelveSecondStepsFindsTorchsOnsets() {
        Fixtures.requireModels("transkun_core_int8.onnx", "transkun_heads.onnx")
        val clips = Json.decodeFromString(Clips.serializer(), Fixtures.text("transkun_notes.json")).clips
        val env = OrtEnvironment.getEnvironment()
        val options = OrtSession.SessionOptions()
        OrtTranskun(env, File(Fixtures.modelsDir, "transkun_core_int8.onnx").path,
                    File(Fixtures.modelsDir, "transkun_heads.onnx").path, options).use { model ->
            val frontend = Frontend(Fixtures.stream("transkun_frontend.bin").use { FrontendConstants.read(it) })
            val transcriber = Transcriber(model, frontend, stepSeconds = 12.0)
            val scores = clips.map { clip ->
                val file = File(Fixtures.asapAudio, clip.path)
                assumeTrue(file.isFile, "ASAP audio not found: $file")
                val notes = transcriber.transcribe(Wav.read(file, clip.seconds.toDouble()))
                val reference = clip.notes.map { it[2].toInt() to it[0] }
                val f = onsetF(reference, notes.map { it.pitch to it.start })
                println("${clip.path}: ${notes.size} notes vs torch's ${reference.size}, onset F $f")
                clip.path to f
            }
            for ((path, f) in scores) assertTrue(f >= MIN_ONSET_F, "$path: onset F $f < $MIN_ONSET_F")
        }
    }

    @Test
    fun onsetMatchingIsOneToOnePerPitch() {
        val reference = listOf(60 to 1.0, 60 to 1.04, 62 to 2.0)
        val estimate = listOf(60 to 1.02, 62 to 2.06, 64 to 2.0)
        // One of the two C onsets matches; D is 60 ms late; E has no reference.
        assertEquals(2.0 * 1 / 6, onsetF(reference, estimate), 1e-12)
        assertEquals(1.0, onsetF(reference, reference), 1e-12)
    }

    private companion object {
        /**
         * Measured per clip, 40 s each: Bach 858 0.984, Beethoven 3-1 0.992, Liszt
         * Ballade 2 (Jin) 0.976 and (Min) 0.980, Mephisto Waltz 0.984. Set a little
         * below the worst.
         */
        const val MIN_ONSET_F = 0.96
        const val TOLERANCE = 0.05

        /**
         * Onset F with a maximum one-to-one matching: per pitch, both onset lists
         * sorted, the earliest unmatched pair within [TOLERANCE] matched first
         * (optimal for points on a line under a fixed window).
         */
        fun onsetF(reference: List<Pair<Int, Double>>, estimate: List<Pair<Int, Double>>): Double {
            if (reference.isEmpty() && estimate.isEmpty()) return 1.0
            val ref = reference.groupBy({ it.first }, { it.second }).mapValues { it.value.sorted() }
            val est = estimate.groupBy({ it.first }, { it.second }).mapValues { it.value.sorted() }
            var matched = 0
            for ((pitch, r) in ref) {
                val e = est[pitch] ?: continue
                var i = 0
                var j = 0
                while (i < r.size && j < e.size) {
                    when {
                        e[j] < r[i] - TOLERANCE -> j++
                        e[j] > r[i] + TOLERANCE -> i++
                        else -> { matched++; i++; j++ }
                    }
                }
            }
            return 2.0 * matched / (reference.size + estimate.size)
        }
    }
}
