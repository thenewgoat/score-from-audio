package dev.scorefromaudio.pipeline

import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import java.nio.FloatBuffer
import kotlin.test.Test
import kotlin.test.assertEquals

class ViterbiTest {
    @Serializable
    private class Expected(val forced: List<Int>, val intervals: List<List<List<Int>>>)

    @Test
    fun reproducesTranskunsViterbiBackward() {
        val score = Fixtures.stream("viterbi_score.npy").use { Npy.read(it) }
        val expected = Json.decodeFromString(Expected.serializer(), Fixtures.text("viterbi_expected.json"))
        val paths = Viterbi.decode(FloatBuffer.wrap(score.floats!!), score.shape[0], score.shape[2], expected.forced.toIntArray())
        assertEquals(expected.intervals, paths.map { track -> track.map { it.toList() } })
    }
}
