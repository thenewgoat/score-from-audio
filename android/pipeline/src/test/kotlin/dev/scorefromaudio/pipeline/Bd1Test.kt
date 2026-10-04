package dev.scorefromaudio.pipeline

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals

/** `export.bd1_ref --fixtures`: the first notes of a held-out piece, bd1's input streams, and every slot it wrote. */
@Serializable
class Bd1Fixture(val piece: String, val notes: List<List<Double>>, val input: List<List<Int>>, val slots: List<List<Int>>) {
    fun detected() = notes.map { Note(it[0], it[1], it[2].toInt(), it[3].toInt()) }

    companion object {
        private val json = Json { ignoreUnknownKeys = true }
        fun load(piece: String) = json.decodeFromString(serializer(), Fixtures.text("bd1_$piece.json"))
    }
}

class Bd1InputTest {
    @Test
    fun encodesAsTheReference() {
        for (piece in BdFixture.PIECES) {
            val fixture = Bd1Fixture.load(piece)
            val input = Bd1Input.encode(fixture.detected())
            assertEquals(fixture.input.size, input.size, piece)
            for ((i, row) in fixture.input.withIndex())
                assertEquals(row, listOf(input.pitch[i], input.onset[i], input.duration[i], input.velocity[i]), "$piece note $i")
        }
    }

    @Test
    fun bucketsLikeTheTokenizer() {
        assertEquals(listOf(0, 10, 45, 81, 98, 199, 199), listOf(0.0, 0.01, 0.1, 0.5, 1.0, 60.0, 100.0).map { Bd1Input.gapBucket(it) })
        assertEquals(listOf(1, 1, 4, 7), listOf(0, 1, 64, 127).map { Bd1Input.velocityBucket(it) })
    }

    @Test
    fun dropsPedalsAndSortsByOnsetThenOffset() {
        val input = Bd1Input.encode(listOf(Note(1.0, 2.0, 60, 80), Note(0.0, 9.0, -64, 127), Note(1.0, 1.5, 64, 80)))
        assertEquals(listOf(64, 60), input.pitch.toList())
    }
}

class Bd1NotatorTest {
    @Test
    fun chunksAsAlignedGreedyDoes() {
        val notator = Bd1Notator(object : Bd1Model {
            override fun encode(input: Bd1Input, from: Int, to: Int) = error("unused")
            override fun newCache() = error("unused")
            override fun step(encoded: Encoded, row: IntArray, cache: Cache) = error("unused")
        })
        assertEquals(listOf(0 to 511, 447 to 958, 894 to 1200), notator.chunkStarts(1200))
        assertEquals(listOf(0 to 30), notator.chunkStarts(30))
        assertEquals(emptyList(), notator.chunkStarts(0))
    }

    @Test
    fun aSpaceIsDroppedAndItsBarLineMovesOn() {
        val input = Bd1Input.encode(List(3) { Note(it.toDouble(), it + 0.5, 60 + it, 70) })
        val space = IntArray(Bd1Streams.COUNT).also { it[Bd1Streams.PITCH] = Bd1Streams.PITCH_SPACE; it[Bd1Streams.BAR] = 30 }
        val note = { p: Int -> IntArray(Bd1Streams.COUNT).also { it[Bd1Streams.PITCH] = p } }
        val rows = Bd1Notator.withoutSpaces(listOf(note(60), space, note(62)), input)
        assertEquals(listOf(60, 62), rows.map { it[Bd1Streams.PITCH] })
        assertEquals(30, rows[1][Bd1Streams.BAR])
        assertEquals(2, rows[1][Bd1Streams.SOURCE])
        assertEquals(70, rows[1][Bd1Streams.VELOCITY])
    }

    @Test
    fun theDecoderNeverWritesAControlToken() {
        val logits = Array(Bd1Streams.COUNT) { FloatArray(if (it == 0) 134 else 4) }
        logits[0][Bd1Streams.PITCH_EOS] = 9f
        logits[0][61] = 1f
        assertEquals(61, Bd1Notator.choose(logits)[Bd1Streams.PITCH])
    }
}

class Bd1ParityTest {
    @Test
    fun reproducesThePythonLoopOnTheHeldOutPieces() {
        Fixtures.requireModels("bd1_encoder.onnx", "bd1_step.onnx")
        OrtBd1(OrtEnvironment.getEnvironment(), File(Fixtures.modelsDir, "bd1_encoder.onnx").path,
               File(Fixtures.modelsDir, "bd1_step.onnx").path, OrtSession.SessionOptions()).use { model ->
            for (piece in BdFixture.PIECES) {
                val fixture = Bd1Fixture.load(piece)
                val started = System.nanoTime()
                val slots = Bd1Notator(model).slots(Bd1Input.encode(fixture.detected()))
                println("$piece: ${slots.size} slots in ${(System.nanoTime() - started) / 1e9} s")
                assertEquals(fixture.slots.size, slots.size)
                for ((i, row) in fixture.slots.withIndex()) assertEquals(row, slots[i].toList(), "$piece slot $i")
            }
        }
    }
}
