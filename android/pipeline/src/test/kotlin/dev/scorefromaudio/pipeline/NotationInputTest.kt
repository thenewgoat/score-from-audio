package dev.scorefromaudio.pipeline

import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlin.test.Test
import kotlin.test.assertEquals

@Serializable
class BdFixture(val piece: String, val notes: List<List<Double>>, val input: List<List<Int>>, val tokens: List<List<Int>>) {
    fun detected() = notes.map { Note(it[0], it[1], it[2].toInt(), it[3].toInt()) }

    companion object {
        val PIECES = listOf("Brahms__Six_Pieces_op_118_2", "Prokofiev__Toccata", "Scriabin__Etudes_op_8_11")
        private val json = Json { ignoreUnknownKeys = true }
        fun load(piece: String) = json.decodeFromString(serializer(), Fixtures.text("bd_$piece.json"))
    }
}

class NotationInputTest {
    @Test
    fun bucketsAsTheirTokenizerDoes() {
        for (piece in BdFixture.PIECES) {
            val fixture = BdFixture.load(piece)
            val input = NotationInput.encode(fixture.detected())
            assertEquals(fixture.input.size, input.size, piece)
            for ((i, row) in fixture.input.withIndex()) {
                assertEquals(row, listOf(input.onset[i], input.duration[i], input.pitch[i], input.velocity[i]), "$piece note $i")
            }
        }
    }

    @Test
    fun dropsPedalsAndSorts() {
        val input = NotationInput.encode(listOf(Note(1.0, 2.0, 64, 80), Note(0.0, 9.0, -64, 127), Note(1.0, 1.5, 60, 80)))
        assertEquals(listOf(60, 64), input.notes.map { it.pitch })
    }
}
