package dev.scorefromaudio.pipeline.score

import kotlin.test.Test
import kotlin.test.assertEquals

class DurationsTest {
    @Test
    fun exactValuesAreOnePiece() {
        assertEquals(listOf(NoteValue("quarter", 1, false, 36)), Durations.split(36))
        assertEquals(listOf(NoteValue("eighth", 0, true, 8)), Durations.split(8))
        assertEquals(listOf(NoteValue("whole", 0, false, 96)), Durations.split(96))
    }

    @Test
    fun everythingElseSplitsIntoPiecesThatSum() {
        for (ticks in 1..150) {
            val pieces = Durations.split(ticks)
            assertEquals(ticks, pieces.sumOf { it.ticks }, "ticks=$ticks")
            if (ticks % 3 == 0) assertEquals(false, pieces.any { it.triplet }, "ticks=$ticks should not need triplets")
        }
    }
}
