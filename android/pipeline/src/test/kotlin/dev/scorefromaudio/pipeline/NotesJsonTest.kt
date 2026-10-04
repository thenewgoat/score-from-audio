package dev.scorefromaudio.pipeline

import kotlin.test.Test
import kotlin.test.assertEquals

class NotesJsonTest {
    @Test
    fun roundTrips() {
        val notes = listOf(Note(0.5, 1.25, 60, 80), Note(0.0, 3.0, -64, 127))
        assertEquals(notes, NotesJson.read(NotesJson.write(notes)))
    }
}
