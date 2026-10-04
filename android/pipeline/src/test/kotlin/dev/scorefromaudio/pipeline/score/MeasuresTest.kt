package dev.scorefromaudio.pipeline.score

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class MeasuresTest {
    @Test
    fun aDownbeatStartsABarAndStatesThePreviousOnesLength() {
        val layout = Measures.layout(fourFour(3))
        assertEquals(listOf(96, 96, 96), layout.bars.map { it.length })
        assertEquals(listOf(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2), layout.notes.map { it.measure })
        assertFalse(layout.pickup)
    }

    @Test
    fun aFallingOffsetStartsABarEvenWithoutADownbeat() {
        val layout = Measures.layout(listOf(tok(0), tok(24), tok(48), tok(72), tok(0), tok(24)))
        assertEquals(2, layout.bars.size)
        assertEquals(96, layout.bars[1].length)
    }

    @Test
    fun aShortFirstBarIsAPickup() {
        val rows = listOf(tok(0)) + fourFour(2).mapIndexed { i, r -> if (i == 0) r.also { it[1] = 25 } else r }
        val layout = Measures.layout(rows)
        assertEquals(listOf(24, 96, 96), layout.bars.map { it.length })
        assertTrue(layout.pickup)
    }

    @Test
    fun anUnwritableBarLengthIsRoundedAndCounted() {
        val layout = Measures.layout(listOf(tok(0), tok(0, downbeat = 101), tok(24)))
        assertEquals(102, layout.bars[0].length)
        assertEquals(2, layout.repairs.barLengthsRounded)
    }

    @Test
    fun keysChangeAtBarsAndUnknownKeepsThePrevious() {
        val rows = fourFour(3).mapIndexed { i, r -> r.also { it[5] = when (i / 4) { 0 -> 7; 1 -> 8; else -> 15 } } }
        assertEquals(listOf(0, 1, 1), Measures.layout(rows).bars.map { it.fifths })
    }

    @Test
    fun anOffsetPastTheBarIsClampedAndCounted() {
        val layout = Measures.layout(listOf(tok(0), tok(120)) + listOf(tok(0, downbeat = 97)))
        assertEquals(90, layout.notes[1].offset)
        assertEquals(1, layout.repairs.offsetsClamped)
    }

    @Test
    fun paddedRowsAreDroppedAndCounted() {
        val layout = Measures.layout(listOf(tok(0), tok(0, pad = 0), tok(24)))
        assertEquals(2, layout.notes.size)
        assertEquals(1, layout.repairs.droppedByModel)
    }

    @Test
    fun unknownHandGoesByPitch() {
        val layout = Measures.layout(listOf(tok(0, pitch = 72, hand = 2), tok(0, pitch = 40, hand = 2)))
        assertEquals(listOf(1, 2), layout.notes.map { it.staff })
    }

    @Test
    fun zeroDurationIsAGraceNote() {
        assertTrue(Measures.layout(listOf(tok(0, duration = 0), tok(0))).notes[0].grace)
    }

    @Test
    fun spelling() {
        assertEquals(-1, Spelling.alter(61, 1, 0).alter)        // D flat
        assertEquals(1, Spelling.alter(61, 3, 0).alter)         // C sharp
        assertEquals(1, Spelling.alter(60, 3, 0).alter)         // B sharp
        assertEquals("B", Spelling.step(60, 1)); assertEquals(3, Spelling.octave(60, 1))
        assertEquals(-1, Spelling.alter(70, 5, -2).alter)       // unknown on a black key in B flat: flat
        assertTrue(Spelling.alter(62, 3, 0).overridden)         // a sharp on D would be C#, not D: overridden
        assertEquals(1, Spelling.keyAlter("F", 1)); assertEquals(0, Spelling.keyAlter("C", 1))
        assertEquals(-1, Spelling.keyAlter("E", -2)); assertEquals(0, Spelling.keyAlter("A", -2))
    }

    @Test
    fun timeSignatures() {
        assertEquals(4 to 4, Measures.timeSignature(96))
        assertEquals(3 to 4, Measures.timeSignature(72))
        assertEquals(5 to 8, Measures.timeSignature(60))
        assertEquals(7 to 16, Measures.timeSignature(42))
    }
}
