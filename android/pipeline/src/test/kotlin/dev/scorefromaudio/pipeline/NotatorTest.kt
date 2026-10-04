package dev.scorefromaudio.pipeline

import dev.scorefromaudio.pipeline.Streams.ACCIDENTAL
import dev.scorefromaudio.pipeline.Streams.DOWNBEAT
import dev.scorefromaudio.pipeline.Streams.OFFSET
import dev.scorefromaudio.pipeline.Streams.PAD
import dev.scorefromaudio.pipeline.Streams.PITCH
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertNotEquals

class NotatorTest {
    /** Predicts, for global note i, pitch i % 128 and pad 1; records what it was asked. */
    private class Fake : NotationModel {
        val encoded = mutableListOf<Pair<Int, Int>>()
        val fed = mutableListOf<MutableList<IntArray>>()
        private inner class E(val from: Int) : Encoded { override fun close() {} }
        private class C : Cache { override var length = 0; override fun close() {} }

        override fun encode(input: NotationInput, from: Int, to: Int): Encoded {
            encoded.add(from to to); fed.add(mutableListOf()); return E(from)
        }
        override fun newCache(): Cache = C()
        override fun step(encoded: Encoded, row: IntArray, cache: Cache): Array<FloatArray> {
            fed.last().add(row.copyOf())
            (cache as C).length++
            val note = (encoded as E).from + cache.length - 1
            return Array(14) { j ->
                val size = if (j == PAD) 1 else Streams.SIZES[j]
                FloatArray(size).also { if (j == PAD) it[0] = 1f else it[if (j == PITCH) note % 128 else 1 % size] = 1f }
            }
        }
    }

    private fun input(n: Int) = NotationInput.encode(List(n) { Note(it * 0.1, it * 0.1 + 0.05, 60, 64) })

    @Test
    fun everyNoteGetsExactlyOneToken() {
        for (n in listOf(1, 64, 65, 512, 513, 1100)) {
            val tokens = Notator(Fake()).notate(input(n))
            assertEquals(n, tokens.size, "n=$n")
            for ((i, row) in tokens.withIndex()) assertEquals(i % 128, row[PITCH], "n=$n note $i")
        }
    }

    @Test
    fun windowsOverlapBy64AndFeedThePreviousTokens() {
        val fake = Fake()
        val tokens = Notator(fake).notate(input(1100))
        assertEquals(listOf(0 to 512, 448 to 960, 896 to 1100), fake.encoded)
        val second = fake.fed[1]
        assertContentEquals(Streams.start(), second[0])
        for (k in 0 until 64) assertContentEquals(tokens[448 + k], second[1 + k])
    }

    private fun logits(offset: Int = 3, downbeat: Int = 0, pitch: Int = 60, accidental: Int = 2, pad: Float = 1f) =
        Array(14) { j ->
            if (j == PAD) floatArrayOf(pad)
            else FloatArray(Streams.SIZES[j]) { k -> -k.toFloat() }.also {
                when (j) {
                    OFFSET -> it[offset] = 10f
                    DOWNBEAT -> it[downbeat] = 10f
                    PITCH -> it[pitch] = 10f
                    ACCIDENTAL -> it[accidental] = 10f
                }
            }
        }

    @Test
    fun aFallingOffsetForcesADownbeat() {
        assertNotEquals(0, Notator.choose(logits(offset = 3, downbeat = 0), previousOffset = 10)[DOWNBEAT])
        assertEquals(0, Notator.choose(logits(offset = 3, downbeat = 0), previousOffset = 2)[DOWNBEAT])
    }

    @Test
    fun accidentalsImpossibleForThePitchAreNeverChosen() {
        // C#4 (61, class 1) forbids 0, 2, 5 and always 0, 4, 6; natural (2) is the favourite and must lose.
        val row = Notator.choose(logits(pitch = 61, accidental = 2), previousOffset = 0)
        assertEquals(1, row[ACCIDENTAL])
    }

    @Test
    fun aNonPositivePadLogitPadsTheWholeRow() {
        for (pad in listOf(-0.5f, 0f)) {
            val row = Notator.choose(logits(pad = pad), previousOffset = 0)
            assertEquals(0, row[PAD])
            for (j in 0 until 13) assertEquals(-1, row[j])
        }
    }
}
