package dev.scorefromaudio.pipeline

import java.nio.FloatBuffer
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class TranscriberTest {
    private val frontend = Frontend(Fixtures.stream("transkun_frontend.bin").use { FrontendConstants.read(it) })
    private val t = Transkun.FRAMES
    private val tracks = Transkun.TRACKS.size

    /** A model whose first segment holds one interval, frames 400..450, on middle C; nothing else anywhere. */
    private inner class OneInterval : TranskunModel {
        var coreCalls = 0
        var headCalls = 0
        override fun core(features: FloatArray): CoreOutput {
            val score = FloatArray(t * t * tracks) { -1f }
            if (coreCalls++ == 0) score[(450 * t + 400) * tracks + Transkun.TRACKS.indexOf(60)] = 10f
            return CoreOutput(FloatBuffer.wrap(score), FloatBuffer.wrap(FloatArray(tracks * t * 256)))
        }
        override fun heads(attr: FloatArray, n: Int): HeadsOutput {
            headCalls++
            return HeadsOutput(IntArray(n) { 70 }, FloatArray(2 * n), BooleanArray(2 * n))
        }
    }

    @Test
    fun silenceGivesNoNotes() {
        val model = OneInterval().apply { coreCalls = 99 }
        val notes = Transcriber(model, frontend).transcribe(Audio(arrayOf(FloatArray(22_050)), 44_100))
        assertTrue(notes.isEmpty())
        assertEquals(0, model.headCalls)
    }

    @Test
    fun oneIntervalBecomesOneNote() {
        val model = OneInterval()
        val notes = Transcriber(model, frontend).transcribe(Audio(arrayOf(FloatArray(44_100)), 44_100))
        assertEquals(3, model.coreCalls)
        assertEquals(1, notes.size)
        val frame = 1024.0 / 44_100
        assertEquals(400 * frame - 8.0, notes[0].start, 1e-9)
        assertEquals(450 * frame - 8.0, notes[0].end, 1e-9)
        assertEquals(60, notes[0].pitch)
        assertEquals(70, notes[0].velocity)
    }

    /** 12 s steps: 4 s of padding, so one second of audio fits one segment and times shift by 4 s. */
    @Test
    fun oneIntervalBecomesOneNoteAtATwelveSecondStep() {
        val model = OneInterval()
        val notes = Transcriber(model, frontend, stepSeconds = 12.0).transcribe(Audio(arrayOf(FloatArray(44_100)), 44_100))
        assertEquals(1, model.coreCalls)
        assertEquals(1, notes.size)
        val frame = 1024.0 / 44_100
        assertEquals(400 * frame - 4.0, notes[0].start, 1e-9)
        assertEquals(450 * frame - 4.0, notes[0].end, 1e-9)
        assertEquals(60, notes[0].pitch)
    }

    /** Segments start every ceil(12 s / hop) hops across the audio and 4 s of padding at each end. */
    @Test
    fun aTwelveSecondStepCutsFewerSegments() {
        val model = OneInterval().apply { coreCalls = 99 }
        Transcriber(model, frontend, stepSeconds = 12.0).transcribe(Audio(arrayOf(FloatArray(30 * 44_100)), 44_100))
        // (30 + 2 * 4) s = 1_675_800 samples, steps of 517 hops = 529_408 samples: starts 0, 1, 2, 3 steps in.
        assertEquals(99 + 4, model.coreCalls)
    }
}
