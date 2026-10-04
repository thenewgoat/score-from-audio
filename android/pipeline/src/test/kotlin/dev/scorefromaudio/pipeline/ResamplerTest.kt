package dev.scorefromaudio.pipeline

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.sin
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertSame
import kotlin.test.assertTrue

class ResamplerTest {
    @Test
    fun fortyEightToFortyFourPointOneKeepsASine() {
        val tone = FloatArray(48_000) { (0.5 * sin(2 * PI * 1000 * it / 48_000.0)).toFloat() }
        val out = Resampler.resample(Audio(arrayOf(tone, tone), 48_000), 44_100)
        assertEquals(44_100, out.sampleRate)
        assertEquals(2, out.channels.size)
        assertEquals(44_100, out.frames)
        var worst = 0.0
        for (i in 200 until 44_100 - 200) worst = maxOf(worst, abs(out.channels[0][i] - 0.5 * sin(2 * PI * 1000 * i / 44_100.0)))
        assertTrue(worst < 2e-3, "worst error $worst")
    }

    @Test
    fun theRightRateIsLeftAlone() {
        val audio = Audio(arrayOf(FloatArray(10)), 44_100)
        assertSame(audio, Resampler.resample(audio, 44_100))
    }
}
