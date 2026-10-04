package dev.scorefromaudio.pipeline

import kotlin.test.Test
import kotlin.test.assertEquals

class WavTest {
    @Test
    fun roundTripsSixteenBitStereo() {
        val left = FloatArray(1000) { (it % 50 - 25) / 32f }
        val right = FloatArray(1000) { -left[it] }
        val audio = Audio(arrayOf(left, right), 48_000)
        val back = Wav.read(Wav.write(audio))
        assertEquals(48_000, back.sampleRate)
        assertEquals(2, back.channels.size)
        for (i in left.indices) assertEquals(left[i], back.channels[0][i], 1f / 32768f)
    }

    @Test
    fun stopsAtMaxSeconds() {
        val audio = Audio(arrayOf(FloatArray(44_100 * 3)), 44_100)
        assertEquals(44_100, Wav.read(Wav.write(audio), maxSeconds = 1.0).frames)
    }
}
