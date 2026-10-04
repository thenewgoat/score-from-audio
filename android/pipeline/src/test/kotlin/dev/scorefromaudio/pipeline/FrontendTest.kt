package dev.scorefromaudio.pipeline

import kotlin.math.abs
import kotlin.math.ln
import kotlin.math.sqrt
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class FrontendTest {
    private val constants = Fixtures.stream("transkun_frontend.bin").use { FrontendConstants.read(it) }
    private val frontend = Frontend(constants)

    @Test
    fun reproducesTranskunsFeatures() {
        val audio = Fixtures.stream("frontend_audio.npy").use { Npy.read(it) }
        val n = audio.shape[1]
        val channels = Array(2) { c -> FloatArray(n) { audio.floats!![c * n + it] } }
        val expected = Fixtures.stream("frontend_features.npy").use { Npy.read(it) }.floats!!
        val features = frontend.features(channels)
        assertEquals(88, features.frames)
        var worst = 0f
        for (i in expected.indices) worst = maxOf(worst, abs(expected[i] - features.data[i]))
        assertTrue(worst < 2e-4f, "worst difference $worst")
    }

    @Test
    fun aSegmentHas691Frames() = assertEquals(691, frontend.frameCount(705_600))

    @Test
    fun monoIsAccepted() {
        val features = frontend.features(arrayOf(FloatArray(20_000) { (it % 100) / 100f }))
        assertEquals(frontend.frameCount(20_000), features.frames)
        assertTrue(features.data.all { it.isFinite() })
    }

    @Test
    fun threadsDoNotChangeABit() {
        val random = Random(3)
        val channels = Array(2) { FloatArray(60_000) { random.nextFloat() - 0.5f } }
        val one = Frontend(constants, threads = 1).use { it.features(channels) }
        val many = Frontend(constants, threads = 4).use { it.features(channels) }
        assertEquals(one.frames, many.frames)
        assertTrue(one.data.contentEquals(many.data))
    }

    @Test
    fun segmentsReuseSpectraAndMatchDirectFeaturesInStereo() = segmentsMatchDirectFeatures(2)

    @Test
    fun segmentsReuseSpectraAndMatchDirectFeaturesInMono() = segmentsMatchDirectFeatures(1)

    /**
     * The app's geometry scaled down: slices of 16 hops and a bit every 12 hops, as 16 s segments
     * (705_600 samples, not a whole number of hops) every 12 s.
     */
    @Test
    fun segmentsMatchDirectFeaturesAtTheAppsStepInStereo() = segmentsMatchDirectFeatures(2, size = 16 * 1024 + 700, step = 12 * 1024)

    @Test
    fun segmentsMatchDirectFeaturesAtTheAppsStepInMono() = segmentsMatchDirectFeatures(1, size = 16 * 1024 + 700, step = 12 * 1024)

    /**
     * Slices every [step] samples of a long signal, zero past its end as
     * Transcriber cuts them: cached interior frames and fresh edge frames
     * together must give what features(slice) gives.
     */
    private fun segmentsMatchDirectFeatures(channelCount: Int, size: Int = 30_000, step: Int = 10 * 1024) {
        val random = Random(4 + channelCount)
        val length = 100_000
        val signal = Array(channelCount) { FloatArray(length) { random.nextFloat() - 0.45f } }
        val segments = frontend.segments(step)
        for (start in 0 until length step step) {
            val slice = Array(channelCount) { ch -> FloatArray(size) { k -> if (start + k < length) signal[ch][start + k] else 0f } }
            val direct = frontend.features(slice)
            val cached = segments.features(slice, start)
            assertEquals(direct.frames, cached.frames)
            var worst = 0f
            for (i in direct.data.indices) worst = maxOf(worst, abs(direct.data[i] - cached.data[i]))
            assertTrue(worst < 1e-5f, "slice at $start: worst difference $worst")
        }
        assertTrue(segments.reused > 0, "no frame was reused")
    }

    /**
     * Half a unit of DC under 1e-3 of noise: the normalisation after the FFT
     * cancels almost all of the power in the low bands, which must still come
     * out finite, equal between cached and direct features, and close to
     * normalising before the FFT as torch does.
     */
    @Test
    fun aLargeDcOffsetSurvivesNormalisingAfterTheFft() {
        val random = Random(9)
        val length = 40_000
        val signal = Array(2) { FloatArray(length) { 0.5f + (random.nextFloat() - 0.5f) * 1e-3f } }
        val size = 16_384
        val step = 4 * 1024
        val segments = frontend.segments(step)
        for (start in 0 until length step step) {
            val slice = Array(2) { ch -> FloatArray(size) { k -> if (start + k < length) signal[ch][start + k] else 0f } }
            val direct = frontend.features(slice)
            val cached = segments.features(slice, start)
            assertTrue(direct.data.all { it.isFinite() } && cached.data.all { it.isFinite() }, "slice at $start")
            assertTrue(direct.data.contentEquals(cached.data), "slice at $start")
        }
        assertTrue(segments.reused > 0, "no frame was reused")

        val slice = Array(2) { ch -> signal[ch].copyOf(size) }
        val expected = normalisingFirst(slice)
        val actual = frontend.features(slice).data
        var worst = 0f
        for (i in expected.indices) worst = maxOf(worst, abs(expected[i] - actual[i]))
        assertTrue(worst < 1e-4f, "worst difference $worst")
    }

    /** The frontend as it was: normalise the samples, then window and FFT each channel alone. */
    private fun normalisingFirst(channels: Array<FloatArray>): FloatArray {
        val size = constants.windowSize
        val hop = 1024
        val n = channels[0].size
        val frames = frontend.frameCount(n)
        fun sample(ch: Int, f: Int, k: Int): Double = (f * hop + k - size / 2).let { if (it in 0 until n) channels[ch][it].toDouble() else 0.0 }
        var sum = 0.0
        var sumSquares = 0.0
        for (ch in channels.indices) for (f in 0 until frames) for (k in 0 until size) { val v = sample(ch, f, k); sum += v; sumSquares += v * v }
        val count = channels.size.toDouble() * frames * size
        val mean = sum / count
        val scale = 1.0 / (sqrt(maxOf((sumSquares - sum * mean) / (count - 1), 0.0)) + 1e-8)
        val fft = Fft(size)
        val nWindows = constants.windows.size
        val out = FloatArray(frames * constants.nMels * nWindows)
        for (f in 0 until frames) for (w in 0 until nWindows) {
            val power = DoubleArray(constants.nBins)
            for (ch in channels.indices) {
                val re = DoubleArray(size) { (sample(ch, f, it) - mean) * scale * constants.windows[w][it] }
                val im = DoubleArray(size)
                fft.transform(re, im)
                for (bin in power.indices) power[bin] += (re[bin] * re[bin] + im[bin] * im[bin]) / size
            }
            for (m in 0 until constants.nMels) {
                var acc = 0.0
                for (bin in power.indices) acc += power[bin] / channels.size * constants.mel[bin * constants.nMels + m]
                out[(f * constants.nMels + m) * nWindows + w] = ((ln(acc + 1e-5) - ln(1e-5)) / -ln(1e-5)).toFloat()
            }
        }
        return out
    }
}
