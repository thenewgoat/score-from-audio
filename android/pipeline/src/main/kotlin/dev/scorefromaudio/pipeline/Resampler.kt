package dev.scorefromaudio.pipeline

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.sin

/**
 * Band-limited resampling by windowed sinc (Blackman, 32 zero crossings each side).
 * Phones mostly record at 48 kHz and Transkun needs 44.1 kHz; Transkun's own
 * command line uses soxr for the same step.
 *
 * The kernel is tabulated once per call ([STEPS] entries per input sample, linearly
 * interpolated), and each output sample's taps are weighted once and applied to
 * every channel, so no trigonometry runs in the inner loop.
 */
object Resampler {
    private const val STEPS = 512

    fun resample(audio: Audio, rate: Int, zeroCrossings: Int = 32): Audio {
        if (audio.sampleRate == rate) return audio
        val ratio = rate.toDouble() / audio.sampleRate
        val cutoff = minOf(1.0, ratio) * 0.95
        val reach = ceil(zeroCrossings / cutoff).toInt()
        // table[k] = kernel(k / STEPS) for k in 0..reach*STEPS, plus one zero so interpolation never overruns.
        val table = DoubleArray(reach * STEPS + 2) { k ->
            val t = k.toDouble() / STEPS
            cutoff * sinc(cutoff * t) * blackman(t / reach)
        }
        val frames = Math.round(audio.frames * ratio).toInt()
        val inputs = audio.channels
        val size = audio.frames
        val outputs = Array(inputs.size) { FloatArray(frames) }
        val weights = DoubleArray(2 * reach + 1)
        for (i in 0 until frames) {
            val center = i / ratio
            val base = floor(center).toInt()
            val lo = maxOf(base - reach + 1, 0)
            val hi = minOf(base + reach, size - 1)
            var n = 0
            for (j in lo..hi) {
                val pos = abs(j - center) * STEPS
                val k = pos.toInt()
                val frac = pos - k
                weights[n++] = table[k] + (table[k + 1] - table[k]) * frac
            }
            for (c in inputs.indices) {
                val x = inputs[c]
                var sum = 0.0
                for (m in 0 until n) sum += x[lo + m] * weights[m]
                outputs[c][i] = sum.toFloat()
            }
        }
        return Audio(outputs, rate)
    }

    private fun sinc(x: Double) = if (x == 0.0) 1.0 else sin(PI * x) / (PI * x)

    private fun blackman(u: Double) = if (u <= -1.0 || u >= 1.0) 0.0 else 0.42 + 0.5 * cos(PI * u) + 0.08 * cos(2 * PI * u)
}
