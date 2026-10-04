package dev.scorefromaudio.pipeline

import java.io.DataInputStream
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.Callable
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.ThreadPoolExecutor
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import kotlin.math.ln
import kotlin.math.sqrt

/** The six analysis windows and the mel matrix, as `export.transkun_export.write_frontend` writes them. */
class FrontendConstants(val windows: Array<FloatArray>, val nBins: Int, val nMels: Int, val mel: FloatArray) {
    val windowSize: Int get() = windows[0].size

    companion object {
        fun read(input: InputStream): FrontendConstants {
            val bytes = DataInputStream(input).use { it.readBytes() }
            require(String(bytes, 0, 4, Charsets.US_ASCII) == "TKF1") { "not a Transkun frontend file" }
            val b = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
            val nWindows = b.getInt(4)
            val size = b.getInt(8)
            val nBins = b.getInt(12)
            val nMels = b.getInt(16)
            b.position(20)
            val floats = b.asFloatBuffer()
            val windows = Array(nWindows) { FloatArray(size).also { w -> floats.get(w) } }
            val mel = FloatArray(nBins * nMels).also { floats.get(it) }
            return FrontendConstants(windows, nBins, nMels, mel)
        }
    }
}

/** Features in the ONNX input layout: [frames][mels][windows], row-major. */
class Features(val frames: Int, val data: FloatArray)

/**
 * Transkun's MelSpectrum, in Kotlin: frame, gain-normalise over the whole
 * framed segment, apply six windows, FFT (orthonormal), take power, average
 * power across channels, project to mels, and log-normalise. Checked against
 * torch in FrontendTest.
 *
 * Normalisation is affine, so each frame's spectra are taken of the raw
 * samples and the segment's mean and scale applied afterwards:
 * FFT((x - mean) scale w) = scale (FFT(x w) - mean FFT(w)). The power, summed
 * over channels and projected to mels, is then
 * scale^2 (P - 2 mean Q + channels mean^2 |FFT(w)|^2) with P the mel power of
 * the raw spectra and Q the mel projection of Re(sum over channels of
 * FFT(x w) conj FFT(w)); both depend on the frame alone, which lets
 * [SegmentFeatures] keep them across overlapping segments.
 *
 * Frames are independent, so they are split across [threads] daemon threads;
 * each frame is computed by the same code whichever thread runs it, so the
 * result does not depend on them. Idle threads end after a second, and
 * [close] ends them at once.
 */
class Frontend(private val c: FrontendConstants, private val hop: Int = 1024,
               private val threads: Int = Runtime.getRuntime().availableProcessors().coerceAtMost(8)) : AutoCloseable {
    private val size = c.windowSize
    private val left = size / 2
    private val nWindows = c.windows.size
    private val fft = Fft(size)
    private val melStart = IntArray(c.nMels)
    private val melWeights: Array<FloatArray>
    /** Mel projection of |FFT(w)|^2, [window][mel]. */
    private val windowPower: DoubleArray
    private val windowRe: Array<DoubleArray>
    private val windowIm: Array<DoubleArray>
    private val pool: ThreadPoolExecutor? = if (threads > 1) {
        ThreadPoolExecutor(threads, threads, 1, TimeUnit.SECONDS, LinkedBlockingQueue()) { r ->
            Thread(r, "frontend").apply { isDaemon = true }
        }.apply { allowCoreThreadTimeOut(true) }
    } else null

    init {
        require(c.nBins <= size / 2 + 1) { "more bins than a real FFT of $size has" }
        val weights = ArrayList<FloatArray>(c.nMels)
        for (m in 0 until c.nMels) {
            var lo = -1
            var hi = -1
            for (bin in 0 until c.nBins) if (c.mel[bin * c.nMels + m] != 0f) { if (lo < 0) lo = bin; hi = bin + 1 }
            if (lo < 0) { lo = 0; hi = 0 }
            melStart[m] = lo
            weights.add(FloatArray(hi - lo) { c.mel[(lo + it) * c.nMels + m] })
        }
        melWeights = weights.toTypedArray()

        windowRe = Array(nWindows) { DoubleArray(size / 2 + 1) }
        windowIm = Array(nWindows) { DoubleArray(size / 2 + 1) }
        val spare = DoubleArray(size / 2 + 1)
        for (a in 0 until nWindows step 2) {
            val b = a + 1
            val re = DoubleArray(size) { c.windows[a][it].toDouble() }
            val im = DoubleArray(size) { if (b < nWindows) c.windows[b][it].toDouble() else 0.0 }
            fft.transformPair(re, im, windowRe[a], windowIm[a],
                              if (b < nWindows) windowRe[b] else spare, if (b < nWindows) windowIm[b] else spare)
        }
        windowPower = DoubleArray(nWindows * c.nMels)
        for (w in 0 until nWindows) {
            val power = DoubleArray(c.nBins) { windowRe[w][it] * windowRe[w][it] + windowIm[w][it] * windowIm[w][it] }
            project(power, windowPower, w * c.nMels)
        }
    }

    fun frameCount(samples: Int): Int = (samples + hop - 1) / hop + 1

    override fun close() { pool?.shutdown() }

    /**
     * What one frame contributes, before normalisation: the sum and sum of
     * squares of its samples (zero padding included) over channels, and P and
     * Q as [window][mel].
     */
    private class Frame(val sum: Double, val sumSquares: Double, val p: DoubleArray, val q: DoubleArray)

    /** Per-thread buffers for one frame's FFTs. */
    private inner class Scratch(channels: Int) {
        val raw = Array(channels) { DoubleArray(size) }
        val re = DoubleArray(size)
        val im = DoubleArray(size)
        val half = Array(4) { DoubleArray(size / 2 + 1) }
        val power = Array(nWindows) { DoubleArray(c.nBins) }
        val sumRe = Array(nWindows) { DoubleArray(c.nBins) }
        val sumIm = Array(nWindows) { DoubleArray(c.nBins) }
        val cross = DoubleArray(c.nBins)
    }

    /** Add the mel projection of [power] to [out] from [offset], one value per mel. */
    private fun project(power: DoubleArray, out: DoubleArray, offset: Int) {
        for (m in 0 until c.nMels) {
            val weights = melWeights[m]
            val lo = melStart[m]
            var acc = 0.0
            for (j in weights.indices) acc += power[lo + j] * weights[j]
            out[offset + m] = acc
        }
    }

    /** Run [body] on every index in 0 until [count], a chunk at a time across the pool, with one [scratch] per task. */
    private fun <S> parallel(count: Int, scratch: () -> S, body: (S, Int) -> Unit) {
        if (pool == null || count < 2) { val own = scratch(); for (i in 0 until count) body(own, i); return }
        val next = AtomicInteger()
        val chunk = 8
        val tasks = List(minOf(threads, (count + chunk - 1) / chunk)) {
            Callable {
                val own = scratch()
                while (true) {
                    val from = next.getAndAdd(chunk)
                    if (from >= count) break
                    for (i in from until minOf(from + chunk, count)) body(own, i)
                }
            }
        }
        for (done in pool.invokeAll(tasks)) done.get()
    }

    /** Frame [f] of [channels], zero outside them. */
    private fun frame(channels: Array<FloatArray>, f: Int, s: Scratch): Frame {
        val n = channels[0].size
        val first = f * hop - left
        var sum = 0.0
        var sumSquares = 0.0
        for (ch in channels.indices) {
            val raw = s.raw[ch]
            val samples = channels[ch]
            for (k in 0 until size) {
                val i = first + k
                val v = if (i in 0 until n) samples[i].toDouble() else 0.0
                raw[k] = v
                sum += v; sumSquares += v * v
            }
        }
        val p = DoubleArray(nWindows * c.nMels)
        val q = DoubleArray(nWindows * c.nMels)
        // A silent frame's spectra are exactly zero.
        if (sumSquares == 0.0) return Frame(sum, sumSquares, p, q)

        for (w in 0 until nWindows) { s.power[w].fill(0.0); s.sumRe[w].fill(0.0); s.sumIm[w].fill(0.0) }
        // Every (window, channel) pair is a real signal; FFT them two at a time.
        val signals = nWindows * channels.size
        for (pair in 0 until (signals + 1) / 2) {
            val a = 2 * pair
            val b = a + 1
            val ra = s.raw[a % channels.size]
            val wa = c.windows[a / channels.size]
            for (k in 0 until size) s.re[k] = ra[k] * wa[k]
            if (b < signals) {
                val rb = s.raw[b % channels.size]
                val wb = c.windows[b / channels.size]
                for (k in 0 until size) s.im[k] = rb[k] * wb[k]
            } else s.im.fill(0.0)
            fft.transformPair(s.re, s.im, s.half[0], s.half[1], s.half[2], s.half[3])
            for ((signal, o) in intArrayOf(a, b).zip(intArrayOf(0, 2))) {
                if (signal >= signals) continue
                val w = signal / channels.size
                val power = s.power[w]
                val sumRe = s.sumRe[w]
                val sumIm = s.sumIm[w]
                val xr = s.half[o]
                val xi = s.half[o + 1]
                for (bin in 0 until c.nBins) {
                    power[bin] += xr[bin] * xr[bin] + xi[bin] * xi[bin]
                    sumRe[bin] += xr[bin]; sumIm[bin] += xi[bin]
                }
            }
        }
        for (w in 0 until nWindows) {
            project(s.power[w], p, w * c.nMels)
            val sumRe = s.sumRe[w]
            val sumIm = s.sumIm[w]
            val wr = windowRe[w]
            val wi = windowIm[w]
            for (bin in 0 until c.nBins) s.cross[bin] = sumRe[bin] * wr[bin] + sumIm[bin] * wi[bin]
            project(s.cross, q, w * c.nMels)
        }
        return Frame(sum, sumSquares, p, q)
    }

    /** Whether frame [f] of an [n]-sample slice lies wholly inside it, so sees no zero padding. */
    private fun interior(f: Int, n: Int): Boolean = f * hop - left >= 0 && f * hop - left + size <= n

    /** Features of [channels], taking any frame [known] has and computing the rest; the frames come back too. */
    private fun compute(channels: Array<FloatArray>, known: (Int) -> Frame?): Pair<Features, Array<Frame?>> {
        val frames = frameCount(channels[0].size)
        val all = arrayOfNulls<Frame>(frames)
        for (f in 0 until frames) all[f] = known(f)
        val missing = (0 until frames).filter { all[it] == null }
        parallel(missing.size, { Scratch(channels.size) }) { s, i -> all[missing[i]] = frame(channels, missing[i], s) }

        var sum = 0.0
        var sumSquares = 0.0
        for (frame in all) { sum += frame!!.sum; sumSquares += frame.sumSquares }
        val count = channels.size.toDouble() * frames * size
        val mean = sum / count
        val std = sqrt(maxOf((sumSquares - sum * mean) / (count - 1), 0.0))
        val scale = 1.0 / (std + 1e-8)

        val out = FloatArray(frames * c.nMels * nWindows)
        val eps = 1e-5
        val logEps = ln(eps)
        val gain = scale * scale / (size * channels.size)
        val constant = channels.size * mean * mean
        parallel(frames, {}) { _, f ->
            val frame = all[f]!!
            for (w in 0 until nWindows) for (m in 0 until c.nMels) {
                val i = w * c.nMels + m
                // The expansion can cancel to a hair below zero where the mean dominates; power cannot be negative.
                val acc = maxOf(gain * (frame.p[i] - 2 * mean * frame.q[i] + constant * windowPower[i]), 0.0)
                out[(f * c.nMels + m) * nWindows + w] = ((ln(acc + eps) - logEps) / -logEps).toFloat()
            }
        }
        return Features(frames, out) to all
    }

    fun features(channels: Array<FloatArray>): Features = compute(channels) { null }.first

    /** Features of successive, overlapping slices of one signal; see [SegmentFeatures]. */
    fun segments(step: Int): SegmentFeatures = SegmentFeatures(step)

    /**
     * Features of slices of one long signal taken every [step] samples,
     * equal to [features] of each slice. A frame wholly inside a slice
     * depends only on the signal, so frames the next slice will also hold
     * whole are kept from one call to the next; frames that reach past
     * either end of a slice see its zero padding and are always computed.
     */
    inner class SegmentFeatures internal constructor(private val step: Int) {
        init { require(step > 0 && step % hop == 0) { "step must be a positive multiple of the hop" } }

        private var cache = HashMap<Int, Frame>()
        /** Frames taken from the cache so far. */
        var reused = 0
            private set

        /** Features of [slice], which starts [start] samples into the signal (a multiple of the hop). */
        fun features(slice: Array<FloatArray>, start: Int): Features {
            require(start % hop == 0) { "a slice must start on a hop" }
            val n = slice[0].size
            val offset = start / hop
            val (features, frames) = compute(slice) { f ->
                if (interior(f, n)) cache[offset + f]?.also { reused++ } else null
            }
            // Keep what is interior here and will be interior in a slice starting a step later.
            val keepFrom = (start + step) / hop + (left + hop - 1) / hop
            val kept = HashMap<Int, Frame>()
            for (f in frames.indices) if (offset + f >= keepFrom && interior(f, n)) kept[offset + f] = frames[f]!!
            cache = kept
            return features
        }
    }
}
