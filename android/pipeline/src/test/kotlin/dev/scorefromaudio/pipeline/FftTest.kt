package dev.scorefromaudio.pipeline

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals

class FftTest {
    @Test
    fun matchesANaiveDft() {
        val n = 64
        val random = Random(1)
        val x = DoubleArray(n) { random.nextDouble(-1.0, 1.0) }
        val re = x.copyOf()
        val im = DoubleArray(n)
        Fft(n).transform(re, im)
        for (k in 0 until n) {
            var r = 0.0
            var i = 0.0
            for (t in 0 until n) {
                r += x[t] * cos(2 * PI * k * t / n)
                i -= x[t] * sin(2 * PI * k * t / n)
            }
            assertEquals(r, re[k], 1e-9)
            assertEquals(i, im[k], 1e-9)
        }
    }

    @Test
    fun aPackedPairEqualsTwoTransforms() {
        val n = 256
        val random = Random(2)
        val x = DoubleArray(n) { random.nextDouble(-1.0, 1.0) }
        val y = DoubleArray(n) { random.nextDouble(-1.0, 1.0) + 0.3 }
        val fft = Fft(n)
        val xRe = x.copyOf(); val xIm = DoubleArray(n); fft.transform(xRe, xIm)
        val yRe = y.copyOf(); val yIm = DoubleArray(n); fft.transform(yRe, yIm)
        val half = n / 2 + 1
        val out = Array(4) { DoubleArray(half) }
        fft.transformPair(x.copyOf(), y.copyOf(), out[0], out[1], out[2], out[3])
        for (k in 0 until half) {
            assertEquals(xRe[k], out[0][k], 1e-12)
            assertEquals(xIm[k], out[1][k], 1e-12)
            assertEquals(yRe[k], out[2][k], 1e-12)
            assertEquals(yIm[k], out[3][k], 1e-12)
        }
    }
}
