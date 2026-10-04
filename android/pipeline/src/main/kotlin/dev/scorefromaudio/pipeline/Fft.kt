package dev.scorefromaudio.pipeline

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin

/** In-place iterative radix-2 FFT of one fixed power-of-two size: X[k] = sum x[t] e^(-2 pi i k t / n). */
class Fft(val n: Int) {
    init { require(n > 1 && n and (n - 1) == 0) { "size must be a power of two" } }

    private val bits = Integer.numberOfTrailingZeros(n)
    private val cosTable = DoubleArray(n / 2) { cos(2 * PI * it / n) }
    private val sinTable = DoubleArray(n / 2) { sin(2 * PI * it / n) }
    private val reversed = IntArray(n) { Integer.reverse(it) ushr (32 - bits) }

    fun transform(re: DoubleArray, im: DoubleArray) {
        for (i in 0 until n) {
            val j = reversed[i]
            if (j > i) {
                val r = re[i]; re[i] = re[j]; re[j] = r
                val m = im[i]; im[i] = im[j]; im[j] = m
            }
        }
        var size = 2
        while (size <= n) {
            val half = size / 2
            val stride = n / size
            var base = 0
            while (base < n) {
                var k = 0
                for (j in base until base + half) {
                    val l = j + half
                    val c = cosTable[k]
                    val s = sinTable[k]
                    val tr = re[l] * c + im[l] * s
                    val ti = im[l] * c - re[l] * s
                    re[l] = re[j] - tr; im[l] = im[j] - ti
                    re[j] += tr; im[j] += ti
                    k += stride
                }
                base += size
            }
            size *= 2
        }
    }

    /**
     * Two real signals for the price of one complex FFT: x goes in the real
     * part and y in the imaginary part of z, and the halves come apart as
     * X[k] = (Z[k] + conj Z[n-k]) / 2 and Y[k] = (Z[k] - conj Z[n-k]) / 2i.
     * Overwrites [re] (holding x) and [im] (holding y); writes bins 0..n/2.
     */
    fun transformPair(re: DoubleArray, im: DoubleArray,
                      xRe: DoubleArray, xIm: DoubleArray, yRe: DoubleArray, yIm: DoubleArray) {
        transform(re, im)
        for (k in 0..n / 2) {
            val j = (n - k) and (n - 1)
            xRe[k] = (re[k] + re[j]) * 0.5; xIm[k] = (im[k] - im[j]) * 0.5
            yRe[k] = (im[k] + im[j]) * 0.5; yIm[k] = (re[j] - re[k]) * 0.5
        }
    }
}
