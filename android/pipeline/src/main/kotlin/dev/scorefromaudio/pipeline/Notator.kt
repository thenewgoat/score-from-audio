package dev.scorefromaudio.pipeline

import dev.scorefromaudio.pipeline.Streams.ACCIDENTAL
import dev.scorefromaudio.pipeline.Streams.DOWNBEAT
import dev.scorefromaudio.pipeline.Streams.OFFSET
import dev.scorefromaudio.pipeline.Streams.PAD
import dev.scorefromaudio.pipeline.Streams.PITCH
import java.util.concurrent.CancellationException

/**
 * The notation decode loop: greedy, windows of 512 notes overlapping by 64,
 * the previous window's last 64 tokens fed as a prefix. Mirrors Beyer & Dai's
 * `infer` and `generate(kv_cache=True, top_k=1)`, except that a window stops
 * after its last note instead of running to 512 steps; a causal decoder's
 * earlier tokens do not depend on later steps. `export/bd_ref.py` is the
 * reference and NotatorParityTest holds this to it.
 */
class Notator(private val model: NotationModel, private val window: Int = 512, private val overlap: Int = 64) {
    fun notate(input: NotationInput, progress: (Float) -> Unit = {}, cancelled: () -> Boolean = { false }): List<IntArray> {
        val n = input.size
        if (n == 0) return emptyList()
        val out = ArrayList<IntArray>(n)
        for (lo in 0 until maxOf(n - overlap, 1) step (window - overlap)) {
            val hi = minOf(lo + window, n)
            val prefix = if (lo == 0) emptyList() else out.subList(lo, lo + overlap).map { it.copyOf() }
            out.addAll(decodeWindow(input, lo, hi, prefix, cancelled))
            progress(out.size.toFloat() / n)
        }
        check(out.size == n) { "decoded ${out.size} tokens for $n notes" }
        return out
    }

    private fun decodeWindow(input: NotationInput, lo: Int, hi: Int, prefix: List<IntArray>,
                             cancelled: () -> Boolean): List<IntArray> =
        model.encode(input, lo, hi).use { encoded ->
            model.newCache().use { cache ->
                var logits = model.step(encoded, Streams.start(), cache)
                for (row in prefix) logits = model.step(encoded, row, cache)
                var previous = prefix.lastOrNull()?.let { if (it[PAD] == 1) it[OFFSET] else 0 } ?: 0
                val count = (hi - lo) - prefix.size
                val rows = ArrayList<IntArray>(count)
                while (true) {
                    if (cancelled()) throw CancellationException("cancelled")
                    val row = choose(logits, previous)
                    rows.add(row)
                    previous = if (row[PAD] == 1) row[OFFSET] else 0
                    if (rows.size == count) break
                    logits = model.step(encoded, row, cache)
                }
                rows
            }
        }

    companion object {
        private val NEVER_ACCIDENTAL = setOf(0, 4, 6)
        private val IMPOSSIBLE_ACCIDENTAL = arrayOf(
            setOf(1, 4), setOf(0, 2, 5), setOf(1, 3), setOf(2, 4, 5), setOf(0, 3), setOf(1, 4),
            setOf(0, 2, 5), setOf(1, 3), setOf(0, 2, 4, 5), setOf(1, 3), setOf(2, 4, 5), setOf(0, 3),
        )

        /** One token row from one step's logits, with the decoder's two masks. */
        fun choose(logits: Array<FloatArray>, previousOffset: Int): IntArray {
            val row = IntArray(14) { -1 }
            for (j in 0 until 13) {
                val scores = logits[j]
                val forbidden: (Int) -> Boolean = when (j) {
                    DOWNBEAT -> { k -> k == 0 && row[OFFSET] < previousOffset }
                    ACCIDENTAL -> {
                        val bad = NEVER_ACCIDENTAL + IMPOSSIBLE_ACCIDENTAL[row[PITCH] % 12]
                        { k -> k in bad }
                    }
                    else -> { _ -> false }
                }
                var best = -1
                for (k in scores.indices) if (!forbidden(k) && (best < 0 || scores[k] > scores[best])) best = k
                row[j] = best
            }
            row[PAD] = if (logits[PAD][0] > 0f) 1 else 0
            if (row[PAD] == 0) for (j in 0 until 13) row[j] = -1
            return row
        }
    }
}
