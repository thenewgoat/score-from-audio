package dev.scorefromaudio.pipeline

import java.nio.FloatBuffer

/**
 * Transkun's semi-CRF decoding (`viterbiBackward`), one track at a time.
 *
 * `score` is [end][begin][track]. The non-event score is identically zero in
 * the exported model, so skipping a frame costs nothing. Arithmetic is float32,
 * as in torch, and ties keep the earlier candidate (skip before any interval).
 */
object Viterbi {
    fun decode(score: FloatBuffer, t: Int, tracks: Int, forced: IntArray): List<List<IntArray>> =
        List(tracks) { track -> decodeTrack(score, t, tracks, track, forced[track]) }

    private fun decodeTrack(score: FloatBuffer, t: Int, tracks: Int, track: Int, start: Int): List<IntArray> {
        fun s(end: Int, begin: Int) = score.get((end * t + begin) * tracks + track)
        fun positive(v: Float) = if (v > 0f) v else 0f

        val q = FloatArray(t)
        val choice = IntArray(t)
        q[t - 1] = positive(s(t - 1, t - 1))
        for (pos in t - 2 downTo 0) {
            var best = q[pos + 1]
            var pick = -1
            for (end in pos + 1 until t) {
                val v = q[end] + s(end, pos)
                if (v > best) { best = v; pick = end - pos - 1 }
            }
            choice[pos] = pick
            q[pos] = best + positive(s(pos, pos))
        }

        val out = ArrayList<IntArray>()
        var j = start
        while (j < t - 1) {
            val pick = choice[j]
            if (s(j, j) > 0f) out.add(intArrayOf(j, j))
            if (pick < 0) {
                j += 1
            } else {
                val end = pick + j + 1
                out.add(intArrayOf(j, end))
                j = end
            }
        }
        if (s(t - 1, t - 1) > 0f) out.add(intArrayOf(t - 1, t - 1))
        return out
    }
}
