package dev.scorefromaudio.pipeline.score

import kotlinx.serialization.Serializable
import kotlin.math.roundToInt

/** A place in the score, in quarter notes from its start, and the time in the recording where it was played. */
@Serializable
data class Anchor(val quarter: Double, val seconds: Double)

/** Where the written notes fall in the recording, and the tempo that follows from it. */
object Timing {
    /**
     * One anchor per score position that starts a note: the median onset of the notes there. Anchors that would run
     * backwards in time are dropped, keeping the longest run that moves forward in both.
     * [onsets] holds each token row's onset in seconds.
     */
    fun anchors(layout: Layout, onsets: DoubleArray): List<Anchor> {
        val byPosition = sortedMapOf<Int, MutableList<Double>>()
        for (n in layout.notes) {
            if (n.grace || n.row !in onsets.indices) continue
            byPosition.getOrPut(layout.position(n)) { ArrayList() }.add(onsets[n.row])
        }
        val points = byPosition.map { (position, times) -> Anchor(position / 24.0, median(times)) }
        return increasing(points)
    }

    /** Quarter notes a minute: the median pace over spans of at least four quarters. Null with too little to go on. */
    fun bpm(anchors: List<Anchor>): Int? {
        val paces = ArrayList<Double>()
        var j = 0
        for (i in anchors.indices) {
            if (j <= i) j = i + 1
            while (j < anchors.size && anchors[j].quarter - anchors[i].quarter < 4.0) j++
            if (j == anchors.size) break
            paces.add((anchors[j].seconds - anchors[i].seconds) / (anchors[j].quarter - anchors[i].quarter))
        }
        if (paces.isEmpty() && anchors.size >= 2) {
            val (a, b) = anchors.first() to anchors.last()
            paces.add((b.seconds - a.seconds) / (b.quarter - a.quarter))
        }
        if (paces.isEmpty()) return null
        return (60.0 / median(paces)).roundToInt().coerceIn(20, 300)
    }

    private fun median(values: List<Double>): Double {
        val sorted = values.sorted()
        val mid = sorted.size / 2
        return if (sorted.size % 2 == 1) sorted[mid] else (sorted[mid - 1] + sorted[mid]) / 2
    }

    /** The longest subsequence whose times strictly increase (points come sorted by quarter). */
    private fun increasing(points: List<Anchor>): List<Anchor> {
        if (points.isEmpty()) return points
        val tails = ArrayList<Int>()          // tails[k]: index of the smallest last time of a run of length k + 1
        val previous = IntArray(points.size) { -1 }
        for (i in points.indices) {
            var lo = 0
            var hi = tails.size
            while (lo < hi) {
                val mid = (lo + hi) / 2
                if (points[tails[mid]].seconds < points[i].seconds) lo = mid + 1 else hi = mid
            }
            if (lo > 0) previous[i] = tails[lo - 1]
            if (lo == tails.size) tails.add(i) else tails[lo] = i
        }
        val out = ArrayList<Anchor>()
        var k = tails.last()
        while (k >= 0) { out.add(points[k]); k = previous[k] }
        return out.reversed()
    }
}
