package dev.scorefromaudio.pipeline

import java.util.concurrent.CancellationException
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.roundToInt

/**
 * Transkun's `transcribe`, in Kotlin: 16 s segments every [stepSeconds], Viterbi
 * forced to start where the previous segment's notes ended, notes cut by a
 * segment boundary merged, overlaps resolved. Mirrors ModelTransformer.transcribe
 * (with `stepInSecond = stepSeconds`) and transcribeFrames line for line;
 * TranskunParityTest holds it to torch's output at Transkun's own 8 s step.
 *
 * A 12 s step cuts 1.5x fewer segments; on the 74 bake-off recordings it scores
 * onset F 0.9880 against 0.9882 at 8 s, offset F 0.9204 against 0.9202.
 */
class Transcriber(private val model: TranskunModel, private val frontend: Frontend,
                  private val stepSeconds: Double = 8.0) {
    init { require(stepSeconds > 0 && stepSeconds <= SEGMENT_SECONDS) { "step must be in (0, $SEGMENT_SECONDS] s" } }

    private class Event(var start: Double, var end: Double, val pitch: Int, val velocity: Int,
                        val hasOnset: Boolean, var hasOffset: Boolean)

    private val order = compareBy<Event>({ it.start }, { it.end }, { it.pitch })

    fun transcribe(audio: Audio, progress: (Float) -> Unit = {}, cancelled: () -> Boolean = { false }): List<Note> {
        require(audio.sampleRate == Transkun.FS) { "Transkun needs ${Transkun.FS} Hz audio, got ${audio.sampleRate}" }
        val fs = Transkun.FS
        val hop = Transkun.HOP
        val padSeconds = SEGMENT_SECONDS - stepSeconds
        val pad = ceil(padSeconds * fs).toInt()
        val length = audio.frames
        val total = length + 2 * pad
        val stepSize = ceil(stepSeconds * fs / hop).toInt() * hop
        val segmentSize = ceil(SEGMENT_SECONDS * fs).toInt()
        val lastFrameIdx = (segmentSize.toDouble() / hop).roundToInt()
        val stepFrames = stepSize / hop
        val tracks = Transkun.TRACKS.size
        var startPos = IntArray(tracks) { floor(padSeconds * fs / hop).toInt() }
        val byPitch = LinkedHashMap<Int, MutableList<Event>>()

        val starts = (0 until total step stepSize).toList()
        val segments = frontend.segments(stepSize)
        for ((index, i) in starts.withIndex()) {
            if (cancelled()) throw CancellationException("cancelled")
            val slice = Array(audio.channels.size) { ch ->
                FloatArray(segmentSize) { k ->
                    val j = i + k - pad
                    if (i + k < total && j in 0 until length) audio.channels[ch][j] else 0f
                }
            }
            val beginTime = i.toDouble() / fs - padSeconds
            val (events, lastP) = segment(segments.features(slice, i), startPos, lastFrameIdx)
            startPos = IntArray(tracks) { max(lastP[it] - stepFrames, 0) }
            for (e in events) {
                e.start = max(e.start + beginTime, 0.0)
                e.end = max(e.end + beginTime, e.start)
            }
            for (e in events) {
                val list = byPitch.getOrPut(e.pitch) { mutableListOf() }
                val last = list.lastOrNull()
                if (last != null && e.start < last.end) {
                    if (e.hasOnset) list[list.size - 1] = e
                    else { last.hasOffset = e.hasOffset; last.end = max(e.end, last.end) }
                    continue
                }
                if (e.hasOnset) list.add(e)
            }
            progress((index + 1f) / starts.size)
        }
        for (list in byPitch.values) list.lastOrNull()?.hasOffset = true
        return resolveOverlapping(byPitch.values.flatten().filter { it.hasOffset })
    }

    private fun segment(features: Features, startPos: IntArray, lastFrameIdx: Int): Pair<List<Event>, IntArray> {
        val t = features.frames
        val tracks = Transkun.TRACKS.size
        val d = Transkun.CONTEXT
        val core = model.core(features.data)
        val paths = Viterbi.decode(core.score, t, tracks, startPos)
        val count = paths.sumOf { it.size }
        if (count == 0) return emptyList<Event>() to IntArray(tracks)

        val attr = FloatArray(count * 3 * d)
        var row = 0
        for ((track, intervals) in paths.withIndex()) for (iv in intervals) {
            val a = (track * t + iv[0]) * d
            val b = (track * t + iv[1]) * d
            val o = row * 3 * d
            for (k in 0 until d) {
                val x = core.ctx.get(a + k)
                val y = core.ctx.get(b + k)
                attr[o + k] = x; attr[o + d + k] = y; attr[o + 2 * d + k] = x * y
            }
            row++
        }
        val heads = model.heads(attr, count)

        val frame = Transkun.HOP.toDouble() / Transkun.FS
        val events = ArrayList<Event>(count)
        val lastP = IntArray(tracks)
        var n = 0
        for ((track, intervals) in paths.withIndex()) {
            var lastEnd = 0.0
            var trackLast = 0
            for (iv in intervals) {
                var start = (iv[0] + heads.ofValue[2 * n].toDouble()) * frame
                var end = (iv[1] + heads.ofValue[2 * n + 1].toDouble()) * frame
                val hasOnset = iv[0] > 0 || heads.ofPresence[2 * n]
                val hasOffset = iv[1] < lastFrameIdx || heads.ofPresence[2 * n + 1]
                start = max(start, lastEnd)
                end = max(end, start + 1e-8)
                lastEnd = end
                events.add(Event(start, end, Transkun.TRACKS[track], heads.velocity[n], hasOnset, hasOffset))
                if (hasOffset) trackLast = iv[1]
                n++
            }
            lastP[track] = trackLast
        }
        events.sortWith(order)
        return events to lastP
    }

    private fun resolveOverlapping(events: List<Event>): List<Note> {
        val sorted = events.sortedWith(order)
        val last = HashMap<Int, Int>()
        for ((index, e) in sorted.withIndex()) {
            last[e.pitch]?.let { if (sorted[it].end > e.start) sorted[it].end = e.start }
            last[e.pitch] = index
        }
        return sorted.sortedWith(order).filter { it.start < it.end }.map { Note(it.start, it.end, it.pitch, it.velocity) }
    }

    private companion object {
        /** The core graph is exported for 16 s segments (691 frames). */
        const val SEGMENT_SECONDS = 16.0
    }
}
