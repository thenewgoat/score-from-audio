package dev.scorefromaudio.app

import android.media.AudioFormat
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import dev.scorefromaudio.pipeline.Audio
import dev.scorefromaudio.pipeline.AudioTracks
import java.io.File
import java.io.IOException
import java.nio.ByteOrder
import java.util.concurrent.CancellationException

/**
 * Any audio or video file the phone can decode -> float samples per channel at the file's own rate.
 * Stops when [cancelled] says so, and gives up on a codec that produces nothing for [IDLE_LIMIT_MS].
 */
object MediaDecoder {
    private const val IDLE_LIMIT_MS = 10_000L

    fun decode(file: File, cancelled: () -> Boolean): Audio {
        val extractor = MediaExtractor()
        try {
            try { extractor.setDataSource(file.path) } catch (e: IOException) { throw UnreadableAudio("This file could not be read") }
            val mimes = List(extractor.trackCount) { extractor.getTrackFormat(it).getString(MediaFormat.KEY_MIME).orEmpty() }
            val track = AudioTracks.pick(mimes) ?: throw UnreadableAudio("This file has no audio track")
            extractor.selectTrack(track)
            val format = extractor.getTrackFormat(track)
            val codec = try { MediaCodec.createDecoderByType(mimes[track]) }
                        catch (e: Exception) { throw UnreadableAudio("This phone cannot decode ${mimes[track]}") }
            var channels = format.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
            var rate = format.getInteger(MediaFormat.KEY_SAMPLE_RATE)
            var encoding = AudioFormat.ENCODING_PCM_16BIT
            val samples = FloatBuilder()
            try {
                codec.configure(format, null, null, 0)
                codec.start()
                val info = MediaCodec.BufferInfo()
                var inputDone = false
                var outputDone = false
                var lastOutput = System.currentTimeMillis()
                while (!outputDone) {
                    if (cancelled()) throw CancellationException("cancelled")
                    if (System.currentTimeMillis() - lastOutput > IDLE_LIMIT_MS) throw UnreadableAudio("This file could not be decoded")
                    if (!inputDone) {
                        val i = codec.dequeueInputBuffer(10_000)
                        if (i >= 0) {
                            val size = extractor.readSampleData(codec.getInputBuffer(i)!!, 0)
                            if (size < 0) {
                                codec.queueInputBuffer(i, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                                inputDone = true
                            } else {
                                codec.queueInputBuffer(i, 0, size, extractor.sampleTime, 0)
                                extractor.advance()
                            }
                        }
                    }
                    val o = codec.dequeueOutputBuffer(info, 10_000)
                    if (o == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                        lastOutput = System.currentTimeMillis()
                        val f = codec.outputFormat
                        channels = f.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
                        rate = f.getInteger(MediaFormat.KEY_SAMPLE_RATE)
                        if (f.containsKey(MediaFormat.KEY_PCM_ENCODING)) encoding = f.getInteger(MediaFormat.KEY_PCM_ENCODING)
                    } else if (o >= 0) {
                        lastOutput = System.currentTimeMillis()
                        val buffer = codec.getOutputBuffer(o)!!.order(ByteOrder.nativeOrder())
                        buffer.position(info.offset)
                        buffer.limit(info.offset + info.size)
                        if (encoding == AudioFormat.ENCODING_PCM_FLOAT) {
                            val floats = buffer.asFloatBuffer()
                            while (floats.hasRemaining()) samples.add(floats.get())
                        } else {
                            val shorts = buffer.asShortBuffer()
                            while (shorts.hasRemaining()) samples.add(shorts.get() / 32768f)
                        }
                        codec.releaseOutputBuffer(o, false)
                        if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) outputDone = true
                    }
                }
            } finally {
                codec.release()
            }
            val interleaved = samples.toArray()
            val frames = interleaved.size / channels
            if (frames == 0) throw UnreadableAudio("This file's audio is empty")
            return Audio(Array(channels) { c -> FloatArray(frames) { interleaved[it * channels + c] } }, rate)
        } finally {
            extractor.release()
        }
    }

    private class FloatBuilder {
        private var data = FloatArray(1 shl 20)
        private var size = 0
        fun add(v: Float) {
            if (size == data.size) data = data.copyOf(size * 2)
            data[size++] = v
        }
        fun toArray(): FloatArray = data.copyOf(size)
    }
}
