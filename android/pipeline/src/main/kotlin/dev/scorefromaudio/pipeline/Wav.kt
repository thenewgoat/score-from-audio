package dev.scorefromaudio.pipeline

import java.io.ByteArrayOutputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** 16-bit PCM WAV, read and written by hand so the same code runs on the JVM and on Android. */
object Wav {
    fun read(file: File, maxSeconds: Double? = null): Audio = read(file.readBytes(), maxSeconds)

    fun read(bytes: ByteArray, maxSeconds: Double? = null): Audio {
        val b = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        require(String(bytes, 0, 4, Charsets.US_ASCII) == "RIFF" && String(bytes, 8, 4, Charsets.US_ASCII) == "WAVE") {
            "not a WAV file"
        }
        var pos = 12
        var channels = 0
        var rate = 0
        var bits = 0
        while (pos + 8 <= bytes.size) {
            val id = String(bytes, pos, 4, Charsets.US_ASCII)
            val size = b.getInt(pos + 4)
            val body = pos + 8
            when (id) {
                "fmt " -> {
                    val format = b.getShort(body).toInt()
                    require(format == 1 || format == -2) { "only PCM WAV is supported (format $format)" }
                    channels = b.getShort(body + 2).toInt()
                    rate = b.getInt(body + 4)
                    bits = b.getShort(body + 14).toInt()
                    require(bits == 16) { "only 16-bit WAV is supported ($bits-bit)" }
                }
                "data" -> {
                    require(channels > 0) { "data chunk before fmt chunk" }
                    val available = minOf(size, bytes.size - body) / (2 * channels)
                    val frames = maxSeconds?.let { minOf(available, (it * rate).toInt()) } ?: available
                    val out = Array(channels) { FloatArray(frames) }
                    for (f in 0 until frames) for (c in 0 until channels) {
                        out[c][f] = b.getShort(body + 2 * (f * channels + c)) / 32768f
                    }
                    return Audio(out, rate)
                }
            }
            pos = body + size + (size and 1)
        }
        error("WAV file has no data chunk")
    }

    fun write(audio: Audio): ByteArray {
        val channels = audio.channels.size
        val dataSize = audio.frames * channels * 2
        val b = ByteBuffer.allocate(44 + dataSize).order(ByteOrder.LITTLE_ENDIAN)
        b.put("RIFF".toByteArray()).putInt(36 + dataSize).put("WAVE".toByteArray())
        b.put("fmt ".toByteArray()).putInt(16).putShort(1).putShort(channels.toShort())
            .putInt(audio.sampleRate).putInt(audio.sampleRate * channels * 2)
            .putShort((channels * 2).toShort()).putShort(16)
        b.put("data".toByteArray()).putInt(dataSize)
        for (f in 0 until audio.frames) for (c in 0 until channels) {
            b.putShort((audio.channels[c][f] * 32768f).toInt().coerceIn(-32768, 32767).toShort())
        }
        return b.array()
    }
}
