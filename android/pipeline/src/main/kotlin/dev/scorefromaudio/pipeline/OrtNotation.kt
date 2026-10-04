package dev.scorefromaudio.pipeline

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import java.nio.LongBuffer

/** NotationModel over bd_encoder.onnx and bd_step.onnx. Tensors are owned by the session results they came from. */
class OrtNotation(
    private val env: OrtEnvironment,
    encoderPath: String,
    stepPath: String,
    options: OrtSession.SessionOptions,
) : NotationModel, AutoCloseable {
    private val encoder = env.createSession(encoderPath, options)
    private val step = env.createSession(stepPath, options)
    private val layers = 4

    private inner class OrtEncoded(val result: OrtSession.Result) : Encoded {
        val cross: Map<String, OnnxTensor> = (0 until layers).flatMap { i -> listOf("cross_k$i", "cross_v$i") }
            .associateWith { result.get(it).get() as OnnxTensor }
        override fun close() = result.close()
    }

    private inner class OrtCache : Cache {
        override var length = 0
        var result: OrtSession.Result? = null
        val empty: List<OnnxTensor> = List(2 * layers) {
            OnnxTensor.createTensor(env, FloatBuffer.allocate(0), longArrayOf(1, 8, 0, 64))
        }
        fun past(i: Int, kv: Char): OnnxTensor =
            result?.get("present_$kv$i")?.get() as OnnxTensor? ?: empty[2 * i + if (kv == 'k') 0 else 1]
        override fun close() { result?.close(); empty.forEach { it.close() } }
    }

    override fun encode(input: NotationInput, from: Int, to: Int): Encoded {
        val n = to - from
        fun tensor(values: IntArray) =
            OnnxTensor.createTensor(env, LongBuffer.wrap(LongArray(n) { values[from + it].toLong() }), longArrayOf(1, n.toLong()))
        val inputs = mapOf("onset" to tensor(input.onset), "duration" to tensor(input.duration),
                           "pitch" to tensor(input.pitch), "velocity" to tensor(input.velocity))
        try {
            return OrtEncoded(encoder.run(inputs))
        } finally {
            inputs.values.forEach { it.close() }
        }
    }

    override fun newCache(): Cache = OrtCache()

    override fun step(encoded: Encoded, row: IntArray, cache: Cache): Array<FloatArray> {
        encoded as OrtEncoded
        cache as OrtCache
        val owned = listOf(
            OnnxTensor.createTensor(env, LongBuffer.wrap(LongArray(13) { row[it].toLong() }), longArrayOf(13)),
            OnnxTensor.createTensor(env, FloatBuffer.wrap(floatArrayOf(row[Streams.PAD].toFloat())), longArrayOf(1)),
            OnnxTensor.createTensor(env, LongBuffer.wrap(longArrayOf(cache.length.toLong())), longArrayOf(1)),
        )
        val feeds = HashMap<String, OnnxTensor>()
        feeds["tokens"] = owned[0]; feeds["pad"] = owned[1]; feeds["pos"] = owned[2]
        for (i in 0 until layers) {
            feeds["past_k$i"] = cache.past(i, 'k'); feeds["past_v$i"] = cache.past(i, 'v')
        }
        feeds.putAll(encoded.cross)
        try {
            val result = step.run(feeds)
            val logits = try {
                Array(14) { j ->
                    val name = if (j == Streams.PAD) "logit_pad" else "logits_${Streams.NAMES[j]}"
                    val buffer = (result.get(name).get() as OnnxTensor).floatBuffer
                    FloatArray(buffer.remaining()).also { buffer.get(it) }
                }
            } catch (e: Throwable) {
                result.close()
                throw e
            }
            cache.result?.close()
            cache.result = result
            cache.length++
            return logits
        } finally {
            owned.forEach { it.close() }
        }
    }

    override fun close() {
        encoder.close()
        step.close()
    }
}
