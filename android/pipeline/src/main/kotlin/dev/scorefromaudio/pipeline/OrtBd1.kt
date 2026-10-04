package dev.scorefromaudio.pipeline

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import java.nio.LongBuffer

/** Bd1Model over bd1_encoder.onnx and bd1_step.onnx. Tensors are owned by the session results they came from. */
class OrtBd1(
    private val env: OrtEnvironment,
    encoderPath: String,
    stepPath: String,
    options: OrtSession.SessionOptions,
) : Bd1Model, AutoCloseable {
    private val encoder = env.createSession(encoderPath, options)
    private val step = env.createSession(stepPath, options)
    private val layers = 4
    private val heads = 8
    private val width = 64

    private inner class OrtEncoded(val result: OrtSession.Result) : Encoded {
        val cross: Map<String, OnnxTensor> = (0 until layers).flatMap { i -> listOf("cross_k$i", "cross_v$i") }
            .associateWith { result.get(it).get() as OnnxTensor }
        override fun close() = result.close()
    }

    private inner class OrtCache : Cache {
        override var length = 0
        var result: OrtSession.Result? = null
        val empty: List<OnnxTensor> = List(2 * layers) {
            OnnxTensor.createTensor(env, FloatBuffer.allocate(0), longArrayOf(1, heads.toLong(), 0, width.toLong()))
        }
        fun past(i: Int, kv: Char): OnnxTensor =
            result?.get("present_$kv$i")?.get() as OnnxTensor? ?: empty[2 * i + if (kv == 'k') 0 else 1]
        override fun close() { result?.close(); empty.forEach { it.close() } }
    }

    override fun encode(input: Bd1Input, from: Int, to: Int): Encoded {
        val n = to - from
        fun tensor(values: IntArray) =
            OnnxTensor.createTensor(env, LongBuffer.wrap(LongArray(n) { values[from + it].toLong() }), longArrayOf(1, n.toLong()))
        val inputs = mapOf("pitch" to tensor(input.pitch), "onset" to tensor(input.onset),
                           "duration" to tensor(input.duration), "velocity" to tensor(input.velocity))
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
            OnnxTensor.createTensor(env, LongBuffer.wrap(LongArray(Bd1Streams.COUNT) { row[it].toLong() }),
                                    longArrayOf(Bd1Streams.COUNT.toLong())),
            OnnxTensor.createTensor(env, LongBuffer.wrap(longArrayOf(cache.length.toLong())), longArrayOf(1)),
        )
        val feeds = HashMap<String, OnnxTensor>()
        feeds["tokens"] = owned[0]; feeds["pos"] = owned[1]
        for (i in 0 until layers) {
            feeds["past_k$i"] = cache.past(i, 'k'); feeds["past_v$i"] = cache.past(i, 'v')
        }
        feeds.putAll(encoded.cross)
        try {
            val result = step.run(feeds)
            val logits = try {
                Array(Bd1Streams.COUNT) { j ->
                    val buffer = (result.get("logits_${Bd1Streams.NAMES[j]}").get() as OnnxTensor).floatBuffer
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
