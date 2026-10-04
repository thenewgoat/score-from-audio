package dev.scorefromaudio.pipeline

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/**
 * TranskunModel over the two exported graphs. The same code runs under the JVM and Android runtimes.
 *
 * The core's input and outputs are pinned: tensors over direct buffers made once, which ONNX Runtime
 * reads from and writes into on every run, so the 234 MB of scores and context are never copied out
 * per segment. Hence [core]'s output is overwritten by the next call.
 *
 * The buffers are held for the model's lifetime and freed when collected, not by [close]: on the JVM
 * they are direct memory; on ART, allocateDirect makes a non-movable array on the managed heap, which
 * counts against the app's (large) heap like any other array.
 */
class OrtTranskun(
    private val env: OrtEnvironment,
    corePath: String,
    headsPath: String,
    options: OrtSession.SessionOptions,
) : TranskunModel, AutoCloseable {
    private val core: OrtSession
    private val heads: OrtSession
    private val input: Pinned
    private val score: Pinned
    private val ctx: Pinned
    private val output: CoreOutput

    init {
        // Allocating the pinned buffers can run out of memory, which the app survives: close what was built.
        val built = ArrayList<AutoCloseable>()
        try {
            core = env.createSession(corePath, options).also { built.add(it) }
            heads = env.createSession(headsPath, options).also { built.add(it) }
            input = Pinned(longArrayOf(1, Transkun.FRAMES.toLong(), 229, 6)).also { built.add(it) }
            score = Pinned(outputShape(core, "score")).also { built.add(it) }
            ctx = Pinned(outputShape(core, "ctx")).also { built.add(it) }
            output = CoreOutput(score.buffer, ctx.buffer)
        } catch (e: Throwable) {
            closeAll(built, e)
            throw e
        }
    }

    override fun core(features: FloatArray): CoreOutput {
        require(features.size == input.buffer.capacity()) { "core takes ${input.buffer.capacity()} features, got ${features.size}" }
        input.buffer.clear()
        input.buffer.put(features)
        // The result does not own pinned outputs: closing it leaves score and ctx intact.
        core.run(mapOf("features" to input.tensor), mapOf("score" to score.tensor, "ctx" to ctx.tensor)).close()
        return output
    }

    override fun heads(attr: FloatArray, n: Int): HeadsOutput =
        OnnxTensor.createTensor(env, FloatBuffer.wrap(attr), longArrayOf(n.toLong(), 768)).use { input ->
            heads.run(mapOf("attr" to input)).use { result ->
                val velocity = (result.get("velocity").get() as OnnxTensor).longBuffer
                @Suppress("UNCHECKED_CAST")
                val presence = (result.get("of_presence").get() as OnnxTensor).value as Array<BooleanArray>
                HeadsOutput(
                    IntArray(n) { velocity.get(it).toInt() },
                    floats(result, "of_value"),
                    BooleanArray(2 * n) { presence[it / 2][it % 2] },
                )
            }
        }

    private fun floats(result: OrtSession.Result, name: String): FloatArray {
        val buffer = (result.get(name).get() as OnnxTensor).floatBuffer
        return FloatArray(buffer.remaining()).also { buffer.get(it) }
    }

    private fun outputShape(session: OrtSession, name: String): LongArray =
        (session.outputInfo.getValue(name).info as TensorInfo).shape

    /**
     * A float tensor over a direct, native-order buffer, which ONNX Runtime reads and writes in place.
     * Kotlin reads [buffer] itself: OnnxTensor.getFloatBuffer would return a copy.
     */
    private inner class Pinned(shape: LongArray) : AutoCloseable {
        val buffer: FloatBuffer
        val tensor: OnnxTensor

        init {
            require(shape.all { it > 0 }) { "core tensors must have static shapes: ${shape.toList()}" }
            val bytes = shape.fold(4L, Long::times)
            require(bytes <= Int.MAX_VALUE) { "core tensor too large: ${shape.toList()}" }
            buffer = ByteBuffer.allocateDirect(bytes.toInt()).order(ByteOrder.nativeOrder()).asFloatBuffer()
            tensor = OnnxTensor.createTensor(env, buffer, shape)
        }

        override fun close() = tensor.close()
    }

    override fun close() = closeAll(listOf(core, heads, input, score, ctx), null)

    /** Closes every resource, last built first, even if one throws; failures go to [primary] or are rethrown. */
    private fun closeAll(resources: List<AutoCloseable>, primary: Throwable?) {
        var failure = primary
        for (resource in resources.asReversed()) {
            try {
                resource.close()
            } catch (e: Throwable) {
                if (failure == null) failure = e else failure.addSuppressed(e)
            }
        }
        if (primary == null && failure != null) throw failure
    }
}
