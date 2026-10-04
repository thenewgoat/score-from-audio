package dev.scorefromaudio.pipeline

import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

class NpyArray(val shape: IntArray, val floats: FloatArray?, val longs: LongArray?)

/** Just enough of the .npy format for the fixtures the export step writes: little-endian f4 and i8, C order. */
object Npy {
    fun read(input: InputStream): NpyArray {
        val bytes = input.readBytes()
        require(bytes[0] == 0x93.toByte() && String(bytes, 1, 5, Charsets.US_ASCII) == "NUMPY") { "not an .npy file" }
        val major = bytes[6].toInt()
        val order = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        val (headerLength, start) = if (major == 1) order.getShort(8).toInt() and 0xffff to 10 else order.getInt(8) to 12
        val header = String(bytes, start, headerLength, Charsets.US_ASCII)
        require("'fortran_order': False" in header) { "Fortran order is not supported" }
        val descr = Regex("'descr':\\s*'([^']+)'").find(header)!!.groupValues[1]
        val shape = Regex("'shape':\\s*\\(([^)]*)\\)").find(header)!!.groupValues[1]
            .split(',').map { it.trim() }.filter { it.isNotEmpty() }.map { it.toInt() }.toIntArray()
        val count = shape.fold(1) { a, b -> a * b }
        val data = ByteBuffer.wrap(bytes, start + headerLength, bytes.size - start - headerLength)
            .slice().order(ByteOrder.LITTLE_ENDIAN)
        return when (descr) {
            "<f4" -> NpyArray(shape, FloatArray(count).also { data.asFloatBuffer().get(it) }, null)
            "<i8" -> NpyArray(shape, null, LongArray(count).also { data.asLongBuffer().get(it) })
            else -> error("unsupported dtype $descr")
        }
    }
}
