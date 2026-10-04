package dev.scorefromaudio.app

import android.content.Context
import java.io.File

/**
 * Copies the models out of the APK so ONNX Runtime can open them by path. The copy is keyed on the
 * package's install/update time, stored in a stamp file: an APK update recopies every model, even a
 * retrained one of the same size, and so does any missing file or an interrupted earlier copy.
 * A recopy first deletes any file in the folder that is no longer a model.
 */
object ModelStore {
    private val NAMES = listOf("transkun_frontend.bin", "transkun_core_int8.onnx", "transkun_heads.onnx", "bd1_encoder.onnx", "bd1_step.onnx")
    private const val STAMP = "installed.stamp"

    fun ensure(context: Context): File {
        val dir = File(context.filesDir, "models").apply { mkdirs() }
        val stamp = File(dir, STAMP)
        @Suppress("DEPRECATION")
        val installed = context.packageManager.getPackageInfo(context.packageName, 0).lastUpdateTime.toString()
        if (stamp.isFile && stamp.readText() == installed && NAMES.all { File(dir, it).isFile }) return dir
        stamp.delete()
        // Whatever an earlier version shipped and this one does not (the fp32 Transkun core, say) goes.
        dir.listFiles()?.filter { it.name !in NAMES }?.forEach { it.delete() }
        for (name in NAMES) {
            val part = File(dir, "$name.part")
            context.assets.open("models/$name").use { input -> part.outputStream().use { input.copyTo(it, 1 shl 20) } }
            val target = File(dir, name)
            target.delete()
            if (!part.renameTo(target)) throw IllegalStateException("Could not install model $name")
        }
        stamp.writeText(installed)
        return dir
    }
}
