package dev.scorefromaudio.desktop

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import dev.scorefromaudio.pipeline.Bd1Input
import dev.scorefromaudio.pipeline.Bd1Notator
import dev.scorefromaudio.pipeline.FrontendConstants
import dev.scorefromaudio.pipeline.Frontend
import dev.scorefromaudio.pipeline.NotationInput
import dev.scorefromaudio.pipeline.Note
import dev.scorefromaudio.pipeline.NotesJson
import dev.scorefromaudio.pipeline.Notator
import dev.scorefromaudio.pipeline.OrtBd1
import dev.scorefromaudio.pipeline.OrtNotation
import dev.scorefromaudio.pipeline.OrtTranskun
import dev.scorefromaudio.pipeline.Transcriber
import dev.scorefromaudio.pipeline.Wav
import dev.scorefromaudio.pipeline.score.ScoreWriter
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import java.io.File
import kotlin.system.exitProcess

private val models = File(System.getenv("MODELS_DIR") ?: "models").absoluteFile
private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
private fun options() = OrtSession.SessionOptions().apply { setIntraOpNumThreads(Runtime.getRuntime().availableProcessors()) }

private fun <T> timed(label: String, block: () -> T): T {
    val started = System.nanoTime()
    return block().also { System.err.println("$label: %.1f s".format((System.nanoTime() - started) / 1e9)) }
}

/** Transkun's own configuration unless asked for the app's: `--int8` core, `--step 12`. */
private class TranskunOptions(val int8: Boolean = false, val stepSeconds: Double = 8.0) {
    val core get() = if (int8) "transkun_core_int8.onnx" else "transkun_core.onnx"
}

private fun transcribe(wav: File, opts: TranskunOptions): List<Note> {
    val audio = Wav.read(wav)
    require(audio.sampleRate == 44_100) { "desktop CLI reads 44.1 kHz WAV only; this is ${audio.sampleRate} Hz" }
    val constants = File(models, "transkun_frontend.bin").inputStream().use { FrontendConstants.read(it) }
    return Frontend(constants).use { frontend ->
        OrtTranskun(env, File(models, opts.core).path, File(models, "transkun_heads.onnx").path, options()).use {
            timed("notes") { Transcriber(it, frontend, opts.stepSeconds).transcribe(audio) }
        }
    }
}

private fun notate(notes: List<Note>, out: File, bd1: Boolean) {
    val tokens = if (bd1) OrtBd1(env, File(models, "bd1_encoder.onnx").path, File(models, "bd1_step.onnx").path, options()).use {
        timed("notation") { Bd1Notator(it).notate(Bd1Input.encode(notes)) }
    } else OrtNotation(env, File(models, "bd_encoder.onnx").path, File(models, "bd_step.onnx").path, options()).use {
        timed("notation") { Notator(it).notate(NotationInput.encode(notes)) }
    }
    File(out.path + ".tokens.json").writeText(Json.encodeToString(tokens.map { it.toList() }))
    val written = timed("writing") { ScoreWriter.write(tokens, out.nameWithoutExtension) }
    out.writeText(written.xml)
    System.err.println("${written.bars} bars; repairs ${written.repairs}")
}

private fun usage(): Nothing {
    System.err.println("usage: transcribe <in.wav> <notes.json> | notate <notes.json> <out.musicxml> | full <in.wav> <out.musicxml>\n" +
                       "       [--int8] [--step <seconds>]   Transkun core and segment step, 0 < step <= 16 (default fp32, 8)\n" +
                       "       [--bd1]                       notate with bd1 (as the app does) rather than Beyer & Dai")
    exitProcess(2)
}

fun main(argv: Array<String>) {
    val args = mutableListOf<String>()
    var opts = TranskunOptions()
    var bd1 = false
    var i = 0
    while (i < argv.size) {
        when (val arg = argv[i++]) {
            "--bd1" -> bd1 = true
            "--int8" -> opts = TranskunOptions(true, opts.stepSeconds)
            "--step" -> opts = TranskunOptions(opts.int8, argv.getOrNull(i++)?.toDoubleOrNull()?.takeIf { it > 0 && it <= 16 } ?: usage())
            else -> args.add(arg)
        }
    }
    args.getOrNull(2)?.let { File(it).absoluteFile.parentFile?.mkdirs() }
    when (args.getOrNull(0)) {
        "transcribe" -> File(args[2]).writeText(NotesJson.write(transcribe(File(args[1]), opts)))
        "notate" -> notate(NotesJson.read(File(args[1]).readText()), File(args[2]), bd1)
        "full" -> notate(transcribe(File(args[1]), opts), File(args[2]), bd1)
        else -> usage()
    }
}
