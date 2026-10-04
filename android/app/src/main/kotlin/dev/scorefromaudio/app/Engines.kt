package dev.scorefromaudio.app

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import dev.scorefromaudio.pipeline.Audio
import dev.scorefromaudio.pipeline.Frontend
import dev.scorefromaudio.pipeline.FrontendConstants
import dev.scorefromaudio.pipeline.Bd1Input
import dev.scorefromaudio.pipeline.Bd1Notator
import dev.scorefromaudio.pipeline.Note
import dev.scorefromaudio.pipeline.OrtBd1
import dev.scorefromaudio.pipeline.OrtTranskun
import dev.scorefromaudio.pipeline.Transcriber
import java.io.File

/** The two model stages, behind an interface so the job runner can be tested without models. */
interface Engines {
    fun notes(audio: Audio, progress: (Float) -> Unit, cancelled: () -> Boolean): List<Note>
    fun tokens(notes: List<Note>, progress: (Float) -> Unit, cancelled: () -> Boolean): List<IntArray>
}

/**
 * Each stage opens its sessions and closes them when done, so the two models are never in memory together.
 * The models folder and the runtime are prepared on first use, inside the NOTES stage, so a failure there
 * (a full disk while copying the models, a runtime that will not load) is recorded as a failed job.
 */
class OrtEngines(prepareModels: () -> File) : Engines {
    private val models by lazy(prepareModels)
    private val env by lazy { OrtEnvironment.getEnvironment() }

    /**
     * [arena] false for Transkun: ORT's CPU arena keeps its largest segment's buffers (about 2.2 GB measured on
     * the desktop) resident until the session closes. The notation model is small and runs many short steps, so it keeps it.
     */
    private fun options(arena: Boolean = true) = OrtSession.SessionOptions().apply {
        setIntraOpNumThreads(Runtime.getRuntime().availableProcessors().coerceAtMost(4))
        setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        setCPUArenaAllocator(arena)
    }

    /**
     * The int8 core at 12 s steps: each core run takes 2.82 s on the phone against fp32's 5.0 s, and there
     * are 1.5x fewer of them. On ASAP, onset F 0.989 against fp32's 0.991 (int8) and 0.9880 against 0.9882
     * (12 s step); see export.transkun_export.write_int8 and Transcriber.
     */
    override fun notes(audio: Audio, progress: (Float) -> Unit, cancelled: () -> Boolean): List<Note> {
        val constants = File(models, "transkun_frontend.bin").inputStream().use { FrontendConstants.read(it) }
        // Four threads, as for ORT: availableProcessors counts the little cores too.
        return Frontend(constants, threads = 4).use { frontend ->
            OrtTranskun(env, File(models, "transkun_core_int8.onnx").path, File(models, "transkun_heads.onnx").path, options(arena = false))
                .use { Transcriber(it, frontend, stepSeconds = 12.0).transcribe(audio, progress, cancelled) }
        }
    }

    /** bd1 (bd1-synth: ASAP, the jazz export and synthetic PDMX), not Beyer & Dai's own checkpoint. */
    override fun tokens(notes: List<Note>, progress: (Float) -> Unit, cancelled: () -> Boolean): List<IntArray> =
        OrtBd1(env, File(models, "bd1_encoder.onnx").path, File(models, "bd1_step.onnx").path, options())
            .use { Bd1Notator(it).notate(Bd1Input.encode(notes), progress, cancelled) }
}
