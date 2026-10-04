package dev.scorefromaudio.pipeline

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals

class NotatorParityTest {
    @Test
    fun reproducesThePythonLoopOnTheHeldOutPieces() {
        Fixtures.requireModels("bd_encoder.onnx", "bd_step.onnx")
        OrtNotation(OrtEnvironment.getEnvironment(), File(Fixtures.modelsDir, "bd_encoder.onnx").path,
                    File(Fixtures.modelsDir, "bd_step.onnx").path, OrtSession.SessionOptions()).use { model ->
            for (piece in BdFixture.PIECES) {
                val fixture = BdFixture.load(piece)
                val started = System.nanoTime()
                val tokens = Notator(model).notate(NotationInput.encode(fixture.detected()))
                println("$piece: ${tokens.size} tokens in ${(System.nanoTime() - started) / 1e9} s")
                assertEquals(fixture.tokens.size, tokens.size)
                for ((i, row) in fixture.tokens.withIndex()) assertEquals(row, tokens[i].toList(), "$piece note $i")
            }
        }
    }
}
