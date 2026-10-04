package dev.scorefromaudio.app.ui

import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class LegalTextTest {
    @Test
    fun editorNotesAreDroppedAndParagraphsSplitOnBlankLines() {
        val text = "// draft\n// fill in\n\nFirst line\nwraps here.\n\n# Heading\n\n- one\n- two\n"
        assertEquals(listOf("First line\nwraps here.", "# Heading", "- one\n- two"), blocks(text))
    }

    @Test
    fun wrappedLinesJoinAndBulletsBreak() {
        assertEquals("First line wraps here.", paragraph("First line\n  wraps here."))
        assertEquals("Intro\n• one\n• two", paragraph("Intro\n- one\n- two"))
    }

    @Test
    fun everyLegalTextHasContent() {
        LegalDoc.entries.forEach { doc ->
            val file = File("src/main/assets/${doc.asset}")
            assertTrue(file.isFile, "missing ${doc.asset}")
            assertTrue(blocks(file.readText()).size > 3, "${doc.asset} is nearly empty")
        }
    }
}
