package dev.scorefromaudio.app

import android.content.Context
import android.content.Intent
import android.graphics.BitmapFactory
import android.graphics.Rect
import android.graphics.pdf.PdfDocument
import android.util.Base64
import androidx.core.content.FileProvider
import java.io.File

/** The score as files to share: MusicXML, MIDI and PDF, named after its title, in the job's export folder. */
object Exports {
    enum class Kind(val extension: String, val mime: String, val label: String) {
        PDF("pdf", "application/pdf", "Sheet music (PDF)"),
        MUSICXML("musicxml", "application/vnd.recordare.musicxml+xml", "MusicXML (MuseScore, Sibelius…)"),
        MIDI("mid", "audio/midi", "MIDI"),
    }

    fun fileName(title: String): String =
        title.replace(Regex("[^\\p{L}\\p{N} ._-]"), "").trim().take(60).ifEmpty { "score" }

    fun file(jobDir: File, title: String, kind: Kind) = File(File(jobDir, "export").apply { mkdirs() }, "${fileName(title)}.${kind.extension}")

    fun musicXml(jobDir: File, title: String): File =
        file(jobDir, title, Kind.MUSICXML).also { File(jobDir, "score.musicxml").copyTo(it, overwrite = true) }

    fun midi(jobDir: File, title: String, base64: String): File =
        file(jobDir, title, Kind.MIDI).also { it.writeBytes(Base64.decode(base64, Base64.DEFAULT)) }

    /** A4 pages from PNG images, each filling its page. */
    fun pdf(jobDir: File, title: String, pages: List<String>): File {
        val out = file(jobDir, title, Kind.PDF)
        val document = PdfDocument()
        try {
            pages.forEachIndexed { i, base64 ->
                val bytes = Base64.decode(base64, Base64.DEFAULT)
                val bitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
                val page = document.startPage(PdfDocument.PageInfo.Builder(595, 842, i + 1).create())
                page.canvas.drawBitmap(bitmap, null, Rect(0, 0, 595, 842), null)
                document.finishPage(page)
                bitmap.recycle()
            }
            out.outputStream().use { document.writeTo(it) }
        } finally {
            document.close()
        }
        return out
    }

    fun share(context: Context, file: File, kind: Kind) {
        val uri = FileProvider.getUriForFile(context, "${context.packageName}.files", file)
        val send = Intent(Intent.ACTION_SEND).setType(kind.mime).putExtra(Intent.EXTRA_STREAM, uri)
            .putExtra(Intent.EXTRA_SUBJECT, file.nameWithoutExtension)
            .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        context.startActivity(Intent.createChooser(send, "Share ${kind.label}"))
    }
}
