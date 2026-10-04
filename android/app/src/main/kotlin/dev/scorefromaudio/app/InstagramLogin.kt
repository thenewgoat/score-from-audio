package dev.scorefromaudio.app

import android.content.Context
import android.webkit.CookieManager
import dev.scorefromaudio.pipeline.Links
import java.io.File

/**
 * The optional Instagram login: only the session cookies Instagram's own page
 * set in the WebView, saved in app-private storage for yt-dlp. The password is
 * typed into Instagram's page and never passes through this code. The cookie
 * file lives in noBackupFilesDir, and the app allows no backups at all (see
 * the manifest), so the session never leaves the phone in a backup.
 */
object InstagramLogin {
    const val URL = "https://www.instagram.com/accounts/login/"
    private const val SITE = "https://www.instagram.com"

    private const val COOKIES = "instagram-cookies.txt"

    fun cookieFile(context: Context): File {
        val file = File(context.noBackupFilesDir, COOKIES)
        // Builds before this one kept the file in filesDir.
        File(context.filesDir, COOKIES).takeIf { it.isFile }?.let { old -> if (file.isFile || !old.renameTo(file)) old.delete() }
        return file
    }

    fun isLoggedIn(context: Context) = cookieFile(context).isFile

    /** True once the WebView holds a session cookie; then saves it for yt-dlp. */
    fun captureIfLoggedIn(context: Context): Boolean {
        val header = CookieManager.getInstance().getCookie(SITE) ?: return false
        if ("sessionid=" !in header) return false
        val year = System.currentTimeMillis() / 1000 + 365L * 24 * 3600
        val file = cookieFile(context)
        val tmp = File(file.path + ".tmp")
        tmp.writeText(Links.netscapeCookies(header, ".instagram.com", year))
        if (!tmp.renameTo(file)) {
            tmp.delete()
            return false
        }
        return true
    }

    fun logOut(context: Context) {
        cookieFile(context).delete()
        CookieManager.getInstance().removeAllCookies { CookieManager.getInstance().flush() }
    }
}
