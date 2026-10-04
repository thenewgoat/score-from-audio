package dev.scorefromaudio.app.ui

import android.annotation.SuppressLint
import android.content.Context
import android.webkit.CookieManager
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.activity.compose.BackHandler
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import dev.scorefromaudio.app.InstagramLogin
import kotlinx.coroutines.delay
import dev.scorefromaudio.app.BuildConfig

/** The legal texts, kept as plain text in assets/legal so they can be reviewed and replaced without touching code. */
enum class LegalDoc(val title: String, val asset: String) {
    TERMS("Terms of use", "legal/terms.txt"),
    PRIVACY("Privacy policy", "legal/privacy.txt"),
    NOTICES("Credits", "legal/notices.txt"),
    LICENCES("Full licence texts", "legal/licences.txt"),
}

/**
 * Whether the current terms were agreed to. Bump [TERMS_VERSION] whenever terms.txt or privacy.txt change in
 * substance, so the next launch asks again.
 */
object Consent {
    const val TERMS_VERSION = 1
    private const val PREFS = "legal"
    private const val KEY = "accepted_terms_version"

    fun accepted(context: Context) = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).getInt(KEY, 0) >= TERMS_VERSION

    fun accept(context: Context) = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit().putInt(KEY, TERMS_VERSION).apply()
}

/**
 * Settings and About: the optional Instagram login, the version, and the terms, privacy policy and credits, each
 * opening as a page of its own.
 */
@Composable
fun SettingsScreen(onBack: () -> Unit) {
    val context = LocalContext.current
    var open by rememberSaveable { mutableStateOf<LegalDoc?>(null) }
    var loggedIn by remember { mutableStateOf(InstagramLogin.isLoggedIn(context)) }
    var loggingIn by rememberSaveable { mutableStateOf(false) }
    if (loggingIn) {
        InstagramLoginPage(onDone = { loggedIn = InstagramLogin.isLoggedIn(context); loggingIn = false })
        return
    }
    open?.let { doc ->
        BackHandler { open = null }
        LegalScreen(doc) { open = null }
        return
    }
    Column(Modifier.fillMaxSize().background(Ink.Paper).verticalScroll(rememberScrollState())) {
        Header("Settings", onBack)
        Column(Modifier.padding(horizontal = 20.dp, vertical = 8.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text("Instagram", style = Type.Body.copy(fontWeight = FontWeight.SemiBold))
            Text(if (loggedIn) "Logged in. Instagram links use this login." else
                "Some Instagram posts need a login. Your password goes to Instagram's own page, never to this app.",
                style = Type.Small)
            Text("Heavy use of an account by a downloader can get it rate-limited.", style = Type.Caption)
            if (loggedIn) OutlinePill("Log out", { InstagramLogin.logOut(context); loggedIn = false })
            else OutlinePill("Log in to Instagram", { loggingIn = true })
        }
        HorizontalDivider(color = Ink.Line, modifier = Modifier.padding(top = 12.dp))
        Column(Modifier.padding(horizontal = 20.dp, vertical = 12.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text("Score from audio ${BuildConfig.VERSION_NAME}", style = Type.Body.copy(fontWeight = FontWeight.SemiBold))
            Text("Free, for non-commercial use. Your recordings are turned into sheet music on this phone and are never uploaded.",
                style = Type.Small)
        }
        HorizontalDivider(color = Ink.Line)
        LegalDoc.entries.forEach { doc ->
            Text(doc.title, style = Type.Body, modifier = Modifier.fillMaxWidth()
                .clickable(role = Role.Button) { open = doc }.padding(horizontal = 20.dp, vertical = 16.dp))
            HorizontalDivider(color = Ink.Line)
        }
    }
}

/**
 * Instagram's own login page in a WebView. Once it has set a session cookie, the cookies are saved for yt-dlp and
 * the page closes. Instagram's page is a single-page app that may log in without a new page load, so the cookie is
 * looked for on every page load and history change, once a second while the page is shown, and once more on leaving.
 */
@SuppressLint("SetJavaScriptEnabled")
@Composable
private fun InstagramLoginPage(onDone: () -> Unit) {
    val context = LocalContext.current
    var done by remember { mutableStateOf(false) }
    fun capture() {
        if (!done && InstagramLogin.captureIfLoggedIn(context)) { done = true; onDone() }
    }
    fun leave() {
        capture()
        if (!done) { done = true; onDone() }
    }
    LaunchedEffect(Unit) {
        while (!done) {
            delay(1_000)
            capture()
        }
    }
    BackHandler { leave() }
    Column(Modifier.fillMaxSize().background(Ink.Paper)) {
        Header("Log in to Instagram", { leave() }, size = 22)
        AndroidView(modifier = Modifier.fillMaxSize(), factory = { ctx ->
            WebView(ctx).apply {
                settings.javaScriptEnabled = true
                settings.domStorageEnabled = true
                CookieManager.getInstance().setAcceptCookie(true)
                webViewClient = object : WebViewClient() {
                    override fun onPageFinished(view: WebView, url: String) = capture()
                    override fun doUpdateVisitedHistory(view: WebView, url: String?, isReload: Boolean) = capture()
                }
                loadUrl(InstagramLogin.URL)
            }
        }, onRelease = { it.destroy() })
    }
}

/** One legal text, scrolled. Lines starting "# " are headings, lines starting "//" are notes for editors and not shown,
 *  and blank lines separate paragraphs. */
@Composable
fun LegalScreen(doc: LegalDoc, onBack: () -> Unit) {
    val context = LocalContext.current
    val blocks = remember(doc) { blocks(context.assets.open(doc.asset).bufferedReader().use { it.readText() }) }
    Column(Modifier.fillMaxSize().background(Ink.Paper)) {
        Header(doc.title, onBack, size = 22)
        // Lazy: the full licence texts run to over a thousand paragraphs.
        LazyColumn(Modifier.weight(1f), contentPadding = PaddingValues(horizontal = 20.dp, vertical = 8.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp)) {
            items(blocks) { block ->
                if (block.startsWith("# ")) Text(block.removePrefix("# "), style = Type.Heading.copy(fontSize = 19.sp),
                    modifier = Modifier.padding(top = 8.dp))
                else Text(paragraph(block), style = Type.Body.copy(fontSize = 15.sp, lineHeight = 22.sp))
            }
        }
    }
}

/** The text's paragraphs and headings, without the "//" editor notes. */
internal fun blocks(text: String): List<String> =
    text.lines().filterNot { it.trimStart().startsWith("//") }.joinToString("\n")
        .split(Regex("\n\\s*\n")).map { it.trim() }.filter { it.isNotEmpty() }

/** Wrapped lines join into one paragraph; a line starting "- " starts a bullet of its own. */
internal fun paragraph(block: String): String =
    block.lines().map { it.trim() }.fold(StringBuilder()) { out, line ->
        when {
            line.startsWith("- ") -> { if (out.isNotEmpty()) out.append('\n'); out.append("• ").append(line.removePrefix("- ")) }
            out.isEmpty() -> out.append(line)
            else -> out.append(' ').append(line)
        }
        out
    }.toString()

/**
 * Shown once, before anything else, until the current terms are agreed to. The texts open in full from here;
 * nothing is recorded or processed before "Agree and continue".
 */
@Composable
fun TermsGate(onAccept: () -> Unit) {
    var open by rememberSaveable { mutableStateOf<LegalDoc?>(null) }
    open?.let { doc ->
        BackHandler { open = null }
        LegalScreen(doc) { open = null }
        return
    }
    Column(Modifier.fillMaxSize().background(Ink.Paper).padding(horizontal = 24.dp, vertical = 32.dp),
        verticalArrangement = Arrangement.spacedBy(14.dp)) {
        Text("Welcome", style = Type.Title)
        Text("Record piano, get sheet music.", style = Type.Small.copy(fontSize = 15.sp))
        Column(Modifier.weight(1f).verticalScroll(rememberScrollState()).padding(top = 12.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)) {
            Text("Before you start:", style = Type.Body.copy(fontWeight = FontWeight.SemiBold))
            Text("• Your music is turned into sheet music on this phone. Your recordings and scores are never uploaded; the app goes online only to fetch a YouTube or Instagram link you give it.", style = Type.Body)
            Text("• The app is free and for non-commercial use only, because the music datasets its models learned from allow only that.", style = Type.Body)
            Text("• Scores are made automatically and will contain mistakes. Only transcribe music you have the right to use.", style = Type.Body)
            Text("Read the Terms of use", style = Type.Body.copy(color = Ink.Blue),
                modifier = Modifier.clickable(role = Role.Button) { open = LegalDoc.TERMS }.padding(vertical = 6.dp))
            Text("Read the Privacy policy", style = Type.Body.copy(color = Ink.Blue),
                modifier = Modifier.clickable(role = Role.Button) { open = LegalDoc.PRIVACY }.padding(vertical = 6.dp))
        }
        Pill("Agree and continue", onAccept, Modifier.fillMaxWidth(), height = 56.dp)
    }
}
