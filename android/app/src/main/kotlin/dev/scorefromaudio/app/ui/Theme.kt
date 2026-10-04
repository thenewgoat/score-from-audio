package dev.scorefromaudio.app.ui

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.defaultMinSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

/** The approved look: warm paper, ink, one blue accent, and red only for recording. */
object Ink {
    val Paper = Color(0xFFF5F1E8)
    val Card = Color(0xFFFFFFFF)
    val Ink = Color(0xFF1D1C1A)
    val Muted = Color(0xFF5E5A52)
    val Line = Color(0xFFDDD6C8)
    val Outline = Color(0xFFC9C1B1)
    val Track = Color(0xFFE2DBCC)
    val Sunken = Color(0xFFE6DFD1)
    val Blue = Color(0xFF2A5D8F)
    val BlueTint = Color(0xFFEAF0F6)
    val BlueDeep = Color(0xFF1E3A57)
    val Red = Color(0xFFB3372B)
    val RedRing = Color(0xFFE3B7B1)
    val Green = Color(0xFF2F6B3A)
    val Warn = Color(0xFF8A4B0F)
}

object Type {
    val Title = TextStyle(fontFamily = FontFamily.Serif, fontWeight = FontWeight.SemiBold, fontSize = 34.sp, color = Ink.Ink)
    val Heading = TextStyle(fontFamily = FontFamily.Serif, fontWeight = FontWeight.SemiBold, fontSize = 24.sp, color = Ink.Ink)
    val Body = TextStyle(fontSize = 16.sp, color = Ink.Ink)
    val Small = TextStyle(fontSize = 14.sp, color = Ink.Muted)
    val Caption = TextStyle(fontSize = 13.sp, color = Ink.Muted)
}

@Composable
fun ScoresTheme(content: @Composable () -> Unit) = MaterialTheme(
    colorScheme = lightColorScheme(
        primary = Ink.Blue, onPrimary = Color.White, background = Ink.Paper, surface = Ink.Card,
        onSurface = Ink.Ink, onBackground = Ink.Ink, error = Ink.Red, surfaceContainerLow = Ink.Card,
    ),
    content = content,
)

/** A filled pill; [color] blue unless it records. At least 48dp tall. */
@Composable
fun Pill(text: String, onClick: () -> Unit, modifier: Modifier = Modifier, color: Color = Ink.Blue, height: Dp = 52.dp,
         enabled: Boolean = true, icon: (@Composable () -> Unit)? = null) {
    Button(onClick = onClick, enabled = enabled, modifier = modifier.height(height), shape = CircleShape,
        colors = ButtonDefaults.buttonColors(containerColor = color, contentColor = Color.White,
            disabledContainerColor = Ink.Track, disabledContentColor = Ink.Muted),
        contentPadding = PaddingValues(horizontal = 18.dp)) {
        PillContent(text, icon, if (height >= 60.dp) 18 else 16)
    }
}

@Composable
fun OutlinePill(text: String, onClick: () -> Unit, modifier: Modifier = Modifier, height: Dp = 48.dp,
                enabled: Boolean = true, icon: (@Composable () -> Unit)? = null) {
    OutlinedButton(onClick = onClick, enabled = enabled, modifier = modifier.height(height), shape = CircleShape,
        border = BorderStroke(1.dp, Ink.Outline),
        colors = ButtonDefaults.outlinedButtonColors(contentColor = Ink.Ink),
        contentPadding = PaddingValues(horizontal = 14.dp)) {
        PillContent(text, icon, 15)
    }
}

@Composable
private fun RowScope.PillContent(text: String, icon: (@Composable () -> Unit)?, size: Int) {
    if (icon != null) { icon(); Box(Modifier.size(8.dp)) }
    Text(text, fontSize = size.sp, fontWeight = FontWeight.SemiBold)
}

/** A choice among a few (beats per bar); the chosen one outlined in blue. */
@Composable
fun Chip(text: String, selected: Boolean, onClick: () -> Unit, modifier: Modifier = Modifier) {
    Box(
        modifier
            .defaultMinSize(minHeight = 48.dp, minWidth = 48.dp)
            .clip(RoundedCornerShape(24.dp))
            .background(if (selected) Ink.BlueTint else Color.Transparent)
            .border(if (selected) 2.dp else 1.dp, if (selected) Ink.Blue else Ink.Outline, RoundedCornerShape(24.dp))
            .clickable(role = Role.RadioButton, onClick = onClick)
            .padding(horizontal = 16.dp),
        contentAlignment = Alignment.Center,
    ) {
        Text(text, fontSize = 15.sp, fontWeight = if (selected) FontWeight.SemiBold else FontWeight.Normal,
            color = if (selected) Ink.BlueDeep else Ink.Ink)
    }
}

/** A 48dp round icon button drawn with [draw], labelled for screen readers. */
@Composable
fun IconTap(label: String, onClick: () -> Unit, modifier: Modifier = Modifier, enabled: Boolean = true,
            draw: androidx.compose.ui.graphics.drawscope.DrawScope.() -> Unit) {
    Box(
        modifier.size(48.dp).clip(CircleShape).clickable(enabled = enabled, role = Role.Button, onClick = onClick)
            .semantics { contentDescription = label },
        contentAlignment = Alignment.Center,
    ) { Canvas(Modifier.size(24.dp), onDraw = draw) }
}

/** The screens' header: a back chevron and a serif title, with room for actions. */
@Composable
fun Header(title: String, onBack: () -> Unit, modifier: Modifier = Modifier, size: Int = 24,
           actions: @Composable RowScope.() -> Unit = {}) {
    Row(modifier.fillMaxWidth().padding(start = 4.dp, end = 8.dp, top = 8.dp, bottom = 4.dp),
        verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(2.dp)) {
        IconTap("Back to scores", onBack) { Icons.chevronLeft(this, Ink.Ink) }
        Text(title, style = Type.Heading.copy(fontSize = size.sp), modifier = Modifier.weight(1f), maxLines = 1,
            overflow = androidx.compose.ui.text.style.TextOverflow.Ellipsis)
        actions()
    }
}

/** Line icons, drawn on a 24-unit grid like the mockups' strokes. */
object Icons {
    private fun androidx.compose.ui.graphics.drawscope.DrawScope.u(v: Float) = v * size.width / 24f
    private fun androidx.compose.ui.graphics.drawscope.DrawScope.line(c: Color, vararg p: Float, w: Float = 2f) {
        for (i in 0 until p.size - 2 step 2) {
            drawLine(c, Offset(u(p[i]), u(p[i + 1])), Offset(u(p[i + 2]), u(p[i + 3])), strokeWidth = u(w), cap = StrokeCap.Round)
        }
    }

    fun chevronLeft(s: androidx.compose.ui.graphics.drawscope.DrawScope, c: Color) = s.line(c, 15f, 6f, 9f, 12f, 15f, 18f)
    fun chevronRight(s: androidx.compose.ui.graphics.drawscope.DrawScope, c: Color) = s.line(c, 9f, 6f, 15f, 12f, 9f, 18f)
    fun check(s: androidx.compose.ui.graphics.drawscope.DrawScope, c: Color, w: Float = 2.5f) = s.line(c, 5f, 12f, 10f, 17f, 19f, 7f, w = w)
    fun share(s: androidx.compose.ui.graphics.drawscope.DrawScope, c: Color) = with(s) {
        line(c, 12f, 3f, 12f, 15f); line(c, 7f, 8f, 12f, 3f, 17f, 8f); line(c, 5f, 14f, 5f, 21f, 19f, 21f, 19f, 14f)
    }
    fun trash(s: androidx.compose.ui.graphics.drawscope.DrawScope, c: Color) = with(s) {
        line(c, 4f, 7f, 20f, 7f); line(c, 9f, 7f, 9f, 4f, 15f, 4f, 15f, 7f); line(c, 6f, 7f, 7f, 20f, 17f, 20f, 18f, 7f)
    }
    fun more(s: androidx.compose.ui.graphics.drawscope.DrawScope, c: Color) = with(s) {
        for (y in listOf(5f, 12f, 19f)) drawCircle(c, u(1.8f), Offset(u(12f), u(y)))
    }
    fun mic(s: androidx.compose.ui.graphics.drawscope.DrawScope, c: Color) = with(s) {
        drawRoundRect(c, Offset(u(9f), u(3f)), androidx.compose.ui.geometry.Size(u(6f), u(11f)),
            androidx.compose.ui.geometry.CornerRadius(u(3f)), style = Stroke(u(2f)))
        drawArc(c, 0f, 180f, false, Offset(u(5f), u(4f)), androidx.compose.ui.geometry.Size(u(14f), u(14f)), style = Stroke(u(2f), cap = StrokeCap.Round))
        line(c, 12f, 18f, 12f, 21f)
    }
    fun file(s: androidx.compose.ui.graphics.drawscope.DrawScope, c: Color) = with(s) {
        line(c, 14f, 3f, 7f, 3f, 5f, 5f, 5f, 19f, 7f, 21f, 17f, 21f, 19f, 19f, 19f, 8f, 14f, 3f); line(c, 14f, 3f, 14f, 8f, 19f, 8f)
    }
    fun sliders(s: androidx.compose.ui.graphics.drawscope.DrawScope, c: Color) = with(s) {
        line(c, 4f, 6f, 14f, 6f); line(c, 18f, 6f, 20f, 6f); drawCircle(c, u(2f), Offset(u(16f), u(6f)), style = Stroke(u(2f)))
        line(c, 4f, 12f, 8f, 12f); line(c, 12f, 12f, 20f, 12f); drawCircle(c, u(2f), Offset(u(10f), u(12f)), style = Stroke(u(2f)))
        line(c, 4f, 18f, 16f, 18f); drawCircle(c, u(2f), Offset(u(18f), u(18f)), style = Stroke(u(2f)))
    }
    fun loop(s: androidx.compose.ui.graphics.drawscope.DrawScope, c: Color) = with(s) {
        line(c, 17f, 2f, 20f, 5f, 17f, 8f); line(c, 4f, 11f, 4f, 9f, 8f, 5f, 20f, 5f)
        line(c, 7f, 22f, 4f, 19f, 7f, 16f); line(c, 20f, 13f, 20f, 15f, 16f, 19f, 4f, 19f)
    }
    fun play(s: androidx.compose.ui.graphics.drawscope.DrawScope, c: Color) = with(s) {
        drawPath(androidx.compose.ui.graphics.Path().apply { moveTo(u(8f), u(5f)); lineTo(u(8f), u(19f)); lineTo(u(19f), u(12f)); close() }, c)
    }
    fun pause(s: androidx.compose.ui.graphics.drawscope.DrawScope, c: Color) = with(s) {
        drawRect(c, Offset(u(6f), u(5f)), androidx.compose.ui.geometry.Size(u(4f), u(14f)))
        drawRect(c, Offset(u(14f), u(5f)), androidx.compose.ui.geometry.Size(u(4f), u(14f)))
    }
    fun stop(s: androidx.compose.ui.graphics.drawscope.DrawScope, c: Color) = with(s) {
        drawRoundRect(c, Offset(u(5f), u(5f)), androidx.compose.ui.geometry.Size(u(14f), u(14f)), androidx.compose.ui.geometry.CornerRadius(u(2.5f)))
    }
    fun dot(s: androidx.compose.ui.graphics.drawscope.DrawScope, c: Color) = with(s) { drawCircle(c, u(7f), Offset(u(12f), u(12f))) }

    /** The library's score thumbnail: a staff with two noteheads. */
    fun staff(s: androidx.compose.ui.graphics.drawscope.DrawScope, low: Boolean) = with(s) {
        for (y in listOf(14f, 18f, 22f, 26f, 30f)) drawLine(Color(0xFF8A8477), Offset(u(8f * 24 / 44), u(y * 24 / 44)), Offset(u(36f * 24 / 44), u(y * 24 / 44)), u(0.6f))
        val notes = if (low) listOf(16f to 28f, 28f to 16f) else listOf(18f to 24f, 27f to 20f)
        for ((x, y) in notes) drawOval(Ink.Ink, Offset(u((x - 3.4f) * 24 / 44), u((y - 2.5f) * 24 / 44)), androidx.compose.ui.geometry.Size(u(6.8f * 24 / 44), u(5f * 24 / 44)))
    }
}
