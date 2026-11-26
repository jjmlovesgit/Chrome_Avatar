package com.pixelavatar.app

import android.app.Notification
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.provider.Settings
import android.service.notification.NotificationListenerService
import android.service.notification.StatusBarNotification
import java.lang.ref.WeakReference

data class GmailSnippet(
    val sender: String,
    val preview: String,
    val expanded: String,
    val account: String,
    val postTimeMs: Long,
) {
    fun bestBody(): String = expanded.ifBlank { preview }.trim()
}

class GmailNotificationListener : NotificationListenerService() {

    companion object {
        const val GMAIL_PKG = "com.google.android.gm"

        @Volatile
        private var activeServiceRef: WeakReference<GmailNotificationListener>? = null

        private val lock = Any()
        private val recentHistory = ArrayDeque<GmailSnippet>()
        private const val MAX_HISTORY = 8

        fun isPermissionGranted(context: Context): Boolean {
            val flat = Settings.Secure.getString(context.contentResolver, "enabled_notification_listeners")
                ?: return false
            val target = ComponentName(context, GmailNotificationListener::class.java).flattenToString()
            val shortTarget = ComponentName(context, GmailNotificationListener::class.java).flattenToShortString()
            return flat.split(':').any { it.equals(target, ignoreCase = true) || it.equals(shortTarget, ignoreCase = true) }
        }

        fun openNotificationAccessSettings(context: Context) {
            val intent = Intent(Settings.ACTION_NOTIFICATION_LISTENER_SETTINGS).apply {
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            }
            runCatching { context.startActivity(intent) }
        }

        /**
         * Returns the most recent Gmail notification snippets (newest first),
         * combining live shade notifications with recently received history.
         */
        fun getLatestEmails(maxCount: Int = 3): List<GmailSnippet> {
            activeServiceRef?.get()?.refreshFromActiveShade()
            synchronized(lock) {
                return recentHistory.take(maxCount)
            }
        }

        fun cachedCount(): Int {
            activeServiceRef?.get()?.refreshFromActiveShade()
            synchronized(lock) { return recentHistory.size }
        }

        private fun recordSnippets(items: List<GmailSnippet>) {
            if (items.isEmpty()) return
            synchronized(lock) {
                for (item in items) {
                    val keyBody = item.bestBody()
                    if (item.sender.isBlank() && keyBody.isBlank()) continue
                    recentHistory.removeAll {
                        it.sender.equals(item.sender, ignoreCase = true) &&
                            it.bestBody().equals(keyBody, ignoreCase = true)
                    }
                    recentHistory.addFirst(item)
                }
                while (recentHistory.size > MAX_HISTORY) {
                    recentHistory.removeLast()
                }
                recentHistory.sortByDescending { it.postTimeMs }
            }
        }

        private fun extractFromSbn(sbn: StatusBarNotification): List<GmailSnippet> {
            if (sbn.packageName != GMAIL_PKG) return emptyList()
            val n = sbn.notification ?: return emptyList()
            val extras = n.extras ?: return emptyList()

            val title = extras.getCharSequence(Notification.EXTRA_TITLE)?.toString()?.trim().orEmpty()
            val text = extras.getCharSequence(Notification.EXTRA_TEXT)?.toString()?.trim().orEmpty()
            val bigText = extras.getCharSequence(Notification.EXTRA_BIG_TEXT)?.toString()?.trim().orEmpty()
            val subText = extras.getCharSequence(Notification.EXTRA_SUB_TEXT)?.toString()?.trim().orEmpty()
            val textLines = extras.getCharSequenceArray(Notification.EXTRA_TEXT_LINES)

            val isGroupSummary = (n.flags and Notification.FLAG_GROUP_SUMMARY) != 0

            // When Gmail bundles multiple unread emails into one InboxStyle summary, parse each line cleanly
            if (isGroupSummary && textLines != null && textLines.isNotEmpty()) {
                return textLines.mapNotNull { lineCs ->
                    val line = lineCs?.toString()?.trim().orEmpty()
                    if (line.isBlank()) null
                    else {
                        val parts = line.split("  ", "\n", limit = 2)
                        val sender = parts.firstOrNull()?.trim().orEmpty()
                        val body = if (parts.size > 1) parts[1].trim() else line
                        GmailSnippet(
                            sender = sender.ifBlank { "Gmail" },
                            preview = body,
                            expanded = body,
                            account = subText,
                            postTimeMs = sbn.postTime,
                        )
                    }
                }
            }

            // Skip empty group summary headers with no message body
            if (isGroupSummary && bigText.isBlank() && text.isBlank()) return emptyList()
            if (title.isBlank() && text.isBlank() && bigText.isBlank()) return emptyList()

            return listOf(
                GmailSnippet(
                    sender = title.ifBlank { "Gmail" },
                    preview = text,
                    expanded = bigText.ifBlank { text },
                    account = subText,
                    postTimeMs = sbn.postTime,
                )
            )
        }
    }

    override fun onListenerConnected() {
        super.onListenerConnected()
        activeServiceRef = WeakReference(this)
        android.util.Log.i("PixelAvatar", "GmailNotificationListener connected!")
        refreshFromActiveShade()
    }

    override fun onListenerDisconnected() {
        super.onListenerDisconnected()
        if (activeServiceRef?.get() === this) activeServiceRef = null
    }

    override fun onNotificationPosted(sbn: StatusBarNotification?) {
        if (sbn == null || sbn.packageName != GMAIL_PKG) return
        val extracted = extractFromSbn(sbn)
        if (extracted.isNotEmpty()) {
            recordSnippets(extracted)
            for (e in extracted) {
                android.util.Log.i(
                    "PixelAvatar",
                    "[GMAIL CAPTURED] sender=${e.sender} | snippet=${e.bestBody().take(120)}",
                )
            }
        }
    }

    private fun refreshFromActiveShade() {
        val active = runCatching { activeNotifications }.getOrNull() ?: return
        val found = mutableListOf<GmailSnippet>()
        for (sbn in active) {
            if (sbn.packageName == GMAIL_PKG) {
                found.addAll(extractFromSbn(sbn))
            }
        }
        if (found.isNotEmpty()) {
            recordSnippets(found)
        }
    }
}
