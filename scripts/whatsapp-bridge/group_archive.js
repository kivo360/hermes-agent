// Scoped message archiver for the Artificial Collective group only.
// Kevin asked for persistence for exactly one chat — everything else flows
// through unchanged and unrecorded. Appends JSON lines; one JSON object per
// message, no buffering (a crash loses at most the message being written).
import { appendFileSync, mkdirSync, readFileSync } from 'fs';
import path from 'path';

const ARCHIVE_CHAT_ID = '120363429357157668@g.us';

// Archive lives next to the WhatsApp session (same home as creds.json) so it
// survives bridge restarts and is easy to find: ~/.hermes/whatsapp/archive/
const ARCHIVE_DIR =
  process.env.HERMES_ARCHIVE_DIR ||
  path.join(process.env.HOME || '~', '.hermes', 'whatsapp', 'archive');
const ARCHIVE_FILE = path.join(ARCHIVE_DIR, 'artificial-collective.jsonl');

let archiveDirReady = false;
function ensureArchiveDir() {
  if (!archiveDirReady) {
    mkdirSync(ARCHIVE_DIR, { recursive: true });
    archiveDirReady = true;
  }
}

// True = this message belongs to the archived group. Cheap first filter.
export function isArchivedChat(chatId) {
  return chatId === ARCHIVE_CHAT_ID;
}

// Append one message to the JSONL archive. Never throws — archiving must
// not break message flow. Returns true on success.
export function archiveGroupMessage(event, extra = {}) {
  try {
    ensureArchiveDir();
    const record = {
      ts: new Date().toISOString(),
      // WhatsApp timestamps are epoch seconds
      wa_ts: event.timestamp ? new Date(Number(event.timestamp) * 1000).toISOString() : null,
      message_id: event.messageId || null,
      chat_id: event.chatId,
      chat_name: event.chatName || null,
      sender_id: event.senderId || null,
      sender_name: event.senderName || null,
      from_me: Boolean(extra.fromMe),
      is_group: Boolean(event.isGroup),
      type: event.mediaType || 'text',
      body: event.body || '',
      has_media: Boolean(event.hasMedia),
      quoted: event.hasQuotedMessage
        ? { id: event.quotedMessageId || null, text: event.quotedText || '' }
        : null,
      media_urls: event.mediaUrls && event.mediaUrls.length ? event.mediaUrls : null,
    };
    appendFileSync(ARCHIVE_FILE, JSON.stringify(record) + '\n');
    return true;
  } catch (err) {
    console.error('[archive] write failed:', err && err.message);
    return false;
  }
}

// GET /archive/stats backing — count lines + first/last timestamps.
export function archiveStats() {
  try {
    const raw = readFileSync(ARCHIVE_FILE, 'utf8').trim();
    if (!raw) return { file: ARCHIVE_FILE, messages: 0 };
    const lines = raw.split('\n');
    const first = JSON.parse(lines[0]);
    const last = JSON.parse(lines[lines.length - 1]);
    return {
      file: ARCHIVE_FILE,
      messages: lines.length,
      first_ts: first.ts,
      last_ts: last.ts,
    };
  } catch {
    return { file: ARCHIVE_FILE, messages: 0 };
  }
}

// GET /archive/messages — read back archived messages.
//   ?limit=N    last N messages (default 50)
//   ?sender=X   substring match on sender name/id
//   ?q=X        substring match on body
export function readArchiveMessages({ limit = 50, sender = null, q = null } = {}) {
  try {
    const raw = readFileSync(ARCHIVE_FILE, 'utf8').trim();
    if (!raw) return [];
    let lines = raw.split('\n');
    if (sender) {
      const needle = String(sender).toLowerCase();
      lines = lines.filter((l) => {
        try {
          const r = JSON.parse(l);
          return (
            (r.sender_name || '').toLowerCase().includes(needle) ||
            (r.sender_id || '').toLowerCase().includes(needle)
          );
        } catch {
          return false;
        }
      });
    }
    if (q) {
      const needle = String(q).toLowerCase();
      lines = lines.filter((l) => {
        try {
          return (JSON.parse(l).body || '').toLowerCase().includes(needle);
        } catch {
          return false;
        }
      });
    }
    const picked = lines.slice(-Math.max(1, Number(limit) || 50));
    return picked.map((l) => {
      try {
        return JSON.parse(l);
      } catch {
        return { unparsed: l };
      }
    });
  } catch {
    return [];
  }
}
