import http from 'node:http';
import { Client, GatewayIntentBits, PermissionFlagsBits } from 'discord.js';

const env = (key, fallback = '') => process.env[key] ?? fallback;
const DISCORD_TOKEN = env('DISCORD_TOKEN');
const TARGET_CHANNEL_ID = env('TARGET_CHANNEL_ID', '1530832778331951184');
const BRAIN_WEBHOOK_SECRET = env('BRAIN_WEBHOOK_SECRET');
const PORT = Number(env('PORT', '10000'));
const MIN_SIGNAL = Number(env('MIN_SIGNAL', '0.05'));
const COOLDOWN_MS = Number(env('ACTION_COOLDOWN_MS', '3000'));
const BRAIN_TIMEOUT_MS = Number(env('BRAIN_TIMEOUT_MS', '30000'));
const BRAIN_PROCESS_TIMEOUT_MS = Number(env('BRAIN_PROCESS_TIMEOUT_MS', '60000'));
const BRAIN_PROCESS_URL = env('BRAIN_PROCESS_URL', 'https://flybot-brain.onrender.com/brain/process');
const BRAIN_RETRIES = Math.max(1, Number(env('BRAIN_RETRIES', '3')));

if (!DISCORD_TOKEN) throw new Error('DISCORD_TOKEN is required');
if (!BRAIN_WEBHOOK_SECRET) throw new Error('BRAIN_WEBHOOK_SECRET is required');

const client = new Client({
  intents: [
    GatewayIntentBits.Guilds,
    GatewayIntentBits.GuildMessages,
    GatewayIntentBits.MessageContent
  ]
});

let lastSignal = null;
let lastMessage = null;
let lastMessageText = null;
let lastNeuralResult = null;
let messagesSent = 0;
let lastActionAt = 0;
let brainConnected = false;
let lastBrainHeartbeat = null;
let lastBrainError = null;
let resolvedChannelId = TARGET_CHANNEL_ID;

const ACTIONS = {
  forward: '🪰 forward',
  left: '🪰 left',
  right: '🪰 right',
  backward: '🪰 backward',
  escape: '🪰 escape',
  groom: '🪰 groom',
  explore: '🪰 explore',
  feed: '🪰 feed',
  idle: '🪰 idle'
};

function brainIsFresh() {
  if (!lastBrainHeartbeat) return false;
  return Date.now() - new Date(lastBrainHeartbeat).getTime() <= BRAIN_TIMEOUT_MS;
}

function pickAction(signals = {}) {
  const candidates = Object.entries(signals)
    .map(([name, value]) => [name, Number(value)])
    .filter(([name, value]) => ACTIONS[name] && Number.isFinite(value) && value > 0);

  if (!candidates.length) return null;
  candidates.sort((a, b) => b[1] - a[1]);

  const [name, score] = candidates[0];
  return score >= MIN_SIGNAL ? { name, score } : null;
}

async function resolveChannel() {
  if (!resolvedChannelId) throw new Error('TARGET_CHANNEL_ID is not configured');

  const channel = await client.channels.fetch(resolvedChannelId);
  if (!channel?.isTextBased?.()) {
    throw new Error(`Channel ${resolvedChannelId} is not a text channel`);
  }

  const permissions = channel.permissionsFor?.(client.user);
  if (permissions && !permissions.has(PermissionFlagsBits.ViewChannel)) {
    throw new Error(`Missing View Channel permission for ${resolvedChannelId}`);
  }
  if (permissions && !permissions.has(PermissionFlagsBits.SendMessages)) {
    throw new Error(`Missing Send Messages permission for ${resolvedChannelId}`);
  }

  return channel;
}

function timeoutSignal(ms) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), ms);
  return { controller, timer };
}

function cleanError(raw, status) {
  const text = String(raw || '').replace(/\s+/g, ' ').trim();
  if (!text) return `FlyBrain HTTP ${status}`;
  if (text.includes('502') || status === 502) return 'FlyBrain service is temporarily unavailable (Render 502).';
  if (text.includes('503') || status === 503) return 'FlyBrain service is temporarily unavailable (Render 503).';
  if (text.includes('504') || status === 504) return 'FlyBrain processing timed out.';
  if (text.includes('501') || status === 501) return 'FlyBrain endpoint does not support POST on the current deployment.';
  return text.slice(0, 800);
}

async function processThroughFlyBrain(text, author) {
  let lastError = null;

  for (let attempt = 1; attempt <= BRAIN_RETRIES; attempt += 1) {
    const { controller, timer } = timeoutSignal(BRAIN_PROCESS_TIMEOUT_MS);
    try {
      const response = await fetch(BRAIN_PROCESS_URL, {
        method: 'POST',
        signal: controller.signal,
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${BRAIN_WEBHOOK_SECRET}`
        },
        body: JSON.stringify({
          message: text,
          author: author?.username || 'unknown',
          timestamp: new Date().toISOString(),
          source: 'Discord realtime message'
        })
      });

      const raw = await response.text();
      let payload;
      try {
        payload = JSON.parse(raw || '{}');
      } catch {
        payload = { error: cleanError(raw, response.status) };
      }

      if (!response.ok) {
        throw new Error(cleanError(payload.error || raw, response.status));
      }

      return payload;
    } catch (error) {
      lastError = error;
      if (attempt < BRAIN_RETRIES) {
        await new Promise(resolve => setTimeout(resolve, 1500 * attempt));
      }
    } finally {
      clearTimeout(timer);
    }
  }

  throw lastError || new Error('FlyBrain processing failed');
}

async function sendFlyMessage(payload) {
  brainConnected = true;
  lastBrainHeartbeat = new Date().toISOString();
  lastBrainError = null;
  lastSignal = payload;

  const action = pickAction(payload?.signals);
  if (!action) {
    return { sent: false, reason: 'No mapped motor activity above threshold.' };
  }

  const now = Date.now();
  if (now - lastActionAt < COOLDOWN_MS) {
    return { sent: false, reason: 'Action cooldown active.', action };
  }

  const channel = await resolveChannel();
  const anatomy = payload?.anatomy ? `\n\`readout=${payload.anatomy}\`` : '';
  const source = payload?.source ? `\n\`source=${payload.source}\`` : '';
  const content = `${ACTIONS[action.name]}\n\`signal=${action.name}\` \`activity=${action.score.toFixed(4)}\`${anatomy}${source}`;
  const message = await channel.send(content);

  lastMessage = message.createdAt.toISOString();
  messagesSent += 1;
  lastActionAt = now;

  return {
    sent: true,
    action,
    messageId: message.id,
    channelId: channel.id
  };
}

function formatBrainReply(input, result) {
  const message = String(input || '').trim().slice(0, 500);
  const action = result?.action?.name || result?.signal?.name || 'none';
  const activity = Number(result?.action?.score ?? result?.activity ?? 0);
  const anatomy = result?.anatomy || 'neural population';
  const mean = Number(result?.neural?.meanFractionFiring ?? result?.meanActivity ?? 0);
  const peak = Number(result?.neural?.peakFractionFiring ?? result?.peakActivity ?? activity);

  const lines = [
    '🪰 **FlyBrain realtime processing**',
    `**Message:** ${message || '[empty]'}`,
    `**Neural result:** ${action === 'none' ? 'no mapped motor response above threshold' : ACTIONS[action] || action}`,
    `**Peak activity:** ${peak.toFixed(4)}`,
    `**Mean activity:** ${mean.toFixed(4)}`,
    `**Readout:** ${anatomy}`,
    '**Source:** FlyWire FAFB v783 + FlyBrain LIF'
  ];

  if (result?.stimulus) {
    lines.splice(3, 0, `**Stimulus:** ${result.stimulus}`);
  }

  return lines.join('\n');
}

async function safeReply(message, content) {
  const text = String(content || '🪰 FlyBrain returned no result.');
  if (text.length <= 1900) {
    return message.reply(text);
  }

  const first = text.slice(0, 1850);
  return message.reply(`${first}\n…`);
}

async function replyToMessage(message) {
  if (message.author?.bot) return;
  if (message.channelId !== TARGET_CHANNEL_ID) return;

  const text = message.content.trim();
  if (!text) return;

  lastMessage = new Date().toISOString();
  lastMessageText = text;

  let result;
  try {
    result = await processThroughFlyBrain(text, message.author);
    lastNeuralResult = result;
    brainConnected = true;
    lastBrainHeartbeat = new Date().toISOString();
    lastBrainError = null;
  } catch (error) {
    lastBrainError = error.message;
    console.error(`[FlyBrain] realtime processing failed: ${error.message}`);
    await safeReply(
      message,
      `🪰 **FlyBrain realtime processing**\n**Message:** ${text.slice(0, 500)}\n**Status:** processing failed\n**Error:** ${error.message}`
    );
    return;
  }

  const reply = await safeReply(message, formatBrainReply(text, result));
  messagesSent += 1;

  console.log(`[Discord] realtime message processed: ${message.author.tag || message.author.id} -> ${text.slice(0, 300)}`);
  console.log(`[FlyBrain] result: ${JSON.stringify(result)}`);
  console.log(`[Discord] reply message id: ${reply.id}`);
}

function readJson(req) {
  return new Promise((resolve, reject) => {
    let body = '';

    req.on('data', chunk => {
      body += chunk;
      if (body.length > 256000) {
        reject(new Error('Request body too large'));
        req.destroy();
      }
    });

    req.on('end', () => {
      try {
        resolve(JSON.parse(body || '{}'));
      } catch {
        reject(new Error('Invalid JSON'));
      }
    });

    req.on('error', reject);
  });
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, `http://${req.headers.host || 'localhost'}`);
  res.setHeader('Content-Type', 'application/json');

  if (req.method === 'GET' && url.pathname === '/health') {
    const connected = brainIsFresh();
    brainConnected = connected;

    res.writeHead(200);
    res.end(JSON.stringify({
      ok: true,
      discordReady: client.isReady(),
      brainConnected: connected,
      lastBrainHeartbeat,
      messagesSent,
      resolvedChannelId,
      lastSignal,
      lastMessage,
      lastMessageText,
      lastNeuralResult,
      lastBrainError
    }));
    return;
  }

  if (req.method === 'POST' && url.pathname === '/brain/heartbeat') {
    if (req.headers.authorization !== `Bearer ${BRAIN_WEBHOOK_SECRET}`) {
      res.writeHead(401);
      res.end(JSON.stringify({ error: 'Unauthorized' }));
      return;
    }

    brainConnected = true;
    lastBrainHeartbeat = new Date().toISOString();
    lastBrainError = null;

    res.writeHead(200);
    res.end(JSON.stringify({ ok: true, heartbeat: lastBrainHeartbeat }));
    return;
  }

  if (req.method === 'POST' && url.pathname === '/brain/input') {
    if (req.headers.authorization !== `Bearer ${BRAIN_WEBHOOK_SECRET}`) {
      res.writeHead(401);
      res.end(JSON.stringify({ error: 'Unauthorized' }));
      return;
    }

    try {
      const payload = await readJson(req);
      const result = await sendFlyMessage(payload);
      res.writeHead(200);
      res.end(JSON.stringify(result));
    } catch (error) {
      lastBrainError = error.message;
      console.error('[brain/input]', error);
      res.writeHead(400);
      res.end(JSON.stringify({ error: error.message }));
    }
    return;
  }

  res.writeHead(404);
  res.end(JSON.stringify({ error: 'Not found' }));
});

client.on('messageCreate', message => {
  void replyToMessage(message).catch(error => console.error('[messageCreate]', error));
});

client.once('ready', async () => {
  console.log(`FlyBot online as ${client.user.tag}`);
  console.log(`Brain gateway listening on :${PORT}`);
  console.log(`Target channel: ${TARGET_CHANNEL_ID}`);
  console.log(`Realtime FlyBrain endpoint: ${BRAIN_PROCESS_URL}`);

  try {
    const channel = await resolveChannel();
    console.log(`[Discord] realtime listener attached to #${channel.name} (${channel.id})`);
  } catch (error) {
    console.error(`[Discord] realtime listener setup failed: ${error.message}`);
  }

  console.log('Waiting for realtime Discord messages and FlyWire connectome activity...');
});

server.listen(PORT, '0.0.0.0');
client.login(DISCORD_TOKEN);
