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

if (!DISCORD_TOKEN) throw new Error('DISCORD_TOKEN is required');
if (!BRAIN_WEBHOOK_SECRET) throw new Error('BRAIN_WEBHOOK_SECRET is required');

// MessageContent is intentionally NOT requested: it is a privileged Discord intent
// and is not needed to detect a message, identify its author, or reply to it.
const client = new Client({
  intents: [GatewayIntentBits.Guilds, GatewayIntentBits.GuildMessages]
});

let lastSignal = null;
let lastMessage = null;
let messagesSent = 0;
let lastActionAt = 0;
let brainConnected = false;
let lastBrainHeartbeat = null;
let lastBrainError = null;
let resolvedChannelId = TARGET_CHANNEL_ID;

const ACTIONS = {
  forward: '🪰 I am moving forward.',
  left: '🪰 I turned left.',
  right: '🪰 I turned right.',
  backward: '🪰 I moved backward.',
  escape: '🪰 Escape response detected!',
  groom: '🪰 Grooming response detected.',
  explore: '🪰 I am exploring.',
  feed: '🪰 Feeding response detected.',
  idle: '🪰 Neural activity returned to baseline.'
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
  if (!channel?.isTextBased?.()) throw new Error(`Channel ${resolvedChannelId} is not a text channel`);
  const permissions = channel.permissionsFor?.(client.user);
  if (permissions && !permissions.has(PermissionFlagsBits.ViewChannel)) {
    throw new Error(`Missing View Channel permission for ${resolvedChannelId}`);
  }
  if (permissions && !permissions.has(PermissionFlagsBits.SendMessages)) {
    throw new Error(`Missing Send Messages permission for ${resolvedChannelId}`);
  }
  return channel;
}

async function sendFlyMessage(payload) {
  brainConnected = true;
  lastBrainHeartbeat = new Date().toISOString();
  lastBrainError = null;
  lastSignal = payload;

  const action = pickAction(payload?.signals);
  if (!action) return { sent: false, reason: 'No mapped motor activity above threshold.' };

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
  return { sent: true, action, messageId: message.id, channelId: channel.id };
}

async function replyToMessage(message) {
  if (message.author?.bot) return;
  if (message.channelId !== TARGET_CHANNEL_ID) return;

  // We do not need message.content. GuildMessages gives us the message event,
  // author and channel, allowing a direct reply without the privileged intent.
  const reply = '🪰 Message received. FlyBrain is listening.';
  await message.reply(reply);
  lastMessage = new Date().toISOString();
  messagesSent += 1;
  console.log(`[Discord] replied to ${message.author.tag || message.author.id} in ${TARGET_CHANNEL_ID}`);
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
      try { resolve(JSON.parse(body || '{}')); }
      catch { reject(new Error('Invalid JSON')); }
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
      lastBrainError
    }));
    return;
  }

  if (req.method === 'POST' && url.pathname === '/brain/heartbeat') {
    if (req.headers.authorization !== `Bearer ${BRAIN_WEBHOOK_SECRET}`) {
      res.writeHead(401); res.end(JSON.stringify({ error: 'Unauthorized' })); return;
    }
    brainConnected = true;
    lastBrainHeartbeat = new Date().toISOString();
    lastBrainError = null;
    res.writeHead(200); res.end(JSON.stringify({ ok: true, heartbeat: lastBrainHeartbeat }));
    return;
  }

  if (req.method === 'POST' && url.pathname === '/brain/input') {
    if (req.headers.authorization !== `Bearer ${BRAIN_WEBHOOK_SECRET}`) {
      res.writeHead(401); res.end(JSON.stringify({ error: 'Unauthorized' })); return;
    }
    try {
      const payload = await readJson(req);
      const result = await sendFlyMessage(payload);
      res.writeHead(200); res.end(JSON.stringify(result));
    } catch (error) {
      lastBrainError = error.message;
      console.error('[brain/input]', error);
      res.writeHead(400); res.end(JSON.stringify({ error: error.message }));
    }
    return;
  }

  res.writeHead(404); res.end(JSON.stringify({ error: 'Not found' }));
});

client.on('messageCreate', message => {
  void replyToMessage(message).catch(error => console.error('[messageCreate]', error));
});

client.once('ready', async () => {
  console.log(`FlyBot online as ${client.user.tag}`);
  console.log(`Brain gateway listening on :${PORT}`);
  console.log(`Target channel: ${TARGET_CHANNEL_ID}`);
  try {
    const channel = await resolveChannel();
    const testMessage = await channel.send('🪰 FlyBot test message — online and listening for messages.');
    lastMessage = testMessage.createdAt.toISOString();
    messagesSent += 1;
    console.log(`[Discord] test message sent to #${channel.name} (${channel.id})`);
  } catch (error) {
    console.error(`[Discord] test message failed: ${error.message}`);
  }
  console.log('Waiting for FlyWire connectome activity...');
});

server.listen(PORT, '0.0.0.0');
client.login(DISCORD_TOKEN);
