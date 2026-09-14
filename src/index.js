import http from 'node:http';
import { Client, GatewayIntentBits, PermissionFlagsBits } from 'discord.js';

const env = (key, fallback = '') => process.env[key] ?? fallback;
const DISCORD_TOKEN = env('DISCORD_TOKEN');
const TARGET_CHANNEL_ID = env('TARGET_CHANNEL_ID');
const BRAIN_WEBHOOK_SECRET = env('BRAIN_WEBHOOK_SECRET');
const PORT = Number(env('PORT', '8080'));
const MIN_SIGNAL = Number(env('MIN_SIGNAL', '0.05'));
const COOLDOWN_MS = Number(env('ACTION_COOLDOWN_MS', '3000'));
const BRAIN_TIMEOUT_MS = Number(env('BRAIN_TIMEOUT_MS', '30000'));
const AUTO_DISCOVER_CHANNEL = env('AUTO_DISCOVER_CHANNEL', 'true').toLowerCase() !== 'false';

if (!DISCORD_TOKEN) throw new Error('DISCORD_TOKEN is required');
if (!BRAIN_WEBHOOK_SECRET) throw new Error('BRAIN_WEBHOOK_SECRET is required');

const client = new Client({ intents: [GatewayIntentBits.Guilds] });
let lastSignal = null;
let lastMessage = null;
let messagesSent = 0;
let lastActionAt = 0;
let brainConnected = false;
let lastBrainHeartbeat = null;
let lastBrainError = null;
let resolvedChannelId = TARGET_CHANNEL_ID || null;

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
  if (resolvedChannelId) {
    try {
      const channel = await client.channels.fetch(resolvedChannelId);
      if (channel?.isTextBased()) {
        const permissions = channel.permissionsFor?.(client.user);
        if (!permissions || permissions.has(PermissionFlagsBits.SendMessages)) return channel;
      }
    } catch (error) {
      console.warn(`[Discord] configured channel unavailable: ${error.message}`);
    }
  }

  if (!AUTO_DISCOVER_CHANNEL) {
    throw new Error('TARGET_CHANNEL_ID is unavailable or not writable');
  }

  for (const guild of client.guilds.cache.values()) {
    try {
      const channels = await guild.channels.fetch();
      const candidates = [...channels.values()]
        .filter(channel => channel?.isTextBased?.())
        .filter(channel => {
          const permissions = channel.permissionsFor?.(client.user);
          return !permissions || permissions.has(PermissionFlagsBits.SendMessages);
        })
        .sort((a, b) => {
          const aName = String(a.name || '').toLowerCase();
          const bName = String(b.name || '').toLowerCase();
          const score = name => {
            if (name.includes('flybot')) return 0;
            if (name.includes('bot')) return 1;
            if (name.includes('ai')) return 2;
            if (name.includes('chat')) return 3;
            return 4;
          };
          return score(aName) - score(bName);
        });

      if (candidates.length) {
        resolvedChannelId = candidates[0].id;
        console.log(`[Discord] auto-selected #${candidates[0].name} (${resolvedChannelId}) in ${guild.name}`);
        return candidates[0];
      }
    } catch (error) {
      console.warn(`[Discord] could not inspect ${guild.name}: ${error.message}`);
    }
  }

  throw new Error('No writable Discord text channel was found');
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
      brainTimeoutMs: BRAIN_TIMEOUT_MS,
      messagesSent,
      resolvedChannelId,
      lastSignal,
      lastMessage,
      lastBrainError
    }));
    return;
  }

  if (req.method === 'GET' && url.pathname === '/brain/status') {
    const connected = brainIsFresh();
    brainConnected = connected;
    res.writeHead(200);
    res.end(JSON.stringify({
      connected,
      lastHeartbeat: lastBrainHeartbeat,
      minSignal: MIN_SIGNAL,
      cooldownMs: COOLDOWN_MS,
      timeoutMs: BRAIN_TIMEOUT_MS,
      resolvedChannelId,
      lastSignal,
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

client.once('ready', () => {
  console.log(`FlyBot online as ${client.user.tag}`);
  console.log(`Brain gateway listening on :${PORT}`);
  console.log(`Target channel: ${TARGET_CHANNEL_ID || 'auto-discovery enabled'}`);
  console.log('Waiting for FlyWire connectome activity...');
});

server.listen(PORT, '0.0.0.0');
client.login(DISCORD_TOKEN);
