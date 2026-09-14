import http from 'node:http';
import { Client, GatewayIntentBits } from 'discord.js';

const env = (key, fallback = '') => process.env[key] ?? fallback;
const DISCORD_TOKEN = env('DISCORD_TOKEN');
const TARGET_CHANNEL_ID = env('TARGET_CHANNEL_ID');
const BRAIN_WEBHOOK_SECRET = env('BRAIN_WEBHOOK_SECRET');
const PORT = Number(env('PORT', '8080'));
const MIN_SIGNAL = Number(env('MIN_SIGNAL', '0.05'));
const COOLDOWN_MS = Number(env('ACTION_COOLDOWN_MS', '3000'));

if (!DISCORD_TOKEN) throw new Error('DISCORD_TOKEN is required');
if (!TARGET_CHANNEL_ID) throw new Error('TARGET_CHANNEL_ID is required');
if (!BRAIN_WEBHOOK_SECRET) throw new Error('BRAIN_WEBHOOK_SECRET is required');

const client = new Client({ intents: [GatewayIntentBits.Guilds] });
let lastSignal = null;
let lastMessage = null;
let messagesSent = 0;
let lastActionAt = 0;
let brainConnected = false;
let lastBrainHeartbeat = null;

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

function pickAction(signals = {}) {
  const candidates = Object.entries(signals)
    .map(([name, value]) => [name, Number(value)])
    .filter(([name, value]) => ACTIONS[name] && Number.isFinite(value));

  if (!candidates.length) return null;
  candidates.sort((a, b) => b[1] - a[1]);

  const [name, score] = candidates[0];
  return score >= MIN_SIGNAL ? { name, score } : null;
}

async function sendFlyMessage(payload) {
  brainConnected = true;
  lastBrainHeartbeat = new Date().toISOString();
  lastSignal = payload;

  const action = pickAction(payload?.signals);
  if (!action) return { sent: false, reason: 'No mapped motor activity above threshold.' };

  const now = Date.now();
  if (now - lastActionAt < COOLDOWN_MS) {
    return { sent: false, reason: 'Action cooldown active.', action };
  }

  const channel = await client.channels.fetch(TARGET_CHANNEL_ID);
  if (!channel?.isTextBased()) throw new Error('TARGET_CHANNEL_ID is not a text channel');

  const anatomy = payload?.anatomy ? `\n\`readout=${payload.anatomy}\`` : '';
  const source = payload?.source ? `\n\`source=${payload.source}\`` : '';
  const content = `${ACTIONS[action.name]}\n\`signal=${action.name}\` \`activity=${action.score.toFixed(4)}\`${anatomy}${source}`;

  const message = await channel.send(content);
  lastMessage = message.createdAt.toISOString();
  messagesSent += 1;
  lastActionAt = now;
  return { sent: true, action, messageId: message.id };
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
    res.writeHead(200);
    res.end(JSON.stringify({
      ok: true,
      discordReady: client.isReady(),
      brainConnected,
      lastBrainHeartbeat,
      messagesSent,
      lastSignal,
      lastMessage
    }));
    return;
  }

  if (req.method === 'GET' && url.pathname === '/brain/status') {
    res.writeHead(200);
    res.end(JSON.stringify({
      connected: brainConnected,
      lastHeartbeat: lastBrainHeartbeat,
      minSignal: MIN_SIGNAL,
      cooldownMs: COOLDOWN_MS,
      lastSignal
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
  console.log('Waiting for FlyWire connectome activity...');
});

server.listen(PORT, '0.0.0.0');
client.login(DISCORD_TOKEN);
