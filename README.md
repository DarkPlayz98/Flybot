# FlyBot 🪰

A Discord bot controlled by signals from a simulated fruit-fly nervous system.

## Architecture

`FlyWire/connectome or other fly-brain simulator → neural/motor activity → /brain/input → FlyBot → Discord`

FlyBot is the bridge. The connectome simulation is intentionally kept separate so we can swap simulators without rewriting the Discord layer.

## Run

```bash
npm install
npm start
```

Required environment variables:

- `DISCORD_TOKEN`
- `TARGET_CHANNEL_ID`
- `BRAIN_WEBHOOK_SECRET`
- `PORT` (optional, defaults to `8080`)

## Brain gateway

POST JSON to `/brain/input` with:

```json
{
  "signals": {
    "forward": 0.8,
    "left": 0.1,
    "right": 0.0,
    "explore": 0.2
  }
}
```

Use `Authorization: Bearer <BRAIN_WEBHOOK_SECRET>`.

The highest positive mapped signal becomes the Discord message. This is an adapter layer, not a claim that the biological fly brain produces English.

## Status

Phase 1: Discord gateway complete.

Phase 2: connect a real public fly connectome/simulation output.

Phase 3: add richer sensory input and behavioral mapping.
