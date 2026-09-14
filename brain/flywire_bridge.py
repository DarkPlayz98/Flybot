#!/usr/bin/env python3
"""Live FlyWire FAFB connectome -> FlyBot bridge.

Runs the real FlyBrain simulator and exposes a tiny HTTP health endpoint so the
same process can run on a Render Web Service without a paid background worker.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/health"):
            body = b'{"status":"ok","service":"flybrain"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, format: str, *args) -> None:
        return


def start_health_server() -> None:
    port = int(env("PORT", "10000"))
    server = ThreadingHTTPServer(("0.0.0.0", port), HealthHandler)
    print(f"[FlyWire] health server listening on port {port}", flush=True)
    threading.Thread(target=server.serve_forever, daemon=True).start()


def post_json(url: str, secret: str, payload: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {secret}",
            "User-Agent": "FlyBot-FlyWire-Bridge/0.3",
        },
    )
    with urllib.request.urlopen(req, timeout=20) as response:
        raw = response.read().decode("utf-8", "replace")
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"raw": raw}


def load_flybrain():
    root = Path(env("FLYBRAIN_PATH", "./vendor/flybrain")).resolve()
    if not root.exists():
        raise RuntimeError(f"FLYBRAIN_PATH does not exist: {root}")

    sys.path.insert(0, str(root))
    from flybrain.circuits import Circuits
    from flybrain.loader import Connectome
    from flybrain.simulator import Simulator

    cache = Path(env("FLYBRAIN_CACHE", str(root / "outputs" / "connectome"))).resolve()
    if not cache.exists():
        raise RuntimeError(f"Connectome cache not found: {cache}")

    device = env("FLYBRAIN_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    connectome = Connectome.load(cache)
    circuits = Circuits(connectome)
    sugar = circuits.sugar_grns()
    proboscis = circuits.proboscis_motor_neurons()

    simulator = Simulator.from_connectome(
        connectome,
        device=device,
        weight_scale=float(env("WEIGHT_SCALE", "0.2")),
    )
    prob_idx = torch.as_tensor(proboscis.indices, dtype=torch.long, device=device)
    sugar_stim = simulator.input_vector(sugar.indices, float(env("STIM_AMP", "2.0")))

    print(
        f"[FlyWire] loaded {len(proboscis)} proboscis motor neurons on {device}",
        flush=True,
    )
    return simulator, prob_idx, sugar_stim, device


def main() -> None:
    start_health_server()

    base = env("FLYBOT_URL").rstrip("/")
    secret = env("BRAIN_WEBHOOK_SECRET")
    if not base or not secret:
        raise RuntimeError("FLYBOT_URL and BRAIN_WEBHOOK_SECRET are required")

    simulator, prob_idx, sugar_stim, device = load_flybrain()
    steps = max(1, int(env("SIM_STEPS", "40")))
    on_steps = min(steps, max(0, int(env("ON_STEPS", "25"))))
    pulse_ms = max(100, int(env("PULSE_MS", "250")))
    reconnect_ms = max(1000, int(env("RECONNECT_MS", "5000")))
    heartbeat_every = max(1, int(env("HEARTBEAT_EVERY", "1")))
    self_ping_every = max(60, int(env("SELF_PING_EVERY", "600")))

    heartbeat_url = f"{base}/brain/heartbeat"
    input_url = f"{base}/brain/input"
    health_url = env("SELF_PING_URL", "").rstrip("/") or None
    simulator.pop.reset()
    pulse = 0
    last_heartbeat = -1
    last_self_ping = 0.0

    while True:
        try:
            now = time.monotonic()

            # Keep a Render free Web Service warm when possible.
            if health_url and now - last_self_ping >= self_ping_every:
                try:
                    with urllib.request.urlopen(health_url + "/health", timeout=10) as response:
                        response.read()
                except Exception as exc:
                    print(f"[FlyWire] self-ping warning: {exc}", file=sys.stderr, flush=True)
                last_self_ping = now

            if pulse == 0 or pulse - last_heartbeat >= heartbeat_every:
                post_json(
                    heartbeat_url,
                    secret,
                    {
                        "source": "FlyWire FAFB v783 + FlyBrain LIF",
                        "pulse": pulse,
                        "device": device,
                    },
                )
                last_heartbeat = pulse

            rates = []
            for step in range(steps):
                spikes = simulator.step(sugar_stim if step < on_steps else None)
                if prob_idx.numel():
                    rates.append(float(spikes[prob_idx].float().mean().item()))
                else:
                    rates.append(0.0)

            peak = max(rates) if rates else 0.0
            mean = float(np.mean(rates)) if rates else 0.0
            signals = {"feed": peak} if peak > 0 else {}

            result = post_json(
                input_url,
                secret,
                {
                    "source": "FlyWire FAFB v783 + FlyBrain LIF",
                    "anatomy": "proboscis motor neurons",
                    "signals": signals,
                    "neural": {
                        "device": device,
                        "pulse": pulse,
                        "motorNeuronPool": int(prob_idx.numel()),
                        "peakFractionFiring": peak,
                        "meanFractionFiring": mean,
                        "steps": steps,
                        "stimulatedSteps": on_steps,
                    },
                },
            )

            print(
                f"[FlyWire] pulse={pulse} feed_peak={peak:.4f} mean={mean:.4f} response={result}",
                flush=True,
            )
            pulse += 1
            elapsed = time.monotonic() - now
            time.sleep(max(0.01, pulse_ms / 1000 - elapsed))

        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            print(f"[FlyWire] network error: {exc}", file=sys.stderr, flush=True)
            time.sleep(reconnect_ms / 1000)
        except KeyboardInterrupt:
            print("[FlyWire] stopped", flush=True)
            return


if __name__ == "__main__":
    main()
