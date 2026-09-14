#!/usr/bin/env python3
"""FlyWire connectome -> FlyBot bridge.

Runs the real step-wise FlyWire LIF simulator from an external flybrain
checkout and sends proboscis motor-neuron activity to FlyBot.

The current biological readout is intentionally narrow and honest:
sugar GRNs are stimulated and the connectome's proboscis motor-neuron pool
becomes the `feed` signal. Locomotion signals are not invented here.
"""
from __future__ import annotations
import json, os, sys, time, urllib.error, urllib.request
from pathlib import Path
import numpy as np
import torch

def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)

def post_json(url: str, secret: str, payload: dict) -> None:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST", headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {secret}",
        "User-Agent": "FlyBot-FlyWire-Bridge/0.1",
    })
    with urllib.request.urlopen(req, timeout=20) as response:
        response.read()

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
    simulator = Simulator.from_connectome(connectome, device=device, weight_scale=float(env("WEIGHT_SCALE", "0.2")))
    prob_idx = torch.as_tensor(proboscis.indices, dtype=torch.long, device=device)
    sugar_stim = simulator.input_vector(sugar.indices, float(env("STIM_AMP", "2.0")))
    print(f"[FlyWire] loaded {len(proboscis)} proboscis motor neurons on {device}", flush=True)
    return simulator, prob_idx, sugar_stim, device

def main() -> None:
    base = env("FLYBOT_URL").rstrip("/")
    secret = env("BRAIN_WEBHOOK_SECRET")
    if not base or not secret:
        raise RuntimeError("FLYBOT_URL and BRAIN_WEBHOOK_SECRET are required")
    simulator, prob_idx, sugar_stim, device = load_flybrain()
    step_ms = int(env("STEP_MS", "40"))
    pulse_ms = int(env("PULSE_MS", "250"))
    on_ms = int(env("ON_MS", "25"))
    heartbeat_url = f"{base}/brain/heartbeat"
    input_url = f"{base}/brain/input"
    simulator.pop.reset()
    pulse = 0
    while True:
        try:
            post_json(heartbeat_url, secret, {"source": "FlyWire LIF connectome"})
            rates = []
            for step in range(step_ms):
                spikes = simulator.step(sugar_stim if step < on_ms else None)
                rates.append(float(spikes[prob_idx].float().mean().item()) if prob_idx.numel() else 0.0)
            peak = max(rates) if rates else 0.0
            mean = float(np.mean(rates)) if rates else 0.0
            post_json(input_url, secret, {
                "source": "FlyWire FAFB connectome + LIF",
                "anatomy": "proboscis motor neurons",
                "signals": {"feed": peak, "idle": max(0.0, 1.0 - peak)},
                "neural": {
                    "device": device,
                    "pulse": pulse,
                    "motorNeuronPool": int(prob_idx.numel()),
                    "peakFractionFiring": peak,
                    "meanFractionFiring": mean,
                },
            })
            print(f"[FlyWire] pulse={pulse} feed_peak={peak:.4f} mean={mean:.4f}", flush=True)
            pulse += 1
            time.sleep(max(0.01, pulse_ms / 1000))
        except (urllib.error.URLError, TimeoutError) as exc:
            print(f"[FlyWire] network error: {exc}", file=sys.stderr, flush=True)
            time.sleep(5)
        except KeyboardInterrupt:
            return

if __name__ == "__main__":
    main()
