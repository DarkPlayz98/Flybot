#!/usr/bin/env python3
"""Live FlyWire FAFB connectome -> FlyBot bridge.

Runs the real FlyBrain simulator and exposes HTTP endpoints for health,
connectome motor input, and realtime Discord-message processing.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


@dataclass
class MessageJob:
    text: str
    result: dict | None = None
    error: str | None = None
    event: threading.Event | None = None


JOBS: deque[MessageJob] = deque()
JOBS_LOCK = threading.Lock()
SIMULATOR = None
DEVICE = "cpu"
PROB_IDX = None
SUGAR_STIM = None
STEPS = 40
ON_STEPS = 25


class HealthHandler(BaseHTTPRequestHandler):
    def _authorized(self) -> bool:
        secret = env("BRAIN_WEBHOOK_SECRET")
        return bool(secret) and self.headers.get("Authorization") == f"Bearer {secret}"

    def _json_response(self, status: int, body: dict) -> None:
        raw = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path in ("/", "/health"):
            self._json_response(200, {
                "status": "ok",
                "service": "flybrain",
                "device": DEVICE,
                "motorNeuronPool": int(PROB_IDX.numel()) if PROB_IDX is not None else 0,
                "queuedMessages": len(JOBS),
            })
            return
        self._json_response(404, {"error": "Not found"})

    def do_POST(self):
        if self.path not in ("/brain/process", "/stimulus"):
            self._json_response(404, {"error": "Not found"})
            return

        if not self._authorized():
            self._json_response(401, {"error": "Unauthorized"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 256000:
                raise ValueError("Request body too large")
            body = self.rfile.read(length).decode("utf-8", "replace")
            payload = json.loads(body or "{}")
            text = str(payload.get("message", payload.get("text", ""))).strip()
            if not text:
                raise ValueError("message is required")

            job = MessageJob(text=text, event=threading.Event())
            with JOBS_LOCK:
                JOBS.append(job)

            timeout = max(1.0, float(env("MESSAGE_PROCESS_TIMEOUT", "45")))
            if not job.event.wait(timeout):
                self._json_response(504, {"error": "FlyBrain processing timeout"})
                return

            if job.error:
                self._json_response(500, {"error": job.error})
                return

            self._json_response(200, job.result or {"error": "No result"})
        except json.JSONDecodeError:
            self._json_response(400, {"error": "Invalid JSON"})
        except Exception as exc:
            self._json_response(400, {"error": str(exc)})

    def log_message(self, format: str, *args) -> None:
        return


def start_health_server() -> None:
    port = int(env("PORT", "10000"))
    server = ThreadingHTTPServer(("0.0.0.0", port), HealthHandler)
    print(f"[FlyWire] HTTP/API server listening on port {port}", flush=True)
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
            "User-Agent": "FlyBot-FlyWire-Bridge/0.5",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        raw = response.read().decode("utf-8", "replace")
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"raw": raw}


def load_flybrain():
    global SIMULATOR, DEVICE, PROB_IDX, SUGAR_STIM, STEPS, ON_STEPS

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

    DEVICE = env("FLYBRAIN_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    connectome = Connectome.load(cache)
    circuits = Circuits(connectome)
    sugar = circuits.sugar_grns()
    proboscis = circuits.proboscis_motor_neurons()

    SIMULATOR = Simulator.from_connectome(
        connectome,
        device=DEVICE,
        weight_scale=float(env("WEIGHT_SCALE", "0.2")),
    )
    PROB_IDX = torch.as_tensor(proboscis.indices, dtype=torch.long, device=DEVICE)
    SUGAR_STIM = SIMULATOR.input_vector(sugar.indices, float(env("STIM_AMP", "2.0")))

    STEPS = max(1, int(env("SIM_STEPS", "40")))
    ON_STEPS = min(STEPS, max(1, int(env("ON_STEPS", "25"))))

    print(f"[FlyWire] loaded {len(proboscis)} proboscis motor neurons on {DEVICE}", flush=True)


def run_simulation(stimulus: torch.Tensor | None) -> tuple[float, float, list[float]]:
    if SIMULATOR is None or PROB_IDX is None:
        raise RuntimeError("FlyBrain simulator is not loaded")

    SIMULATOR.pop.reset()
    rates: list[float] = []
    for step in range(STEPS):
        spikes = SIMULATOR.step(stimulus if step < ON_STEPS else None)
        if PROB_IDX.numel():
            rates.append(float(spikes[PROB_IDX].float().mean().item()))
        else:
            rates.append(0.0)

    peak = max(rates) if rates else 0.0
    mean = float(np.mean(rates)) if rates else 0.0
    return peak, mean, rates


def message_stimulus(text: str) -> tuple[torch.Tensor, str, float]:
    """Create a deterministic sensory stimulus from the actual Discord text."""
    normalized = " ".join(text.split())
    if not normalized:
        raise ValueError("message is empty")

    lower = normalized.lower()
    unique = len(set(normalized))
    length_factor = min(len(normalized), 240) / 240.0
    diversity_factor = min(unique, 80) / 80.0
    punctuation_factor = min(sum(ch in "!?.,:;" for ch in normalized), 12) / 12.0
    vowel_factor = min(sum(ch in "aeiou" for ch in lower), 40) / 40.0
    digit_factor = min(sum(ch.isdigit() for ch in normalized), 10) / 10.0

    # Every non-empty message gets a stimulus. This prevents arbitrary text such
    # as "Ay bro?" from becoming a zero-input event while keeping the actual
    # simulator responsible for the neural result.
    factor = (
        0.20
        + 0.30 * length_factor
        + 0.20 * diversity_factor
        + 0.10 * punctuation_factor
        + 0.15 * vowel_factor
        + 0.05 * digit_factor
    )

    if any(word in lower.split() for word in ("food", "feed", "sugar", "eat", "hungry")):
        factor += 0.15

    factor = float(max(0.20, min(1.0, factor)))
    stimulus = SUGAR_STIM * factor
    description = f"message-derived sensory stimulus; factor={factor:.3f}"
    return stimulus, description, factor


def process_message_job(job: MessageJob) -> None:
    try:
        stimulus, stimulus_name, factor = message_stimulus(job.text)
        peak, mean, rates = run_simulation(stimulus)

        job.result = {
            "ok": True,
            "message": job.text,
            "source": "FlyWire FAFB v783 + FlyBrain LIF",
            "anatomy": "proboscis motor neurons",
            "status": "processed",
            "stimulus": stimulus_name,
            "stimulusFactor": factor,
            "action": {"name": "feed", "score": peak} if peak > 0 else None,
            "peakActivity": peak,
            "meanActivity": mean,
            "neural": {
                "device": DEVICE,
                "motorNeuronPool": int(PROB_IDX.numel()),
                "peakFractionFiring": peak,
                "meanFractionFiring": mean,
                "steps": STEPS,
                "stimulatedSteps": ON_STEPS,
            },
            "sampledRates": rates[:12],
        }

        print(
            f"[FlyWire] MESSAGE {job.text!r} -> factor={factor:.3f} "
            f"peak={peak:.4f} mean={mean:.4f}",
            flush=True,
        )
    except Exception as exc:
        job.error = str(exc)
        print(f"[FlyWire] message processing error: {exc}", file=sys.stderr, flush=True)
    finally:
        if job.event:
            job.event.set()


def run_legacy_connectome_loop(base: str, secret: str) -> None:
    pulse_ms = max(250, int(env("PULSE_MS", "5000")))
    reconnect_ms = max(1000, int(env("RECONNECT_MS", "5000")))
    heartbeat_every = max(1, int(env("HEARTBEAT_EVERY", "1")))
    pulse = 0
    last_heartbeat = -1

    while True:
        try:
            # Realtime Discord jobs always have priority. The autonomous loop is
            # only retained as a background heartbeat/demo and no longer races
            # each user message through the same endpoint.
            with JOBS_LOCK:
                job = JOBS.popleft() if JOBS else None
            if job is not None:
                process_message_job(job)
                continue

            if pulse == 0 or pulse - last_heartbeat >= heartbeat_every:
                post_json(
                    f"{base}/brain/heartbeat",
                    secret,
                    {
                        "source": "FlyWire FAFB v783 + FlyBrain LIF",
                        "pulse": pulse,
                        "device": DEVICE,
                    },
                )
                last_heartbeat = pulse

            time.sleep(pulse_ms / 1000)
            pulse += 1
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            print(f"[FlyWire] network error: {exc}", file=sys.stderr, flush=True)
            time.sleep(reconnect_ms / 1000)
        except KeyboardInterrupt:
            print("[FlyWire] stopped", flush=True)
            return
        except Exception as exc:
            print(f"[FlyWire] loop error: {exc}", file=sys.stderr, flush=True)
            time.sleep(reconnect_ms / 1000)


def main() -> None:
    start_health_server()

    base = env("FLYBOT_URL").rstrip("/")
    secret = env("BRAIN_WEBHOOK_SECRET")
    if not base or not secret:
        raise RuntimeError("FLYBOT_URL and BRAIN_WEBHOOK_SECRET are required")

    load_flybrain()
    print("[FlyWire] realtime Discord-message processing is enabled", flush=True)
    run_legacy_connectome_loop(base, secret)


if __name__ == "__main__":
    main()
