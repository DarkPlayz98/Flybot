#!/usr/bin/env python3
"""Realtime FlyWire FAFB v783 -> lightweight LIF bridge.

Uses the real cached FlyWire FAFB v783 sparse connectome and the same normalized
leaky integrate-and-fire equations used by FlyBrain, but executes the sparse
matrix/vector math with SciPy + NumPy. This avoids PyTorch's extra memory peaks
and makes the full real connectome practical on Render's 512 MiB free instance.
"""
from __future__ import annotations

import gc
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
import pandas as pd
import scipy.sparse as sp

# Reduce native thread memory on tiny Render instances.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


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
WEIGHTS: sp.csr_matrix | None = None
N_NEURONS = 0
SUGAR_IDX: np.ndarray | None = None
PROB_IDX: np.ndarray | None = None
SUGAR_STIM: np.ndarray | None = None
STEPS = 40
ON_STEPS = 25
READY = False
LOAD_ERROR = None
LAST_MESSAGE = None


# Same normalized LIF defaults as FlyBrain's flybrain/neurons.py.
TAU_MS = 20.0
DT_MS = 1.0
V_REST = 0.0
V_THRESHOLD = 1.0
V_RESET = 0.0
REFRACTORY_STEPS = 2
DECAY = DT_MS / TAU_MS


class LIFPopulation:
    """Memory-light NumPy implementation of the same FlyBrain LIF dynamics."""

    def __init__(self, n: int):
        self.n = int(n)
        self.v = np.zeros(self.n, dtype=np.float32)
        self.refractory = np.zeros(self.n, dtype=np.int16)
        self.spikes = np.zeros(self.n, dtype=np.float32)

    def reset(self) -> None:
        self.v.fill(V_REST)
        self.refractory.fill(0)
        self.spikes.fill(0)

    def step(self, current: np.ndarray) -> np.ndarray:
        not_ref = self.refractory == 0

        self.v[not_ref] += DECAY * (-(self.v[not_ref] - V_REST) + current[not_ref])

        spike_mask = not_ref & (self.v >= V_THRESHOLD)
        self.v[spike_mask] = V_RESET

        self.refractory[:] = np.maximum(self.refractory.astype(np.int16) - 1, 0)
        self.refractory[spike_mask] = REFRACTORY_STEPS

        self.spikes.fill(0.0)
        self.spikes[spike_mask] = 1.0
        return self.spikes


POPULATION: LIFPopulation | None = None


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
            self._json_response(200 if READY else 503, {
                "status": "ok" if READY else "loading",
                "service": "flybrain",
                "ready": READY,
                "engine": "SciPy sparse + NumPy LIF",
                "source": "FlyWire FAFB v783",
                "neurons": N_NEURONS,
                "proboscisMotorNeurons": int(PROB_IDX.size) if PROB_IDX is not None else 0,
                "sugarInputNeurons": int(SUGAR_IDX.size) if SUGAR_IDX is not None else 0,
                "queuedMessages": len(JOBS),
                "lastMessage": LAST_MESSAGE,
                "loadError": LOAD_ERROR,
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

        if not READY:
            self._json_response(503, {
                "error": "FlyBrain is still loading",
                "loadError": LOAD_ERROR,
            })
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


def start_http_server() -> None:
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
            "User-Agent": "FlyBot-FlyWire-Bridge/0.7",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        raw = response.read().decode("utf-8", "replace")
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"raw": raw}


def build_runtime(root: Path) -> None:
    """Load the real sparse matrix once, sign it in-place, then run NumPy LIF."""
    global WEIGHTS, N_NEURONS, SUGAR_IDX, PROB_IDX, SUGAR_STIM
    global STEPS, ON_STEPS, READY, LOAD_ERROR, POPULATION

    try:
        cache = Path(env("FLYBRAIN_CACHE", str(root / "outputs" / "connectome"))).resolve()
        matrix_path = cache / "syn_counts.npz"
        nodes_path = cache / "nodes.parquet"

        if not matrix_path.exists() or not nodes_path.exists():
            raise RuntimeError(f"Connectome cache incomplete: {cache}")

        print(f"[FlyWire] loading real FAFB v783 sparse cache", flush=True)
        weights = sp.load_npz(matrix_path).tocsr()
        N_NEURONS = int(weights.shape[0])
        print(
            f"[FlyWire] sparse matrix: {N_NEURONS:,} neurons / {weights.nnz:,} stored edges",
            flush=True,
        )

        nodes = pd.read_parquet(
            nodes_path,
            columns=["nt_sign", "class", "sub_class", "super_class"],
        )
        if len(nodes) != N_NEURONS:
            raise RuntimeError(
                f"Node/matrix size mismatch: nodes={len(nodes)} matrix={N_NEURONS}"
            )

        nt_sign = nodes["nt_sign"].fillna(0).to_numpy(dtype=np.int8, copy=False)
        sugar_mask = (
            nodes["class"].astype("string").eq("gustatory")
            & nodes["sub_class"].astype("string").eq("sugar/water")
        ).fillna(False).to_numpy(dtype=bool)
        prob_mask = (
            nodes["super_class"].astype("string").eq("motor")
            & nodes["sub_class"].astype("string").eq("proboscis_motor_neuron")
        ).fillna(False).to_numpy(dtype=bool)

        SUGAR_IDX = np.flatnonzero(sugar_mask).astype(np.int64, copy=False)
        PROB_IDX = np.flatnonzero(prob_mask).astype(np.int64, copy=False)

        if SUGAR_IDX.size == 0 or PROB_IDX.size == 0:
            raise RuntimeError(
                f"Required circuits missing: sugar={SUGAR_IDX.size}, proboscis={PROB_IDX.size}"
            )

        print(
            f"[FlyWire] circuit lookup: {SUGAR_IDX.size} sugar/water GRNs, "
            f"{PROB_IDX.size} proboscis motor neurons",
            flush=True,
        )

        # Keep the matrix sparse and avoid a second signed copy.
        if weights.data.dtype != np.float32:
            weights.data = weights.data.astype(np.float32, copy=True)
        np.multiply(
            weights.data,
            nt_sign[weights.indices],
            out=weights.data,
            casting="unsafe",
        )
        weights.data *= float(env("WEIGHT_SCALE", "0.2"))

        del nodes, nt_sign, sugar_mask, prob_mask
        gc.collect()

        WEIGHTS = weights
        POPULATION = LIFPopulation(N_NEURONS)
        SUGAR_STIM = np.zeros(N_NEURONS, dtype=np.float32)
        SUGAR_STIM[SUGAR_IDX] = float(env("STIM_AMP", "2.0"))

        STEPS = max(1, int(env("SIM_STEPS", "30")))
        ON_STEPS = min(STEPS, max(1, int(env("ON_STEPS", "20"))))

        READY = True
        LOAD_ERROR = None
        print(
            f"[FlyWire] REAL runtime ready: {N_NEURONS:,} neurons, "
            f"{WEIGHTS.nnz:,} signed edges, SciPy sparse + NumPy LIF",
            flush=True,
        )
    except Exception as exc:
        LOAD_ERROR = str(exc)
        READY = False
        print(f"[FlyWire] runtime load error: {exc}", file=sys.stderr, flush=True)
        raise


def run_simulation(stimulus: np.ndarray) -> tuple[float, float, list[float]]:
    if WEIGHTS is None or POPULATION is None or PROB_IDX is None:
        raise RuntimeError("FlyBrain runtime is not loaded")

    POPULATION.reset()
    rates: list[float] = []

    for step in range(STEPS):
        # The actual connectome computation: I[t] = W @ spikes[t-1].
        current = WEIGHTS.dot(POPULATION.spikes)
        if step < ON_STEPS:
            current += stimulus
        spikes = POPULATION.step(current)
        rates.append(float(spikes[PROB_IDX].mean()) if PROB_IDX.size else 0.0)

    peak = max(rates) if rates else 0.0
    mean = float(np.mean(rates)) if rates else 0.0
    return peak, mean, rates


def message_stimulus(text: str) -> tuple[np.ndarray, str, float]:
    """Map the actual Discord message into a deterministic sensory stimulus."""
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
    global LAST_MESSAGE
    try:
        stimulus, stimulus_name, factor = message_stimulus(job.text)
        peak, mean, rates = run_simulation(stimulus)
        LAST_MESSAGE = job.text

        job.result = {
            "ok": True,
            "status": "processed",
            "message": job.text,
            "source": "FlyWire FAFB v783 + FlyBrain LIF equations",
            "engine": "SciPy sparse + NumPy LIF",
            "anatomy": "proboscis motor neurons",
            "stimulus": stimulus_name,
            "stimulusFactor": factor,
            "action": {"name": "feed", "score": peak} if peak > 0 else None,
            "peakActivity": peak,
            "meanActivity": mean,
            "neural": {
                "neurons": N_NEURONS,
                "storedEdges": int(WEIGHTS.nnz),
                "motorNeuronPool": int(PROB_IDX.size),
                "sugarInputPool": int(SUGAR_IDX.size),
                "peakFractionFiring": peak,
                "meanFractionFiring": mean,
                "steps": STEPS,
                "stimulatedSteps": ON_STEPS,
                "tauMs": TAU_MS,
                "dtMs": DT_MS,
                "refractorySteps": REFRACTORY_STEPS,
            },
            "sampledRates": rates[:12],
        }

        print(
            f"[FlyWire] MESSAGE {job.text!r} -> stimulus={factor:.3f} "
            f"peak={peak:.4f} mean={mean:.4f}",
            flush=True,
        )
    except Exception as exc:
        job.error = str(exc)
        print(f"[FlyWire] message processing error: {exc}", file=sys.stderr, flush=True)
    finally:
        if job.event:
            job.event.set()


def worker_loop() -> None:
    base = env("FLYBOT_URL").rstrip("/")
    secret = env("BRAIN_WEBHOOK_SECRET")
    heartbeat_interval = max(10.0, float(env("HEARTBEAT_INTERVAL", "30")))
    next_heartbeat = 0.0

    while True:
        try:
            with JOBS_LOCK:
                job = JOBS.popleft() if JOBS else None

            if job is not None:
                process_message_job(job)
                continue

            now = time.monotonic()
            if base and secret and now >= next_heartbeat:
                try:
                    post_json(
                        f"{base}/brain/heartbeat",
                        secret,
                        {"source": "FlyWire FAFB v783 + FlyBrain LIF", "engine": "SciPy sparse + NumPy"},
                    )
                except Exception as exc:
                    print(f"[FlyWire] heartbeat warning: {exc}", file=sys.stderr, flush=True)
                next_heartbeat = now + heartbeat_interval

            time.sleep(0.05)
        except Exception as exc:
            print(f"[FlyWire] worker error: {exc}", file=sys.stderr, flush=True)
            time.sleep(1.0)


def main() -> None:
    start_http_server()

    root = Path(env("FLYBRAIN_PATH", "./vendor/flybrain")).resolve()
    if not root.exists():
        raise RuntimeError(f"FLYBRAIN_PATH does not exist: {root}")

    print("[FlyWire] starting memory-safe REAL FAFB v783 runtime", flush=True)
    build_runtime(root)

    threading.Thread(target=worker_loop, daemon=True).start()
    print("[FlyWire] realtime Discord-message processing is ENABLED", flush=True)

    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
