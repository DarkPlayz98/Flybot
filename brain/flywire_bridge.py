#!/usr/bin/env python3
"""Realtime FlyWire FAFB v783 -> FlyBrain LIF bridge.

The service uses the real FlyWire-derived sparse connectome cache, but builds
its Torch CSR runtime without creating an additional signed scipy matrix. This
keeps the free Render instance below its 512 MiB memory ceiling.
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
import torch

# Keep BLAS/OpenMP thread overhead small on Render's tiny instance.
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
SIMULATOR = None
DEVICE = "cpu"
PROB_IDX = None
SUGAR_IDX = None
STEPS = 40
ON_STEPS = 25
READY = False
LOAD_ERROR = None
LAST_MESSAGE = None


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
                "device": DEVICE,
                "motorNeuronPool": int(PROB_IDX.numel()) if PROB_IDX is not None else 0,
                "sugarInputPool": int(SUGAR_IDX.numel()) if SUGAR_IDX is not None else 0,
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
            self._json_response(503, {"error": "FlyBrain is still loading", "loadError": LOAD_ERROR})
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
            "User-Agent": "FlyBot-FlyWire-Bridge/0.6",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        raw = response.read().decode("utf-8", "replace")
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"raw": raw}


def build_runtime(root: Path):
    """Load only the metadata needed for the named circuits and reuse the
    cached sparse CSR arrays directly as Torch buffers.

    The old implementation created:
      1. unsigned scipy CSR
      2. a second signed scipy CSR
      3. a Torch CSR copy

    On a 512 MiB instance that startup peak was enough to cause Render 502s.
    Here we mutate the cached CSR data to signed float32 in-place, build Torch
    tensors from those same NumPy buffers, and immediately discard metadata.
    """
    global SIMULATOR, DEVICE, PROB_IDX, SUGAR_IDX, STEPS, ON_STEPS, READY, LOAD_ERROR

    try:
        cache = Path(env("FLYBRAIN_CACHE", str(root / "outputs" / "connectome"))).resolve()
        matrix_path = cache / "syn_counts.npz"
        nodes_path = cache / "nodes.parquet"

        if not matrix_path.exists() or not nodes_path.exists():
            raise RuntimeError(f"Connectome cache incomplete: {cache}")

        print(f"[FlyWire] loading sparse cache: {matrix_path}", flush=True)
        syn_counts = sp.load_npz(matrix_path).tocsr()
        print(
            f"[FlyWire] sparse matrix loaded: shape={syn_counts.shape} nnz={syn_counts.nnz:,}",
            flush=True,
        )

        # Read only columns needed to identify input/output populations and signs.
        nodes = pd.read_parquet(
            nodes_path,
            columns=["root_id", "nt_sign", "class", "sub_class", "super_class"],
        )
        if len(nodes) != syn_counts.shape[0]:
            raise RuntimeError(
                f"Node/matrix size mismatch: nodes={len(nodes)} matrix={syn_counts.shape[0]}"
            )

        node_root = nodes["root_id"].to_numpy(dtype=np.int64, copy=False)
        nt_sign = nodes["nt_sign"].fillna(0).to_numpy(dtype=np.int8, copy=False)

        sugar_mask = (
            nodes["class"].astype("string").eq("gustatory")
            & nodes["sub_class"].astype("string").eq("sugar/water")
        ).fillna(False).to_numpy(dtype=bool)
        prob_mask = (
            nodes["super_class"].astype("string").eq("motor")
            & nodes["sub_class"].astype("string").eq("proboscis_motor_neuron")
        ).fillna(False).to_numpy(dtype=bool)

        sugar_indices = np.flatnonzero(sugar_mask).astype(np.int64, copy=False)
        prob_indices = np.flatnonzero(prob_mask).astype(np.int64, copy=False)

        if sugar_indices.size == 0:
            raise RuntimeError("No sugar/water gustatory receptor neurons found in cache")
        if prob_indices.size == 0:
            raise RuntimeError("No proboscis motor neurons found in cache")

        print(
            f"[FlyWire] circuits: sugar_input={sugar_indices.size} proboscis_motor={prob_indices.size}",
            flush=True,
        )

        # Convert matrix values in-place to signed float32 weights.
        # CSR indices point to the presynaptic column for every stored edge.
        if syn_counts.data.dtype != np.float32:
            syn_counts.data = syn_counts.data.astype(np.float32, copy=True)
        syn_counts.data *= nt_sign[syn_counts.indices].astype(np.float32, copy=False)
        syn_counts.eliminate_zeros()
        scale = float(env("WEIGHT_SCALE", "0.2"))
        syn_counts.data *= scale

        # Release pandas metadata before constructing the live simulator.
        del nodes, node_root, nt_sign, sugar_mask, prob_mask
        gc.collect()

        DEVICE = "cpu"  # Render free instances have no CUDA device.

        # Reuse the NumPy CSR buffers for the Torch CSR when possible.
        crow = torch.from_numpy(syn_counts.indptr)
        col = torch.from_numpy(syn_counts.indices)
        val = torch.from_numpy(syn_counts.data)
        weight_tensor = torch.sparse_csr_tensor(
            crow,
            col,
            val,
            size=syn_counts.shape,
            device=DEVICE,
            dtype=torch.float32,
            check_invariants=False,
        )

        # Import the real FlyBrain LIF implementation, but avoid
        # Simulator.from_connectome() because that method constructs another
        # scipy signed matrix first.
        sys.path.insert(0, str(root))
        from flybrain.neurons import LIFPopulation
        from flybrain.simulator import Simulator

        simulator = Simulator.__new__(Simulator)
        simulator.device = torch.device(DEVICE)
        simulator.dtype = torch.float32
        simulator.weight_scale = 1.0
        simulator.W = weight_tensor
        simulator.n = int(weight_tensor.shape[0])
        simulator.pop = LIFPopulation(simulator.n, device=DEVICE, dtype=torch.float32)

        SIMULATOR = simulator
        SUGAR_IDX = torch.from_numpy(sugar_indices)
        PROB_IDX = torch.from_numpy(prob_indices)

        STEPS = max(1, int(env("SIM_STEPS", "40")))
        ON_STEPS = min(STEPS, max(1, int(env("ON_STEPS", "25"))))

        # Build the external sugar stimulus once. This is only a dense 139K
        # vector, tiny compared with the sparse connectivity.
        sugar_stim = torch.zeros(simulator.n, dtype=torch.float32, device=DEVICE)
        sugar_stim[SUGAR_IDX] = float(env("STIM_AMP", "2.0"))
        globals()["SUGAR_STIM"] = sugar_stim

        READY = True
        LOAD_ERROR = None
        print(
            f"[FlyWire] REAL connectome runtime ready: {simulator.n:,} neurons, "
            f"{weight_tensor._nnz():,} signed edges, CPU LIF, weight_scale={scale}",
            flush=True,
        )
    except Exception as exc:
        LOAD_ERROR = str(exc)
        READY = False
        print(f"[FlyWire] runtime load error: {exc}", file=sys.stderr, flush=True)
        raise


SUGAR_STIM = None


def run_simulation(stimulus: torch.Tensor | None) -> tuple[float, float, list[float]]:
    if SIMULATOR is None or PROB_IDX is None:
        raise RuntimeError("FlyBrain simulator is not loaded")

    SIMULATOR.pop.reset()
    rates: list[float] = []
    with torch.no_grad():
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
    """Convert the actual Discord text into a sensory input to the real LIF network."""
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
                "neurons": int(SIMULATOR.n),
                "motorNeuronPool": int(PROB_IDX.numel()),
                "sugarInputPool": int(SUGAR_IDX.numel()),
                "peakFractionFiring": peak,
                "meanFractionFiring": mean,
                "steps": STEPS,
                "stimulatedSteps": ON_STEPS,
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
                        {"source": "FlyWire FAFB v783 + FlyBrain LIF", "device": DEVICE},
                    )
                except Exception as exc:
                    print(f"[FlyWire] heartbeat warning: {exc}", file=sys.stderr, flush=True)
                next_heartbeat = now + heartbeat_interval

            time.sleep(0.05)
        except Exception as exc:
            print(f"[FlyWire] worker error: {exc}", file=sys.stderr, flush=True)
            time.sleep(1.0)


def main() -> None:
    # Start HTTP immediately so Render sees an open port while the connectome loads.
    start_http_server()

    root = Path(env("FLYBRAIN_PATH", "./vendor/flybrain")).resolve()
    if not root.exists():
        raise RuntimeError(f"FLYBRAIN_PATH does not exist: {root}")

    print("[FlyWire] starting memory-optimized REAL FAFB v783 runtime", flush=True)
    build_runtime(root)

    thread = threading.Thread(target=worker_loop, daemon=True)
    thread.start()

    print("[FlyWire] realtime Discord-message processing is ENABLED", flush=True)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
