import threading
import time

import torch


class InferenceRequest:
    __slots__ = ("tensor", "event", "policy", "value")

    def __init__(self, tensor):
        self.tensor = tensor
        self.event = threading.Event()
        self.policy = None
        self.value = None


class InferenceServer:
    """Batched GPU inference server for self-play.

    One dedicated thread runs the model; game threads submit tensors and block
    until results are ready.
    """

    def __init__(self, model, device, batch_size=512, max_wait_ms=1.0):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.max_wait_s = max_wait_ms / 1000.0

        self._queue = []
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._model_lock = threading.Lock()
        self._running = True

        # Stats
        self._total_batches = 0
        self._total_requests = 0
        self._batch_size_sum = 0
        self._min_batch = float('inf')
        self._max_batch = 0
        self._total_inference_ms = 0.0
        self._total_wait_ms = 0.0
        self._stats_lock = threading.Lock()
        self._stats_start = time.time()

    def submit(self, tensor):
        """Submit a single board tensor for evaluation. Blocks until result is ready."""
        req = InferenceRequest(tensor)
        with self._cond:
            self._queue.append(req)
            self._cond.notify()
        req.event.wait()
        return req.policy, req.value

    def run(self):
        """Main inference loop — run on a dedicated thread."""
        self.model.eval()
        while self._running:
            batch = self._collect_batch()
            if batch:
                self._process_batch(batch)

    def _collect_batch(self):
        """Wait for at least one request, then collect up to batch_size."""
        with self._cond:
            while not self._queue and self._running:
                self._cond.wait(timeout=0.1)
            if not self._running and not self._queue:
                return []

            wait_start = time.monotonic()
            deadline = wait_start + self.max_wait_s
            while len(self._queue) < self.batch_size and self._running:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._cond.wait(timeout=remaining)

            wait_ms = (time.monotonic() - wait_start) * 1000.0
            batch = self._queue[:self.batch_size]
            self._queue = self._queue[self.batch_size:]

        with self._stats_lock:
            self._total_wait_ms += wait_ms

        return batch

    def _process_batch(self, batch):
        """Run forward pass on a batch of requests and distribute results."""
        t0 = time.monotonic()
        tensors = torch.stack([req.tensor for req in batch]).to(self.device)

        with self._model_lock:
            with torch.no_grad():
                policies, values = self.model(tensors)

        policies = torch.softmax(policies, dim=1).cpu()
        values = values.cpu()
        inference_ms = (time.monotonic() - t0) * 1000.0

        bs = len(batch)
        with self._stats_lock:
            self._total_batches += 1
            self._total_requests += bs
            self._batch_size_sum += bs
            self._min_batch = min(self._min_batch, bs)
            self._max_batch = max(self._max_batch, bs)
            self._total_inference_ms += inference_ms

        for i, req in enumerate(batch):
            req.policy = policies[i]
            req.value = values[i].item()
            req.event.set()

    def get_stats(self):
        """Return stats dict and reset counters."""
        with self._stats_lock:
            elapsed = time.time() - self._stats_start
            n = self._total_batches
            stats = {
                "elapsed_s": round(elapsed, 1),
                "total_batches": n,
                "total_requests": self._total_requests,
                "avg_batch_size": round(self._batch_size_sum / n, 1) if n else 0,
                "min_batch_size": self._min_batch if n else 0,
                "max_batch_size": self._max_batch if n else 0,
                "avg_inference_ms": round(self._total_inference_ms / n, 2) if n else 0,
                "avg_wait_ms": round(self._total_wait_ms / n, 2) if n else 0,
                "throughput_req_per_s": round(self._total_requests / elapsed, 0) if elapsed > 0 else 0,
                "queue_depth": len(self._queue),
            }
            # Reset
            self._total_batches = 0
            self._total_requests = 0
            self._batch_size_sum = 0
            self._min_batch = float('inf')
            self._max_batch = 0
            self._total_inference_ms = 0.0
            self._total_wait_ms = 0.0
            self._stats_start = time.time()
        return stats

    def update_weights(self, state_dict):
        """Update the inference model weights (called by trainer between iterations)."""
        with self._model_lock:
            self.model.load_state_dict(state_dict)
            self.model.eval()

    def stop(self):
        """Signal the inference loop to shut down."""
        self._running = False
        with self._cond:
            self._cond.notify_all()
