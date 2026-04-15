import argparse
import io
import json
import os
import random
import threading
import time
from collections import deque

import torch
import torch.multiprocessing as mp

from models.base import BaseModel
from self_play import play_game
from train import train
from utils.board_utils import add_board_argument, get_board_spec, validate_checkpoint_board


class InferenceServer:
    """Batched GPU inference server using mp.Queues for cross-process communication.

    Runs as a thread in the main process (owns the GPU). Workers in separate
    processes submit requests via request_queue and receive responses on their
    per-worker response_queue.
    """

    def __init__(self, model, device, request_queue, response_queues,
                 batch_size=256, max_wait_ms=1.0):
        self.model = model
        self.device = device
        self.request_queue = request_queue
        self.response_queues = response_queues
        self.batch_size = batch_size
        self.max_wait_s = max_wait_ms / 1000.0
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

    def run(self):
        """Main inference loop — run on a dedicated thread."""
        self.model.eval()
        while self._running:
            batch = self._collect_batch()
            if batch:
                self._process_batch(batch)

    def _collect_batch(self):
        """Wait for at least one request, then collect up to batch_size."""
        # Block until at least one request arrives
        try:
            first = self.request_queue.get(timeout=0.1)
        except Exception:
            return []
        if first is None:
            return []

        batch = [first]
        wait_start = time.monotonic()
        deadline = wait_start + self.max_wait_s

        while len(batch) < self.batch_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                item = self.request_queue.get(timeout=remaining)
                if item is None:
                    break
                batch.append(item)
            except Exception:
                break

        wait_ms = (time.monotonic() - wait_start) * 1000.0
        with self._stats_lock:
            self._total_wait_ms += wait_ms

        return batch

    def _process_batch(self, batch):
        """Run forward pass on a batch of requests and distribute results."""
        t0 = time.monotonic()
        worker_ids = [item[0] for item in batch]
        tensors = torch.stack([item[1] for item in batch]).to(self.device)

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

        for i, wid in enumerate(worker_ids):
            self.response_queues[wid].put((policies[i], values[i].item()))

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
                "queue_depth": self.request_queue.qsize(),
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


def self_play_worker(worker_id, request_queue, response_queue, results_queue,
                     stop_event, mcts_sims, c_puct, board_name):
    """Self-play worker process. Plays games and sends results back via queue."""
    # Re-seed RNGs so forked processes don't all play identical games
    seed = os.getpid() + worker_id
    random.seed(seed)
    torch.manual_seed(seed)

    board_spec = get_board_spec(board_name)

    def evaluate_fn(tensor):
        request_queue.put((worker_id, tensor))
        policy, value = response_queue.get()
        return policy, value

    while not stop_event.is_set():
        samples = play_game(evaluate_fn, mcts_sims=mcts_sims, c_puct=c_puct, board_spec=board_spec)
        game_len = len(samples)

        # Serialize tensors to bytes to avoid leaking shared-memory file descriptors
        buf = io.BytesIO()
        torch.save(samples, buf)
        results_queue.put((worker_id, game_len, buf.getvalue()))


def trainer_loop(replay_buffer, results_queue, server, train_model, args,
                 training_log, device, start_iteration, stop_event, board_spec):
    """Runs training iterations, consuming completed games from results_queue."""
    samples_since_train = 0
    games_played = 0
    game_lengths = []

    sp_start = time.time()
    for iteration in range(start_iteration, args.iterations):
        # Collect samples until we have enough
        while not stop_event.is_set():
            try:
                worker_id, game_len, samples_data = results_queue.get(timeout=1.0)
            except Exception:
                continue

            games_played += 1
            game_lengths.append(game_len)

            # Deserialize samples from bytes
            samples = torch.load(io.BytesIO(samples_data), weights_only=False)
            is_draw = samples[0][2].item() == 0.0 if samples else False
            label = "(draw)" if is_draw else "(decisive)"
            replay_buffer.extend(samples)
            samples_since_train += len(samples)
            buf_len = len(replay_buffer)
            print(f"  [Worker {worker_id}] Game {games_played} done — {game_len} moves {label} (buffer: {buf_len})")

            if samples_since_train >= args.min_samples:
                break

        if stop_event.is_set():
            break

        # Snapshot buffer and reset counter
        buffer_snapshot = list(replay_buffer)
        new_samples = samples_since_train
        samples_since_train = 0
        iter_game_lengths = list(game_lengths)
        game_lengths.clear()

        sp_elapsed = time.time() - sp_start
        iter_start = sp_start  # iteration starts when self-play starts

        # Inference server stats
        inf_stats = server.get_stats()

        print(f"\n{'='*60}")
        print(f"  Self-play wait time: {sp_elapsed:.1f}s")
        print(f"  Inference server: {inf_stats['total_batches']} batches, "
              f"avg batch size: {inf_stats['avg_batch_size']}, "
              f"min/max: {inf_stats['min_batch_size']}/{inf_stats['max_batch_size']}")
        print(f"  Avg inference: {inf_stats['avg_inference_ms']:.2f}ms/batch, "
              f"avg wait: {inf_stats['avg_wait_ms']:.2f}ms, "
              f"throughput: {inf_stats['throughput_req_per_s']:.0f} req/s, "
              f"queue depth: {inf_stats['queue_depth']}")
        print(f"Training iteration {iteration}/{args.iterations - 1}")
        print(f"  {new_samples} new samples, {len(buffer_snapshot)} total in buffer, {games_played} games played")
        print(f"{'='*60}")

        # Train
        train_start = time.time()
        epoch_losses = train(
            train_model,
            buffer_snapshot,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=device,
            decisive_weight=args.decisive_weight,
        )
        train_elapsed = time.time() - train_start
        print(f"  Training time: {train_elapsed:.1f}s")

        # Publish new weights to inference server
        server.update_weights(train_model.state_dict())

        # Checkpoint
        ckpt_path = os.path.join(args.results_dir, f"model_iter_{iteration}.pt")
        torch.save({
            "iteration": iteration,
            "model_state_dict": train_model.state_dict(),
            "args": vars(args),
            "board_shape": [board_spec.rows, board_spec.cols],
            "policy_size": board_spec.policy_size,
            "input_channels": board_spec.input_channels,
        }, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

        # Replay buffer saved separately
        buffer_path = os.path.join(args.results_dir, "replay_buffer.pt")
        torch.save(list(replay_buffer), buffer_path)

        iter_elapsed = time.time() - iter_start
        print(f"  Iteration wall time: {iter_elapsed:.1f}s (self-play: {sp_elapsed:.1f}s, train: {train_elapsed:.1f}s)")

        # Reset self-play timer for next iteration
        sp_start = time.time()

        # Log stats
        avg_game_length = sum(iter_game_lengths) / len(iter_game_lengths) if iter_game_lengths else 0
        iter_stats = {
            "iteration": iteration,
            "self_play": {
                "num_games": len(iter_game_lengths),
                "game_lengths": iter_game_lengths,
                "avg_game_length": avg_game_length,
                "total_samples": new_samples,
            },
            "training": epoch_losses,
            "final_loss": epoch_losses[-1]["total"],
            "timing": {
                "self_play_wait_s": round(sp_elapsed, 2),
                "training_s": round(train_elapsed, 2),
                "iteration_s": round(iter_elapsed, 2),
            },
        }
        training_log.append(iter_stats)

        log_path = os.path.join(args.results_dir, "training_log.json")
        with open(log_path, "w") as f:
            json.dump(training_log, f, indent=2)

    # Signal workers to stop
    stop_event.set()


def main():
    mp.set_start_method("spawn" if os.name == "nt" else "fork", force=True)

    parser = argparse.ArgumentParser(description="AlphaZero parallel training pipeline")
    add_board_argument(parser)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=20)
    parser.add_argument("--min-samples", type=int, default=2048,
                        help="New samples required before each training iteration")
    parser.add_argument("--mcts-sims", type=int, default=150)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-res-blocks", type=int, default=3)
    parser.add_argument("--num-channels", type=int, default=32)
    parser.add_argument("--c-puct", type=float, default=1.4)
    parser.add_argument("--results-dir", type=str, default="results/")
    parser.add_argument("--decisive-weight", type=float, default=3.0,
                        help="Sampling weight for decisive (non-draw) training samples")
    parser.add_argument("--buffer-size", type=int, default=30000,
                        help="Replay buffer capacity (FIFO)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--inference-batch-size", type=int, default=256,
                        help="Max batch size for inference server")
    parser.add_argument("--max-wait-ms", type=float, default=5.0,
                        help="Max time (ms) inference server waits to fill a batch")
    args = parser.parse_args()
    board_spec = get_board_spec(args.board)
    args.board = board_spec.name

    os.makedirs(args.results_dir, exist_ok=True)

    # IPC queues — create before any CUDA init so forked workers don't inherit GPU state
    request_queue = mp.Queue()
    response_queues = [mp.Queue() for _ in range(args.num_workers)]
    results_queue = mp.Queue()
    stop_event = mp.Event()

    # Spawn self-play worker processes BEFORE touching CUDA to avoid fork+CUDA deadlock
    workers = []
    for i in range(args.num_workers):
        p = mp.Process(
            target=self_play_worker,
            args=(i, request_queue, response_queues[i], results_queue,
                  stop_event, args.mcts_sims, args.c_puct, board_spec.name),
            daemon=True,
        )
        p.start()
        workers.append(p)

    # Device setup — now safe to init CUDA since workers are already forked
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Create training model (for gradient updates)
    train_model = BaseModel(
        input_channels=board_spec.input_channels,
        board_shape=(board_spec.rows, board_spec.cols),
        policy_size=board_spec.policy_size,
        num_res_blocks=args.num_res_blocks,
        num_channels=args.num_channels,
    )
    train_model.to(device)

    # Replay buffer (local to main process)
    replay_buffer = deque(maxlen=args.buffer_size)
    training_log = []
    start_iteration = 0

    # Resume from checkpoint
    if args.resume:
        checkpoint = torch.load(args.resume, weights_only=False)
        validate_checkpoint_board(checkpoint.get("args", {}), board_spec)
        train_model.load_state_dict(checkpoint["model_state_dict"])
        start_iteration = checkpoint.get("iteration", 0) + 1
        buffer_path = os.path.join(args.results_dir, "replay_buffer.pt")
        if os.path.exists(buffer_path):
            replay_buffer.extend(torch.load(buffer_path, weights_only=False))
        log_path = os.path.join(args.results_dir, "training_log.json")
        if os.path.exists(log_path):
            with open(log_path) as f:
                training_log = json.load(f)
        print(f"Resumed from {args.resume} (starting at iteration {start_iteration}, "
              f"buffer: {len(replay_buffer)} samples)")

    # Create inference model (separate instance, lives inside server)
    inference_model = BaseModel(
        input_channels=board_spec.input_channels,
        board_shape=(board_spec.rows, board_spec.cols),
        policy_size=board_spec.policy_size,
        num_res_blocks=args.num_res_blocks,
        num_channels=args.num_channels,
    )
    inference_model.to(device)
    inference_model.load_state_dict(train_model.state_dict())
    inference_model.eval()

    # Start inference server (thread in main process — owns the GPU)
    server = InferenceServer(
        inference_model, device,
        request_queue, response_queues,
        batch_size=args.inference_batch_size,
        max_wait_ms=args.max_wait_ms,
    )
    inference_thread = threading.Thread(target=server.run, daemon=True)
    inference_thread.start()

    print(f"Device: {device}")
    print(f"Config: {vars(args)}")
    print(f"Model params: {sum(p.numel() for p in train_model.parameters()):,}")
    print(f"Starting {args.num_workers} self-play worker processes with batched inference "
          f"(batch_size={args.inference_batch_size}, max_wait={args.max_wait_ms}ms)")

    # Run trainer on main thread
    try:
        trainer_loop(replay_buffer, results_queue, server, train_model, args,
                     training_log, device, start_iteration, stop_event, board_spec)
    finally:
        # Graceful shutdown
        stop_event.set()
        server.stop()
        for p in workers:
            p.join(timeout=10)
        # Terminate stragglers
        for p in workers:
            if p.is_alive():
                p.terminate()

    print(f"\nDone. Results saved in {args.results_dir}/")


if __name__ == "__main__":
    main()
