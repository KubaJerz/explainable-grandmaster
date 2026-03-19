import argparse
import json
import os
import random
import time
from collections import deque
import torch

from models.base import BaseModel
from self_play import generate_games
from train import train


def main():
    parser = argparse.ArgumentParser(description="AlphaZero iteration pipeline")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--games-per-iter", type=int, default=25)
    parser.add_argument("--mcts-sims", type=int, default=150)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-res-blocks", type=int, default=19)
    parser.add_argument("--c-puct", type=float, default=1.0)
    parser.add_argument("--results-dir", type=str, default="results/")
    parser.add_argument("--buffer-size", type=int, default=50000, help="Replay buffer capacity (FIFO)")
    parser.add_argument("--draw-keep-ratio", type=float, default=0.25,
                        help="Probability of keeping samples from drawn games (0-1)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Initialize or load model
    model = BaseModel(input_channels=119, num_res_blocks=args.num_res_blocks)
    model.to(device)
    start_iter = 0
    training_log = []
    replay_buffer = deque(maxlen=args.buffer_size)

    if args.resume:
        checkpoint = torch.load(args.resume, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_iter = checkpoint.get("iteration", 0) + 1
        buffer_path = os.path.join(args.results_dir, "replay_buffer.pt")
        if os.path.exists(buffer_path):
            replay_buffer.extend(torch.load(buffer_path, weights_only=False))
        log_path = os.path.join(args.results_dir, "training_log.json")
        if os.path.exists(log_path):
            with open(log_path) as f:
                training_log = json.load(f)
        print(f"Resumed from {args.resume} (starting at iteration {start_iter}, buffer: {len(replay_buffer)} samples)")

    print(f"Device: {device}")
    print(f"Config: {vars(args)}")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    for iteration in range(start_iter, args.iterations):
        print(f"\n{'='*60}")
        print(f"Iteration {iteration}/{args.iterations - 1}")
        print(f"{'='*60}")

        iter_start = time.time()

        # self play
        print("Self-play phase:")
        model.eval()

        def evaluate_fn(tensor):
            with torch.no_grad():
                t = tensor.unsqueeze(0).to(device)
                policy, value = model(t)
                return torch.softmax(policy.squeeze(), dim=0).cpu(), value.item()

        sp_start = time.time()
        samples, sp_stats = generate_games(
            evaluate_fn,
            num_games=args.games_per_iter,
            mcts_sims=args.mcts_sims,
            c_puct=args.c_puct,
        )
        sp_elapsed = time.time() - sp_start
        print(f"  Collected {sp_stats['total_samples']} samples (avg game length: {sp_stats['avg_game_length']:.1f})")
        print(f"  Self-play time: {sp_elapsed:.1f}s")

        # Downsample draws to reduce draw dominance in training
        kept_samples = []
        # Group samples by game using game_lengths from stats
        offset = 0
        draws_skipped = 0
        for gl in sp_stats["game_lengths"]:
            game_samples = samples[offset:offset + gl]
            offset += gl
            is_draw = game_samples[0][2].item() == 0.0 if game_samples else False
            if is_draw and random.random() > args.draw_keep_ratio:
                draws_skipped += 1
                continue
            kept_samples.extend(game_samples)
        if draws_skipped > 0:
            print(f"  Skipped {draws_skipped} drawn games, keeping {len(kept_samples)}/{len(samples)} samples")
        samples = kept_samples

        # accumulate into replay buffer
        replay_buffer.extend(samples)
        print(f"  Replay buffer: {len(replay_buffer)}/{replay_buffer.maxlen} samples")

        # train
        print("Training phase:")
        train_start = time.time()
        epoch_losses = train(
            model,
            list(replay_buffer),
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=device,
        )
        train_elapsed = time.time() - train_start
        print(f"  Training time: {train_elapsed:.1f}s")

        # checkpoint (model only)
        ckpt_path = os.path.join(args.results_dir, f"model_iter_{iteration}.pt")
        torch.save({
            "iteration": iteration,
            "model_state_dict": model.state_dict(),
            "args": vars(args),
        }, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

        # replay buffer saved separately (overwritten each iteration)
        buffer_path = os.path.join(args.results_dir, "replay_buffer.pt")
        torch.save(list(replay_buffer), buffer_path)

        iter_elapsed = time.time() - iter_start
        print(f"  Iteration total: {iter_elapsed:.1f}s")

        # log stats
        iter_stats = {
            "iteration": iteration,
            "self_play": sp_stats,
            "training": epoch_losses,
            "final_loss": epoch_losses[-1]["total"],
            "timing": {
                "self_play_s": round(sp_elapsed, 2),
                "training_s": round(train_elapsed, 2),
                "iteration_s": round(iter_elapsed, 2),
            },
        }
        training_log.append(iter_stats)

        log_path = os.path.join(args.results_dir, "training_log.json")
        with open(log_path, "w") as f:
            json.dump(training_log, f, indent=2)

    print(f"\nDone. Results saved in {args.results_dir}/")


if __name__ == "__main__":
    main()
