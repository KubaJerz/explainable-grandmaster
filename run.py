import argparse
import json
import os
import random
import time
from collections import deque
import torch

from models.base import BaseModel
from self_play import generate_games, play_match
from train import train
from utils.game_utils import INPUT_CHANNELS


def main():
    parser = argparse.ArgumentParser(description="AlphaZero iteration pipeline")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--games-per-iter", type=int, default=25)
    parser.add_argument("--mcts-sims", type=int, default=150)
    parser.add_argument("--steps-per-iter", type=int, default=150,
                        help="Mini-batch gradient steps per iteration")
    parser.add_argument("--sample-without-replacement", action="store_true",
                        help="Sample buffer without replacement (default: with replacement)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--lr-milestones", type=int, nargs="*", default=[115, 170],
                        help="Iterations at which LR drops by 10x")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-res-blocks", type=int, default=3)
    parser.add_argument("--num-channels", type=int, default=32)
    parser.add_argument("--c-puct", type=float, default=1.0)
    parser.add_argument("--results-dir", type=str, default="results/")
    parser.add_argument("--buffer-size", type=int, default=50000, help="Replay buffer capacity (FIFO)")
    parser.add_argument("--draw-keep-ratio", type=float, default=0.5,
                        help="Probability of keeping samples from drawn games (0-1)")
    parser.add_argument("--value-weight", type=float, default=0.25,
                        help="Loss weight on value head (Leela uses 0.25)")
    parser.add_argument("--gate-games", type=int, default=20,
                        help="Number of evaluation games challenger vs champion")
    parser.add_argument("--gate-threshold", type=float, default=0.55,
                        help="Win-rate (draws=0.5) needed to promote challenger")
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

    # Initialize or load model. We keep two copies:
    #   model      — the network being trained (challenger)
    #   best_model — the gated champion that drives self-play
    model = BaseModel(input_channels=INPUT_CHANNELS, num_res_blocks=args.num_res_blocks, num_channels=args.num_channels)
    best_model = BaseModel(input_channels=INPUT_CHANNELS, num_res_blocks=args.num_res_blocks, num_channels=args.num_channels)
    model.to(device)
    best_model.to(device)
    best_model.load_state_dict(model.state_dict())
    start_iter = 0
    training_log = []
    replay_buffer = deque(maxlen=args.buffer_size)

    if args.resume:
        checkpoint = torch.load(args.resume, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        # Fall back to model weights if checkpoint predates gating.
        best_state = checkpoint.get("best_model_state_dict", checkpoint["model_state_dict"])
        best_model.load_state_dict(best_state)
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

        # self play (uses gated best_model)
        print("Self-play phase:")
        best_model.eval()

        def best_eval_fn(tensor):
            with torch.no_grad():
                t = tensor.unsqueeze(0).to(device)
                policy, value = best_model(t)
                return torch.softmax(policy.squeeze(), dim=0).cpu(), value.item()

        sp_start = time.time()
        samples, sp_stats = generate_games(
            best_eval_fn,
            num_games=args.games_per_iter,
            mcts_sims=args.mcts_sims,
            c_puct=args.c_puct,
        )
        sp_elapsed = time.time() - sp_start
        illegal = sp_stats["illegal"]
        print(f"  Collected {sp_stats['total_samples']} samples (avg game length: {sp_stats['avg_game_length']:.1f})")
        print(f"  Illegal-argmax rate: {illegal['argmax_illegal_rate']:.2%}, "
              f"mean illegal mass: {illegal['mean_illegal_mass']:.4f}")
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
        current_lr = args.lr * (0.1 ** sum(1 for m in args.lr_milestones if iteration >= m))
        if current_lr != args.lr:
            print(f"  LR annealed to {current_lr:g} (milestones: {args.lr_milestones})")
        train_start = time.time()
        epoch_losses = train(
            model,
            list(replay_buffer),
            steps_per_iter=args.steps_per_iter,
            batch_size=args.batch_size,
            lr=current_lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            value_weight=args.value_weight,
            sample_with_replacement=not args.sample_without_replacement,
            device=device,
        )
        train_elapsed = time.time() - train_start
        print(f"  Training time: {train_elapsed:.1f}s")

        # gating: trained model (challenger) vs best_model (champion)
        print("Gating phase:")
        model.eval()

        def challenger_eval_fn(tensor):
            with torch.no_grad():
                t = tensor.unsqueeze(0).to(device)
                policy, value = model(t)
                return torch.softmax(policy.squeeze(), dim=0).cpu(), value.item()

        gate_start = time.time()
        ch_wins, ch_losses_count, ch_draws = play_match(
            challenger_eval_fn,
            best_eval_fn,
            num_games=args.gate_games,
            mcts_sims=args.mcts_sims,
            c_puct=args.c_puct,
        )
        gate_elapsed = time.time() - gate_start
        win_rate = (ch_wins + 0.5 * ch_draws) / args.gate_games if args.gate_games else 0.0
        promoted = win_rate >= args.gate_threshold
        print(f"  Challenger: {ch_wins}W-{ch_losses_count}L-{ch_draws}D "
              f"(win rate {win_rate:.2%}, threshold {args.gate_threshold:.0%})")
        if promoted:
            best_model.load_state_dict(model.state_dict())
            print("  -> Challenger PROMOTED to best_model")
        else:
            print("  -> Champion retained")
        print(f"  Gating time: {gate_elapsed:.1f}s")

        # checkpoint (current trained model + gated best)
        ckpt_path = os.path.join(args.results_dir, f"model_iter_{iteration}.pt")
        torch.save({
            "iteration": iteration,
            "model_state_dict": model.state_dict(),
            "best_model_state_dict": best_model.state_dict(),
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
            "lr": current_lr,
            "self_play": sp_stats,
            "training": epoch_losses,
            "final_loss": epoch_losses[-1]["total"],
            "gating": {
                "games": args.gate_games,
                "challenger_wins": ch_wins,
                "challenger_losses": ch_losses_count,
                "draws": ch_draws,
                "win_rate": win_rate,
                "promoted": promoted,
            },
            "timing": {
                "self_play_s": round(sp_elapsed, 2),
                "training_s": round(train_elapsed, 2),
                "gating_s": round(gate_elapsed, 2),
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
