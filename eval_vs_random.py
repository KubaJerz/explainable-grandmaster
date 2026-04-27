import argparse
import json
import os
import random
import time

import torch

from mcts.mcts import MCTS
from models.base import BaseModel
from utils.game_utils import (
    INPUT_CHANNELS,
    index_to_move,
    initial_game_state,
    is_terminal,
)
from utils.silverman import WHITE

MAX_EVAL_MOVES = 512


def detect_input_channels(state_dict):
    return state_dict["stem.0.weight"].shape[1]


def load_model(checkpoint_path, device, which="best"):
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
    ckpt_args = checkpoint.get("args", {}) or {}

    num_res_blocks = ckpt_args.get("num_res_blocks", 3)
    num_channels = ckpt_args.get("num_channels", 32)

    if which == "best":
        state = checkpoint.get("best_model_state_dict", checkpoint["model_state_dict"])
    else:
        state = checkpoint["model_state_dict"]

    input_channels = detect_input_channels(state)
    if input_channels != INPUT_CHANNELS:
        raise ValueError(
            f"Checkpoint has {input_channels} input channels, current code expects {INPUT_CHANNELS}"
        )

    model = BaseModel(
        input_channels=input_channels,
        num_res_blocks=num_res_blocks,
        num_channels=num_channels,
    )
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def white_result(board):
    result = board.result(claim_draw=True)
    if result == "1-0":
        return 1.0
    if result == "0-1":
        return -1.0
    return 0.0


def play_game_vs_random(evaluate_fn, model_is_white, mcts_sims, c_puct):
    game_state = initial_game_state()
    move_num = 0

    while not is_terminal(game_state.board) and move_num < MAX_EVAL_MOVES:
        is_model_turn = (game_state.board.turn == WHITE) == model_is_white

        if is_model_turn:
            mcts = MCTS(evaluate_fn, c_puct=c_puct, tau=0.01, dirichlet_epsilon=0.0)
            action = mcts.mcts_search(game_state, mcts_sims)
            move = index_to_move(action, game_state.board)
        else:
            legal = list(game_state.board.legal_moves)
            move = random.choice(legal)

        game_state = game_state.apply_move(move)
        move_num += 1

    z_white = white_result(game_state.board) if is_terminal(game_state.board) else 0.0
    return z_white if model_is_white else -z_white, move_num


def evaluate_checkpoint(checkpoint_path, device, num_games, mcts_sims, c_puct, which="best"):
    model = load_model(checkpoint_path, device, which=which)

    def evaluate_fn(tensor):
        with torch.no_grad():
            t = tensor.unsqueeze(0).to(device)
            policy, value = model(t)
            return torch.softmax(policy.squeeze(), dim=0).cpu(), value.item()

    wins, draws, losses = 0, 0, 0
    for i in range(num_games):
        model_is_white = (i % 2 == 0)
        result, _ = play_game_vs_random(evaluate_fn, model_is_white, mcts_sims, c_puct)

        if result > 0:
            wins += 1
        elif result < 0:
            losses += 1
        else:
            draws += 1

    score = wins + 0.5 * draws
    return {
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score": score,
        "score_pct": round(score / num_games * 100, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate saved checkpoints against a random player")
    parser.add_argument("--results-dir", type=str, default="results/sgd_gate_run")
    parser.add_argument("--every", type=int, default=10)
    parser.add_argument("--num-games", type=int, default=20)
    parser.add_argument("--mcts-sims", type=int, default=50)
    parser.add_argument("--c-puct", type=float, default=1.4)
    parser.add_argument("--which", choices=["best", "challenger"], default="best",
                        help="Which network in the checkpoint to evaluate: 'best' = gated champion that "
                             "drove self-play (default), 'challenger' = the just-trained network that iter")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")
    print(f"Evaluating: {args.which}")

    checkpoints = []
    for fname in os.listdir(args.results_dir):
        if fname.startswith("model_iter_") and fname.endswith(".pt"):
            iteration = int(fname.replace("model_iter_", "").replace(".pt", ""))
            if iteration % args.every == 0:
                checkpoints.append((iteration, os.path.join(args.results_dir, fname)))
    checkpoints.sort()

    if not checkpoints:
        print("No matching checkpoints found.")
        return

    print(f"Found {len(checkpoints)} checkpoints to evaluate (every {args.every}th)")
    print(f"Games per checkpoint: {args.num_games} | MCTS sims: {args.mcts_sims}")
    print()

    output_path = args.output or os.path.join(args.results_dir, f"eval_vs_random_{args.which}.json")
    results = []
    evaluated_iters = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            results = json.load(f)
        evaluated_iters = {r["iteration"] for r in results}
        print(f"Resuming: {len(evaluated_iters)} checkpoints already evaluated")

    for iteration, ckpt_path in checkpoints:
        if iteration in evaluated_iters:
            continue

        print(f"Iteration {iteration}...", end=" ", flush=True)
        t0 = time.time()
        try:
            stats = evaluate_checkpoint(ckpt_path, device, args.num_games, args.mcts_sims, args.c_puct, which=args.which)
        except Exception as e:
            print(f"FAILED: {e}")
            continue
        elapsed = time.time() - t0

        entry = {"iteration": iteration, **stats, "elapsed_s": round(elapsed, 1)}
        results.append(entry)
        results.sort(key=lambda r: r["iteration"])

        print(f"+{stats['wins']} ={stats['draws']} -{stats['losses']} ({stats['score_pct']}%) [{elapsed:.1f}s]")

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nDone. Results saved to {output_path}")


if __name__ == "__main__":
    main()
