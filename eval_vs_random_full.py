import argparse
import json
import os
import random
import re
import time

import torch

from mcts.mcts import MCTS
from models.base import BaseModel
from utils.game_utils import (
    INPUT_CHANNELS,
    index_to_move,
    initial_game_state,
    is_terminal,
    terminal_state_evaluation,
    terminal_state_evaluation_large,
)
from utils.piece_square_tables import PIECE_SQUARE_TABLES, PIECE_VALUES
from utils.silverman import BLACK, BOARD_WIDTH, WHITE, square_file, square_rank

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


def make_model_agent(label, model, device, mcts_sims, c_puct):
    def evaluate_fn(tensor):
        with torch.no_grad():
            t = tensor.unsqueeze(0).to(device)
            policy, value = model(t)
            return torch.softmax(policy.squeeze(0), dim=0).cpu(), value.item()

    def move_fn(game_state):
        mcts = MCTS(evaluate_fn, c_puct=c_puct, tau=0.01, dirichlet_epsilon=0.0)
        action = mcts.mcts_search(game_state, mcts_sims)
        return index_to_move(action, game_state.board)

    return {
        "label": label,
        "move_fn": move_fn,
    }


def get_random_move(game_state):
    legal_moves = list(game_state.board.legal_moves)
    if not legal_moves:
        raise ValueError("Random agent requested a move in a terminal position")
    return random.choice(legal_moves)


def evaluate_with_piece_square_tables(board):
    white_score = 0
    black_score = 0

    for square_index, piece in enumerate(board.squares):
        if piece is None:
            continue

        row = square_rank(square_index)
        col = square_file(square_index)
        if piece.color == BLACK:
            col = BOARD_WIDTH - 1 - col

        score = (
            # PIECE_VALUES[piece.piece_type]
            PIECE_SQUARE_TABLES[piece.piece_type][row][col]
        )
        if piece.color == WHITE:
            white_score += score
        else:
            black_score += score

    net_score = white_score - black_score
    return float(net_score if board.turn == WHITE else -net_score)


def alphabeta_value(board, alpha, beta):
    if board.is_game_over():
        return terminal_state_evaluation(board)

    best_value = float("-inf")
    for move in board.legal_moves:
        next_board = board.copy()
        next_board.push(move)
        value = -alphabeta_value(next_board, -beta, -alpha)
        if value > best_value:
            best_value = value
        if best_value > alpha:
            alpha = best_value
        if alpha >= beta:
            break
    return best_value


def alphabeta_n_value(board, depth_remaining, alpha, beta):
    if board.is_game_over():
        return terminal_state_evaluation_large(board)
    if depth_remaining <= 0:
        return evaluate_with_piece_square_tables(board)

    best_value = float("-inf")
    for move in board.legal_moves:
        next_board = board.copy()
        next_board.push(move)
        value = -alphabeta_n_value(next_board, depth_remaining - 1, -beta, -alpha)
        if value > best_value:
            best_value = value
        if best_value > alpha:
            alpha = best_value
        if alpha >= beta:
            break
    return best_value


def get_alphabeta_move(game_state):
    best_move = None
    best_value = float("-inf")
    alpha = float("-inf")
    beta = float("inf")

    for move in game_state.board.legal_moves:
        next_board = game_state.board.copy()
        next_board.push(move)
        value = -alphabeta_value(next_board, -beta, -alpha)
        if value > best_value:
            best_value = value
            best_move = move
        if best_value > alpha:
            alpha = best_value

    if best_move is None:
        raise ValueError("Alpha-beta requested a move in a terminal position")

    return best_move


def get_alphabeta_n_move(game_state, depth):
    best_move = None
    best_value = float("-inf")
    alpha = float("-inf")
    beta = float("inf")

    for move in game_state.board.legal_moves:
        next_board = game_state.board.copy()
        next_board.push(move)
        value = -alphabeta_n_value(next_board, depth - 1, -beta, -alpha)
        if value > best_value:
            best_value = value
            best_move = move
        if best_value > alpha:
            alpha = best_value

    if best_move is None:
        raise ValueError("Alpha-beta-n requested a move in a terminal position")

    return best_move


def resolve_agent(
    spec,
    device,
    mcts_sims,
    c_puct,
    alphabeta_depth,
    checkpoint_which="best",
):
    normalized = spec.strip().lower()

    if normalized == "human":
        raise ValueError(
            "eval_vs_random is headless and does not support 'human'. "
            "Use 'random', 'alphabeta', 'alphabeta-n', or a checkpoint path."
        )

    if normalized in {"alphabeta", "alpha-beta"}:
        return {
            "label": "alpha-beta",
            "move_fn": lambda game_state: get_alphabeta_move(game_state),
        }

    if normalized in {"alphabeta-n", "alpha-beta-n"}:
        return {
            "label": f"alpha-beta-{alphabeta_depth}",
            "move_fn": lambda game_state: get_alphabeta_n_move(game_state, alphabeta_depth),
        }

    if normalized == "random":
        return {
            "label": "random",
            "move_fn": lambda game_state: get_random_move(game_state),
        }

    model = load_model(spec, device, which=checkpoint_which)
    checkpoint_name = os.path.basename(spec)
    return make_model_agent(
        f"{checkpoint_name} ({checkpoint_which})",
        model,
        device,
        mcts_sims,
        c_puct,
    )


def opponent_id(spec, alphabeta_depth, checkpoint_which):
    normalized = spec.strip().lower()
    if normalized == "random":
        return "random"
    if normalized in {"alphabeta", "alpha-beta"}:
        return "alphabeta"
    if normalized in {"alphabeta-n", "alpha-beta-n"}:
        return f"alphabeta_n_d{alphabeta_depth}"

    base = os.path.splitext(os.path.basename(spec.rstrip(os.sep)))[0] or "checkpoint"
    safe_base = re.sub(r"[^A-Za-z0-9_.-]+", "_", base)
    return f"checkpoint_{safe_base}_{checkpoint_which}"


def play_game(evaluated_agent, opponent_agent, evaluated_is_white):
    game_state = initial_game_state()
    move_num = 0

    while not is_terminal(game_state.board) and move_num < MAX_EVAL_MOVES:
        is_evaluated_turn = (game_state.board.turn == WHITE) == evaluated_is_white
        agent = evaluated_agent if is_evaluated_turn else opponent_agent

        move = agent["move_fn"](game_state)
        game_state = game_state.apply_move(move)
        move_num += 1

    z_white = white_result(game_state.board) if is_terminal(game_state.board) else 0.0
    return z_white if evaluated_is_white else -z_white, move_num


def evaluate_checkpoint(
    checkpoint_path,
    device,
    num_games,
    mcts_sims,
    c_puct,
    opponent_agent,
    which="best",
):
    model = load_model(checkpoint_path, device, which=which)
    evaluated_agent = make_model_agent(
        f"{os.path.basename(checkpoint_path)} ({which})",
        model,
        device,
        mcts_sims,
        c_puct,
    )

    wins, draws, losses = 0, 0, 0
    for i in range(num_games):
        evaluated_is_white = (i % 2 == 0)
        result, _ = play_game(evaluated_agent, opponent_agent, evaluated_is_white)

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
    parser = argparse.ArgumentParser(
        description="Evaluate saved checkpoints against a headless agent"
    )
    parser.add_argument("--results-dir", type=str, default="results/sgd_gate_run")
    parser.add_argument("--every", type=int, default=10)
    parser.add_argument("--num-games", type=int, default=20)
    parser.add_argument("--mcts-sims", type=int, default=50)
    parser.add_argument("--c-puct", type=float, default=1.4)
    parser.add_argument(
        "--which",
        choices=["best", "challenger"],
        default="best",
        help=(
            "Which network in the checkpoint to evaluate: 'best' = gated champion "
            "that drove self-play (default), 'challenger' = the just-trained "
            "network for that iteration"
        ),
    )
    parser.add_argument(
        "--opponent",
        type=str,
        default="random",
        help="Opponent spec: 'random', 'alphabeta', 'alphabeta-n', or a checkpoint path",
    )
    parser.add_argument(
        "--opponent-which",
        choices=["best", "challenger"],
        default="best",
        help="Which network to use when --opponent is a checkpoint path",
    )
    parser.add_argument(
        "--alphabeta-depth",
        type=int,
        default=4,
        help="Ply depth for the 'alphabeta-n' opponent before switching to piece-square evaluation",
    )
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument("--output", type=str, default=None)
    output_group.add_argument(
        "--add-to",
        type=str,
        default=None,
        help=(
            "Append results to an existing JSON file and start after its last "
            "evaluated iteration"
        ),
    )
    args = parser.parse_args()

    if args.alphabeta_depth < 1:
        raise ValueError("--alphabeta-depth must be at least 1")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    opponent_agent = resolve_agent(
        args.opponent,
        device,
        args.mcts_sims,
        args.c_puct,
        args.alphabeta_depth,
        checkpoint_which=args.opponent_which,
    )

    print(f"Device: {device}")
    print(f"Evaluating: {args.which}")
    print(f"Opponent: {opponent_agent['label']}")

    checkpoints = []
    for fname in os.listdir(args.results_dir):
        if fname.startswith("model_iter_") and fname.endswith(".pt"):
            iteration = int(fname.replace("model_iter_", "").replace(".pt", ""))
            if iteration % args.every == 0:
                checkpoints.append((iteration, os.path.join(args.results_dir, fname)))
    checkpoints.sort()

    output_name = (
        f"eval_vs_{opponent_id(args.opponent, args.alphabeta_depth, args.opponent_which)}"
        f"_{args.which}.json"
    )
    output_path = args.add_to or args.output or os.path.join(args.results_dir, output_name)
    results = []
    evaluated_iters = set()
    last_iteration = None
    if os.path.exists(output_path):
        with open(output_path) as f:
            results = json.load(f)
        evaluated_iters = {r["iteration"] for r in results}
        if results:
            last_iteration = max(evaluated_iters)
        if args.add_to and last_iteration is not None:
            print(
                f"Appending to {output_path} from iteration {last_iteration + 1}"
            )
        else:
            print(f"Resuming: {len(evaluated_iters)} checkpoints already evaluated")

    if args.add_to and last_iteration is not None:
        checkpoints = [(it, path) for it, path in checkpoints if it > last_iteration]

    if not checkpoints:
        print("No matching checkpoints found.")
        return

    print(f"Found {len(checkpoints)} checkpoints to evaluate (every {args.every}th)")
    print(f"Games per checkpoint: {args.num_games} | MCTS sims: {args.mcts_sims}")
    print()

    for iteration, ckpt_path in checkpoints:
        if iteration in evaluated_iters:
            continue

        print(f"Iteration {iteration}...", end=" ", flush=True)
        t0 = time.time()
        try:
            stats = evaluate_checkpoint(
                ckpt_path,
                device,
                args.num_games,
                args.mcts_sims,
                args.c_puct,
                opponent_agent,
                which=args.which,
            )
        except Exception as e:
            print(f"FAILED: {e}")
            continue
        elapsed = time.time() - t0

        entry = {"iteration": iteration, **stats, "elapsed_s": round(elapsed, 1)}
        results.append(entry)
        results.sort(key=lambda r: r["iteration"])

        print(
            f"+{stats['wins']} ={stats['draws']} -{stats['losses']} "
            f"({stats['score_pct']}%) [{elapsed:.1f}s]"
        )

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nDone. Results saved to {output_path}")


if __name__ == "__main__":
    main()
