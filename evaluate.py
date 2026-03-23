import argparse
import os
from datetime import datetime

import chess
import chess.engine
import chess.pgn
import torch

from models.base import BaseModel
from mcts.mcts import MCTS
from utils.game_utils import GameState, index_to_move, is_terminal

MAX_MOVES = 512


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    num_res_blocks = checkpoint["args"]["num_res_blocks"]
    num_channels = checkpoint["args"].get("num_channels", 256)
    model = BaseModel(input_channels=119, num_res_blocks=num_res_blocks, num_channels=num_channels)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def play_game(model, engine, model_is_white, mcts_sims, c_puct, time_limit, device, opponent_str):
    """Play a single game between the model and Stockfish.
    Returns (score, pgn_game) where score is +1 (model win), -1 (model loss), 0 (draw)
    and pgn_game is a chess.pgn.Game with the full move history."""
    board = chess.Board()
    game_state = GameState(board.copy())

    def evaluate_fn(tensor):
        with torch.no_grad():
            t = tensor.unsqueeze(0).to(device)
            policy, value = model(t)
            return torch.softmax(policy.squeeze(), dim=0).cpu(), value.item()

    mcts = MCTS(evaluate_fn, c_puct=c_puct, tau=0.01)

    # Set up PGN game
    pgn_game = chess.pgn.Game()
    pgn_game.headers["White"] = "AlphaZero" if model_is_white else f"Stockfish ({opponent_str})"
    pgn_game.headers["Black"] = f"Stockfish ({opponent_str})" if model_is_white else "AlphaZero"
    pgn_game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
    pgn_game.headers["Event"] = "Evaluation"
    pgn_node = pgn_game

    move_count = 0
    while not is_terminal(game_state.board) and move_count < MAX_MOVES:
        is_model_turn = (game_state.board.turn == chess.WHITE) == model_is_white

        if is_model_turn:
            action = mcts.mcts_search(game_state, mcts_sims)
            move = index_to_move(action, game_state.board)
        else:
            result = engine.play(game_state.board, chess.engine.Limit(time=time_limit))
            move = result.move

        pgn_node = pgn_node.add_variation(move)
        game_state = game_state.apply_move(move)
        move_count += 1

    # Determine result
    board = game_state.board
    if not board.is_game_over(claim_draw=True):
        pgn_game.headers["Result"] = "1/2-1/2"
        return 0, pgn_game

    result_str = board.result(claim_draw=True)
    pgn_game.headers["Result"] = result_str
    if result_str == "1/2-1/2":
        return 0, pgn_game
    if result_str == "1-0":
        return (1 if model_is_white else -1), pgn_game
    else:  # "0-1"
        return (-1 if model_is_white else 1), pgn_game


def main():
    parser = argparse.ArgumentParser(description="Evaluate model against Stockfish")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--stockfish-path", type=str, default="stockfish", help="Path to Stockfish binary")
    parser.add_argument("--elo", type=int, default=1350, help="Stockfish ELO limit (ignored if --skill-level is set)")
    parser.add_argument("--skill-level", type=int, default=None, help="Stockfish Skill Level 0-20 (overrides --elo)")
    parser.add_argument("--num-games", type=int, default=10, help="Total games to play")
    parser.add_argument("--mcts-sims", type=int, default=50, help="MCTS simulations per move")
    parser.add_argument("--c-puct", type=float, default=1.0, help="Exploration constant")
    parser.add_argument("--time-limit", type=float, default=0.1, help="Stockfish time per move (seconds)")
    parser.add_argument("--pgn-dir", type=str, default="results/games", help="Directory to save PGN files")
    args = parser.parse_args()

    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")

    # Load model
    model = load_model(args.checkpoint, device)
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Start Stockfish
    try:
        engine = chess.engine.SimpleEngine.popen_uci(args.stockfish_path)
    except FileNotFoundError:
        print(f"Stockfish not found at '{args.stockfish_path}'.")
        print("Install with: sudo apt install stockfish")
        print("Or pass --stockfish-path /path/to/stockfish")
        return

    try:
        if args.skill_level is not None:
            engine.configure({"Skill Level": args.skill_level})
            opponent_str = f"Skill Level {args.skill_level}"
        else:
            engine.configure({"UCI_LimitStrength": True, "UCI_Elo": args.elo})
            opponent_str = f"ELO {args.elo}"
        print(f"Stockfish: {opponent_str}")
        print(f"Games: {args.num_games} | MCTS sims: {args.mcts_sims} | c_puct: {args.c_puct}")
        print()

        os.makedirs(args.pgn_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.skill_level is not None:
            pgn_path = os.path.join(args.pgn_dir, f"eval_skill{args.skill_level}_{timestamp}.pgn")
        else:
            pgn_path = os.path.join(args.pgn_dir, f"eval_elo{args.elo}_{timestamp}.pgn")

        wins, draws, losses = 0, 0, 0
        games_as_white = args.num_games // 2
        games_as_black = args.num_games - games_as_white

        with open(pgn_path, "w") as pgn_file:
            for i in range(args.num_games):
                model_is_white = i < games_as_white
                color_str = "White" if model_is_white else "Black"
                print(f"Game {i+1}/{args.num_games} (model plays {color_str})...", end=" ", flush=True)

                result, pgn_game = play_game(
                    model, engine, model_is_white, args.mcts_sims,
                    args.c_puct, args.time_limit, device, opponent_str,
                )

                if result == 1:
                    wins += 1
                    print("Win")
                elif result == -1:
                    losses += 1
                    print("Loss")
                else:
                    draws += 1
                    print("Draw")

                print(pgn_game, file=pgn_file)
                print(file=pgn_file)  # blank line between games

        # Summary
        total = wins + draws + losses
        score = wins + 0.5 * draws
        print(f"\n{'='*40}")
        print(f"Results: +{wins} ={draws} -{losses} / {total}")
        print(f"Win: {wins/total*100:.1f}% | Draw: {draws/total*100:.1f}% | Loss: {losses/total*100:.1f}%")
        print(f"Score: {score}/{total} ({score/total*100:.1f}%)")
        print(f"Games saved to: {pgn_path}")

    finally:
        engine.quit()


if __name__ == "__main__":
    main()
