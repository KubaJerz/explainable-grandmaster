import argparse
import random

import torch

from mcts.mcts import MCTS
from models.base import BaseModel
from utils.game_utils import (
    INPUT_CHANNELS,
    GameState,
    index_to_move,
    terminal_state_evaluation,
    terminal_state_evaluation_large,
)
from utils.piece_square_tables import PIECE_SQUARE_TABLES, PIECE_VALUES
from utils.gui import ChessGUI
from utils.silverman import BLACK, BOARD_WIDTH, WHITE, square_file, square_rank


def load_checkpoint(model_path, device):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = BaseModel(
        input_channels=INPUT_CHANNELS,
        num_res_blocks=checkpoint["args"]["num_res_blocks"],
        num_channels=checkpoint["args"].get("num_channels", 128),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return checkpoint, model


def get_model_move(game_state, model, mcts_sims, c_puct):
    def evaluate_fn(tensor):
        tensor = tensor.unsqueeze(0).to(next(model.parameters()).device)
        policy, value = model(tensor)
        return torch.softmax(policy.squeeze(0), dim=0), value.item()

    mcts = MCTS(evaluate_fn, c_puct=c_puct, tau=0.01)
    action = mcts.mcts_search(game_state, mcts_sims)
    return index_to_move(action, game_state.board)


def get_random_move(game_state):
    legal_moves = game_state.board.legal_moves
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

        score = PIECE_VALUES[piece.piece_type] + PIECE_SQUARE_TABLES[piece.piece_type][row][col]
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


def resolve_agent(spec, color_name, device, mcts_sims, c_puct, alphabeta_depth):
    normalized = spec.strip().lower()

    if normalized == "human":
        return {
            "label": f"{color_name} human",
            "move_fn": None,
            "is_human": True,
        }

    if normalized in {"alphabeta", "alpha-beta"}:
        return {
            "label": f"{color_name} alpha-beta",
            "move_fn": lambda game_state: get_alphabeta_move(game_state),
            "is_human": False,
        }

    if normalized in {"alphabeta-n", "alpha-beta-n"}:
        return {
            "label": f"{color_name} alpha-beta-{alphabeta_depth}",
            "move_fn": lambda game_state: get_alphabeta_n_move(game_state, alphabeta_depth),
            "is_human": False,
        }

    if normalized == "random":
        return {
            "label": f"{color_name} random",
            "move_fn": lambda game_state: get_random_move(game_state),
            "is_human": False,
        }

    _, model = load_checkpoint(spec, device)
    return {
        "label": f"{color_name} checkpoint",
        "move_fn": lambda game_state, model=model: get_model_move(game_state, model, mcts_sims, c_puct),
        "is_human": False,
    }


def play_arena(
    white_spec,
    black_spec,
    mcts_sims=150,
    c_puct=1.0,
    move_delay_ms=400,
    alphabeta_depth=4,
):
    """Run a GUI game between two participants."""
    if alphabeta_depth < 1:
        raise ValueError("--alphabeta-depth must be at least 1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    white_agent = resolve_agent(
        white_spec,
        "White",
        device,
        mcts_sims,
        c_puct,
        alphabeta_depth,
    )
    black_agent = resolve_agent(
        black_spec,
        "Black",
        device,
        mcts_sims,
        c_puct,
        alphabeta_depth,
    )

    if white_agent["is_human"] and black_agent["is_human"]:
        gui = ChessGUI(ai_callback=None, human_color="both", ai_color="none")
        gui.root.title("Silverman 5x4 Human Arena")
        gui.run()
        return

    if white_agent["is_human"] or black_agent["is_human"]:
        human_color = "white" if white_agent["is_human"] else "black"
        ai_color = "black" if human_color == "white" else "white"
        ai_agent = black_agent if human_color == "white" else white_agent

        gui = ChessGUI(
            ai_callback=ai_agent["move_fn"],
            ai_color=ai_color,
            human_color=human_color,
        )
        gui.root.title("Silverman 5x4 Human Arena")
        gui.run()
        return

    gui = ChessGUI(ai_callback=None, human_color="white", ai_color="none")
    gui.root.title("Silverman 5x4 Agent Arena")

    def step_game():
        if gui.board.is_game_over():
            gui.update_status()
            return

        current_agent = white_agent if gui.board.turn == WHITE else black_agent
        gui.status_label.config(text=f"{current_agent['label']} is thinking...")
        gui.root.update_idletasks()

        game_state = GameState(gui.board, gui.history)
        move = current_agent["move_fn"](game_state)
        gui.make_move(move)

        if not gui.board.is_game_over():
            gui.root.after(move_delay_ms, step_game)

    gui.root.after(move_delay_ms, step_game)
    gui.run()


def main():
    parser = argparse.ArgumentParser(description="Run a GUI game between agents and/or a human")
    parser.add_argument(
        "--white-model",
        type=str,
        required=True,
        help="Checkpoint path, 'human', 'random', 'alphabeta', or 'alphabeta-n' for White",
    )
    parser.add_argument(
        "--black-model",
        type=str,
        required=True,
        help="Checkpoint path, 'human', 'random', 'alphabeta', or 'alphabeta-n' for Black",
    )
    parser.add_argument("--mcts-sims", type=int, default=150, help="MCTS simulations per move")
    parser.add_argument("--c-puct", type=float, default=1.0, help="MCTS exploration constant")
    parser.add_argument("--move-delay-ms", type=int, default=400, help="Delay between moves in the GUI")
    parser.add_argument(
        "--alphabeta-depth",
        type=int,
        default=4,
        help="Ply depth for the 'alphabeta-n' agent before switching to piece-square evaluation",
    )
    args = parser.parse_args()

    play_arena(
        args.white_model,
        args.black_model,
        mcts_sims=args.mcts_sims,
        c_puct=args.c_puct,
        move_delay_ms=args.move_delay_ms,
        alphabeta_depth=args.alphabeta_depth,
    )


if __name__ == "__main__":
    main()
