import argparse
import sys

import berserk
import chess
import torch

from mcts.mcts import MCTS
from models.base import BaseModel
from utils.board_utils import add_board_argument, get_board_spec, validate_checkpoint_board
from utils.game_utils import GameState, index_to_move, is_terminal
from utils.gui import ChessGUI


def load_model(model_path, device, board_spec):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    validate_checkpoint_board(checkpoint.get("args", {}), board_spec)
    checkpoint_args = checkpoint["args"]
    model = BaseModel(
        input_channels=checkpoint.get("input_channels", board_spec.input_channels),
        board_shape=tuple(checkpoint.get("board_shape", [board_spec.rows, board_spec.cols])),
        policy_size=checkpoint.get("policy_size", board_spec.policy_size),
        num_res_blocks=checkpoint_args["num_res_blocks"],
        num_channels=checkpoint_args.get("num_channels", 128),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def get_ai_move(game_state, model, mcts_sims, c_puct):
    def evaluate_fn(tensor):
        tensor = tensor.unsqueeze(0).to(next(model.parameters()).device)
        policy_logits, value = model(tensor)
        policy = torch.softmax(policy_logits.squeeze(0), dim=0).cpu()
        return policy, value.item()

    mcts = MCTS(evaluate_fn, c_puct=c_puct, tau=0.01)
    action = mcts.mcts_search(game_state, mcts_sims)
    return index_to_move(action, game_state.board, game_state.board_spec)


def play_gui(model_path, board_spec, mcts_sims=800, c_puct=1.0, human_color="white"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device, board_spec)

    ai_color = "white" if human_color == "black" else "black"

    def ai_callback(game_state):
        return get_ai_move(game_state, model, mcts_sims, c_puct)

    gui = ChessGUI(board_spec=board_spec, ai_callback=ai_callback, ai_color=ai_color, human_color=human_color)
    gui.run()


def play_remote(model_path, token, board_spec, mcts_sims=800, c_puct=1.0):
    if not board_spec.is_standard:
        raise ValueError("Remote play is only supported for the standard 8x8 board.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device, board_spec)

    session = berserk.TokenSession(token)
    client = berserk.Client(session)

    print("Connected to Lichess. Waiting for challenges...")

    game_id = None
    for event in client.board.stream_incoming_events():
        if event["type"] == "challenge":
            challenge = event["challenge"]
            if challenge["variant"]["key"] == "standard":
                print(f"Accepting challenge from {challenge['challenger']['name']}")
                client.challenges.accept(challenge["id"])
                game_id = challenge["id"]
                break

    if not game_id:
        print("No challenge accepted.")
        return

    for event in client.board.stream_incoming_events():
        if event["type"] == "gameStart":
            print(f"Game started: {event['game']['id']}")
            break

    board = board_spec.create_board()
    game_state = GameState(board, board_spec=board_spec)
    history = []

    for event in client.board.stream_game_state(game_id):
        if event["type"] == "gameFull":
            board = board_spec.create_board()
            history = []
        elif event["type"] == "gameState":
            moves = event.get("moves", "").split()
            board = board_spec.create_board()
            history = []
            for uci_move in moves:
                history = [board_spec.copy_board(board)] + history[:6]
                board_spec.apply_move_inplace(board, board_spec.parse_uci(uci_move, board))

        game_state = GameState(board, history, board_spec)

        if is_terminal(board, board_spec):
            print("Game ended.")
            break

        account = client.account.get()
        is_our_turn = (board.turn == chess.WHITE and event.get("white", {}).get("id") == account["id"]) or \
                      (board.turn == chess.BLACK and event.get("black", {}).get("id") == account["id"])

        if is_our_turn:
            print("AI is thinking...")
            ai_move = get_ai_move(game_state, model, mcts_sims, c_puct)
            client.board.make_move(game_id, board_spec.uci(ai_move, board))
            print(f"AI plays: {board_spec.uci(ai_move, board)}")


def main():
    parser = argparse.ArgumentParser(description="Play chess with pretrained model")
    add_board_argument(parser)
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--mode", choices=["local", "remote"], required=True, help="Play mode")
    parser.add_argument("--mcts-sims", type=int, default=150, help="MCTS simulations per move")
    parser.add_argument("--c-puct", type=float, default=1.4, help="MCTS exploration constant")
    parser.add_argument("--token", type=str, default="")
    parser.add_argument("--human-color", choices=["white", "black"], default="white", help="Human player's color in local mode")
    args = parser.parse_args()

    board_spec = get_board_spec(args.board)

    if args.mode == "remote" and not args.token:
        print("Token required for remote mode.")
        sys.exit(1)

    if args.mode == "local":
        play_gui(args.model, board_spec, args.mcts_sims, args.c_puct, args.human_color)
    else:
        play_remote(args.model, args.token, board_spec, args.mcts_sims, args.c_puct)


if __name__ == "__main__":
    main()
