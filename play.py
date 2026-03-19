import argparse
import torch
import chess
import sys
from pathlib import Path

from models.base import BaseModel
from utils.game_utils import GameState, index_to_move, is_terminal
from mcts.mcts import MCTS
from utils.gui import ChessGUI
import berserk

def load_model(model_path, device):
    """Load pretrained model from checkpoint."""
    # checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    # model = BaseModel(input_channels=119, num_res_blocks=checkpoint.get('num_res_blocks', 5))
    # model.load_state_dict(checkpoint['model_state_dict'])
    model = BaseModel(input_channels=119, num_res_blocks=5)  # Use default architecture
    model.to(device)
    model.eval()
    return model


def get_ai_move(game_state, model, mcts_sims, c_puct):
    """Get AI move using MCTS."""
    def evaluate_fn(tensor):
        tensor = tensor.unsqueeze(0).to(next(model.parameters()).device)
        p, v = model(tensor)
        return p.squeeze(0), v.item()
    
    mcts = MCTS(evaluate_fn, c_puct=c_puct, tau=0.01)  # Low temperature for best move
    action = mcts.mcts_search(game_state, mcts_sims)
    return index_to_move(action, game_state.board)


def play_gui(model_path, mcts_sims=800, c_puct=1.0, human_color='white'):
    """Play locally against the AI using GUI."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)

    ai_color = 'white' if human_color == 'black' else 'black'

    def ai_callback(game_state):
        return get_ai_move(game_state, model, mcts_sims, c_puct)

    gui = ChessGUI(ai_callback=ai_callback, ai_color=ai_color, human_color=human_color)
    gui.run()


def play_remote(model_path, token, mcts_sims=800, c_puct=1.0):
    """Play on Lichess using berserk API."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)

    session = berserk.TokenSession(token)
    client = berserk.Client(session)

    print("Connected to Lichess. Waiting for challenges...")

    game_id = None
    for event in client.board.stream_incoming_events():
        if event['type'] == 'challenge':
            challenge = event['challenge']
            if challenge['variant']['key'] == 'standard':
                print(f"Accepting challenge from {challenge['challenger']['name']}")
                client.challenges.accept(challenge['id'])
                game_id = challenge['id']
                break

    if not game_id:
        print("No challenge accepted.")
        return

    # Wait for game start
    for event in client.board.stream_incoming_events():
        if event['type'] == 'gameStart':
            print(f"Game started: {event['game']['id']}")
            break

    board = chess.Board()
    game_state = GameState(board)
    history = []

    for event in client.board.stream_game_state(game_id):
        if event['type'] == 'gameFull':
            board = chess.Board(event['state']['board']['fen'])
            history = []
        elif event['type'] == 'gameState':
            board = chess.Board(event['board']['fen'])

        # Update history
        history = [board.copy()] + history[:6]
        game_state = GameState(board, history)

        if is_terminal(board):
            print("Game ended.")
            break

        # Check if it's our turn
        account = client.account.get()
        is_our_turn = (board.turn == chess.WHITE and event.get('white', {}).get('id') == account['id']) or \
                      (board.turn == chess.BLACK and event.get('black', {}).get('id') == account['id'])

        if is_our_turn:
            print("AI is thinking...")
            ai_move = get_ai_move(game_state, model, mcts_sims, c_puct)
            client.board.make_move(game_id, ai_move.uci())
            print(f"AI plays: {ai_move.uci()}")


def main():
    parser = argparse.ArgumentParser(description="Play chess with pretrained model")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--mode", choices=["local", "remote"], required=True, help="Play mode")
    parser.add_argument("--mcts-sims", type=int, default=800, help="MCTS simulations per move")
    parser.add_argument("--c-puct", type=float, default=1.0, help="MCTS exploration constant")
    parser.add_argument("--token", type=str, default='')
    parser.add_argument("--human-color", choices=["white", "black"], default="white", help="Human player's color in local modes")

    args = parser.parse_args()

    if args.mode == "remote" and not args.token:
        print("Token required for remote mode.")
        sys.exit(1)

    if args.mode == "local":
        play_gui(args.model, args.mcts_sims, args.c_puct, args.human_color)
    elif args.mode == "remote":
        play_remote(args.model, args.token, args.mcts_sims, args.c_puct)


if __name__ == "__main__":
    main()