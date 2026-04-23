import argparse
import torch

from models.base import BaseModel
from utils.game_utils import INPUT_CHANNELS, index_to_move
from mcts.mcts import MCTS
from utils.gui import ChessGUI

def load_model(model_path, device):
    """Load pretrained model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = BaseModel(
        input_channels=INPUT_CHANNELS,
        num_res_blocks=checkpoint["args"]["num_res_blocks"],
        num_channels=checkpoint["args"].get("num_channels", 128),
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def get_ai_move(game_state, model, mcts_sims, c_puct):
    """Get AI move using MCTS."""
    def evaluate_fn(tensor):
        tensor = tensor.unsqueeze(0).to(next(model.parameters()).device)
        p, v = model(tensor)
        return torch.softmax(p.squeeze(0), dim=0), v.item()
    
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


def main():
    parser = argparse.ArgumentParser(description="Play chess with pretrained model")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--mcts-sims", type=int, default=150, help="MCTS simulations per move")
    parser.add_argument("--c-puct", type=float, default=1.0, help="MCTS exploration constant")
    parser.add_argument("--human-color", choices=["white", "black"], default="white", help="Human player's color in local modes")

    args = parser.parse_args()
    play_gui(args.model, args.mcts_sims, args.c_puct, args.human_color)


if __name__ == "__main__":
    main()
