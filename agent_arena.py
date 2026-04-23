import argparse

import torch

from mcts.mcts import MCTS
from models.base import BaseModel
from utils.game_utils import INPUT_CHANNELS, GameState, index_to_move
from utils.gui import ChessGUI
from utils.silverman import WHITE


def load_model(model_path, device):
    """Load a pretrained model checkpoint."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = BaseModel(
        input_channels=INPUT_CHANNELS,
        num_res_blocks=checkpoint["args"]["num_res_blocks"],
        num_channels=checkpoint["args"].get("num_channels", 128),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def get_ai_move(game_state, model, mcts_sims, c_puct):
    """Get a move from a model using MCTS."""

    def evaluate_fn(tensor):
        tensor = tensor.unsqueeze(0).to(next(model.parameters()).device)
        policy, value = model(tensor)
        return torch.softmax(policy.squeeze(0), dim=0), value.item()

    mcts = MCTS(evaluate_fn, c_puct=c_puct, tau=0.01)
    action = mcts.mcts_search(game_state, mcts_sims)
    return index_to_move(action, game_state.board)


def play_arena(white_model_path, black_model_path, mcts_sims=150, c_puct=1.0, move_delay_ms=400):
    """Run a GUI game between two model checkpoints."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    white_model = load_model(white_model_path, device)
    black_model = load_model(black_model_path, device)

    gui = ChessGUI(ai_callback=None, human_color="white")
    gui.root.title("Silverman 5x4 Agent Arena")

    def step_game():
        if gui.board.is_game_over():
            gui.update_status()
            return

        current_model = white_model if gui.board.turn == WHITE else black_model
        current_name = "White model" if gui.board.turn == WHITE else "Black model"
        gui.status_label.config(text=f"{current_name} is thinking...")
        gui.root.update_idletasks()

        game_state = GameState(gui.board, gui.history)
        move = get_ai_move(game_state, current_model, mcts_sims, c_puct)
        gui.make_move(move)

        if not gui.board.is_game_over():
            gui.root.after(move_delay_ms, step_game)

    gui.root.after(move_delay_ms, step_game)
    gui.run()


def main():
    parser = argparse.ArgumentParser(description="Benchmark two model checkpoints against each other")
    parser.add_argument("--white-model", type=str, required=True, help="Path to the white model checkpoint")
    parser.add_argument("--black-model", type=str, required=True, help="Path to the black model checkpoint")
    parser.add_argument("--mcts-sims", type=int, default=150, help="MCTS simulations per move")
    parser.add_argument("--c-puct", type=float, default=1.0, help="MCTS exploration constant")
    parser.add_argument("--move-delay-ms", type=int, default=400, help="Delay between moves in the GUI")
    args = parser.parse_args()

    play_arena(
        args.white_model,
        args.black_model,
        mcts_sims=args.mcts_sims,
        c_puct=args.c_puct,
        move_delay_ms=args.move_delay_ms,
    )


if __name__ == "__main__":
    main()
