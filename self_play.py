import torch
import chess

from utils.game_utils import GameState, move_to_index, index_to_move, is_terminal, terminal_state_evaluation
from mcts.mcts import MCTS

MAX_MOVES = 512


def play_game(model, mcts_sims=800, c_puct=1.0, tau_threshold=30, device="cpu"):
    """Play a single self-play game using MCTS, returning training data.

    Runs till end of game or max moves.

    Args:
        model: the neural network (policy + value heads)
        mcts_sims: number of MCTS simulations per move
        c_puct: exploration constant
        tau_threshold: move number after which temperature drops to ~0

    Returns:
        list of (state_tensor, policy_target, value_target) tuples
    """
    game_state = GameState(chess.Board())
    trajectory = []  # (tensor, mcts_policy, side_to_move)

    move_num = 0
    while not is_terminal(game_state.board) and move_num < MAX_MOVES:
        # Temperature schedule: tau=1 for first N moves, then near-zero
        tau = 1.0 if move_num < tau_threshold else 0.01

        #run the mcts
        mcts = MCTS(model, c_puct=c_puct, tau=tau, device=device)
        action = mcts.mcts_search(game_state, mcts_sims)
        mcts_policy = mcts.get_policy()

        # Store position data to use for training later
        state_tensor = game_state.encode()
        side = game_state.board.turn  # chess.WHITE or chess.BLACK
        trajectory.append((state_tensor, mcts_policy, side))

        # apply the chosen action
        move = index_to_move(action, game_state.board)
        game_state = game_state.apply_move(move)
        move_num += 1

    # Determine game result from white's perspective
    if is_terminal(game_state.board):
        result = game_state.board.result()
        if result == "1-0":
            z_white = 1.0
        elif result == "0-1":
            z_white = -1.0
        else:
            z_white = 0.0
    else:
        # Hit move cap — adjudicate as draw
        z_white = 0.0

    # value targets "z" from each position's side-to-move perspective given the outcoem of the game
    # this is from page 
    training_data = []
    for state_tensor, mcts_policy, side in trajectory:
        value_target = z_white if side == chess.WHITE else -z_white
        training_data.append((state_tensor, mcts_policy, torch.tensor(value_target, dtype=torch.float32)))

    return training_data


def generate_games(model, num_games, mcts_sims=800, c_puct=1.0, tau_threshold=30, device="cpu"):
    """Generate multiple self-play games and collect all training samples.

    Returns:
        samples: list of (state_tensor, policy_target, value_target)
        stats: dict with game_lengths
    """
    all_samples = []
    game_lengths = []

    for i in range(num_games):
        print(f"  Self-play game {i+1}/{num_games}", end="", flush=True)
        samples = play_game(model, mcts_sims=mcts_sims, c_puct=c_puct, tau_threshold=tau_threshold, device=device)
        game_lengths.append(len(samples))
        all_samples.extend(samples)
        print(f" — {len(samples)} moves")

    stats = {
        "num_games": num_games,
        "game_lengths": game_lengths,
        "avg_game_length": sum(game_lengths) / len(game_lengths) if game_lengths else 0,
        "total_samples": len(all_samples),
    }
    return all_samples, stats
