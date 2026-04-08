import random

import torch

from mcts.mcts import MCTS
from utils.board_utils import get_board_spec
from utils.game_utils import GameState, index_to_move, is_terminal

MAX_MOVES = 512


def play_game(evaluate_fn, mcts_sims=800, c_puct=1.0, tau_threshold=30,
              playout_cap_fraction=0.25, full_search_prob=0.25, board_spec=None):
    """Play a single self-play game using MCTS, returning training data."""
    board_spec = board_spec or get_board_spec()
    game_state = GameState(board_spec.create_board(), board_spec=board_spec)
    trajectory = []

    capped_sims = max(1, int(mcts_sims * playout_cap_fraction))

    move_num = 0
    while not is_terminal(game_state.board, game_state.board_spec) and move_num < MAX_MOVES:
        tau = 1.0 if move_num < tau_threshold else 0.01
        is_full_search = random.random() < full_search_prob
        sims = mcts_sims if is_full_search else capped_sims

        mcts = MCTS(evaluate_fn, c_puct=c_puct, tau=tau)
        action = mcts.mcts_search(game_state, sims)
        mcts_policy = mcts.get_policy(prune_noise_visits=True) if is_full_search else None

        state_tensor = game_state.encode()
        side_to_move_is_white = game_state.board_spec.turn_is_white(game_state.board)
        trajectory.append((state_tensor, mcts_policy, side_to_move_is_white))

        move = index_to_move(action, game_state.board, game_state.board_spec)
        game_state = game_state.apply_move(move)
        move_num += 1

    z_white = (
        game_state.board_spec.white_result_from_terminal(game_state.board)
        if is_terminal(game_state.board, game_state.board_spec)
        else 0.0
    )

    training_data = []
    for state_tensor, mcts_policy, side_to_move_is_white in trajectory:
        if mcts_policy is None:
            continue
        value_target = game_state.board_spec.value_target_from_white_result(z_white, side_to_move_is_white)
        training_data.append((state_tensor, mcts_policy, torch.tensor(value_target, dtype=torch.float32)))

    return training_data


def generate_games(evaluate_fn, num_games, mcts_sims=800, c_puct=1.0, tau_threshold=30, board_spec=None):
    """Generate multiple self-play games and collect all training samples."""
    board_spec = board_spec or get_board_spec()
    all_samples = []
    game_lengths = []

    for i in range(num_games):
        print(f"  Self-play game {i+1}/{num_games}", end="", flush=True)
        samples = play_game(
            evaluate_fn,
            mcts_sims=mcts_sims,
            c_puct=c_puct,
            tau_threshold=tau_threshold,
            board_spec=board_spec,
        )
        game_lengths.append(len(samples))
        all_samples.extend(samples)
        print(f" - {len(samples)} moves")

    stats = {
        "num_games": num_games,
        "game_lengths": game_lengths,
        "avg_game_length": sum(game_lengths) / len(game_lengths) if game_lengths else 0,
        "total_samples": len(all_samples),
    }
    return all_samples, stats
