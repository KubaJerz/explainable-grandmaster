import random

import torch

from utils.game_utils import index_to_move, initial_game_state, is_terminal, move_to_index
from mcts.mcts import MCTS
from utils.silverman import WHITE

MAX_MOVES = 512


def _illegal_stats_from_root(mcts, board):
    """Compute (argmax_illegal, illegal_mass) from raw root priors against legal moves."""
    raw = mcts.root_raw_priors
    legal_mask = torch.zeros_like(raw)
    for move in board.legal_moves:
        legal_mask[move_to_index(move)] = 1.0
    raw_argmax = int(torch.argmax(raw).item())
    argmax_illegal = legal_mask[raw_argmax].item() == 0.0
    illegal_mass = float((raw * (1.0 - legal_mask)).sum().item())
    return argmax_illegal, illegal_mass


def play_game(evaluate_fn, mcts_sims=800, c_puct=1.0, tau_threshold=30,
              playout_cap_fraction=0.25, full_search_prob=0.25):
    """Play a single self-play game using MCTS, returning training data.

    Runs till end of game or max moves. Uses playout cap randomization (KataGo):
    each turn randomly gets either full sims or reduced sims. Only full-search
    turns contribute policy targets; all turns contribute value targets.

    Args:
        evaluate_fn: callable(tensor) -> (policy, value) for board evaluation
        mcts_sims: number of MCTS simulations per move (full search)
        c_puct: exploration constant
        tau_threshold: move number after which temperature drops to ~0
        playout_cap_fraction: fraction of mcts_sims for capped (quick) turns
        full_search_prob: probability of a turn getting full search

    Returns:
        list of (state_tensor, policy_target, value_target, policy_weight) tuples
    """
    game_state = initial_game_state()
    trajectory = []  # (tensor, mcts_policy, side_to_move, has_policy_target)

    capped_sims = max(1, int(mcts_sims * playout_cap_fraction))

    illegal_argmax_count = 0
    illegal_mass_sum = 0.0
    eval_count = 0

    move_num = 0
    while not is_terminal(game_state.board) and move_num < MAX_MOVES:
        # Temperature schedule: tau=1 for first N moves, then near-zero
        tau = 1.0 if move_num < tau_threshold else 0.01

        # Playout cap randomization: full search or capped search
        is_full_search = random.random() < full_search_prob
        sims = mcts_sims if is_full_search else capped_sims

        # Run the MCTS
        mcts = MCTS(evaluate_fn, c_puct=c_puct, tau=tau)
        action = mcts.mcts_search(game_state, sims)

        # Track illegal-move predictions from the raw network output.
        argmax_illegal, illegal_mass = _illegal_stats_from_root(mcts, game_state.board)
        illegal_argmax_count += int(argmax_illegal)
        illegal_mass_sum += illegal_mass
        eval_count += 1

        # Only full-search turns train the policy head.
        if is_full_search:
            mcts_policy = mcts.get_policy(prune_noise_visits=True)
        else:
            mcts_policy = torch.zeros_like(mcts.root.visit_counts)

        # Store position data to use for training later.
        state_tensor = game_state.encode()
        side = game_state.board.turn
        trajectory.append((state_tensor, mcts_policy, side, is_full_search))

        # Apply the chosen action.
        move = index_to_move(action, game_state.board)
        game_state = game_state.apply_move(move)
        move_num += 1

    # Determine game result from white's perspective.
    if is_terminal(game_state.board):
        result = game_state.board.result()
        if result == "1-0":
            z_white = 1.0
        elif result == "0-1":
            z_white = -1.0
        else:
            z_white = 0.0
    else:
        # Hit move cap - adjudicate as draw.
        z_white = 0.0

    # All turns contribute value targets. Only full-search turns contribute
    # policy loss through a nonzero policy weight.
    training_data = []
    for state_tensor, mcts_policy, side, has_policy_target in trajectory:
        value_target = z_white if side == WHITE else -z_white
        training_data.append((
            state_tensor,
            mcts_policy,
            torch.tensor(value_target, dtype=torch.float32),
            torch.tensor(1.0 if has_policy_target else 0.0, dtype=torch.float32),
        ))

    illegal_stats = {
        "evals": eval_count,
        "argmax_illegal": illegal_argmax_count,
        "illegal_mass_sum": illegal_mass_sum,
    }
    return training_data, illegal_stats


def generate_games(evaluate_fn, num_games, mcts_sims=800, c_puct=1.0, tau_threshold=30):
    """Generate multiple self-play games and collect all training samples.

    Returns:
        samples: list of (state_tensor, policy_target, value_target, policy_weight)
        stats: dict with game_lengths
    """
    all_samples = []
    game_lengths = []
    total_evals = 0
    total_argmax_illegal = 0
    total_illegal_mass = 0.0

    for i in range(num_games):
        print(f"  Self-play game {i+1}/{num_games}", end="", flush=True)
        samples, illegal_stats = play_game(evaluate_fn, mcts_sims=mcts_sims, c_puct=c_puct, tau_threshold=tau_threshold)
        game_lengths.append(len(samples))
        all_samples.extend(samples)
        total_evals += illegal_stats["evals"]
        total_argmax_illegal += illegal_stats["argmax_illegal"]
        total_illegal_mass += illegal_stats["illegal_mass_sum"]
        print(f" - {len(samples)} moves")

    stats = {
        "num_games": num_games,
        "game_lengths": game_lengths,
        "avg_game_length": sum(game_lengths) / len(game_lengths) if game_lengths else 0,
        "total_samples": len(all_samples),
        "illegal": {
            "evals": total_evals,
            "argmax_illegal": total_argmax_illegal,
            "argmax_illegal_rate": (total_argmax_illegal / total_evals) if total_evals else 0.0,
            "mean_illegal_mass": (total_illegal_mass / total_evals) if total_evals else 0.0,
        },
    }
    return all_samples, stats


def play_match(eval_a, eval_b, num_games=20, mcts_sims=150, c_puct=1.0):
    """Headless head-to-head match between two evaluators.

    Plays num_games with alternating colors (eval_a white on even-indexed games).
    Greedy MCTS with no Dirichlet noise so play is deterministic-ish and reflects
    each agent's best play. Returns (a_wins, b_wins, draws).
    """
    a_wins, b_wins, draws = 0, 0, 0
    for game_idx in range(num_games):
        a_is_white = (game_idx % 2 == 0)
        white_eval = eval_a if a_is_white else eval_b
        black_eval = eval_b if a_is_white else eval_a

        game_state = initial_game_state()
        move_num = 0
        while not is_terminal(game_state.board) and move_num < MAX_MOVES:
            current_eval = white_eval if game_state.board.turn == WHITE else black_eval
            mcts = MCTS(current_eval, c_puct=c_puct, tau=0.01, dirichlet_epsilon=0.0)
            action = mcts.mcts_search(game_state, mcts_sims)
            move = index_to_move(action, game_state.board)
            game_state = game_state.apply_move(move)
            move_num += 1

        if is_terminal(game_state.board):
            result = game_state.board.result(claim_draw=True)
            if result == "1-0":
                if a_is_white:
                    a_wins += 1
                else:
                    b_wins += 1
            elif result == "0-1":
                if a_is_white:
                    b_wins += 1
                else:
                    a_wins += 1
            else:
                draws += 1
        else:
            draws += 1
    return a_wins, b_wins, draws
