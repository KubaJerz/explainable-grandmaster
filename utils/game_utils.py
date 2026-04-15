import torch

from utils.board_utils import get_board_spec_for_board


def board_to_tensor(board, history=None):
    return get_board_spec_for_board(board).encode(board, history)


class GameState:
    """Wraps a board object and optional history for NN encoding."""

    def __init__(self, board, history=None, board_spec=None):
        self.board = board
        self.board_spec = board_spec or get_board_spec_for_board(board)
        self.history = history or []

    def apply_move(self, move):
        new_board = self.board_spec.apply_move_copy(self.board, move)
        return GameState(new_board, [], self.board_spec)

    def encode(self):
        return self.board_spec.encode(self.board, self.history)


def move_to_index(move, board, board_spec=None):
    active_spec = board_spec or get_board_spec_for_board(board)
    return active_spec.move_to_index(move, board)


def index_to_move(index, board, board_spec=None):
    active_spec = board_spec or get_board_spec_for_board(board)
    return active_spec.index_to_move(index, board)


def make_output_valid(policy, board, board_spec=None):
    active_spec = board_spec or get_board_spec_for_board(board)
    mask = torch.zeros_like(policy)
    for move in active_spec.legal_moves(board):
        mask[active_spec.move_to_index(move, board)] = 1.0

    policy = policy * mask
    total = policy.sum()
    if total > 0:
        return policy / total
    return mask / mask.sum()


def is_terminal(board, board_spec=None):
    active_spec = board_spec or get_board_spec_for_board(board)
    return active_spec.is_terminal(board)


def terminal_state_evaluation(board, board_spec=None):
    active_spec = board_spec or get_board_spec_for_board(board)
    return active_spec.terminal_state_evaluation(board)
