import torch

from utils.silverman import (
    BLACK,
    BOARD_HEIGHT,
    BOARD_WIDTH,
    KING,
    NUM_SQUARES,
    PAWN,
    QUEEN,
    ROOK,
    Board,
    Move,
    square_file,
    square_rank,
)


PIECE_ORDER = [PAWN, ROOK, QUEEN, KING]
NUM_HISTORY_STEPS = 4
PLANES_PER_STEP = len(PIECE_ORDER) * 2
INPUT_CHANNELS = NUM_HISTORY_STEPS * PLANES_PER_STEP + 5
ACTION_SIZE = NUM_SQUARES * NUM_SQUARES


def board_to_tensor(board, history=None):
    tensor = torch.zeros(INPUT_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH, dtype=torch.float32)

    boards = [board] + (history or [])
    boards = boards[:NUM_HISTORY_STEPS]
    current_color = board.turn

    for time_index, historic_board in enumerate(boards):
        offset = time_index * PLANES_PER_STEP
        _encode_pieces(tensor, offset, historic_board, current_color)

    tensor[NUM_HISTORY_STEPS * PLANES_PER_STEP] = 1.0 if board.is_repetition(1) else 0.0
    tensor[NUM_HISTORY_STEPS * PLANES_PER_STEP + 1] = 1.0 if board.is_repetition(2) else 0.0
    tensor[NUM_HISTORY_STEPS * PLANES_PER_STEP + 2] = 1.0 if current_color else 0.0
    tensor[NUM_HISTORY_STEPS * PLANES_PER_STEP + 3] = board.fullmove_number / 100.0
    tensor[NUM_HISTORY_STEPS * PLANES_PER_STEP + 4] = board.halfmove_clock / 100.0

    return tensor


def _encode_pieces(tensor, offset, board, perspective):
    flip_files = perspective == BLACK
    for piece_index, piece_type in enumerate(PIECE_ORDER):
        for square_index in board.pieces(piece_type, perspective):
            row = square_rank(square_index)
            col = square_file(square_index)
            if flip_files:
                col = BOARD_WIDTH - 1 - col
            tensor[offset + piece_index, row, col] = 1.0

        for square_index in board.pieces(piece_type, not perspective):
            row = square_rank(square_index)
            col = square_file(square_index)
            if flip_files:
                col = BOARD_WIDTH - 1 - col
            tensor[offset + len(PIECE_ORDER) + piece_index, row, col] = 1.0


class GameState:
    def __init__(self, board, history=None):
        self.board = board
        self.history = history or []

    def apply_move(self, move):
        new_history = [self.board.copy()] + self.history[: NUM_HISTORY_STEPS - 2]
        new_board = self.board.copy()
        new_board.push(move)
        return GameState(new_board, new_history)

    def encode(self):
        return board_to_tensor(self.board, self.history)


def move_to_index(move):
    return move.from_square * NUM_SQUARES + move.to_square


def index_to_move(index, board):
    from_square = index // NUM_SQUARES
    to_square = index % NUM_SQUARES
    piece_type = board.piece_type_at(from_square)
    promotion = None
    if piece_type == PAWN:
        target_file = square_file(to_square)
        if board.turn and target_file == BOARD_WIDTH - 1:
            promotion = QUEEN
        elif not board.turn and target_file == 0:
            promotion = QUEEN
    return Move(from_square, to_square, promotion=promotion)


def make_output_valid(policy, board):
    mask = torch.zeros_like(policy)
    for move in board.legal_moves:
        mask[move_to_index(move)] = 1.0

    policy = policy * mask
    total = policy.sum()
    if total > 0:
        return policy / total

    legal_total = mask.sum()
    if legal_total == 0:
        return mask
    return mask / legal_total


def is_terminal(board):
    return board.is_game_over(claim_draw=True)


def terminal_state_evaluation(board):
    result = board.result(claim_draw=True)
    if result == "1/2-1/2":
        return 0.0
    if (result == "1-0" and board.turn) or (result == "0-1" and not board.turn):
        return 1.0
    return -1.0


def initial_game_state():
    return GameState(Board())
