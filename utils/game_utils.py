import torch
import chess


"""
Utility functions for chess game state encoding, move encoding, and MCTS integration.

# The input representation is from Klein's "Neural Networks for Chess" (pg. 163-165).

So first 6 planes enncode white pieces (pawn, rook, knight, bishop, queen, king).
Next 6 encode black pieces.
The is repated for the last eight moves so 12*8 = 96 planes for board state + history.
Then 2 planes for repetition.
Then auxiliary planes for side to move, move count, castling rights, and halfmove clock.
Total 119 planes.
"""


# Piece order within each timestep's planes
PIECE_ORDER = [chess.PAWN, chess.ROOK,  chess.KNIGHT, chess.BISHOP, chess.QUEEN, chess.KING]


def board_to_tensor(board, history=None):
    """
    Encode a chess.Board (and optional history) into the 119-plane input representation.

    Plane layout per timestep (14 planes):
        0-5:   current player's pieces (P, R, N, B,  Q, K)
        6-11:  opponent's pieces (P, R, N, B,  Q, K)
        12:    1-fold repetition
        13:    2-fold repetition

    Additional planes (7):
        112: color (all 1s if white to move, all 0s if black)
        113: white kingside castling (These planes are set to all ones if the right to castling exists, and to zeros otherwise.)
        114: white queenside castling
        115: black kingside castling
        116: black queenside castling
        117: total move count (fullmove number, NOT normalized)
        118: move count without progress (for 50-move rule, NOT normalized)

    Args:
        board: chess.Board for the current position.
        history: list of up to 7 previous chess.Board states (most recent first).
                 If None, unfilled timesteps are zeros.

    Returns:
        torch.Tensor of shape (119, 8, 8)
    """
    tensor = torch.zeros(119, 8, 8, dtype=torch.float32)

    boards = [board] + (history or [])
    # Pad to 8 timesteps
    boards = boards[:8]

    current_color = board.turn

    for t, b in enumerate(boards):
        offset = t * 14
        _encode_pieces(tensor, offset, b, current_color)
        # Repetition planes
        if b.is_repetition(1):
            tensor[offset + 12] = 1.0
        if b.is_repetition(2):
            tensor[offset + 13] = 1.0

    # Additional info planes
    tensor[112] = 1.0 if current_color == chess.WHITE else 0.0
    tensor[113] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    tensor[114] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    tensor[115] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    tensor[116] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
    tensor[117] = board.fullmove_number 
    tensor[118] = board.halfmove_clock 

    return tensor


def _encode_pieces(tensor, offset, board, perspective):
    """Write piece planes into tensor at the given offset."""
    flip = (perspective == chess.BLACK)
    for i, piece_type in enumerate(PIECE_ORDER):
        # Current player's pieces
        for sq in board.pieces(piece_type, perspective):
            row, col = divmod(sq, 8)
            if flip:
                row = 7 - row
            tensor[offset + i][row][col] = 1.0
        # Opponent's pieces
        for sq in board.pieces(piece_type, not perspective):
            row, col = divmod(sq, 8)
            if flip:
                row = 7 - row
            tensor[offset + 6 + i][row][col] = 1.0


class GameState:
    """Wraps a chess.Board + history for NN encoding."""
    def __init__(self, board, history=None):
        self.board = board            # chess.Board (source of truth)
        self.history = history or []  # list of up to 7 previous chess.Boards

    def apply_move(self, move):
        new_history = [self.board.copy()] + self.history[:6]
        new_board = self.board.copy()
        new_board.push(move)
        return GameState(new_board, new_history)

    def encode(self):
        return board_to_tensor(self.board, self.history)


# ── Move encoding ──────────────────────────────────────────────────────
# As per Klein's implementation,  on page 165-167 of "Neural Networks for Chess".
# Policy vector has 4672 entries = 64 squares × 73 move types.
#
# Move types per source square (73 total):
#   0-55:  Queen-like moves — 8 directions × 7 distances
#   56-63: Knight moves — 8 possible L-shapes
#   64-72: Underpromotions — 3 directions (left-capture, forward, right-capture)
#                          × 3 piece types (knight, bishop, rook)
#          (queen promotions are encoded as normal queen-like moves)
#
# Directions for queen-like moves (index order):
#   0=N, 1=NE, 2=E, 3=SE, 4=S, 5=SW, 6=W, 7=NW

QUEEN_DIRECTIONS = [
    (1, 0),   # N
    (1, 1),   # NE
    (0, 1),   # E
    (-1, 1),  # SE
    (-1, 0),  # S
    (-1, -1), # SW
    (0, -1),  # W
    (1, -1),  # NW
]

KNIGHT_MOVES = [
    (2, 1), (2, -1), (-2, 1), (-2, -1),
    (1, 2), (1, -2), (-1, 2), (-1, -2),
]

UNDERPROMO_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]
UNDERPROMO_DIRECTIONS = [-1, 0, 1]  # file deltas: left-capture, forward, right-capture


def move_to_index(move):
    """Convert a chess.Move to a policy vector index (0-4671)."""
    #ranks = rows
    # files = columns
    from_sq = move.from_square
    to_sq = move.to_square

    from_rank, from_file = chess.square_rank(from_sq), chess.square_file(from_sq)
    to_rank, to_file = chess.square_rank(to_sq), chess.square_file(to_sq)

    distance_row = to_rank - from_rank
    distance_col = to_file - from_file

    # Underpromotions (knight, bishop, rook — queen promos use queen-like encoding)
    if move.promotion and move.promotion != chess.QUEEN:
        piece_idx = UNDERPROMO_PIECES.index(move.promotion)
        dir_idx = UNDERPROMO_DIRECTIONS.index(distance_col)
        move_type = 64 + dir_idx * 3 + piece_idx
        return from_sq * 73 + move_type

    # Knight moves
    if (distance_row, distance_col) in KNIGHT_MOVES:
        knight_idx = KNIGHT_MOVES.index((distance_row, distance_col))
        move_type = 56 + knight_idx
        return from_sq * 73 + move_type

    # Queen-like moves (includes queen promotions)
    distance = max(abs(distance_row), abs(distance_col))
    direction = (_sign(distance_row), _sign(distance_col))
    dir_idx = QUEEN_DIRECTIONS.index(direction)
    move_type = dir_idx * 7 + (distance - 1)
    return from_sq * 73 + move_type


def index_to_move(index, board):
    """Convert a policy vector index (0-4671) back to a chess.Move."""
    from_sq = index // 73
    move_type = index % 73

    from_rank = chess.square_rank(from_sq)
    from_file = chess.square_file(from_sq)

    # Underpromotions
    if move_type >= 64:
        underpromo_idx = move_type - 64
        dir_idx = underpromo_idx // 3
        piece_idx = underpromo_idx % 3
        df = UNDERPROMO_DIRECTIONS[dir_idx]
        # Pawns promote by moving one rank forward
        dr = 1 if board.turn == chess.WHITE else -1
        to_sq = chess.square(from_file + df, from_rank + dr)
        return chess.Move(from_sq, to_sq, promotion=UNDERPROMO_PIECES[piece_idx])

    # Knight moves
    if move_type >= 56:
        knight_idx = move_type - 56
        dr, df = KNIGHT_MOVES[knight_idx]
        to_sq = chess.square(from_file + df, from_rank + dr)
        return chess.Move(from_sq, to_sq)

    # Queen-like moves
    dir_idx = move_type // 7
    distance = (move_type % 7) + 1
    dr, df = QUEEN_DIRECTIONS[dir_idx]
    to_rank = from_rank + dr * distance
    to_file = from_file + df * distance
    to_sq = chess.square(to_file, to_rank)

    # Queen promotion: pawn reaching last rank with a queen-like move
    promotion = None
    is_pawn = board.piece_type_at(from_sq) == chess.PAWN
    last_rank = 7 if board.turn == chess.WHITE else 0
    if is_pawn and to_rank == last_rank:
        promotion = chess.QUEEN

    return chess.Move(from_sq, to_sq, promotion=promotion)


# ── Game utilities ─────────────────────────────────────────────────────

def make_output_valid(policy, board):
    """Mask illegal moves: zero out everything except legal moves, then renormalize."""
    mask = torch.zeros_like(policy)
    for move in board.legal_moves:
        mask[move_to_index(move)] = 1.0

    policy = policy * mask

    total = policy.sum()
    if total > 0:
        policy = policy / total
    else:
        policy = mask / mask.sum() #unifrom policy of all zeros

    return policy


def is_terminal(board):
    """Check if the game is over."""
    return board.is_game_over()


def terminal_state_evaluation(board):
    """Return game result from the perspective of the player to move.
    +1 = current player wins, -1 = current player loses, 0 = draw."""
    result = board.result()
    if result == "1/2-1/2":
        return 0.0
    if (result == "1-0" and board.turn == chess.WHITE) or (result == "0-1" and board.turn == chess.BLACK):
        return 1.0
    return -1.0


def _sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    return 0
