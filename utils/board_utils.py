from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import chess

from minichess.chess.fastchess import Chess as MiniChessBoard
from minichess.chess.fastchess_utils import (
    WHITE as MINI_WHITE,
    INVERSE_PIECE_LOOKUP,
    castling_masks,
    chess_move_to_uci,
    king_moves,
    knight_moves,
    load_board,
    pawn_attacks,
    pawn_moves_double,
    pawn_moves_single,
    piece_matrix_to_legal_moves,
    promotion_masks,
    straight_line_moves,
    diagonal_line_moves,
    uci_move_to_native_move,
)
from minichess.chess.magic import save_magic_bitboards
from minichess.chess.move_utils import (
    calculate_all_moves,
    flat_move_to_partial,
    move_to_index as mini_move_delta_to_index,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
MINICHESS_ROOT = REPO_ROOT / "minichess"
BOARDS_DIR = MINICHESS_ROOT / "boards"
MAGICS_DIR = MINICHESS_ROOT / "chess" / "magics"

STANDARD_INPUT_CHANNELS = 19
STANDARD_POLICY_SIZE = 4672
STANDARD_PIECE_ORDER = [
    chess.PAWN,
    chess.ROOK,
    chess.KNIGHT,
    chess.BISHOP,
    chess.QUEEN,
    chess.KING,
]
STANDARD_QUEEN_DIRECTIONS = [
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
]
STANDARD_KNIGHT_MOVES = [
    (2, 1), (2, -1), (-2, 1), (-2, -1),
    (1, 2), (1, -2), (-1, 2), (-1, -2),
]
STANDARD_UNDERPROMO_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]
STANDARD_UNDERPROMO_DIRECTIONS = [-1, 0, 1]


def _sign(x: int) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def _attach_spec(board, board_spec: "BoardSpec"):
    setattr(board, "_board_spec", board_spec)
    return board


def get_board_spec_for_board(board) -> "BoardSpec":
    board_spec = getattr(board, "_board_spec", None)
    if board_spec is not None:
        return board_spec
    if isinstance(board, chess.Board):
        return get_board_spec("8x8standard")
    raise ValueError("Board spec is required for non-standard board objects.")


def available_board_names() -> List[str]:
    return sorted(path.stem for path in BOARDS_DIR.glob("*.board"))


def _resolve_board_path(board_name: str) -> Path:
    candidate = Path(board_name)
    if candidate.suffix == ".board" and candidate.exists():
        return candidate.resolve()
    if candidate.exists() and candidate.is_file():
        return candidate.resolve()

    named = BOARDS_DIR / f"{board_name}.board"
    if named.exists():
        return named.resolve()

    raise ValueError(
        f"Unknown board '{board_name}'. Available boards: {', '.join(available_board_names())}"
    )


def add_board_argument(parser):
    parser.add_argument(
        "--board",
        type=str,
        default="8x8standard",
        help="Board variant to use. Can be a board name like 5x4silverman or a path to a .board file.",
    )
    return parser


@dataclass(frozen=True)
class BoardSpec:
    name: str
    rows: int
    cols: int
    input_channels: int
    policy_size: int
    move_cap: int
    is_standard: bool
    board_path: Optional[str] = None
    all_moves: Optional[np.ndarray] = None
    all_moves_inv: Optional[np.ndarray] = None

    def create_board(self):
        if self.is_standard:
            return _attach_spec(chess.Board(), self)

        board_path = Path(self.board_path)
        board_stem = str(board_path.with_suffix(""))
        bitboards, piece_lookup, dims = load_board(board_stem)
        diag_hash, diag_magics, diag_shift, straight_hash, straight_magics, straight_shift = _load_or_create_magics(dims)
        empty_masks, attack_masks, castling_rights = castling_masks(dims, board_stem)
        board = MiniChessBoard(
            bitboards=bitboards,
            piece_lookup=piece_lookup,
            dims=dims,
            diagonal_hash=diag_hash,
            diagonal_magics=diag_magics,
            diagonal_shift=diag_shift,
            straight_hash=straight_hash,
            straight_magics=straight_magics,
            straight_shift=straight_shift,
            PAWN_MOVES_SINGLE=pawn_moves_single(dims),
            PAWN_MOVES_DOUBLE=pawn_moves_double(dims),
            PAWN_ATTACKS=pawn_attacks(dims),
            KNIGHT_MOVES=knight_moves(dims),
            KING_MOVES=king_moves(dims),
            DIAGONAL_MOVES=diagonal_line_moves(dims),
            STRAIGHT_MOVES=straight_line_moves(dims),
            CASTLING_EMPTY_MASKS=empty_masks,
            CASTLING_ATTACK_MASKS=attack_masks,
            PROMOTION_MASKS=promotion_masks(dims),
            castling_rights=castling_rights,
        )
        return _attach_spec(board, self)

    def copy_board(self, board):
        return _attach_spec(board.copy(), self)

    def apply_move_copy(self, board, move):
        new_board = self.copy_board(board)
        self.apply_move_inplace(new_board, move)
        return _attach_spec(new_board, self)

    def apply_move_inplace(self, board, move):
        if self.is_standard:
            board.push(move)
        else:
            i, j, dx, dy, promotion = move
            board.make_move(i, j, dx, dy, promotion)
        return _attach_spec(board, self)

    def encode(self, board, history=None) -> torch.Tensor:
        if self.is_standard:
            return _standard_board_to_tensor(board, history)
        state_array = np.ascontiguousarray(board.agent_board_state())
        state = torch.from_numpy(state_array).permute(2, 0, 1).to(torch.float32)
        return state

    def turn_is_white(self, board) -> bool:
        return bool(board.turn)

    def is_terminal(self, board) -> bool:
        if self.is_standard:
            return board.is_game_over(claim_draw=True)
        return board.game_result() is not None

    def result(self, board):
        if self.is_standard:
            return board.result(claim_draw=True)
        return board.game_result()

    def terminal_state_evaluation(self, board) -> float:
        if self.is_standard:
            result = board.result(claim_draw=True)
            if result == "1/2-1/2":
                return 0.0
            if (result == "1-0" and board.turn == chess.WHITE) or (result == "0-1" and board.turn == chess.BLACK):
                return 1.0
            return -1.0

        result = board.game_result()
        if result is None or result == 0:
            return 0.0
        return float(result) if board.turn == MINI_WHITE else -float(result)

    def value_target_from_white_result(self, white_result: float, side_to_move_is_white: bool) -> float:
        return white_result if side_to_move_is_white else -white_result

    def white_result_from_terminal(self, board) -> float:
        if self.is_standard:
            result = board.result(claim_draw=True)
            if result == "1-0":
                return 1.0
            if result == "0-1":
                return -1.0
            return 0.0

        result = board.game_result()
        if result is None or result == 0:
            return 0.0
        return float(result)

    def legal_moves(self, board) -> List:
        if self.is_standard:
            return list(board.legal_moves)
        moves, promotions = board.legal_moves()
        return [self._normalize_mini_move(move) for move in piece_matrix_to_legal_moves(moves, promotions)]

    def move_to_index(self, move, board) -> int:
        if self.is_standard:
            return _standard_move_to_index(move)

        i, j, dx, dy, promotion = self._normalize_mini_move(move)
        move_index = mini_move_delta_to_index(self.all_moves, dx, dy, promotion, board.turn)
        return int(np.ravel_multi_index((i, j, move_index), (self.rows, self.cols, self.move_cap)))

    def index_to_move(self, index: int, board):
        if self.is_standard:
            return _standard_index_to_move(index, board)

        i, j, dx, dy, promotion = flat_move_to_partial(self.all_moves_inv, (self.rows, self.cols), int(index), board.turn)
        return (int(i), int(j), int(dx), int(dy), int(promotion))

    def uci(self, move, board) -> str:
        if self.is_standard:
            return move.uci()
        return chess_move_to_uci(self._tuple_to_native_move(move), board.dims)

    def parse_uci(self, uci_move: str, board):
        if self.is_standard:
            return chess.Move.from_uci(uci_move)
        return self._normalize_mini_move(uci_move_to_native_move(uci_move, board))

    def piece_symbol_at(self, board, row: int, col: int) -> Optional[str]:
        if self.is_standard:
            square = chess.square(col, self.rows - 1 - row)
            piece = board.piece_at(square)
            return piece.symbol() if piece else None

        piece, color = board.any_piece_at(row, col)
        if piece == -1:
            return None
        symbol = INVERSE_PIECE_LOOKUP[piece]
        return symbol.upper() if color == MINI_WHITE else symbol

    def legal_moves_from(self, board, row: int, col: int) -> Dict[Tuple[int, int], object]:
        if self.is_standard:
            square = chess.square(col, self.rows - 1 - row)
            return {
                (self.rows - 1 - chess.square_rank(move.to_square), chess.square_file(move.to_square)): move
                for move in board.legal_moves
                if move.from_square == square
            }

        move_map: Dict[Tuple[int, int], Tuple[int, int, int, int, int]] = {}
        for move in self.legal_moves(board):
            i, j, dx, dy, promotion = move
            if i != row or j != col:
                continue
            target = (i + dx, j + dy)
            existing = move_map.get(target)
            if existing is None or promotion > existing[4]:
                move_map[target] = move
        return move_map

    def status_text(self, board) -> str:
        if self.is_standard:
            if board.is_checkmate():
                winner = "White" if board.turn == chess.BLACK else "Black"
                return f"Checkmate! {winner} wins."
            if board.is_stalemate():
                return "Stalemate! Draw."
            if board.is_insufficient_material():
                return "Draw by insufficient material."
            if board.is_check():
                turn = "White" if board.turn == chess.WHITE else "Black"
                return f"{turn} is in check."
            turn = "White" if board.turn == chess.WHITE else "Black"
            return f"{turn} to move."

        result = board.game_result()
        if result == 1:
            return "Checkmate! White wins."
        if result == -1:
            return "Checkmate! Black wins."
        if result == 0:
            return "Draw."
        _, _ = board.legal_moves()
        if board.any_checkers:
            turn = "White" if board.turn == MINI_WHITE else "Black"
            return f"{turn} is in check."
        turn = "White" if board.turn == MINI_WHITE else "Black"
        return f"{turn} to move."

    def is_game_over(self, board) -> bool:
        if self.is_standard:
            return board.is_game_over(claim_draw=True)
        return board.game_result() is not None

    def _tuple_to_native_move(self, move):
        i, j, dx, dy, promotion = self._normalize_mini_move(move)
        return (np.int8(i), np.int8(j)), (np.int8(dx), np.int8(dy)), np.int8(promotion)

    @staticmethod
    def _normalize_mini_move(move) -> Tuple[int, int, int, int, int]:
        if len(move) == 5:
            return tuple(int(v) for v in move)
        (origin, delta, promotion) = move
        return int(origin[0]), int(origin[1]), int(delta[0]), int(delta[1]), int(promotion)


def get_board_spec(board_name: str = "8x8standard") -> BoardSpec:
    board_path = _resolve_board_path(board_name)
    board_stem = board_path.stem
    with board_path.open() as handle:
        lines = handle.readlines()
    rows = len(lines)
    first_line = ""
    if lines:
        first_line = lines[0].rstrip("\n")
    cols = len(first_line)

    if board_stem == "8x8standard":
        return BoardSpec(
            name=board_stem,
            rows=rows,
            cols=cols,
            input_channels=STANDARD_INPUT_CHANNELS,
            policy_size=STANDARD_POLICY_SIZE,
            move_cap=73,
            is_standard=True,
            board_path=str(board_path),
        )

    all_moves, all_moves_inv = calculate_all_moves((rows, cols))
    move_cap = int(all_moves_inv.shape[0])
    return BoardSpec(
        name=board_stem,
        rows=rows,
        cols=cols,
        input_channels=19,
        policy_size=rows * cols * move_cap,
        move_cap=move_cap,
        is_standard=False,
        board_path=str(board_path),
        all_moves=all_moves,
        all_moves_inv=all_moves_inv,
    )


def validate_checkpoint_board(checkpoint_args, board_spec: BoardSpec):
    checkpoint_board = checkpoint_args.get("board", "8x8standard")
    if checkpoint_board != board_spec.name:
        raise ValueError(
            f"Checkpoint was trained for board '{checkpoint_board}', but '{board_spec.name}' was requested."
        )


def _load_or_create_magics(dims):
    magics_path = MAGICS_DIR / f"{dims[0]}x{dims[1]}"
    diagonals_path = magics_path / "diagonals.npz"
    straights_path = magics_path / "straights.npz"

    if not diagonals_path.exists() or not straights_path.exists():
        save_magic_bitboards(dims, str(MINICHESS_ROOT))

    diag_data = np.load(diagonals_path)
    straight_data = np.load(straights_path)
    return (
        diag_data["hash_table"],
        diag_data["magics"],
        int(diag_data["shift"]),
        straight_data["hash_table"],
        straight_data["magics"],
        int(straight_data["shift"]),
    )


def _standard_board_to_tensor(board, history=None):
    tensor = torch.zeros(STANDARD_INPUT_CHANNELS, 8, 8, dtype=torch.float32)

    current_color = board.turn

    # Channels 0-11: piece planes for current board only
    _encode_standard_pieces(tensor, 0, board, current_color)

    # Channels 12-15: castling rights
    tensor[12] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    tensor[13] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    tensor[14] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    tensor[15] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0

    # Channel 16: en passant square
    if board.ep_square is not None:
        row, col = divmod(board.ep_square, 8)
        if current_color == chess.BLACK:
            row = 7 - row
        tensor[16][row][col] = 1.0

    # Channel 17: halfmove clock (normalized)
    tensor[17] = board.halfmove_clock / 20.0

    # Channel 18: turn indicator
    tensor[18] = 1.0 if current_color == chess.WHITE else 0.0

    return tensor


def _encode_standard_pieces(tensor, offset, board, perspective):
    flip = perspective == chess.BLACK
    for i, piece_type in enumerate(STANDARD_PIECE_ORDER):
        for sq in board.pieces(piece_type, perspective):
            row, col = divmod(sq, 8)
            if flip:
                row = 7 - row
            tensor[offset + i][row][col] = 1.0
        for sq in board.pieces(piece_type, not perspective):
            row, col = divmod(sq, 8)
            if flip:
                row = 7 - row
            tensor[offset + 6 + i][row][col] = 1.0


def _standard_move_to_index(move):
    from_sq = move.from_square
    to_sq = move.to_square

    from_rank, from_file = chess.square_rank(from_sq), chess.square_file(from_sq)
    to_rank, to_file = chess.square_rank(to_sq), chess.square_file(to_sq)
    distance_row = to_rank - from_rank
    distance_col = to_file - from_file

    if move.promotion and move.promotion != chess.QUEEN:
        piece_idx = STANDARD_UNDERPROMO_PIECES.index(move.promotion)
        dir_idx = STANDARD_UNDERPROMO_DIRECTIONS.index(distance_col)
        move_type = 64 + dir_idx * 3 + piece_idx
        return from_sq * 73 + move_type

    if (distance_row, distance_col) in STANDARD_KNIGHT_MOVES:
        knight_idx = STANDARD_KNIGHT_MOVES.index((distance_row, distance_col))
        move_type = 56 + knight_idx
        return from_sq * 73 + move_type

    distance = max(abs(distance_row), abs(distance_col))
    direction = (_sign(distance_row), _sign(distance_col))
    dir_idx = STANDARD_QUEEN_DIRECTIONS.index(direction)
    move_type = dir_idx * 7 + (distance - 1)
    return from_sq * 73 + move_type


def _standard_index_to_move(index, board):
    from_sq = index // 73
    move_type = index % 73

    from_rank = chess.square_rank(from_sq)
    from_file = chess.square_file(from_sq)

    if move_type >= 64:
        underpromo_idx = move_type - 64
        dir_idx = underpromo_idx // 3
        piece_idx = underpromo_idx % 3
        df = STANDARD_UNDERPROMO_DIRECTIONS[dir_idx]
        dr = 1 if board.turn == chess.WHITE else -1
        to_sq = chess.square(from_file + df, from_rank + dr)
        return chess.Move(from_sq, to_sq, promotion=STANDARD_UNDERPROMO_PIECES[piece_idx])

    if move_type >= 56:
        knight_idx = move_type - 56
        dr, df = STANDARD_KNIGHT_MOVES[knight_idx]
        to_sq = chess.square(from_file + df, from_rank + dr)
        return chess.Move(from_sq, to_sq)

    dir_idx = move_type // 7
    distance = (move_type % 7) + 1
    dr, df = STANDARD_QUEEN_DIRECTIONS[dir_idx]
    to_rank = from_rank + dr * distance
    to_file = from_file + df * distance
    to_sq = chess.square(to_file, to_rank)

    promotion = None
    is_pawn = board.piece_type_at(from_sq) == chess.PAWN
    last_rank = 7 if board.turn == chess.WHITE else 0
    if is_pawn and to_rank == last_rank:
        promotion = chess.QUEEN

    return chess.Move(from_sq, to_sq, promotion=promotion)
