from __future__ import annotations

from dataclasses import dataclass


BOARD_WIDTH = 5
BOARD_HEIGHT = 4
NUM_SQUARES = BOARD_WIDTH * BOARD_HEIGHT

WHITE = True
BLACK = False

PAWN = 1
ROOK = 2
QUEEN = 3
KING = 4

PIECE_SYMBOLS = {
    PAWN: "P",
    ROOK: "R",
    QUEEN: "Q",
    KING: "K",
}

ORTHOGONAL_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIAGONAL_DIRS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
QUEEN_DIRS = ORTHOGONAL_DIRS + DIAGONAL_DIRS
KING_DIRS = QUEEN_DIRS

WHITE_PROMOTION_FILE = BOARD_WIDTH - 1
BLACK_PROMOTION_FILE = 0


def square(file_index: int, rank_index: int) -> int:
    return rank_index * BOARD_WIDTH + file_index


def square_file(square_index: int) -> int:
    return square_index % BOARD_WIDTH


def square_rank(square_index: int) -> int:
    return square_index // BOARD_WIDTH


def square_name(square_index: int) -> str:
    return f"{chr(ord('a') + square_file(square_index))}{square_rank(square_index) + 1}"


def on_board(file_index: int, rank_index: int) -> bool:
    return 0 <= file_index < BOARD_WIDTH and 0 <= rank_index < BOARD_HEIGHT


@dataclass(frozen=True)
class Piece:
    piece_type: int
    color: bool

    def symbol(self) -> str:
        symbol = PIECE_SYMBOLS[self.piece_type]
        return symbol if self.color == WHITE else symbol.lower()


@dataclass(frozen=True)
class Move:
    from_square: int
    to_square: int
    promotion: int | None = None

    def uci(self) -> str:
        suffix = ""
        if self.promotion is not None:
            suffix = PIECE_SYMBOLS[self.promotion].lower()
        return f"{square_name(self.from_square)}{square_name(self.to_square)}{suffix}"


class Board:
    def __init__(
        self,
        squares: list[Piece | None] | None = None,
        turn: bool = WHITE,
        halfmove_clock: int = 0,
        fullmove_number: int = 1,
        position_counts: dict[tuple, int] | None = None,
    ):
        if squares is None:
            self.squares = [None] * NUM_SQUARES
            self.turn = WHITE
            self.halfmove_clock = 0
            self.fullmove_number = 1
            self._setup_start_position()
            self.position_counts: dict[tuple, int] = {}
            self._record_position()
        else:
            self.squares = list(squares)
            self.turn = turn
            self.halfmove_clock = halfmove_clock
            self.fullmove_number = fullmove_number
            self.position_counts = dict(position_counts or {})

    def _setup_start_position(self) -> None:
        home_rank_pieces = [ROOK, QUEEN, KING, ROOK]
        for rank_index, piece_type in enumerate(home_rank_pieces):
            self.set_piece_at(square(0, rank_index), Piece(piece_type, WHITE))
            self.set_piece_at(square(1, rank_index), Piece(PAWN, WHITE))
            self.set_piece_at(square(3, rank_index), Piece(PAWN, BLACK))
            self.set_piece_at(square(4, rank_index), Piece(piece_type, BLACK))

    def copy(self) -> "Board":
        return Board(
            squares=self.squares,
            turn=self.turn,
            halfmove_clock=self.halfmove_clock,
            fullmove_number=self.fullmove_number,
            position_counts=self.position_counts,
        )

    def piece_at(self, square_index: int) -> Piece | None:
        return self.squares[square_index]

    def set_piece_at(self, square_index: int, piece: Piece | None) -> None:
        self.squares[square_index] = piece

    def piece_type_at(self, square_index: int) -> int | None:
        piece = self.piece_at(square_index)
        return piece.piece_type if piece else None

    def pieces(self, piece_type: int, color: bool) -> list[int]:
        return [
            square_index
            for square_index, piece in enumerate(self.squares)
            if piece is not None and piece.piece_type == piece_type and piece.color == color
        ]

    @property
    def legal_moves(self) -> list[Move]:
        moves = []
        for move in self._generate_pseudo_legal_moves(self.turn):
            test_board = self.copy()
            test_board._push_no_validation(move)
            if not test_board.is_in_check(self.turn):
                moves.append(move)
        return moves

    def push(self, move: Move) -> None:
        if move not in self.legal_moves:
            raise ValueError(f"Illegal move: {move.uci()}")
        self._push_no_validation(move)
        self._record_position()

    def _push_no_validation(self, move: Move) -> None:
        moving_piece = self.piece_at(move.from_square)
        if moving_piece is None:
            raise ValueError("No piece on source square")

        target_piece = self.piece_at(move.to_square)
        self.set_piece_at(move.from_square, None)
        if move.promotion is not None:
            moving_piece = Piece(move.promotion, moving_piece.color)
        self.set_piece_at(move.to_square, moving_piece)

        if moving_piece.piece_type == PAWN or target_piece is not None:
            self.halfmove_clock = 0
        else:
            self.halfmove_clock += 1

        if self.turn == BLACK:
            self.fullmove_number += 1
        self.turn = not self.turn

    def is_check(self) -> bool:
        return self.is_in_check(self.turn)

    def is_in_check(self, color: bool) -> bool:
        king_squares = self.pieces(KING, color)
        if not king_squares:
            return True
        return self.is_square_attacked(king_squares[0], not color)

    def is_square_attacked(self, square_index: int, by_color: bool) -> bool:
        target_file = square_file(square_index)
        target_rank = square_rank(square_index)

        for from_square in self.pieces(PAWN, by_color):
            file_index = square_file(from_square)
            rank_index = square_rank(from_square)
            forward = 1 if by_color == WHITE else -1
            for rank_delta in (-1, 1):
                attack_file = file_index + forward
                attack_rank = rank_index + rank_delta
                if attack_file == target_file and attack_rank == target_rank:
                    return True

        for from_square in self.pieces(KING, by_color):
            file_index = square_file(from_square)
            rank_index = square_rank(from_square)
            for file_delta, rank_delta in KING_DIRS:
                if file_index + file_delta == target_file and rank_index + rank_delta == target_rank:
                    return True

        for from_square in self.pieces(ROOK, by_color):
            if self._ray_attacks_square(from_square, square_index, ORTHOGONAL_DIRS):
                return True

        for from_square in self.pieces(QUEEN, by_color):
            if self._ray_attacks_square(from_square, square_index, QUEEN_DIRS):
                return True

        return False

    def _ray_attacks_square(self, from_square: int, target_square: int, directions: tuple[tuple[int, int], ...]) -> bool:
        file_index = square_file(from_square)
        rank_index = square_rank(from_square)
        for file_delta, rank_delta in directions:
            current_file = file_index + file_delta
            current_rank = rank_index + rank_delta
            while on_board(current_file, current_rank):
                current_square = square(current_file, current_rank)
                if current_square == target_square:
                    return True
                if self.piece_at(current_square) is not None:
                    break
                current_file += file_delta
                current_rank += rank_delta
        return False

    def _generate_pseudo_legal_moves(self, color: bool) -> list[Move]:
        moves = []
        for from_square, piece in enumerate(self.squares):
            if piece is None or piece.color != color:
                continue
            if piece.piece_type == PAWN:
                moves.extend(self._pawn_moves(from_square, piece))
            elif piece.piece_type == ROOK:
                moves.extend(self._sliding_moves(from_square, color, ORTHOGONAL_DIRS))
            elif piece.piece_type == QUEEN:
                moves.extend(self._sliding_moves(from_square, color, QUEEN_DIRS))
            elif piece.piece_type == KING:
                moves.extend(self._king_moves(from_square, color))
        return moves

    def _pawn_moves(self, from_square: int, piece: Piece) -> list[Move]:
        moves = []
        file_index = square_file(from_square)
        rank_index = square_rank(from_square)
        forward = 1 if piece.color == WHITE else -1
        next_file = file_index + forward

        if on_board(next_file, rank_index):
            to_square = square(next_file, rank_index)
            if self.piece_at(to_square) is None:
                moves.append(self._make_move(from_square, to_square, piece))

        for rank_delta in (-1, 1):
            target_rank = rank_index + rank_delta
            if not on_board(next_file, target_rank):
                continue
            to_square = square(next_file, target_rank)
            target_piece = self.piece_at(to_square)
            if target_piece is not None and target_piece.color != piece.color:
                moves.append(self._make_move(from_square, to_square, piece))

        return moves

    def _sliding_moves(self, from_square: int, color: bool, directions: tuple[tuple[int, int], ...]) -> list[Move]:
        moves = []
        file_index = square_file(from_square)
        rank_index = square_rank(from_square)
        for file_delta, rank_delta in directions:
            current_file = file_index + file_delta
            current_rank = rank_index + rank_delta
            while on_board(current_file, current_rank):
                to_square = square(current_file, current_rank)
                target_piece = self.piece_at(to_square)
                if target_piece is None:
                    moves.append(Move(from_square, to_square))
                else:
                    if target_piece.color != color:
                        moves.append(Move(from_square, to_square))
                    break
                current_file += file_delta
                current_rank += rank_delta
        return moves

    def _king_moves(self, from_square: int, color: bool) -> list[Move]:
        moves = []
        file_index = square_file(from_square)
        rank_index = square_rank(from_square)
        for file_delta, rank_delta in KING_DIRS:
            target_file = file_index + file_delta
            target_rank = rank_index + rank_delta
            if not on_board(target_file, target_rank):
                continue
            to_square = square(target_file, target_rank)
            target_piece = self.piece_at(to_square)
            if target_piece is None or target_piece.color != color:
                moves.append(Move(from_square, to_square))
        return moves

    def _make_move(self, from_square: int, to_square: int, piece: Piece) -> Move:
        promotion = None
        if piece.piece_type == PAWN:
            target_file = square_file(to_square)
            if piece.color == WHITE and target_file == WHITE_PROMOTION_FILE:
                promotion = QUEEN
            elif piece.color == BLACK and target_file == BLACK_PROMOTION_FILE:
                promotion = QUEEN
        return Move(from_square, to_square, promotion=promotion)

    def is_checkmate(self) -> bool:
        return self.is_check() and len(self.legal_moves) == 0

    def is_stalemate(self) -> bool:
        return not self.is_check() and len(self.legal_moves) == 0

    def is_insufficient_material(self) -> bool:
        non_kings = [piece for piece in self.squares if piece is not None and piece.piece_type != KING]
        return len(non_kings) == 0

    def is_repetition(self, count: int) -> bool:
        return self.position_counts.get(self._position_key(), 0) >= count + 1

    def is_game_over(self, claim_draw: bool = True) -> bool:
        if self.is_checkmate() or self.is_stalemate() or self.is_insufficient_material():
            return True
        if claim_draw and (self.is_repetition(2) or self.halfmove_clock >= 40):
            return True
        return False

    def result(self, claim_draw: bool = True) -> str:
        if self.is_checkmate():
            return "0-1" if self.turn == WHITE else "1-0"
        if self.is_stalemate() or self.is_insufficient_material():
            return "1/2-1/2"
        if claim_draw and (self.is_repetition(2) or self.halfmove_clock >= 40):
            return "1/2-1/2"
        return "*"

    def _position_key(self) -> tuple:
        board_key = tuple(
            None if piece is None else (piece.piece_type, piece.color)
            for piece in self.squares
        )
        return board_key, self.turn

    def _record_position(self) -> None:
        key = self._position_key()
        self.position_counts[key] = self.position_counts.get(key, 0) + 1
