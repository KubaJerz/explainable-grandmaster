from utils.silverman import BOARD_HEIGHT, BOARD_WIDTH, KING, PAWN, QUEEN, ROOK


PIECE_VALUES = {
    PAWN: 100,
    ROOK: 500,
    QUEEN: 900,
    KING: 20000,
}


def _table(rows):
    if len(rows) != BOARD_HEIGHT:
        raise ValueError("Piece-square table row count must match board height")
    for row in rows:
        if len(row) != BOARD_WIDTH:
            raise ValueError("Piece-square table column count must match board width")
    return rows


PIECE_SQUARE_TABLES = {
    PAWN: _table(
        [
            [0, 10, 20, 35, 50],
            [5, 15, 25, 40, 55],
            [5, 15, 25, 40, 55],
            [0, 10, 20, 35, 50],
        ]
    ),
    ROOK: _table(
        [
            [0, 5, 10, 10, 5],
            [5, 10, 15, 15, 10],
            [5, 10, 15, 15, 10],
            [0, 5, 10, 10, 5],
        ]
    ),
    QUEEN: _table(
        [
            [0, 10, 15, 15, 10],
            [10, 20, 25, 25, 20],
            [10, 20, 25, 25, 20],
            [0, 10, 15, 15, 10],
        ]
    ),
    KING: _table(
        [
            [20, 30, 20, 10, 0],
            [10, 20, 10, 0, -10],
            [10, 20, 10, 0, -10],
            [20, 30, 20, 10, 0],
        ]
    ),
}
