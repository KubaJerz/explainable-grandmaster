import io
import tkinter as tk

import cairosvg
from PIL import Image, ImageTk

from utils.silverman import BOARD_HEIGHT, BOARD_WIDTH, BLACK, Board, WHITE, square


class ChessGUI:
    def __init__(self, board=None, ai_callback=None, ai_color="black", human_color="white"):
        self.board = board or Board()
        self.ai_callback = ai_callback
        self.ai_color = ai_color.lower()
        self.human_color = human_color.lower()
        self.selected_square = None
        self.legal_moves = []
        self.history = []
        self.flip = self.human_color == "black"
        self.square_size = 90

        self.root = tk.Tk()
        self.root.title("Silverman 5x4 Chess")

        self.piece_images = {}
        for piece_symbol in ["P", "R", "Q", "K", "p", "r", "q", "k"]:
            color = "w" if piece_symbol.isupper() else "b"
            filename = f"pieces/{color}{piece_symbol.upper()}.svg"
            with open(filename, "rb") as svg_file:
                png_data = cairosvg.svg2png(bytestring=svg_file.read())
            image = Image.open(io.BytesIO(png_data))
            image = image.resize((70, 70), Image.LANCZOS)
            self.piece_images[piece_symbol] = ImageTk.PhotoImage(image)

        canvas_width = BOARD_WIDTH * self.square_size
        canvas_height = BOARD_HEIGHT * self.square_size
        self.canvas = tk.Canvas(self.root, width=canvas_width, height=canvas_height)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_click)
        self.status_label = tk.Label(self.root, text="")
        self.status_label.pack()

        self.draw_board()
        self.update_status()

        if self._is_ai_turn() and self.ai_callback:
            self._perform_ai_move()

    def _display_file(self, file_index):
        return BOARD_WIDTH - 1 - file_index if self.flip else file_index

    def _display_rank(self, rank_index):
        return BOARD_HEIGHT - 1 - rank_index

    def draw_board(self):
        self.canvas.delete("all")
        for rank_index in range(BOARD_HEIGHT):
            for file_index in range(BOARD_WIDTH):
                display_file = self._display_file(file_index)
                display_rank = self._display_rank(rank_index)
                x1 = display_file * self.square_size
                y1 = display_rank * self.square_size
                x2 = x1 + self.square_size
                y2 = y1 + self.square_size
                color = "#b58863" if (file_index + rank_index) % 2 == 0 else "#f0d9b5"

                board_square = square(file_index, rank_index)
                if self.selected_square == board_square:
                    color = "#f7ec59"
                elif board_square in self.legal_moves:
                    color = "#7edc89"

                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")

                piece = self.board.piece_at(board_square)
                if piece:
                    self.canvas.create_image(
                        x1 + self.square_size // 2,
                        y1 + self.square_size // 2,
                        image=self.piece_images[piece.symbol()],
                        anchor="center",
                    )

    def update_status(self):
        if self.board.is_checkmate():
            winner = "White" if self.board.turn == BLACK else "Black"
            self.status_label.config(text=f"Checkmate! {winner} wins.")
        elif self.board.is_stalemate():
            self.status_label.config(text="Stalemate! Draw.")
        elif self.board.is_insufficient_material():
            self.status_label.config(text="Draw by insufficient material.")
        elif self.board.is_check():
            turn = "White" if self.board.turn == WHITE else "Black"
            self.status_label.config(text=f"{turn} is in check.")
        else:
            turn = "White" if self.board.turn == WHITE else "Black"
            self.status_label.config(text=f"{turn} to move.")

    def on_click(self, event):
        if not self._is_human_turn():
            return

        file_index = event.x // self.square_size
        rank_from_top = event.y // self.square_size
        if not (0 <= file_index < BOARD_WIDTH and 0 <= rank_from_top < BOARD_HEIGHT):
            return

        board_file = BOARD_WIDTH - 1 - file_index if self.flip else file_index
        board_rank = BOARD_HEIGHT - 1 - rank_from_top
        clicked_square = square(board_file, board_rank)

        if self.selected_square is None:
            piece = self.board.piece_at(clicked_square)
            if piece and piece.color == self.board.turn:
                self.selected_square = clicked_square
                self.legal_moves = [
                    move.to_square
                    for move in self.board.legal_moves
                    if move.from_square == clicked_square
                ]
                self.draw_board()
            return

        if clicked_square in self.legal_moves:
            for move in self.board.legal_moves:
                if move.from_square == self.selected_square and move.to_square == clicked_square:
                    self.make_move(move)
                    break

        self.selected_square = None
        self.legal_moves = []
        self.draw_board()

    def make_move(self, move):
        previous_board = self.board.copy()
        self.board.push(move)
        self.history = [previous_board] + self.history[:6]
        self.update_status()
        self.selected_square = None
        self.legal_moves = []
        self.draw_board()

        if self._is_ai_turn() and self.ai_callback and not self.board.is_game_over():
            self._perform_ai_move()

    def _perform_ai_move(self):
        self.status_label.config(text="AI is thinking...")
        self.root.update()
        game_state = self.create_game_state()
        ai_move = self.ai_callback(game_state)
        previous_board = self.board.copy()
        self.board.push(ai_move)
        self.history = [previous_board] + self.history[:6]
        self.update_status()
        self.draw_board()

    def _is_human_turn(self):
        return (self.board.turn == WHITE and self.human_color == "white") or (
            self.board.turn == BLACK and self.human_color == "black"
        )

    def _is_ai_turn(self):
        return (self.board.turn == WHITE and self.ai_color == "white") or (
            self.board.turn == BLACK and self.ai_color == "black"
        )

    def create_game_state(self):
        from utils.game_utils import GameState

        return GameState(self.board, self.history)

    def update_board(self, board):
        self.board = board
        self.history = []
        self.draw_board()
        self.update_status()

    def run(self):
        self.root.mainloop()
