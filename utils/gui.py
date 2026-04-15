import io
import tkinter as tk

import cairosvg
from PIL import Image, ImageTk

from utils.game_utils import GameState


class ChessGUI:
    def __init__(self, board_spec, board=None, ai_callback=None, ai_color="black", human_color="white"):
        self.board_spec = board_spec
        self.board = board or board_spec.create_board()
        self.ai_callback = ai_callback
        self.ai_color = ai_color.lower()
        self.human_color = human_color.lower()
        self.selected_square = None
        self.legal_moves = {}
        self.history = []
        self.flip = self.human_color == "black"

        self.square_size = 50
        self.root = tk.Tk()
        self.root.title(f"Chess Board - {board_spec.name}")

        self.piece_images = {}
        for piece_symbol in ["P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"]:
            color = "w" if piece_symbol.isupper() else "b"
            filename = f"pieces/{color}{piece_symbol.upper()}.svg"
            with open(filename, "rb") as handle:
                svg_data = handle.read()
            png_data = cairosvg.svg2png(bytestring=svg_data)
            image = Image.open(io.BytesIO(png_data))
            image = image.resize((int(self.square_size * 0.8), int(self.square_size * 0.8)), Image.LANCZOS)
            self.piece_images[piece_symbol] = ImageTk.PhotoImage(image)

        canvas_width = self.board_spec.cols * self.square_size
        canvas_height = self.board_spec.rows * self.square_size
        self.canvas = tk.Canvas(self.root, width=canvas_width, height=canvas_height)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_click)
        self.status_label = tk.Label(self.root, text="")
        self.status_label.pack()
        self.draw_board()
        self.update_status()

        if self._is_ai_turn():
            self._make_ai_move()

    def draw_board(self):
        self.canvas.delete("all")
        for display_row in range(self.board_spec.rows):
            for display_col in range(self.board_spec.cols):
                board_row, board_col = self._display_to_board(display_row, display_col)
                x1 = display_col * self.square_size
                y1 = display_row * self.square_size
                x2 = x1 + self.square_size
                y2 = y1 + self.square_size
                color = "#b58863" if (board_row + board_col) % 2 == 0 else "#f0d9b5"

                if self.selected_square == (board_row, board_col):
                    color = "#ffff00"
                if (board_row, board_col) in self.legal_moves:
                    color = "#00ff00"

                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color)

                piece_symbol = self.board_spec.piece_symbol_at(self.board, board_row, board_col)
                if piece_symbol:
                    self.canvas.create_image(
                        x1 + self.square_size // 2,
                        y1 + self.square_size // 2,
                        image=self.piece_images[piece_symbol],
                        anchor="center",
                    )

    def update_status(self):
        self.status_label.config(text=self.board_spec.status_text(self.board))

    def on_click(self, event):
        display_col = event.x // self.square_size
        display_row = event.y // self.square_size
        if display_col < 0 or display_col >= self.board_spec.cols or display_row < 0 or display_row >= self.board_spec.rows:
            return
        row, col = self._display_to_board(display_row, display_col)

        if not self._is_human_turn():
            return

        if self.selected_square is None:
            legal_moves = self.board_spec.legal_moves_from(self.board, row, col)
            if legal_moves:
                self.selected_square = (row, col)
                self.legal_moves = legal_moves
                self.draw_board()
            return

        move = self.legal_moves.get((row, col))
        self.selected_square = None
        self.legal_moves = {}
        if move is not None:
            self.make_move(move)
        else:
            self.draw_board()

    def make_move(self, move):
        self.board_spec.apply_move_inplace(self.board, move)
        self.history = []
        self.update_status()
        self.selected_square = None
        self.legal_moves = {}
        self.draw_board()

        if self._is_ai_turn() and not self.board_spec.is_game_over(self.board):
            self._make_ai_move()

    def create_game_state(self):
        return GameState(self.board, self.history, self.board_spec)

    def update_board(self, board):
        self.board = board
        self.history = []
        self.draw_board()
        self.update_status()

    def run(self):
        self.root.mainloop()

    def _is_human_turn(self):
        return self._turn_color() == self.human_color

    def _is_ai_turn(self):
        return self.ai_callback is not None and self._turn_color() == self.ai_color

    def _make_ai_move(self):
        self.status_label.config(text="AI is thinking...")
        self.root.update()
        ai_move = self.ai_callback(self.create_game_state())
        self.board_spec.apply_move_inplace(self.board, ai_move)
        self.history = []
        self.update_status()
        self.draw_board()

    def _turn_color(self):
        return "white" if self.board_spec.turn_is_white(self.board) else "black"

    def _display_to_board(self, display_row, display_col):
        if self.flip:
            return self.board_spec.rows - 1 - display_row, self.board_spec.cols - 1 - display_col
        return display_row, display_col
