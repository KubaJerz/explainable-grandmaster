import tkinter as tk
import chess
import io
from PIL import Image, ImageTk
import cairosvg


class ChessGUI:
    def __init__(self, board=None, ai_callback=None, ai_color='black', human_color='white'):
        self.board = board or chess.Board()
        self.ai_callback = ai_callback  # Function to get AI move: ai_callback(game_state) -> move
        self.ai_color = ai_color.lower()
        self.human_color = human_color.lower()
        self.selected_square = None
        self.legal_moves = []
        self.history = []
        self.flip = (self.human_color == 'black')  # Flip if human plays black

        self.root = tk.Tk()
        self.root.title("Chess Board")

        # Load piece images
        self.piece_images = {}
        for piece_symbol in ['P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']:
            color = 'w' if piece_symbol.isupper() else 'b'
            filename = f"pieces/{color}{piece_symbol.upper()}.svg"
            with open(filename, 'rb') as f:
                svg_data = f.read()
            png_data = cairosvg.svg2png(bytestring=svg_data)
            image = Image.open(io.BytesIO(png_data))
            image = image.resize((40, 40), Image.LANCZOS)
            self.piece_images[piece_symbol] = ImageTk.PhotoImage(image)

        self.canvas = tk.Canvas(self.root, width=400, height=400)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_click)
        self.status_label = tk.Label(self.root, text="")
        self.status_label.pack()
        self.draw_board()
        self.update_status()

        # If it's AI's turn at the start, make AI move
        human_turn = (self.board.turn == chess.WHITE and self.human_color == 'white') or \
                     (self.board.turn == chess.BLACK and self.human_color == 'black')
        if not human_turn and self.ai_callback:
            self.status_label.config(text="AI is thinking...")
            self.root.update()
            game_state = self.create_game_state()
            ai_move = self.ai_callback(game_state)
            self.board.push(ai_move)
            self.history = [self.board.copy()] + self.history[:6]
            self.update_status()
            self.draw_board()

    def draw_board(self):
        self.canvas.delete("all")
        square_size = 50
        for row in range(8):
            for col in range(8):
                x1 = col * square_size
                if self.flip:
                    y1 = row * square_size  # Black at bottom
                else:
                    y1 = (7 - row) * square_size  # White at bottom
                x2 = x1 + square_size
                y2 = y1 + square_size
                color = "#b58863" if (row + col) % 2 == 0 else "#f0d9b5"
                
                # Highlight selected square
                if self.selected_square is not None and chess.square(col, row) == self.selected_square:
                    color = "#ffff00"  # Yellow for selected
                
                # Highlight legal moves
                square = chess.square(col, row)
                if square in self.legal_moves:
                    color = "#00ff00"  # Green for legal moves
                
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color)

                # Draw piece
                piece = self.board.piece_at(square)
                if piece:
                    img = self.piece_images[piece.symbol()]
                    self.canvas.create_image(x1 + square_size//2, y1 + square_size//2,
                                           image=img, anchor="center")

    def update_status(self):
        if self.board.is_checkmate():
            winner = "White" if self.board.turn == chess.BLACK else "Black"
            self.status_label.config(text=f"Checkmate! {winner} wins.")
        elif self.board.is_stalemate():
            self.status_label.config(text="Stalemate! Draw.")
        elif self.board.is_insufficient_material():
            self.status_label.config(text="Draw by insufficient material.")
        elif self.board.is_check():
            turn = "White" if self.board.turn == chess.WHITE else "Black"
            self.status_label.config(text=f"{turn} is in check.")
        else:
            turn = "White" if self.board.turn == chess.WHITE else "Black"
            self.status_label.config(text=f"{turn} to move.")

    def on_click(self, event):
        square_size = 50
        col = event.x // square_size
        if self.flip:
            row = event.y // square_size
        else:
            row = 7 - (event.y // square_size)
        square = chess.square(col, row)

        # Only allow human moves when it's their turn
        human_turn = (self.board.turn == chess.WHITE and self.human_color == 'white') or \
                     (self.board.turn == chess.BLACK and self.human_color == 'black')

        if not human_turn:
            return

        if self.selected_square is None:
            # Select piece
            piece = self.board.piece_at(square)
            if piece and piece.color == self.board.turn:
                self.selected_square = square
                self.legal_moves = [move.to_square for move in self.board.legal_moves if move.from_square == square]
                self.draw_board()
        else:
            # Try to move
            if square in self.legal_moves:
                move = chess.Move(self.selected_square, square)
                # Check for promotion
                if self.board.piece_at(self.selected_square).piece_type == chess.PAWN and (chess.square_rank(square) == 0 or chess.square_rank(square) == 7):
                    # Auto promote to queen for simplicity
                    move = chess.Move(self.selected_square, square, promotion=chess.QUEEN)
                self.make_move(move)
            self.selected_square = None
            self.legal_moves = []
            self.draw_board()

    def make_move(self, move):
        self.board.push(move)
        self.history = [self.board.copy()] + self.history[:6]
        self.update_status()
        
        # Clear any highlights
        self.selected_square = None
        self.legal_moves = []
        self.draw_board()
        
        # Check if AI should move
        ai_turn = (self.board.turn == chess.WHITE and self.ai_color == 'white') or \
                  (self.board.turn == chess.BLACK and self.ai_color == 'black')
        if ai_turn and self.ai_callback and not self.board.is_game_over():
            self.status_label.config(text="AI is thinking...")
            self.root.update()
            # For simplicity, call AI synchronously (may freeze GUI)
            game_state = self.create_game_state()
            ai_move = self.ai_callback(game_state)
            self.board.push(ai_move)
            self.history = [self.board.copy()] + self.history[:6]
            self.update_status()
            self.draw_board()

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