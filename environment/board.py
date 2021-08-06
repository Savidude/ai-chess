import pygame
import sys

import numpy as np

from . import constants
from . import piece

bp = piece.Pawn(constants.TEAM_BLACK)
wp = piece.Pawn(constants.TEAM_WHITE)
bk = piece.King(constants.TEAM_BLACK)
wk = piece.King(constants.TEAM_WHITE)
br = piece.Rook(constants.TEAM_BLACK)
wr = piece.Rook(constants.TEAM_WHITE)
bb = piece.Bishop(constants.TEAM_BLACK)
wb = piece.Bishop(constants.TEAM_WHITE)
bq = piece.Queen(constants.TEAM_BLACK)
wq = piece.Queen(constants.TEAM_WHITE)
bn = piece.Knight(constants.TEAM_BLACK)
wn = piece.Knight(constants.TEAM_WHITE)


class Move:
    def __init__(self, team, piece_type, from_pos, to_pos, reward):
        self.team = team
        self.piece_type = piece_type
        self.from_pos = from_pos
        self.to_pos = to_pos
        self.reward = reward


class Board:
    def __init__(self):
        state = [[None for i in range(constants.COLUMNS)] for i in range(constants.ROWS)]

        state[0] = [br, bn, bb, bq, bk, bb, bn, br]  # populate top black pieces
        state[7] = [wr, wn, wb, wq, wk, wb, wn, wr]  # populate top white pieces

        # populate pawns
        for i in range(constants.ROWS):
            state[1][i] = bp
            state[6][i] = wp

        self.state = state
        self.history = []

    def print_state(self):
        for row in range(constants.ROWS):
            row_str = ""
            for col in range(constants.COLUMNS):
                piece_str = str(self.get_piece(row, col))
                if piece_str == "None":
                    piece_str = "00"
                row_str += piece_str + "\t"
            print(row_str)

    def get_state(self):
        b = np.zeros((constants.NUM_PIECES, constants.ROWS, constants.COLUMNS))
        for i in range(constants.NUM_PIECES):
            for row in range(constants.ROWS):
                for col in range(constants.COLUMNS):
                    board_piece = self.get_piece(row, col)
                    if board_piece is not None:
                        board_piece_encoding = board_piece.get_encoding()
                    else:
                        board_piece_encoding = piece.piece_dict["."]
                    b[i][row][col] = board_piece_encoding[i]
        return b

    def get_valid_pieces(self, team):
        p = np.zeros((constants.ROWS, constants.COLUMNS))
        for row in range(constants.ROWS):
            for col in range(constants.COLUMNS):
                board_piece = self.get_piece(row, col)
                if board_piece is not None and board_piece.get_team() == team:
                    p[row][col] = 1
        return p

    def get_valid_moves(self, team):
        moves_dict = {
            'Pawn': np.zeros((constants.ROWS, constants.COLUMNS)),
            'Rook': np.zeros((constants.ROWS, constants.COLUMNS)),
            'Knight': np.zeros((constants.ROWS, constants.COLUMNS)),
            'Bishop': np.zeros((constants.ROWS, constants.COLUMNS)),
            'King': np.zeros((constants.ROWS, constants.COLUMNS)),
            'Queen': np.zeros((constants.ROWS, constants.COLUMNS))
        }

        for row in range(constants.ROWS):
            for col in range(constants.COLUMNS):
                board_piece = self.get_piece(row, col)
                if board_piece is not None and board_piece.get_team() == team:
                    valid_moves = board_piece.valid_moves(board=self, pos=(row, col))
                    piece_name = board_piece.__class__.__name__
                    moves_dict[piece_name] += valid_moves

        """
        convert all values > 0 in the valid moves matrix to 1, as the same square could be valid for multiple pieces 
        of the same type
        """
        for key in moves_dict:
            valid_moves = moves_dict[key]
            valid_moves[valid_moves > 0] = 1
            moves_dict[key] = valid_moves

        return moves_dict

    def take_action(self, team, from_pos, to_pos):
        from_row, from_col = from_pos
        moving_piece = self.get_piece(from_row, from_col)
        if moving_piece is None or moving_piece.get_team() != team:
            raise Exception("Invalid piece moved")

        to_row, to_col = to_pos
        target_piece = self.get_piece(to_row, to_col)
        if target_piece is not None and target_piece.get_team() == team:
            raise Exception("Trying to kill piece from the same team")

        reward = 0
        if target_piece is not None:
            reward = target_piece

        self.state[from_row][from_col] = None
        self.state[to_row][to_col] = moving_piece

        move = Move(team=team, piece_type=moving_piece, from_pos=from_pos, to_pos=to_pos, reward=reward)
        self.history.append(move)

    def get_piece(self, row, col):
        return self.state[row][col]

# class Node:
#     def __init__(self, row, col, width):
#         self.row = row
#         self.col = col
#         self.x = int(row * width)
#         self.y = int(col * width)
#         self.colour = constants.WHITE
#         self.occupied = None
#
#     def draw(self, window):
#         pygame.draw.rect(window, self.colour, (self.x, self.y, constants.WIDTH / 8, constants.HEIGHT / 8))
#
#     def setup(self, window, board):
#         piece = board.get_piece(self.col, self.row)
#         if piece is None:
#             pass
#         else:
#             window.blit(pygame.image.load(piece.image), (self.x, self.y))


# def make_grid(rows, columns, width):
#     grid = []
#     node_width = width // rows
#
#     for row in range(rows):
#         grid.append([])
#         for col in range(columns):
#             node = Node(row, col, node_width)
#             grid[row].append(node)
#             if (col + row) % 2 == 1:
#                 grid[row][col].colour = constants.GREY
#     return grid


# def draw_grid(window, rows, width):
#     node_width = width // 8
#     for i in range(rows):
#         pygame.draw.line(window, constants.BLACK, (0, i * node_width), (width, i * node_width))
#         for j in range(rows):
#             pygame.draw.line(window, constants.BLACK, (j * node_width, 0), (j * node_width, width))
#
#
# def update_display(window, grid, rows, width, board):
#     for row in grid:
#         for node in row:
#             node.draw(window)
#             node.setup(window, board)
#     draw_grid(window, rows, width)
#     pygame.display.update()


# def main(window, board):
#     moves = 0
#     selected = False
#     piece_to_move = []
#
#     grid = make_grid(constants.ROWS, constants.COLUMNS, constants.WIDTH)
#
#     while True:
#         pygame.time.delay(50)  # reduce CPU load
#
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 pygame.quit()
#                 sys.exit()
#
#         update_display(window, grid, constants.ROWS, constants.WIDTH, board)
#
#
# if __name__ == "__main__":
#     board = Board()
#     print(board.get_state())
#     # board.print_state()
#     window = pygame.display.set_mode((constants.HEIGHT, constants.WIDTH))
#     main(window, board)
