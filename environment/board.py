# import pygame
import sys

import numpy as np
import json

from environment import constants
from environment import piece

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
    def __init__(self, team, piece_type, from_pos, to_pos, reward, killed):
        self.team = team
        self.piece_type = piece_type
        self.from_pos = from_pos
        self.to_pos = to_pos
        self.reward = reward
        self.killed = killed

    def __str__(self):
        move_dict = {
            'team': self.team,
            'piece': str(self.piece_type),
            'from_pos': self.from_pos,
            'to_pos': self.to_pos
        }
        return json.dumps(move_dict)


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
            constants.PIECE_PAWN: np.zeros((constants.ROWS, constants.COLUMNS)),
            constants.PIECE_ROOK: np.zeros((constants.ROWS, constants.COLUMNS)),
            constants.PIECE_KNIGHT: np.zeros((constants.ROWS, constants.COLUMNS)),
            constants.PIECE_BISHOP: np.zeros((constants.ROWS, constants.COLUMNS)),
            constants.PIECE_KING: np.zeros((constants.ROWS, constants.COLUMNS)),
            constants.PIECE_QUEEN: np.zeros((constants.ROWS, constants.COLUMNS))
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

    def take_action(self, team, from_pos, to_pos, replay=False):
        done = False
        from_row, from_col = from_pos
        moving_piece = self.get_piece(from_row, from_col)
        if (moving_piece is None or moving_piece.get_team() != team) and not replay:
            raise Exception("Invalid piece moved")

        to_row, to_col = to_pos
        target_piece = self.get_piece(to_row, to_col)
        if (target_piece is not None and target_piece.get_team() == team) and not replay:
            raise Exception("Trying to kill piece from the same team")

        reward = 0
        killed_piece = None
        if target_piece is not None:
            reward = target_piece.value
            killed_piece = target_piece
            if target_piece.__class__.__name__ == constants.PIECE_KING:
                done = True

        self.state[from_row][from_col] = None
        self.state[to_row][to_col] = moving_piece

        move = Move(team=team, piece_type=moving_piece, from_pos=from_pos, to_pos=to_pos, reward=reward,
                    killed=killed_piece)
        self.history.append(move)
        return done

    def revert_action(self):
        last_move = self.history[len(self.history) - 1]
        to_row, to_col = last_move.to_pos
        from_row, from_col = last_move.from_pos
        moved_piece = self.get_piece(to_row, to_col)

        self.state[from_row][from_col] = moved_piece
        self.state[to_row][to_col] = last_move.killed

        del self.history[-1]

    def get_piece(self, row, col):
        return self.state[row][col]
