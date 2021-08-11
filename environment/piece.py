import numpy as np

from . import constants

# One hot encoding of all pieces
piece_dict = {
    'wp': [1, 0, 0, 0, 0, 0],
    'bp': [-1, 0, 0, 0, 0, 0],
    'wn': [0, 1, 0, 0, 0, 0],
    'bn': [0, -1, 0, 0, 0, 0],
    'wb': [0, 0, 1, 0, 0, 0],
    'bb': [0, 0, -1, 0, 0, 0],
    'wr': [0, 0, 0, 1, 0, 0],
    'br': [0, 0, 0, -1, 0, 0],
    'wq': [0, 0, 0, 0, 1, 0],
    'bq': [0, 0, 0, 0, -1, 0],
    'wk': [0, 0, 0, 0, 0, 1],
    'bk': [0, 0, 0, 0, 0, -1],
    '.': [0, 0, 0, 0, 0, 0],
}


def is_on_board(row, col):
    return -1 < row < constants.ROWS and -1 < col < constants.COLUMNS


def rook_valid_moves(team, board, pos):
    valid_pos = np.zeros((constants.ROWS, constants.COLUMNS))
    row, col = pos

    cross = [[[row + i, col] for i in range(1, 8 - row)],  # down
             [[row - i, col] for i in range(1, row + 1)],  # up
             [[row, col + i] for i in range(1, 8 - col)],  # right
             [[row, col - i] for i in range(1, col + 1)]]  # left

    for direction in cross:
        for position in direction:
            if is_on_board(position[0], position[1]):
                piece = board.get_piece(position[0], position[1])
                if piece is None:
                    valid_pos[position[0], position[1]] = 1
                else:
                    if piece.team != team:
                        valid_pos[position[0], position[1]] = 1
                    break
    return valid_pos


def bishop_valid_moves(team, board, pos):
    valid_pos = np.zeros((constants.ROWS, constants.COLUMNS))
    row, col = pos

    diagonals = [[[row + i, col + i] for i in range(1, 8)],
                 [[row + i, col - i] for i in range(1, 8)],
                 [[row - i, col + i] for i in range(1, 8)],
                 [[row - i, col - i] for i in range(1, 8)]]

    for direction in diagonals:
        for position in direction:
            if is_on_board(position[0], position[1]):
                piece = board.get_piece(position[0], position[1])
                if piece is None:
                    valid_pos[position[0], position[1]] = 1
                else:
                    if piece.team != team:
                        valid_pos[position[0], position[1]] = 1
                    break
    return valid_pos


class Piece:
    def __init__(self, team, image):
        self.team = team
        self.image = image
        self.value = 0

    def get_team(self):
        return self.team

    def get_value(self):
        return self.value

    def get_encoding(self):
        return piece_dict[str(self)]

    def is_valid_move(self, board, pos, target):
        row, col = target
        valid_pos = self.valid_moves(board, pos)
        return valid_pos[row][col] == 1

    def valid_moves(self, board, pos):
        pass


class Pawn(Piece):
    def __init__(self, team):
        if team == constants.TEAM_WHITE:
            image = "../assets/images/pawn_white.png"
        elif team == constants.TEAM_BLACK:
            image = "../assets/images/pawn_black.png"
        super().__init__(team, image)
        self.value = 1

    def __str__(self):
        return self.team + "p"

    def valid_moves(self, board, pos):
        valid_pos = np.zeros((constants.ROWS, constants.COLUMNS))
        row, col = pos

        if self.team == constants.TEAM_BLACK:
            if row == 1:
                if board.get_piece(row + 2, col) is None and board.get_piece(row + 1, col) is None:
                    valid_pos[row + 2, col] = 1

            bottom3 = [[row + 1, col + i] for i in range(-1, 2)]
            for i in range(len(bottom3)):
                position = bottom3[i]
                if is_on_board(position[0], position[1]):
                    if i % 2 == 0:
                        piece = board.get_piece(position[0], position[1])
                        if piece is not None:
                            if piece.get_team() != constants.TEAM_BLACK:
                                valid_pos[position[0], position[1]] = 1
                    else:
                        piece = board.get_piece(position[0], position[1])
                        if piece is None:
                            valid_pos[position[0], position[1]] = 1
        elif self.team == constants.TEAM_WHITE:
            if row == 6:
                if board.get_piece(row - 2, col) is None and board.get_piece(row - 1, col) is None:
                    valid_pos[row - 2, col] = 1

            top3 = [[row - 1, col + i] for i in range(-1, 2)]
            for i in range(len(top3)):
                position = top3[i]
                if is_on_board(position[0], position[1]):
                    if i % 2 == 0:
                        piece = board.get_piece(position[0], position[1])
                        if piece is not None:
                            if piece.get_team() != constants.TEAM_WHITE:
                                valid_pos[position[0], position[1]] = 1
                    else:
                        piece = board.get_piece(position[0], position[1])
                        if piece is None:
                            valid_pos[position[0], position[1]] = 1
        return valid_pos


class Rook(Piece):
    def __init__(self, team):
        if team == constants.TEAM_WHITE:
            image = "../assets/images/rook_white.png"
        elif team == constants.TEAM_BLACK:
            image = "../assets/images/rook_black.png"
        super().__init__(team, image)
        self.value = 5

    def __str__(self):
        return self.team + "r"

    def valid_moves(self, board, pos):
        return rook_valid_moves(self.team, board, pos)


class Knight(Piece):
    def __init__(self, team):
        if team == constants.TEAM_WHITE:
            image = "../assets/images/knight_white.png"
        elif team == constants.TEAM_BLACK:
            image = "../assets/images/knight_black.png"
        super().__init__(team, image)
        self.value = 3

    def __str__(self):
        return self.team + "n"

    def valid_moves(self, board, pos):
        valid_pos = np.zeros((constants.ROWS, constants.COLUMNS))
        row, col = pos

        for r in range(-2, 3):
            for c in range(-2, 3):
                if r ** 2 + c ** 2 == 5:
                    if is_on_board(row + r, col + c):
                        piece = board.get_piece(row + r, col + c)
                        if piece is None:
                            valid_pos[row + r][col + c] = 1
                        elif piece.team != self.team:
                            valid_pos[row + r][col + c] = 1
        return valid_pos


class Bishop(Piece):
    def __init__(self, team):
        if team == constants.TEAM_WHITE:
            image = "../assets/images/bishop_white.png"
        elif team == constants.TEAM_BLACK:
            image = "../assets/images/bishop_black.png"
        super().__init__(team, image)
        self.value = 3

    def __str__(self):
        return self.team + "b"

    def valid_moves(self, board, pos):
        return bishop_valid_moves(self.team, board, pos)


class King(Piece):
    def __init__(self, team):
        if team == constants.TEAM_WHITE:
            image = "../assets/images/king_white.png"
        elif team == constants.TEAM_BLACK:
            image = "../assets/images/king_black.png"
        super().__init__(team, image)
        self.value = 255

    def __str__(self):
        return self.team + "k"

    def valid_moves(self, board, pos):
        valid_pos = np.zeros((constants.ROWS, constants.COLUMNS))
        row, col = pos

        for r in range(3):
            for c in range(3):
                if is_on_board(row - 1 + r, col - 1 + c):
                    piece = board.get_piece(row - 1 + r, col - 1 + c)
                    if piece is None:
                        valid_pos[row - 1 + r][col - 1 + c] = 1
                    elif piece.team != self.team:
                        valid_pos[row - 1 + r][col - 1 + c] = 1
        return valid_pos


class Queen(Piece):
    def __init__(self, team):
        if team == constants.TEAM_WHITE:
            image = "../assets/images/queen_white.png"
        elif team == constants.TEAM_BLACK:
            image = "../assets/images/queen_black.png"
        super().__init__(team, image)
        self.value = 9

    def __str__(self):
        return self.team + "q"

    def valid_moves(self, board, pos):
        valid_pos = rook_valid_moves(self.team, board, pos)
        return valid_pos + bishop_valid_moves(self.team, board, pos)
