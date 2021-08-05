from brain.move_selector import mCNN
from brain.piece_selector import pCNN

import torch
import torch.nn.functional as F


def flatten_validity_matrix(matrix):
    t = torch.from_numpy(matrix).float()
    t = t.reshape(-1, 64)
    return t


def get_validated_moves(piece_selected_moves, piece_valid_moves):
    piece_valid_moves_flat = flatten_validity_matrix(piece_valid_moves)
    piece_validated_moves = piece_selected_moves * piece_valid_moves_flat
    return piece_selected_moves, piece_validated_moves


class Agent:
    def __init__(self, device, team):
        self.device = device
        self.team = team

        self.piece_selector = pCNN()

        self.pawn_move_selector = mCNN()
        self.rook_move_selector = mCNN()
        self.knight_move_selector = mCNN()
        self.bishop_move_selector = mCNN()
        self.queen_move_selector = mCNN()
        self.king_move_selector = mCNN()

    def select_pieces(self, state, valid_pieces):
        # Inserts an additional dimension that represents a batch of size of 1
        selected_pieces = self.piece_selector(state.unsqueeze(0)).to(self.device)
        valid_pieces_flat = flatten_validity_matrix(valid_pieces)
        validated_pieces = selected_pieces * valid_pieces_flat
        return selected_pieces, validated_pieces

    def select_moves(self, state, valid_moves):
        moves_dict = {}

        pawn_selected_moves = self.pawn_move_selector(state.unsqueeze(0)).to(self.device)
        moves_dict["Pawn"] = get_validated_moves(pawn_selected_moves, valid_moves["Pawn"])

        rook_selected_moves = self.rook_move_selector(state.unsqueeze(0)).to(self.device)
        moves_dict["Rook"] = get_validated_moves(rook_selected_moves, valid_moves["Rook"])

        knight_selected_moves = self.knight_move_selector(state.unsqueeze(0)).to(self.device)
        moves_dict["Knight"] = get_validated_moves(knight_selected_moves, valid_moves["Knight"])

        bishop_selected_moves = self.bishop_move_selector(state.unsqueeze(0)).to(self.device)
        moves_dict["Bishop"] = get_validated_moves(bishop_selected_moves, valid_moves["Bishop"])

        queen_selected_moves = self.queen_move_selector(state.unsqueeze(0)).to(self.device)
        moves_dict["Queen"] = get_validated_moves(queen_selected_moves, valid_moves["Queen"])

        king_selected_moves = self.king_move_selector(state.unsqueeze(0)).to(self.device)
        moves_dict["King"] = get_validated_moves(king_selected_moves, valid_moves["King"])

        return moves_dict
