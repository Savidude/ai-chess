from brain.move_selector import MoveSelector
from brain.piece_selector import pCNN

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

learning_rate = 0.01


def flatten_validity_matrix(matrix):
    t = torch.from_numpy(matrix).float()
    t = t.reshape(-1, 64)
    return t


def get_validated_moves(piece_selected_moves, piece_valid_moves):
    piece_valid_moves_flat = flatten_validity_matrix(piece_valid_moves)
    piece_validated_moves = piece_selected_moves * piece_valid_moves_flat
    return piece_selected_moves, piece_validated_moves


def get_tensor_index_from_pos(pos):
    row, col = pos
    return (row * 8) + col


class Agent:
    def __init__(self, device, team):
        self.device = device
        self.team = team

        self.piece_selector = pCNN().to(device=device)

        self.move_selectors = {
            'Pawn': MoveSelector(device=device),
            'Rook': MoveSelector(device=device),
            'Knight': MoveSelector(device=device),
            'Bishop': MoveSelector(device=device),
            'Queen': MoveSelector(device=device),
            'King': MoveSelector(device=device),
        }

        self.loss_func = nn.MSELoss()

    def select_pieces(self, state, valid_pieces, detach=False):
        if detach:
            selected_pieces = self.piece_selector(state.unsqueeze(0)).to(self.device).detach()
        else:
            # Inserts an additional dimension that represents a batch of size of 1
            selected_pieces = self.piece_selector(state.unsqueeze(0)).to(self.device)
        valid_pieces_flat = flatten_validity_matrix(valid_pieces)
        validated_pieces = selected_pieces * valid_pieces_flat
        return selected_pieces, validated_pieces

    def select_moves(self, state, valid_moves, detach=False):
        moves_dict = {}
        for key in self.move_selectors:
            if detach:
                selected_moves = self.move_selectors[key].network(state.unsqueeze(0)).to(self.device).detach()
            else:
                selected_moves = self.move_selectors[key].network(state.unsqueeze(0)).to(self.device)

            moves_dict[key] = get_validated_moves(selected_moves, valid_moves[key])

        return moves_dict

    def optimize_piece_selector(self, piece_values, piece_q_val, piece_pos):
        selected_pieces, validated_pieces = piece_values
        i = get_tensor_index_from_pos(piece_pos)
        validated_pieces[0][i] = torch.tensor(piece_q_val)

        loss = self.loss_func(selected_pieces, validated_pieces)
        optimizer = optim.Adam(params=self.piece_selector.parameters(), lr=learning_rate)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def optimize_move_selectors(self, move_values, move_q_val, move_pos, piece_moved):
        for key in move_values:
            selected_moves, validated_moves = move_values[key]
            if key == piece_moved:
                i = get_tensor_index_from_pos(move_pos)
                validated_moves[0][i] = move_q_val

            loss = self.move_selectors[key].loss_func(selected_moves, validated_moves)
            optimizer = optim.Adam(params=self.move_selectors[key].network.parameters(), lr=learning_rate)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
