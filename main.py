from agent import Agent
from environment.board import Board
from environment.board import Move

import torch

TEAM_WHITE = "w"
TEAM_BLACK = "b"


def get_pos_by_index(index):
    row = index // 8
    col = index % 8
    return row, col


class ChessEnvManager:
    def __init__(self, device):
        self.device = device
        self.env = Board()
        self.white = Agent(device, TEAM_WHITE)
        self.black = Agent(device, TEAM_BLACK)
        self.turns = 0

    def perform_action(self, team):
        state = torch.from_numpy(self.env.get_state()).float()
        valid_pieces = self.env.get_valid_pieces(team)
        valid_moves = self.env.get_valid_moves(team)
        agent = self.get_agent(team)

        selected_pieces, validated_pieces = agent.select_pieces(state, valid_pieces)
        moves = agent.select_moves(state, valid_moves)
        from_pos, to_pos = self.select_action(validated_pieces, moves)
        self.env.take_action(team, from_pos, to_pos)

        self.turns += 1

    def get_agent(self, team):
        agent = None
        if team == TEAM_WHITE:
            if self.turns % 2 != 0:
                raise Exception("It is not white's turn to take an action")
            agent = self.white
        elif team == TEAM_BLACK:
            if self.turns % 2 != 1:
                raise Exception("It is not black's turn to take an action")
            agent = self.black
        return agent

    def select_action(self, validated_pieces, moves):
        possible_actions = {}
        validated_pieces_sorted = torch.sort(validated_pieces, descending=True)

        piece_index = 0
        piece_value = validated_pieces_sorted.values[0][piece_index].item()

        while piece_value > 0:
            piece_sorted_index = validated_pieces_sorted.indices[0][piece_index].item()
            p_row, p_col = get_pos_by_index(piece_sorted_index)

            piece = self.env.get_piece(p_row, p_col)
            piece_name = piece.__class__.__name__
            _, piece_validated_moves = moves[piece_name]
            validated_moves_sorted = torch.sort(piece_validated_moves, descending=True)

            move_index = 0
            move_value = validated_moves_sorted.values[0][move_index].item()

            while move_value > 0:
                move_sorted_index = validated_moves_sorted.indices[0][move_index].item()
                m_row, m_col = get_pos_by_index(move_sorted_index)
                if piece.is_valid_move(self.env, (p_row, p_col), (m_row, m_col)):
                    action_value = piece_value * move_value
                    possible_actions[action_value] = (p_row, p_col), (m_row, m_col)
                    break

                move_index += 1
                move_value = validated_moves_sorted.values[0][move_index].item()

            piece_index += 1
            piece_value = validated_pieces_sorted.values[0][piece_index].item()

        max_action_value = max(possible_actions, key=possible_actions.get)
        return possible_actions[max_action_value]

    def evaluate_reward(self, team):
        move_history = self.env.history
        last_opponent_move = move_history[self.turns - 1]
        last_team_move = move_history[self.turns - 2]

        if last_team_move.team != team or last_opponent_move.team == team:
            raise Exception("Invalid team has moved")
        reward = last_team_move.reward - last_opponent_move.reward
        return reward


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    em = ChessEnvManager(device)

    em.perform_action(TEAM_WHITE)
    em.perform_action(TEAM_BLACK)

    while True:
        white_reward = em.evaluate_reward(TEAM_WHITE)


if __name__ == "__main__":
    main()
