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


def calculate_action_value(piece_value, move_value):
    if piece_value > 0 and move_value > 0:
        return piece_value * move_value
    elif piece_value < 0 and move_value < 0:
        return piece_value * move_value * -2
    # the piece shouldn't be moved, but it is advantageous to have the piece in the given location
    elif piece_value < 0 < move_value:
        return piece_value * move_value
    # it is disadvantageous to have the piece in the given location, but the piece should be moved
    elif move_value < 0 < piece_value:
        return piece_value * move_value * 1.5


class Action:
    def __init__(self, from_pos, to_pos, piece_val, move_val, moved_piece):
        self.from_pos = from_pos
        self.to_pos = to_pos
        self.piece_value = piece_val
        self.move_value = move_val
        self.moved_piece = moved_piece


class ChessEnvManager:
    def __init__(self, device):
        self.device = device
        self.env = Board()
        self.white = Agent(device, TEAM_WHITE)
        self.black = Agent(device, TEAM_BLACK)
        self.turns = 0

        # hyperparameters
        self.gamma = 0.99

    def perform_action(self, team):
        state = torch.from_numpy(self.env.get_state()).float()
        valid_pieces = self.env.get_valid_pieces(team)
        valid_moves = self.env.get_valid_moves(team)
        agent = self.get_agent(team)

        selected_pieces, validated_pieces = agent.select_pieces(state, valid_pieces)
        moves = agent.select_moves(state, valid_moves)
        action = self.select_action(validated_pieces, moves)
        self.env.take_action(team, action.from_pos, action.to_pos)

        self.turns += 1
        self.env.print_state()
        print()
        return (selected_pieces, validated_pieces), moves, action

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

        for piece_index in range(len(validated_pieces_sorted.values[0])):
            piece_value = validated_pieces_sorted.values[0][piece_index].item()
            if piece_value == 0:
                continue
            piece_sorted_index = validated_pieces_sorted.indices[0][piece_index].item()
            p_row, p_col = get_pos_by_index(piece_sorted_index)

            piece = self.env.get_piece(p_row, p_col)
            piece_name = piece.__class__.__name__
            _, piece_validated_moves = moves[piece_name]
            validated_moves_sorted = torch.sort(piece_validated_moves, descending=True)

            for move_index in range(len(validated_moves_sorted.values[0])):
                move_value = validated_moves_sorted.values[0][move_index].item()
                if move_value == 0:
                    continue
                move_sorted_index = validated_moves_sorted.indices[0][move_index].item()
                m_row, m_col = get_pos_by_index(move_sorted_index)
                if piece.is_valid_move(self.env, (p_row, p_col), (m_row, m_col)):
                    action_value = calculate_action_value(piece_value, move_value)
                    possible_actions[action_value] = Action(from_pos=(p_row, p_col), to_pos=(m_row, m_col),
                                                            piece_val=piece_value, move_val=move_value,
                                                            moved_piece=piece_name)
                    break

        # return action with the largest action_value
        return sorted(possible_actions.items(), reverse=True)[0][1]

    def evaluate_reward(self, team):
        move_history = self.env.history
        last_opponent_move = move_history[self.turns - 1]
        last_team_move = move_history[self.turns - 2]

        if last_team_move.team != team or last_opponent_move.team == team:
            raise Exception("Invalid team has moved")
        reward = last_team_move.reward - last_opponent_move.reward
        return reward

    def evaluate_q_val(self, team, reward):
        new_state = torch.from_numpy(self.env.get_state()).float()
        valid_pieces = self.env.get_valid_pieces(team)
        valid_moves = self.env.get_valid_moves(team)
        agent = self.get_agent(team)

        _, validated_pieces = agent.select_pieces(new_state, valid_pieces, detach=True)
        moves = agent.select_moves(new_state, valid_moves, detach=True)
        action = self.select_action(validated_pieces, moves)

        piece_q_val = reward + self.gamma * action.piece_value
        move_q_val = reward + self.gamma * action.move_value
        return piece_q_val, move_q_val

    def optimize_agent_piece_network(self, team, piece_values, piece_q_val, piece_pos):
        agent = self.get_agent(team)
        agent.optimize_piece_selector(piece_values, piece_q_val, piece_pos)

    def optimize_agent_move_networks(self, team, move_values, move_q_val, move_pos, piece_moved):
        agent = self.get_agent(team)
        agent.optimize_move_selectors(move_values, move_q_val, move_pos, piece_moved)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    em = ChessEnvManager(device)

    white_piece_values, white_move_values, white_action = em.perform_action(TEAM_WHITE)
    black_piece_values, black_move_values, black_action = em.perform_action(TEAM_BLACK)

    while True:
        white_reward = em.evaluate_reward(TEAM_WHITE)
        white_piece_q_val, white_move_q_val = em.evaluate_q_val(TEAM_WHITE, white_reward)
        em.optimize_agent_piece_network(TEAM_WHITE, white_piece_values, white_piece_q_val, white_action.from_pos)
        em.optimize_agent_move_networks(TEAM_WHITE, white_move_values, white_move_q_val, white_action.to_pos,
                                        white_action.moved_piece)
        white_piece_values, white_move_values, white_action = em.perform_action(TEAM_WHITE)

        black_reward = em.evaluate_reward(TEAM_BLACK)
        black_piece_q_val, black_move_q_val = em.evaluate_q_val(TEAM_BLACK, black_reward)
        em.optimize_agent_piece_network(TEAM_BLACK, black_piece_values, black_piece_q_val, black_action.from_pos)
        em.optimize_agent_move_networks(TEAM_BLACK, black_move_values, black_move_q_val, black_action.to_pos,
                                        black_action.moved_piece)
        black_piece_values, black_move_values, black_action = em.perform_action(TEAM_BLACK)


if __name__ == "__main__":
    main()
