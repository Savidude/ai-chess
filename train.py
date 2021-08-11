import torch

from environment.board import Board
from environment.board import Move

import utils

TEAM_WHITE = "w"
TEAM_BLACK = "b"

MOVE_LIMIT = 100
NUM_EPISODES = 100


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
    def __init__(self, white_agent, black_agent):
        self.env = Board()
        self.white = white_agent
        self.black = black_agent
        self.turns = 0

        self.done = False

        # hyperparameters
        self.gamma = 0.99

    def reset_environment(self):
        self.env = Board()
        self.turns = 0

    def perform_action(self, team):
        state = torch.from_numpy(self.env.get_state()).float()
        valid_pieces = self.env.get_valid_pieces(team)
        valid_moves = self.env.get_valid_moves(team)
        agent = self.get_agent(team)

        selected_pieces, validated_pieces = agent.select_pieces(state, valid_pieces)
        moves = agent.select_moves(state, valid_moves)
        action = self.select_action(validated_pieces, moves)
        done = self.env.take_action(team, action.from_pos, action.to_pos)

        self.turns += 1
        if self.turns == MOVE_LIMIT:
            done = True
        self.done = done
        return (selected_pieces, validated_pieces), moves, action, done

    def get_agent(self, team):
        agent = None
        if team == TEAM_WHITE:
            if self.turns % 2 != 0 and not self.done:
                raise Exception("It is not white's turn to take an action")
            agent = self.white
        elif team == TEAM_BLACK:
            if self.turns % 2 != 1 and not self.done:
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

    def optimize_networks(self, team, piece_values, move_values, action):
        reward = self.evaluate_reward(team)
        agent = self.get_agent(team)
        piece_q_val, move_q_val = self.evaluate_q_val(team, agent, reward)

        agent.optimize_piece_selector(piece_values, piece_q_val, action.from_pos)
        agent.optimize_move_selectors(move_values, move_q_val, action.to_pos, action.moved_piece)

    def evaluate_reward(self, team):
        move_history = self.env.history
        if not self.done:
            last_opponent_move = move_history[self.turns - 1]
            last_team_move = move_history[self.turns - 2]

            if last_team_move.team != team or last_opponent_move.team == team:
                raise Exception("Invalid team has moved")
            reward = last_team_move.reward - last_opponent_move.reward
            return reward
        else:
            last_move = move_history[self.turns - 1]
            if last_move.team == team:
                return last_move.reward
            else:
                return last_move.reward * -1

    def evaluate_q_val(self, team, agent, reward):
        new_state = torch.from_numpy(self.env.get_state()).float()
        valid_pieces = self.env.get_valid_pieces(team)
        valid_moves = self.env.get_valid_moves(team)

        _, validated_pieces = agent.select_pieces(new_state, valid_pieces, detach=True)
        moves = agent.select_moves(new_state, valid_moves, detach=True)
        action = self.select_action(validated_pieces, moves)

        piece_q_val = reward + self.gamma * action.piece_value
        move_q_val = reward + self.gamma * action.move_value
        return piece_q_val, move_q_val


def train(rank, white_agent, black_agent):
    torch.manual_seed(rank)
    em = ChessEnvManager(white_agent, black_agent)

    for episode in range(NUM_EPISODES):
        em.reset_environment()

        white_piece_values, white_move_values, white_action, _ = em.perform_action(TEAM_WHITE)
        black_piece_values, black_move_values, black_action, _ = em.perform_action(TEAM_BLACK)

        while True:
            em.optimize_networks(TEAM_WHITE, white_piece_values, white_move_values, white_action)
            white_piece_values, white_move_values, white_action, done = em.perform_action(TEAM_WHITE)
            if done:
                em.optimize_networks(TEAM_WHITE, white_piece_values, white_move_values, white_action)
                em.optimize_networks(TEAM_BLACK, black_piece_values, black_move_values, black_action)
                break

            em.optimize_networks(TEAM_BLACK, black_piece_values, black_move_values, black_action)
            black_piece_values, black_move_values, black_action, done = em.perform_action(TEAM_BLACK)
            if done:
                em.optimize_networks(TEAM_WHITE, white_piece_values, white_move_values, white_action)
                em.optimize_networks(TEAM_BLACK, black_piece_values, black_move_values, black_action)
                break

        history = em.env.history
        utils.write_game_to_file(history, rank, episode)
        print("Completed rank {}, episode {}".format(rank, episode))
