import pygame
import sys
import json

from environment import board
from environment import constants


class Node:
    def __init__(self, row, col, width):
        self.row = row
        self.col = col
        self.x = int(row * width)
        self.y = int(col * width)
        self.colour = constants.WHITE

    def draw(self, window):
        pygame.draw.rect(window, self.colour, (self.x, self.y, constants.WIDTH / 8, constants.HEIGHT / 8))

    def setup(self, window, game):
        piece = game.get_piece(self.row, self.col)
        if piece is None:
            pass
        else:
            window.blit(pygame.image.load(piece.image), (self.x, self.y))


def make_grid(rows, columns, width):
    grid = []
    node_width = width // rows

    for row in range(rows):
        grid.append([])
        for col in range(columns):
            node = Node(row, col, node_width)
            grid[row].append(node)
            if (col + row) % 2 == 1:
                grid[row][col].colour = constants.GREY
    return grid


def draw_grid(window, rows, width):
    node_width = width // 8
    for i in range(rows):
        pygame.draw.line(window, constants.BLACK, (0, i * node_width), (width, i * node_width))
        for j in range(rows):
            pygame.draw.line(window, constants.BLACK, (j * node_width, 0), (j * node_width, width))


def update_display(window, grid, rows, width, game):
    for row in grid:
        for node in row:
            node.draw(window)
            node.setup(window, game)
    draw_grid(window, rows, width)
    pygame.display.update()


def main(window, game, history_file):
    moves = []
    with open(history_file) as history:
        for line in history:
            move = json.loads(line.strip())
            moves.append(move)

    grid = make_grid(constants.ROWS, constants.COLUMNS, constants.WIDTH)
    update_display(window, grid, constants.ROWS, constants.WIDTH, game)

    # for move in moves:
    #     pygame.time.delay(1000)
    #     _ = game.take_action(move['team'], move['from_pos'], move['to_pos'])
    #     update_display(window, grid, constants.ROWS, constants.WIDTH, game)

    current_move_index = 0
    while True:
        pygame.time.delay(50)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    move = moves[current_move_index]
                    _ = game.take_action(move['team'], move['to_pos'], move['from_pos'], replay=True)
                    current_move_index -= 1
                if event.key == pygame.K_RIGHT:
                    move = moves[current_move_index]
                    _ = game.take_action(move['team'], move['from_pos'], move['to_pos'], replay=True)
                    current_move_index += 1

        update_display(window, grid, constants.ROWS, constants.WIDTH, game)


if __name__ == "__main__":
    game = board.Board()
    window = pygame.display.set_mode((constants.HEIGHT, constants.WIDTH))

    history_file = sys.argv[1]
    main(window, game, history_file)
