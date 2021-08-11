import math
import os
import time

HISTORY_DIR = "history"


def write_game_to_file(history, rank, episode):
    if ((rank + 1) * (episode + 1)) % 10 == 0:
        create_dir_if_not_exist(HISTORY_DIR)

        moves = ""
        for move in history:
            moves += str(move) + "\n"

        episode_file = os.path.join(HISTORY_DIR, "{}-episode-{}-{}.txt".format(math.floor(time.time()), rank, episode))
        f = open(episode_file, "w")
        f.write(moves)
        f.close()


def create_dir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)
