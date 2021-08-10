import os

HISTORY_DIR = "history"


def write_game_to_file(history, episode):
    create_dir_if_not_exist(HISTORY_DIR)

    moves = ""
    for move in history:
        moves += str(move) + "\n"

    episode_file = os.path.join(HISTORY_DIR, "episode-{}.txt".format(episode))
    f = open(episode_file, "w")
    f.write(moves)
    f.close()


def create_dir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)
