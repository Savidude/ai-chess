def write_game_to_file(history):
    moves = ""
    for move in history:
        moves += str(move) + "\n"

    f = open("history.txt", "w")
    f.write(moves)
    f.close()
