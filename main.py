from agent import Agent
from train import train

import torch
import torch.multiprocessing as mp

from brain.piece_selector import pCNN
from brain.move_selector import mCNN

TEAM_WHITE = "w"
TEAM_BLACK = "b"

NUM_PROCESSES = 16


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # white_agent = Agent(device, TEAM_WHITE)
    # black_agent = Agent(device, TEAM_BLACK)

    white_piece_selector = pCNN().to(device=device).share_memory()
    white_p_move_selector = mCNN().to(device=device).share_memory()
    white_r_move_selector = mCNN().to(device=device).share_memory()
    white_n_move_selector = mCNN().to(device=device).share_memory()
    white_b_move_selector = mCNN().to(device=device).share_memory()
    white_q_move_selector = mCNN().to(device=device).share_memory()
    white_k_move_selector = mCNN().to(device=device).share_memory()

    black_piece_selector = pCNN().to(device=device).share_memory()
    black_p_move_selector = mCNN().to(device=device).share_memory()
    black_r_move_selector = mCNN().to(device=device).share_memory()
    black_n_move_selector = mCNN().to(device=device).share_memory()
    black_b_move_selector = mCNN().to(device=device).share_memory()
    black_q_move_selector = mCNN().to(device=device).share_memory()
    black_k_move_selector = mCNN().to(device=device).share_memory()

    # train(0, white_agent, black_agent) # uncomment to debug on a single thread

    torch.multiprocessing.set_start_method('forkserver')
    processes = []
    for rank in range(NUM_PROCESSES):
        p = mp.Process(target=train, args=(rank, device, white_piece_selector, white_p_move_selector,
                                           white_r_move_selector, white_b_move_selector, white_n_move_selector,
                                           white_q_move_selector, white_k_move_selector,
                                           black_piece_selector, black_p_move_selector,
                                           black_r_move_selector, black_b_move_selector, black_n_move_selector,
                                           black_q_move_selector, black_k_move_selector
                                           ))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
