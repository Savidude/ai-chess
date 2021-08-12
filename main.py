from agent import Agent
from train import train

import torch
import torch.multiprocessing as mp

TEAM_WHITE = "w"
TEAM_BLACK = "b"

NUM_PROCESSES = 16


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    white_agent = Agent(device, TEAM_WHITE)
    black_agent = Agent(device, TEAM_BLACK)

    # train(0, white_agent, black_agent) # uncomment to debug on a single thread

    torch.multiprocessing.set_start_method('spawn')
    processes = []
    for rank in range(NUM_PROCESSES):
        p = mp.Process(target=train, args=(rank, white_agent, black_agent))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
