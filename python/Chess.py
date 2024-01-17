import numpy as np
import torch


class Chess:

    def __init__(self, n):
        self.size = n
        self.board = np.full((n, n), -1)
        self.player = 1

    def tensor(self):
        res = np.zeros((3, self.size, self.size))
        res[0][self.board == 1] = 1
        res[1][self.board == 0] = 1
        res[2] = self.player
        return torch.tensor(res)

    def hash(self):
        return hash((self.board.tobytes(), self.player))

    def can_play(self, pos):
        x, y = pos[:2]

        pass

    #def state(self):
