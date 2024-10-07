import numpy as np
import random
import mmh3


class Tictac:
    def __init__(self):
        self.n = 3
        self.o = 1
        self.o_hash = random.randint(0, 2 ** 32 - 1)
        self.board = np.full((3, 3), -1)

    def next(self):
        self.o ^= 1

    def play(self, x, y):
        if self.board[x, y] != -1:
            return False
        self.board[x, y] = self.o
        self.next()
        return True

    def state(self):
        for i in range(3):
            res = self.board[i, 0]
            for j in range(3):
                if self.board[i, j] != self.board[i, 0] or self.board[i, j] == -1:
                    res = 2        
            if res != 2:
                return res
            
        for i in range(3):
            res = self.board[0, i]
            for j in range(3):
                if self.board[j, i] != self.board[0, i] or self.board[j, i] == -1:
                    res = 2
            if res != 2:
                return res
        if np.all(self.board[[0, 1, 2], [0, 1, 2]] == self.board[1, 1]):
            return self.board[1, 1]
        if np.all(self.board[[0, 1, 2], [2, 1, 0]] == self.board[1, 1]):
            return self.board[1, 1]
        for i in range(3):
            for j in range(3):
                if self.board[i, j] == -1:
                    return -1
        return 2

    def hash(self):
        if self.o == 1:
            return mmh3.hash(self.board, signed=False) ^ self.o_hash
        else:
            return mmh3.hash(self.board, signed=False)

