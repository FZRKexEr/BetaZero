import numpy as np
import random
import mmh3
import torch


class Othello:
    dx = [0, 1, 1, 1, 0, -1, -1, -1]
    dy = [1, 1, 0, -1, -1, -1, 0, 1]

    def __init__(self, n):
        self.n = n
        self.o = 1
        self.o_hash = random.randint(0, 2 ** 32 - 1)
        self.board = np.full((n, n), -1)
        self.board[[(n - 1) // 2, n // 2], [(n - 1) // 2, n // 2]] = 0
        self.board[[(n - 1) // 2, n // 2], [n // 2, (n - 1) // 2]] = 1

    def flip_once(self, color, x, y):
        res = []
        for i in range(8):
            temp = []
            nx = x + self.dx[i]
            ny = y + self.dy[i]
            while True:
                if nx >= self.n or ny >= self.n or nx < 0 or ny < 0 or self.board[nx, ny] == -1:
                    temp.clear()
                    break
                if self.board[nx, ny] == color:
                    break
                if self.board[nx, ny] == (color ^ 1):
                    temp.append((nx, ny))

                nx += self.dx[i]
                ny += self.dy[i]
            res += temp
        return res

    def next(self):
        self.o ^= 1

    # 在 x, y 落子, 返回是否落子成功
    def play(self, x, y):
        if x < 0 or y < 0 or x >= self.n or y >= self.n:
            return False
        if self.board[x, y] != -1:
            return False
        self.board[x, y] = self.o
        res = self.flip_once(self.o, x, y)
        if not res:
            self.board[x, y] = -1
            return False
        for itx, ity in res:
            self.board[itx, ity] = self.o
        self.next()
        return True

    # 当前棋局的状态: (-1 = 未完成), (0 = 白棋胜), (1 = 黑棋胜), (2 = 平局)
    def state(self):
        cnt = [0, 0]
        for i in range(self.n):
            for j in range(self.n):
                if self.board[i, j] != -1:
                    cnt[self.board[i, j]] += 1
                    continue
                res0 = self.flip_once(0, i, j)
                res1 = self.flip_once(1, i, j)
                if res0 or res1:
                    return -1
        if cnt[0] > cnt[1]:
            return 0
        if cnt[0] < cnt[1]:
            return 1
        return 2

    # 不包含对称信息的 hash
    def hash_old(self):
        hash_value = mmh3.hash(self.board, signed=False)
        if self.o == 1:
            return hash_value ^ self.o_hash
        else:
            return hash_value

    # 包含对称信息的 hash
    def hash(self):
        hash_list = set()

        for i in range(4):
            hash_list.add(mmh3.hash(np.ascontiguousarray(np.rot90(self.board, k=i)).tobytes(), signed=False))
            hash_list.add(mmh3.hash(np.ascontiguousarray(np.fliplr(np.rot90(self.board, k=i))).tobytes(), signed=False))
            hash_list.add(mmh3.hash(np.ascontiguousarray(np.flipud(np.rot90(self.board, k=i))).tobytes(), signed=False))

        hash_list = list(hash_list)
        hash_list.sort()

        if self.o == 1:
            return mmh3.hash(str(hash_list).encode('utf-8'), signed=False) ^ self.o_hash
        else:
            return mmh3.hash(str(hash_list).encode('utf-8'), signed=False)

    # 转换成 tensor
    def to_tensor(self):
        board = np.zeros((self.n, self.n, 3))
        board[:, :, 0] = (self.board == 1)  # 第一层, 所有等于1的位置
        board[:, :, 1] = (self.board == 0)  # 第二层, 所有等于0的位置
        board[:, :, 2].fill(self.o)         # 第三层，等于self.o的值
        return torch.from_numpy(board)
