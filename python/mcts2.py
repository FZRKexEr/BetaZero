# 新版纯 mcts

import threading
import math
import copy
import random
from collections import defaultdict


class MCTS2:
    CPUCT = 0.1

    def __init__(self):
        self.N = defaultdict(int)
        self.Q = defaultdict(float)
        self.running = False

    # 停止搜索
    def stop_search(self):
        self.running = False

    def get_valid_actions(self, node):
        actions = []
        for i in range(node.n):
            for j in range(node.n):
                next_node = copy.deepcopy(node)
                if next_node.play(i, j):
                    actions.append((i, j, next_node.hash()))
        return actions

    # 返回快速走子的结局, 这个函数需要被 nnet 替代
    def quick_analysis(self, node):
        node = copy.deepcopy(node)
        while node.state() == -1:
            choices = self.get_valid_actions(node)
            if not choices:
                node.next()
                continue
            i, j, hash_value = random.choice(choices)
            node.play(i, j)
        return node.state()

    def get_reward(self, o, ending):
        if o == ending:
            return 1
        elif (o ^ 1) == ending:
            return -1
        else:
            return 0

    def update(self, hash_value, v):
        self.Q[hash_value] = (self.Q[hash_value] * self.N[hash_value] + v) / (self.N[hash_value] + 1)
        self.N[hash_value] += 1

    def get_pi(self, node):
        pi = []
        N_total = 0
        actions = self.get_valid_actions(node)
        for i, j, hash_value in actions:
            N_total += self.N[hash_value]
        for i, j, hash_value in actions:
            pi.append((i, j, self.N[hash_value] / N_total))
        return pi

    # 返回单次mcts搜索结局
    def search(self, node):
        node = copy.deepcopy(node)
        current_hash = node.hash()

        if node.state() != -1:
            res = node.state()
            v = self.get_reward(node.o, res)
            self.update(current_hash, v)
            return res

        if current_hash not in self.N:  # 遇到需要快速评估的节点了
            self.N[current_hash] += 1
            res = self.quick_analysis(node)
            v = self.get_reward(node.o, res)
            self.update(current_hash, v)
            return res

        max_u, res_x, res_y = -float("inf"), -1, -1

        N_total = sum(self.N[x[2]] for x in self.get_valid_actions(node))

        for x, y, hash_value in self.get_valid_actions(node):
            u = -self.Q[hash_value] + self.CPUCT * 1 * math.sqrt(N_total) / (1 + self.N[hash_value])
            if u >= max_u:
                max_u, res_x, res_y = u, x, y

        if res_x == -1 and res_y == -1:
            node.next()
        else:
            node.play(res_x, res_y)

        res = self.search(node)
        v = self.get_reward(node.o ^ 1, res)
        self.update(current_hash, v)

        return res

    def benchmark(self, node):
        node = copy.deepcopy(node)
        self.running = True
        timer = threading.Timer(1, self.stop_search)  # 计时线程
        timer.start()
        cnt = 0
        while self.running:
            self.search(node)
            cnt += 1
        timer.cancel()
        return cnt

    def run(self, node):
        node = copy.deepcopy(node)

        self.running = True
        timer = threading.Timer(2, self.stop_search)  # 计时线程
        timer.start()
        while self.running:
            self.search(node)
        timer.cancel()

        actions = self.get_valid_actions(node)
        search_result = []

        min_q, x, y = float("inf"), -1, -1
        for i, j, hash_value in actions:  # 找到估值最小的点
            # print(i, j, hash_value)
            search_result.append((i, j, self.Q[hash_value], self.N[hash_value]))
            if self.Q[hash_value] <= min_q:
                min_q, x, y = self.Q[hash_value], i, j

        return x, y, search_result
