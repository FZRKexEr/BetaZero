# 旧版纯mcts，有bug

import threading
import math
import copy
import random
import functools
from collections import defaultdict

class MCTS:
    CONFIDENCE = 0.1

    def __init__(self, time_limit):
        self.visNode = defaultdict(int)
        self.winNode = defaultdict(int)
        self.running = False
        self.time_limit = time_limit
        self.init()

    def init(self):
        self.visNode.clear()
        self.winNode.clear()

    # 停止搜索
    def stop_search(self):
        self.running = False

    def get_choices(self, node):
        choices = []
        for i in range(node.n):
            for j in range(node.n):
                next_node = copy.deepcopy(node)
                if next_node.play(i, j):
                    choices.append((i, j, next_node.hash()))
        return choices

    def uct_choice(self, father, select):
        assert select

        def uct(item):
            son = item[2]
            son_loss = self.visNode[son] - self.winNode[son]
            return son_loss / (self.visNode[son] + 1) + math.sqrt(
                self.CONFIDENCE * math.log(self.visNode[father] + 1) / (self.visNode[son] + 1))

        select.sort(key=uct)
        return select[-1][0], select[-1][1]

    def select_and_expand(self, node):
        node = copy.deepcopy(node)

        search_path = [(node.hash(), node.o)]
        while node.state() == -1:
            choices = self.get_choices(node)
            select = []
            expand = []
            for i, j, hash_value in choices:
                if hash_value in self.visNode:
                    select.append((i, j, hash_value))
                else:
                    expand.append((i, j, hash_value))
            if not choices:  # 没有选择, 停一手
                node.next()
                search_path.append((node.hash(), node.o))
            elif not expand:  # 不需要 expand, uct 选择一个
                i, j = self.uct_choice(node.hash(), select)
                node.play(i, j)
                search_path.append((node.hash(), node.o))
            else:  # 随机拓展一个
                i, j, hash_value = random.choice(expand)
                node.play(i, j)
                search_path.append((node.hash(), node.o))
                break
        return node, search_path

    def quick_analysis(self, node):
        node = copy.deepcopy(node)
        while node.state() == -1:
            choices = self.get_choices(node)
            if not choices:
                node.next()
                continue
            i, j, hash_value = random.choice(choices)
            node.play(i, j)
        return node.state()

    def run(self, node):
        node = copy.deepcopy(node)

        assert node.state() == -1
        self.init()
        self.running = True

        timer = threading.Timer(self.time_limit, self.stop_search)  # 计时线程
        timer.start()

        cnt = 0
        while self.running:
            expanding_node, search_path = self.select_and_expand(node)  # 得到待扩展节点
            res = self.quick_analysis(expanding_node)  # 快速分析待拓展节点
            for hash_value, o in search_path:  # 更新搜索路径上的节点
                self.visNode[hash_value] += 1
                if o == res:
                    self.winNode[hash_value] += 1
            cnt = cnt + 1

        timer.cancel()

        # print("搜索量:", cnt)
        # print("胜率: ", self.winNode[node.hash()] / self.visNode[node.hash()])

        choices = self.get_choices(node)
        search_result = []

        x, y, best_vis = -1, -1, -1
        for i, j, hash_value in choices:
            search_result.append(
                (i, j, (1 - self.winNode[hash_value] / (self.visNode[hash_value] + 1)) * 100, self.visNode[hash_value]))
            if self.visNode[hash_value] > best_vis:
                x, y, best_vis = i, j, self.visNode[hash_value]

        return x, y, search_result
