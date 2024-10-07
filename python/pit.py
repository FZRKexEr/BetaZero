# 比较两个 engine 的胜负
# 可选是否调用界面

class PIT:
    # 棋局，引擎A, 引擎B
    def __init__(self, node, engineA, engineB):
        self.node = node
        self.engineA = engineA
        self.engineB = engineB

    def get_result(self):
        while self.node.state() == -1:
            if self.node.o == 1:             # A 下黑子
                x, y, search_result = self.engineA.run(self.node)
                print("A 落子", x, y)
                if x == -1 and y == -1:
                    self.node.next()              # 停一手
                else:
                    self.node.play(x, y)
            else:
                x, y, search_result = self.engineB.run(self.node)
                print("B 落子", x, y)
                if x == -1 and y == -1:
                    self.node.next()
                else:
                    self.node.play(x, y)

        return self.node.state()

