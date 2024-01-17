class MCTS:
    def __init__(self, game, net):
        self.net = net
        self.Qsa = {}  # 从状态 s 访问 a 的平局估值
        self.Ns = {}  # 一个状态的访问次数
        self.Nsa = {}  # 从状态s访问a的次数
        self.Ps = {}  # 对于状态s，它的下一步的概率分布
        self.Ns = {}  # 对于状态s，它的下一步的合法位置

    def rollout(self, game):
        self.net.forward(game.tensor())
        pass

    def search(self):
        pass

