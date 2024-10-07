from othello import Othello
from tictac import Tictac
from mcts2 import MCTS2
from mcts import MCTS
from human_play import Game
from pit import PIT

# 定义游戏和引擎
chess = Othello(8)
engine = MCTS2()

# 测试性能
print("每秒搜索到的结局个数:", engine.benchmark(chess))

# 人类下棋
game = Game(chess, engine, 0)
game.play()

# 引擎对比
# game = PIT(chess, MCTS2(), MCTS(1))
# result = game.get_result()
# print("结局: ", result)

