# 有界面的游戏

import copy
import numpy as np
from threading import Thread
from ui import UI


# 后台进行搜索，传入棋面和引擎
class Search(Thread):
    def __init__(self, node, engine):
        Thread.__init__(self)
        self.x = None
        self.y = None
        self.search_result = None
        self.node = node
        self.engine = engine

    def run(self):
        self.x, self.y, self.search_result = self.engine.run(self.node)

    def get_result(self):
        return self.x, self.y, self.search_result


class Game:
    def __init__(self, node, engine, human_o):
        self.chess = node
        self.engine = engine
        self.human_o = human_o

    def play(self):
        ui = UI(670, 670, self.chess.n, 'BetaZero')
        mcts = None
        search_info = []

        while True:
            if ui.check_quit():  # 触发退出
                break

            ui.draw_screen()
            ui.draw_grid()
            ui.draw_mouse()
            ui.draw_board(self.chess.board)

            if mcts is not None:  # mcts已经启动了
                if not mcts.is_alive():  # mcts已经运行结束了
                    mcts.join()
                    x, y, search_info = mcts.get_result()
                    print("mcts 落子:", x, y)
                    if x == -1 and y == -1:
                        self.chess.next()
                    else:
                        self.chess.play(x, y)
                    mcts = None
            else:  # mcts 未启动
                if self.chess.state() == -1 and self.chess.o != self.human_o:
                    mcts = Search(self.chess, self.engine)
                    mcts.start()

                # if chess.state() == -1 and chess.o == 1:
                #     mcts = Search(chess, MCTS2())
                #     mcts.start()

            if self.chess.o == self.human_o:  # user 落子
                if ui.is_pressed():
                    y, x = ui.get_mouse_pos()
                    if not self.engine.get_valid_actions(self.chess):
                        self.chess.next()
                    else:
                        res = self.chess.play(x, y)
                        if res:  # 合法落子
                            print("play:", x, y)

            ui.draw_search_info(search_info)
            ui.update()
