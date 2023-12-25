//
// Created by LiZnB on 2023/12/23.
//

#ifndef BETAZERO_TICTAC_H
#define BETAZERO_TICTAC_H

#include "chess.h"

// 3 * 3 tictac
class Tictac : public chess {
public:
  Tictac();

  // 尝试在 (x, y) 落子(不会修改棋盘)，如果落子成功，返回新的棋盘的 hash, 否则返回 -1
  long long try_play(int x, int y);

  // 判断棋盘状态: -1 未完成, 0 白子胜利, 1 黑子胜利, 2 平局
  int end();

  // 在 (x, y) 落子
  void play(int x, int y);

  // 停一手
  void pass();
};

#endif //BETAZERO_TICTAC_H
