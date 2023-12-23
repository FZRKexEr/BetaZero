//
// Created by LiZnB on 2023/12/23.
//

#ifndef BETAZERO_OTHELLO_H
#define BETAZERO_OTHELLO_H

#include "chess.h"

class Othello : public chess {
private:
  int dx[8] = {0, 1, 1, 1, 0, -1, -1, -1}, dy[8] = {1, 1, 0, -1, -1, -1, 0, 1};

public:
  // 初始化开局
  Othello(int n);

  // 翻转棋盘，返回需要翻转的位置
  vector<array<int, 2>> flip_once(vector<vector<int>> temp, int x, int y);

  // 尝试落子, 返回落子后的 hash 值，如果落子失败，返回 -1
  long long try_play(int x, int y);

  // 判断棋局状态，返回 -1 未结束，0 白棋胜, 1 黑棋胜, 2 平局
  int end();

  // 在 x, y 落子
  void play(int x, int y);

  // 停一手
  void pass();

};


#endif //BETAZERO_OTHELLO_H
