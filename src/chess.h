//
// Created by LiZnB on 2023/12/23.
//

#ifndef BETAZERO_CHESS_H
#define BETAZERO_CHESS_H

#include <bits/stdc++.h>
using namespace std;

// 定义棋盘的状态
// hash: 棋盘状态的 hash值，包含棋面和先后手
// ohash: 先手的 hash 值, 后手 hash 值默认是0
// size: 棋盘大小
// o: 下一个落子的是 1 先手, 0 后手
// zobrist: 每个位置的 zobrist hash 值
// display() 展示棋盘

class chess {
public:
  vector<vector<int>> board;
  long long hash, ohash;
  int size, o;
  vector<vector<array<long long, 2>>> zobrist;
  chess(int n);
  void display();
};

#endif //BETAZERO_CHESS_H
