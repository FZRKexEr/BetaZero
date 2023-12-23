//
// Created by LiZnB on 2023/12/23.
//

#include "chess.h"

chess::chess(int n) : size(n) {
  mt19937 seed(time(0));
  uniform_int_distribution<long long> R(1, LLONG_MAX);
  o = 1, hash = ohash = R(seed);
  zobrist.resize(size, vector<array<long long, 2>>(size));
  board.resize(size, vector<int>(size, -1));
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      zobrist[i][j] = {R(seed), R(seed)};
    }
  }
}

void chess::display() {
  cout << "   ";
  for (int i = 0; i < size; i++) {
    cout << setiosflags(ios::right) << setw(3) << (char) (i + 'A');
  }
  cout << endl;
  for (int i = 0; i < size; i++) {
    cout << setiosflags(ios::right) << setw(3) << i + 1;
    for (int j = 0; j < size; j++) {
      if (board[i][j] == 1) cout << "  x";
      if (board[i][j] == 0) cout << "  o";
      if (board[i][j] == -1) cout << "  _";
    }
    cout << endl;
  }
  cout << "Hash: " << hash << endl;
  cout << "等待 player" << o << " 落子" << endl;
}