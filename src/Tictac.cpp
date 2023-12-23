//
// Created by LiZnB on 2023/12/23.
//

#include "Tictac.h"

Tictac::Tictac(int n) : chess(3) {}

long long Tictac::try_play(int x, int y) {
  if (board[x][y] != -1) return -1;
  long long t_hash = hash ^ zobrist[x][y][o] ^ ohash;
  return t_hash;
}

int Tictac::end() {
  for (int i = 0; i < 3; i++) {
    int ok = board[i][0];
    for (int j = 0; j < 3; j++) {
      if (board[i][j] != board[i][0] || board[i][j] == -1) ok = 2;
    }
    if (ok != 2) return ok;
  }
  for (int i = 0; i < 3; i++) {
    int ok = board[0][i];
    for (int j = 0; j < 3; j++) {
      if (board[j][i] != board[0][i] || board[j][i] == -1) ok = 2;
    }
    if (ok != 2) return ok;
  }
  if (board[0][0] == board[1][1] && board[1][1] == board[2][2] && board[0][0] != -1) return board[0][0];
  if (board[0][2] == board[1][1] && board[1][1] == board[2][0] && board[0][2] != -1) return board[0][0];
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      if (board[i][j] == -1) return -1;
    }
  }
  return 2;
}

void Tictac::play(int x, int y) {
  hash ^= zobrist[x][y][o] ^ ohash;
  board[x][y] = o;
  o ^= 1;
}

void Tictac::pass() {
  o ^= 1;
  hash ^= ohash;
}