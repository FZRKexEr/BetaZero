//
// Created by LiZnB on 2023/12/23.
//

#include "Othello.h"

Othello::Othello(int n) : chess(n) {
  board[(size - 1) / 2][(size - 1) / 2] = 0;
  board[(size - 1) / 2][size / 2] = 1;
  board[size / 2][(size - 1) / 2] = 1;
  board[size / 2][size / 2] = 0;
  hash ^= zobrist[(size - 1) / 2][(size - 1) / 2][0];
  hash ^= zobrist[(size - 1) / 2][size / 2][1];
  hash ^= zobrist[size / 2][(size - 1) / 2][1];
  hash ^= zobrist[size / 2][size / 2][0];
};

vector<array<int, 2>> Othello::flip_once(int color, int x, int y) {
  vector<array<int, 2>> res;
  for (int i = 0; i < 8; i++) {
    vector<array<int, 2>> tque;
    for (int j = 1;; j++) {
      int nx = x + dx[i] * j;
      int ny = y + dy[i] * j;
      if (nx >= size || ny >= size || nx < 0 || ny < 0 || board[nx][ny] == -1) {
        tque.clear();
        break;
      }
      if (board[nx][ny] == color) break;
      if (board[nx][ny] == (color ^ 1)) tque.push_back({nx, ny});
    }
    res.insert(res.end(), tque.begin(), tque.end());
  }
  return res;
}

long long Othello::try_play(int x, int y) {
  if (board[x][y] != -1) return -1ll;
  vector<vector<int>> temp = board;
  long long t_hash = hash ^ zobrist[x][y][o] ^ ohash;
  vector<array<int, 2>> res = flip_once(o, x, y);
  if (res.empty()) return -1ll;
  for (auto &it: res) {
    t_hash ^= (zobrist[it[0]][it[1]][0] ^ zobrist[it[0]][it[1]][1]);
  }
  return t_hash;
}
int Othello::end() {
  array<int, 2> cnt = {0, 0};
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      if (board[i][j] != -1) {
        cnt[board[i][j]]++;
        continue;
      }
      vector<vector<int>> temp = board;
      auto res1 = flip_once(1, i, j);
      auto res0 = flip_once(0, i, j);
      if (!res1.empty() || !res0.empty()) return -1;
    }
  }
  if (cnt[0] > cnt[1]) return 0;
  if (cnt[0] < cnt[1]) return 1;
  return 2;
}

void Othello::play(int x, int y) {
  hash ^= zobrist[x][y][o] ^ ohash;
  board[x][y] = o;
  vector<array<int, 2>> res = flip_once(o, x, y);
  for (auto &it: res) {
    board[it[0]][it[1]] = o;
    hash ^= (zobrist[it[0]][it[1]][0] ^ zobrist[it[0]][it[1]][1]);
  }
  o ^= 1;
}

void Othello::pass() {
  o ^= 1;
  hash ^= ohash;
}