#include "Tictac.h"
#include "Othello.h"
#include "MCTS.h"
#include <bits/stdc++.h>
using namespace std;

using Chess = Othello;

class Human {
public:
  int search_nums;
  Human() : search_nums(0) {}

  // 返回一个 x, y 合法的位置, 但不保证按照棋类的规则可以落子。
  array<int, 2> play(const Chess& game) {
    string pos; cin >> pos;
    if (!isupper(pos[0])) return {-1, -1};
    if (!isdigit(pos[1])) return {-1, -1};
    array<int, 2> res{pos[1] - '0' - 1, pos[0] - 'A'};
    if (res[0] >= game.size) return {-1, -1};
    if (res[1] >= game.size) return {-1, -1};
    return res;
  }
};

void self_play() {
  Chess game(8);
  while (game.end() == -1) {
    game.display();

    MCTS<Chess> player0;
    MCTS<Chess> player1;

    array<int, 2> res = game.o ? player1.play(game) : player0.play(game);
    cout << endl;
    cout << "获胜概率: " << (game.o ? player1.win_per: player0.win_per) * 100 << "%" << endl;
    cout << "平局概率: " << (game.o ? player1.draw_per: player0.draw_per) * 100 << "%" <<  endl;
    cout << "搜索量: " << (game.o ? player1.search_nums : player0.search_nums) << endl;
    cout << "Select+Expand 用时: " << (game.o ? player1.select_time: player0.select_time) << endl;
    cout << "Quick Move 用时: " << (game.o ? player1.move_time: player0.move_time) << endl;

    // 判断人类是否输入的 (x, y) 不合法, 输入不合法 (x, y) 默认为停一手
    if (res[0] == -1 && res[1] == -1) {
      game.pass();
      continue;
    }
    // 判断返回的位置按照棋类规则是否能落子 (主要是避免人类错误落子), 不按规则默认停一手。
    if (game.try_play(res[0], res[1]) != -1) {
      game.play(res[0], res[1]);
    } else {
      cout << "不能落在这里" << endl;
    }
  }
  game.display();

  if (game.end() == 2) cout << "和棋" << endl;
  if (game.end() == 1) cout << "x 胜利" << endl;
  if (game.end() == 0) cout << "o 胜利" << endl;
}

int main() {

  self_play();

  return 0;
}