#include "Tictac.h"
#include "Othello.h"
#include "MCTS.h"
#include <bits/stdc++.h>
using namespace std;

using Chess = Othello;

class Human {
public:
  int search_nums;
  double select_time, move_time;
  double win_per, draw_per;

  Human() : search_nums(0), select_time(0), move_time(0), win_per(0), draw_per(0) {}

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

// 判断类型是否是 Human
template <typename T>
struct IsHuman : std::is_same<T, Human> {};

template <typename T0, typename T1>
int play_once(T0 player0, T1 player1, bool display) {
  Chess game;
  while (game.end() == -1) {
    if (display) game.display();
    array<int, 2> res = game.o ? player1.play(game) : player0.play(game);

    if (display) {
      cout << endl;
      cout << "落子 " << (char)(res[1] + 'A') << res[0] + 1 << endl;
      cout << "获胜概率: " << (game.o ? player1.win_per : player0.win_per) * 100 << "%" << endl;
      cout << "平局概率: " << (game.o ? player1.draw_per : player0.draw_per) * 100 << "%" << endl;
      cout << "搜索量: " << (game.o ? player1.search_nums : player0.search_nums) << endl;
      cout << "Select+Expand 用时: " << (game.o ? player1.select_time : player0.select_time) << endl;
      cout << "Quick Move 用时: " << (game.o ? player1.move_time : player0.move_time) << endl;
    }

    // 判断人类是否输入的 (x, y) 不合法, 输入不合法 (x, y) 默认为停一手
    if (res[0] == -1 && res[1] == -1) {
      game.pass();
      continue;
    }
    // 判断返回的位置按照棋类规则是否能落子 (主要是避免人类错误落子), 不按规则默认停一手。
    if (game.try_play(res[0], res[1]) != -1) {
      game.play(res[0], res[1]);
    } else {
      cerr << "Error: 不能落在这里" << endl;
    }
  }
  if (display) game.display();
  return game.end();
}

// 对弈 cnt 次, 每次思考时间，搜索数量上限，是否展示棋局

void test(int cnt, double time_limit, int search_limit, bool display) {
  int cnt_mcts = 0, cnt_next = 0;
  int cnt_draw = 0;
  for (int i = 1; i <= cnt; i++) {
    cout << "正在进行第 " << i << " 场对弈" << endl;

    MCTS<Chess> player0(time_limit, search_limit);
    MCTS<Chess> player1(time_limit, search_limit);

    int res = play_once(player1, player0, display);
    if (res == 2) cout << "和棋" << endl, cnt_draw++;
    if (res == 1) cout << "player0 执黑胜" << endl, cnt_next++;
    if (res == 0) cout << "player1 执白胜" << endl, cnt_mcts++;

    res = play_once(player0, player1, display);
    if (res == 2) cout << "和棋" << endl, cnt_draw++;
    if (res == 1) cout << "player1 执黑胜" << endl, cnt_mcts++;
    if (res == 0) cout << "player0 执白胜" << endl, cnt_next++;
  }
  cout << "mcts:mcts_next = " << cnt_mcts << ":" << cnt_next << endl;
}

int main() {


  MCTS<Chess> bench(1, 100000);
  cout << bench.benchmark() << endl;

  test(1, 1, 100000, true);

  return 0;



  return 0;
}