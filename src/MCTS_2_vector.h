//
// Created by LiZnB on 2023/12/26.
//
// 用 vector 替换 gp_hash_table, 几乎没有性能提升。复杂度瓶颈不在 hash 表 ?

#ifndef BETAZERO_MCTS_2_VECTOR_H
#define BETAZERO_MCTS_2_VECTOR_H

#include <bits/stdc++.h>
using namespace std;

template <typename Chess>
class MCTS_2_vector{
private:
  mt19937 seed;
  long long CLK;
  int Search_Times = 100000; // 对弈到终局的次数
  const double Confidence = 0.1; // 值越大，越偏向探索。
  double Time_Limit = 1.5; // 每一步时间限制 1s
  const int N = 1 << 20;
  vector<int> vis, win;

public:
  int search_nums; // 在时限下搜索到终点的次数
  double select_time, move_time; // 选择用时和快速移动用时
  double win_per, draw_per;

  // 初始化
  MCTS_2_vector(double time_limit, int search_limit) : Time_Limit(time_limit), Search_Times(search_limit), vis(N, 0), win(N, 0) {
    win_per = -1.0;
    draw_per = -1.0;
    search_nums = 0;
    select_time = 0.0;
    move_time = 0.0;
    random_device rd;
    seed = mt19937(rd());
  };

  // uct: son.win(father win or draw) / son.vis + sqrt(c * log(fa.vis) / son.vis)
  double uct(long long u, long long v) {
    // win[v] 表示儿子胜利，儿子没胜利表示父亲平局或者胜利。这里需要的是父亲的平局或者胜利。
    int v_win = vis[v & (N - 1)] - win[v & (N - 1)];
    double res = 1.0 * v_win / vis[v & (N - 1)] + sqrt(Confidence * log(1.0 * vis[u & (N - 1)]) / vis[v & (N - 1)]);
    return res;
  }

  // 快速走子，返回结局 0， 1， 2
  int quick_move(Chess node) {
    if (node.end() != -1) return node.end();
    vector<array<long long, 3>> expand;
    for (int i = 0; i < node.size; i++) {
      for (int j = 0; j < node.size; j++) {
        long long v = node.try_play(i, j);
        if (v == -1) continue;
        expand.push_back({i, j, v});
      }
    }
    if (expand.empty()) {
      node.pass();
      return quick_move(node);
    }
    uniform_int_distribution<int> R(0, expand.size() - 1);
    auto [x, y, v] = expand[R(seed)];
    node.play(x, y);
    return quick_move(node);
  }

  // select + expand
  pair<Chess, vector<pair<long long, int>>>select_and_expand(Chess game) {
    Chess node= game;
    vector<pair<long long, int>> path;
    path.emplace_back(game.hash, game.o);

    while (node.end() == -1) {
      vector<array<long long, 3>> select, expand;

      for (int i = 0; i < node.size; i++) {
        for (int j = 0; j < node.size; j++) {
          long long v = node.try_play(i, j);
          if (v == -1) continue;
          if (vis[v & (N - 1)] == 0) expand.push_back({i, j, v});
          else select.push_back({i, j, v});
        }
      }
      if (expand.empty() && select.empty()) {
        // 没有选择，停一手
        node.pass();
        path.emplace_back(node.hash, node.o);
      } else if (expand.empty()) {
        // uct 下一层
        sort(select.begin(), select.end(), [&](auto &i, auto &j) {
          return uct(node.hash, i[2]) < uct(node.hash, j[2]);
        });
        auto [x, y, v] = select.back();
        node.play(x, y);
        path.emplace_back(node.hash, node.o);
      } else {
        // 拓展一个，立刻返回
        uniform_int_distribution<int> R(0, (int) expand.size() - 1);
        auto [x, y, v] = expand[R(seed)];
        node.play(x, y);
        path.emplace_back(node.hash, node.o);
        return make_pair(node, path);
      }
    }
    return make_pair(node, path);
  }

  // 分析一个局面，走 Search_Times 次到终局，选出最优选择
  array<int, 2> play(Chess game) {
    assert(game.end() == -1);
    CLK = clock();
    for (search_nums = 1; search_nums <= Search_Times; search_nums++) {
      if ((double) (clock() - CLK) / CLOCKS_PER_SEC > Time_Limit - 0.05) break;
      double t1 = clock();
      auto [node, path] = select_and_expand(game);
      double t2 = clock();
      int res = quick_move(node);
      double t3 = clock();

      select_time += (t2 - t1) / CLOCKS_PER_SEC;
      move_time += (t3 - t2) / CLOCKS_PER_SEC;

      for (auto &[hash, o] : path) {
        vis[hash & (N - 1)]++;
        // 注意：不能把平局当成 win, 这样 uct 会选不出最优解
        if (o == res) win[hash & (N - 1)]++;
      }
    }
    vector<array<long long, 3>> select;
    for (int i = 0; i < game.size; i++) {
      for (int j = 0; j < game.size; j++) {
        long long v = game.try_play(i, j);
        if (v == -1) continue;
        if (vis[v & (N - 1)]) {
          select.push_back({i, j, v});
        }
      }
    }
    win_per = 1.0 * win[game.hash & (N - 1)] / vis[game.hash & (N - 1)];
    // 没有选择就停一手
    if (select.empty()) return {-1, -1};
    sort(select.begin(), select.end(), [&](auto &i, auto &j) {
      return uct(game.hash, i[2]) < uct(game.hash, j[2]);
    });
    auto [x, y, v] = select.back();

    draw_per = abs(1.0 - 1.0 * win[v & (N - 1)] / vis[v & (N - 1)] - win_per);

    return (array<int, 2>) {(int)x, (int)y};
  }

  // 在初始状态进行一次搜索，计算每秒搜索的结局个数
  double benchmark() {
    Chess game;
    double T_backup = Time_Limit;
    int S_backup = Search_Times;
    Time_Limit = 1000000;
    Search_Times = 1000; // 搜到1000个结局终止
    play(game);
    double res = search_nums * 1.0 / (select_time + move_time);
    Time_Limit = T_backup;
    Search_Times = S_backup;
    vis.clear();
    win.clear();
    search_nums = 0;
    select_time = 0;
    move_time = 0;
    return res;
  }
};

#endif //BETAZERO_MCTS_2_VECTOR_H
