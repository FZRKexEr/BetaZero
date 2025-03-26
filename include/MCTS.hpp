#pragma once

#include <unordered_map>
#include <vector>
#include <utility>
#include <cmath>
#include <limits>
#include <algorithm>
#include <random>
#include <memory>
#include <cassert>
#include "BoardGame.h"

// MCTS 基类
template <typename GameType>
class MCTS {
protected:
    // 核心数据结构
    std::unordered_map<uint64_t, int> N;      // 节点访问次数
    std::unordered_map<uint64_t, float> Q;    // 节点预估价值
    
    float cpuct;                              // 探索常数

public:
    // 构造函数
    MCTS(float cpuct = 5.0f) : cpuct(cpuct) {}
    
    // 虚析构函数
    virtual ~MCTS() = default;

    virtual void reset() = 0; 
    
    // 核心搜索方法
    float search(GameType game, bool isRoot = false) {
        // 获取当前状态的哈希值
        uint64_t currentHash = game.getHash();
        
        // 检查游戏是否结束
        int gameState = game.getGameState();
        if (gameState != -1) {  // 游戏已结束
            int currentPlayer = game.getCurrentPlayer();
            float reward;
            if (gameState == 2) { // 平局
                reward = 0.0f;
            } else if (gameState == currentPlayer) { // 当前玩家胜利
                reward = 1.0f;
            } else { // 当前玩家失败
                reward = -1.0f;
            }
            
            update(currentHash, reward);
            return reward;
        }
        
        // 如果是新节点，评估并返回
        if (N.find(currentHash) == N.end()) {
            float value = evaluateLeaf(game);
            update(currentHash, value);
            return value;
        }
        
        // 获取所有合法动作
        auto validMoves = game.getValidMoves();
        if (validMoves.empty()) {
            // 无合法着法，必须跳过
            game.pass();
            float value = -search(game); // 负号是因为对手的价值与我方相反
            update(currentHash, value);
            return value;
        }
        
        // 计算所有合法动作的总访问次数
        int totalVisits = 0;
        for (const auto& move : validMoves) {
            GameType nextGame = game;
            nextGame.makeMove(move.first, move.second);
            uint64_t nextHash = nextGame.getHash();
            totalVisits += N.count(nextHash) ? N[nextHash] : 0;
        }
        
        totalVisits = std::max(1, totalVisits); // 避免除以零
        // 这里 totalVisits 可能为0，比如当前节点在上一次search中第一次访问到.

        // 选择UCT值最高的动作
        float bestValue = -std::numeric_limits<float>::max();
        std::pair<int, int> bestMove = {-1, -1};

        if (isRoot) { // 添加 DirichletNoise, 只会添加一次
            addDirichletNoise(currentHash, game);    
        }

        // 选择最佳行动
        for (const auto& move : validMoves) {
            GameType nextGame = game;
            if (nextGame.makeMove(move.first, move.second)) {
                uint64_t nextHash = nextGame.getHash();

                // 计算UCT值
                float prior = getPrior(currentHash, move);
                float uct = calculateUCT(nextHash, prior, totalVisits);

                if (uct > bestValue) {
                    bestValue = uct;
                    bestMove = move;
                }
            }
        }
        
        // 执行最佳行动
        game.makeMove(bestMove.first, bestMove.second);
        float value = -search(game, false);
        
        // 更新统计信息
        update(currentHash, value);
        return value;
    }
    
    // 获取节点访问次数
    int getVisitCount(uint64_t hash) const {
        auto it = N.find(hash);
        if (it != N.end()) {
            return it->second;
        }
        return 0;
    }
    
    // 获取节点估值
    float getValue(uint64_t hash) const {
        auto it = Q.find(hash);
        if (it != Q.end()) {
            return it->second;
        }
        return 0.0f;
    }
    
protected:
    // 更新节点统计信息
    void update(uint64_t hash, float value) {
        Q[hash] = (Q[hash] * N[hash] + value) / (N[hash] + 1);
        N[hash] += 1;
    }
    
    // 计算UCT值
    float calculateUCT(uint64_t hashValue, float prior, int totalVisits) const {
        // 如果节点未被访问过，Q值为0，通过先验概率和访问次数控制探索
        if (N.find(hashValue) == N.end()) {
            return cpuct * prior * std::sqrt(static_cast<float>(totalVisits));
        }
        
        // UCT公式: Q + cpuct * P * sqrt(总访问次数) / (1 + 当前节点访问次数)
        // Q.at(hashValue) 是对手视角的Q,取相反数得到我方视角的Q
        return -Q.at(hashValue) + 
               cpuct * prior * std::sqrt(static_cast<float>(totalVisits)) / 
               (1.0f + static_cast<float>(N.at(hashValue)));
    }
    
    // 获取先验概率 - 基类中返回默认值，派生类可重写
    virtual float getPrior(uint64_t stateHash, const std::pair<int, int>& action) const {
        return 1.0f;
    }
    
    // 评估叶节点 - 必须由派生类实现
    virtual float evaluateLeaf(GameType& game) = 0;

    // 基类中默认不添加 DirichletNoise    
    virtual void addDirichletNoise(uint64_t hash, GameType& game) { return; }
};
