#pragma once

#include "MCTS.hpp"
#include <random>

template <typename GameType>
class PureMCTS : public MCTS<GameType> {
private:
    std::mt19937 rng;

public:
    // 构造函数，使用较小的cpuct值
    PureMCTS(float cpuct = 0.1f) : MCTS<GameType>(cpuct), rng(std::random_device{}()) {}

    void reset() override {
        this->N.clear();
        this->Q.clear();
    }

protected:
    // 实现随机模拟的叶节点评估
    float evaluateLeaf(GameType& game) override {
        return randomRollout(game);
    }
    
private:
    // 随机走子直到游戏结束
    float randomRollout(GameType game) {
        int gameState = game.getGameState();
        int originalPlayer = game.getCurrentPlayer();
        
        // 快速模拟到游戏结束
        while (gameState == -1) {
            auto moves = game.getValidMoves();
            if (moves.empty()) {
                game.pass();
            } else {
                // 随机选择一个合法着法
                std::uniform_int_distribution<> dist(0, moves.size() - 1);
                auto move = moves[dist(rng)];
                game.makeMove(move.first, move.second);
            }
            gameState = game.getGameState();
        }
        
        // 从最初玩家的角度计算结果
        if (gameState == 2) { // 平局
            return 0.0f;
        } else if (gameState == originalPlayer) { // 原始玩家胜利
            return 1.0f;
        } else { // 原始玩家失败
            return -1.0f;
        }
    }
};