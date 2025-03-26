#pragma once

#include "MCTS.hpp"
#include "NeuralNetwork.hpp"
#include <memory>
#include <array>


template <typename GameType, typename NN>
class NeuralNetworkMCTS : public MCTS<GameType> {
private:
    // 神经网络模型
    std::shared_ptr<NN> network;
    
    // 使用模板获取棋盘大小
    static constexpr int BOARD_SIZE = GameType::BOARD_SIZE;
    
    // 先验概率缓存 - 改用固定大小的std::array
    std::unordered_map<uint64_t, std::array<float, BOARD_SIZE * BOARD_SIZE>> P;

public:
    NeuralNetworkMCTS(std::shared_ptr<NN> network, float cpuct = 5.0f) 
        : MCTS<GameType>(cpuct), network(network) {}

    void reset() override {
        this->Q.clear();
        this->N.clear();
        P.clear();
    }

protected:
    // 实现神经网络评估
    float evaluateLeaf(GameType& game) override {
        auto state = game.toTensor();
        auto [value, policy] = network->predict(state);
        
        uint64_t stateHash = game.getHash();
        auto validMoves = game.getValidMoves();
        
        // 为当前状态创建固定大小的概率数组
        std::array<float, BOARD_SIZE * BOARD_SIZE> probs{};
        probs.fill(0.0f); // 初始化为0
        
        // 只为合法着法分配概率
        if (!validMoves.empty()) {
            // 计算合法招法的总概率
            float totalProb = 0.0f;
            for (const auto& move : validMoves) {
                int index = move.first * BOARD_SIZE + move.second;
                if (index >= 0 && index < policy.size()) {
                    totalProb += policy[index];
                }
            }
            
            // 归一化概率
            if (totalProb < 1e-6) { // 防止除零错误, 此时说明神经网络的输出概率很小
                float uniformProb = 1.0f / validMoves.size();
                for (const auto& move : validMoves) {
                    int index = move.first * BOARD_SIZE + move.second;
                    probs[index] = uniformProb;
                }
            } else {
                for (const auto& move : validMoves) {
                    int index = move.first * BOARD_SIZE + move.second;
                    if (index >= 0 && index < policy.size()) {
                        probs[index] = policy[index] / totalProb;
                    }
                }
            }
        }
        
        // 保存概率数组
        P[stateHash] = probs;
        
        return value;
    }
    
    // 重写获取先验概率的方法
    float getPrior(uint64_t stateHash, const std::pair<int, int>& action) const override {
        auto stateIt = P.find(stateHash);
        if (stateIt != P.end()) {
            int index = action.first * BOARD_SIZE + action.second;
            if (index >= 0 && index < BOARD_SIZE * BOARD_SIZE) {
                return stateIt->second[index];
            }
        }
        return 1.0f;  // 默认先验概率, 其实不应该走到这里
    }

    void addDirichletNoise(uint64_t stateHash, GameType& game) override {
        auto it = P.find(stateHash);
        if (it == P.end()) return;
        
        auto& priors = it->second;
        auto validMoves = game.getValidMoves();
        
        // 生成Dirichlet噪声
        std::vector<float> noise(validMoves.size());
        float alpha = 0.3f; // 黑白棋推荐参数
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::gamma_distribution<float> gamma(alpha, 1.0f);
        
        float noiseSum = 0.0f;
        for (size_t i = 0; i < validMoves.size(); ++i) {
            noise[i] = gamma(gen);
            noiseSum += noise[i];
        }
        
        // 归一化噪声
        for (auto& n : noise) {
            n /= noiseSum;
        }
        
        // 混合噪声与原始先验概率
        float epsilon = 0.25f; // 混合参数
        size_t i = 0;
        for (const auto& move : validMoves) {
            int index = move.first * BOARD_SIZE + move.second;
            if (index >= 0 && index < BOARD_SIZE * BOARD_SIZE) {
                priors[index] = (1.0f - epsilon) * priors[index] + epsilon * noise[i++];
            }
        }
    }
};