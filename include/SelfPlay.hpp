#pragma once

#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <atomic>
#include <memory>
#include <random>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <ctime>
#include "NeuralNetworkMCTS.hpp"
#include "BatchNeuralNetwork.hpp"
#include "TrainingData.hpp"

// 自我对弈模块，支持模板化的棋盘游戏
template<typename GameType>
class SelfPlay {
public:
    struct SelfPlayConfig {
        int numGames;           // 要进行的对局数量
        int numSimulations;     // 每个动作的MCTS模拟次数
        bool useDataAugmentation; // 是否使用数据增强
        std::string outputDir;  // 训练数据输出目录
        
        SelfPlayConfig() 
            : numGames(256), numSimulations(1000), 
              useDataAugmentation(true),
              outputDir("data") {}
    };

    SelfPlay(const std::string& modelPath, const SelfPlayConfig& config = SelfPlayConfig())
        : config(config) {
        
        std::filesystem::create_directories(config.outputDir);

        network = std::make_shared<BatchNeuralNetwork>(
            modelPath, 
            GameType::BOARD_SIZE,
            3,
            true,
            config.numGames
        );
    }
    
    // 析构函数
    ~SelfPlay() {
        shouldStop = true;
        for (auto& thread : threads) {
            if (thread.joinable()) {
                thread.join();
            }
        }
    }
    
    // run函数
    void run() {
        auto startTime = std::chrono::steady_clock::now();
        gamesCompleted = 0;
        shouldStop = false;
        trainingData.clear();

        threads.clear();
        for (int i = 0; i < config.numGames; ++i) {
            threads.emplace_back(&SelfPlay::workerThread, this, i);
        }

        for (auto& thread : threads) {
            thread.join();
        }

        auto endTime = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed = endTime - startTime;

        {
            std::lock_guard<std::mutex> lock(consoleMutex);
            std::cout << "\n自我对弈完成！" << std::endl;
            std::cout << "总游戏数: " << gamesCompleted << std::endl;
            std::cout << "总样本数: " << trainingData.size() << std::endl;
            std::cout << "总耗时: " << elapsed.count() << " 秒" << std::endl;
            std::cout << "平均每局耗时: " << (elapsed.count() / gamesCompleted) << " 秒" << std::endl;
        }
    }
    
    const TrainingData<GameType>& getTrainingData() const { return trainingData; }
    
    bool saveData(const std::string& filename = "") const {
        if (trainingData.size() == 0) {
            std::cerr << "没有数据可保存！" << std::endl;
            return false;
        }
        
        std::string outputFilename = filename;
        if (outputFilename.empty()) {
            // 生成基于时间戳的文件名
            auto now = std::chrono::system_clock::now();
            auto time_t_now = std::chrono::system_clock::to_time_t(now);
            std::tm* local_time = std::localtime(&time_t_now);
            
            std::ostringstream oss;
            oss << config.outputDir << "/selfplay_" 
                << (local_time->tm_year + 1900) 
                << std::setw(2) << std::setfill('0') << (local_time->tm_mon + 1)
                << std::setw(2) << std::setfill('0') << local_time->tm_mday
                << "_"
                << std::setw(2) << std::setfill('0') << local_time->tm_hour
                << std::setw(2) << std::setfill('0') << local_time->tm_min
                << ".data";
            outputFilename = oss.str();
        }
        
        bool success = trainingData.saveToFile(outputFilename);
        if (success) {
            std::cout << "训练数据已保存到: " << outputFilename << std::endl;
            std::cout << "样本数量: " << trainingData.size() << std::endl;
        } else {
            std::cerr << "保存训练数据失败！" << std::endl;
        }
        
        return success;
    }

private:
    SelfPlayConfig config;
    std::shared_ptr<BatchNeuralNetwork> network;
    TrainingData<GameType> trainingData;
    std::mutex dataMutex;
    std::mutex consoleMutex;
    std::vector<std::thread> threads;
    std::atomic<int> gamesCompleted{0};
    std::atomic<bool> shouldStop{false};

    // 工作线程函数
    void workerThread(int threadId) {
        try {
            if (!shouldStop) {
                std::random_device rd;
                std::mt19937 rng(rd() + threadId);
                
                // 执行一场对弈
                int samplesCollected = executeEpisode(threadId, rng);

                // 更新游戏完成计数
                int completed = ++gamesCompleted;

                {
                    std::lock_guard<std::mutex> lock(consoleMutex);
                    std::cout << "\r游戏进度: " << completed << "/" << config.numGames 
                              << " (" << std::fixed << std::setprecision(1) 
                              << (completed * 100.0f / config.numGames) << "%), "
                              << "线程 " << threadId << " 完成游戏, 样本: " << samplesCollected
                              << ", 活跃线程: " << network->getActiveThreads();

                    {
                        std::lock_guard<std::mutex> dataLock(dataMutex);
                        std::cout << ", 总样本: " << trainingData.size() << " ";
                    }
                    std::cout << std::flush;
                }
            }
        }
        catch (const std::exception& e) {
            network->threadFinished();  // 确保异常时也通知线程结束
            std::cerr << "线程 " << threadId << " 发生异常: " << e.what() << std::endl;
        }
    }

    // 执行一场对弈
    int executeEpisode(int gameId, std::mt19937& rng) {
        GameType game;
        std::vector<std::vector<float>> gameStates;
        std::vector<std::vector<float>> gamePolicies;
        std::vector<int> gameCurrentPlayers;
        
        // 创建MCTS实例
        auto mcts = std::make_shared<NeuralNetworkMCTS<GameType, BatchNeuralNetwork>>(network);
        
        int moveCount = 0;
        
        // KataGo风格的温度策略
        // 计算棋盘宽度
        int boardSize = game.getBoardSize();
        
        // 计算开局随机落子步数r，服从均值为0.04*b²的指数分布
        std::exponential_distribution<float> exp_dist(1.0f / (0.04f * boardSize * boardSize));
        int randomMoveCount = std::round(exp_dist(rng));
        
        // 初始温度为0.8，最终温度为0.2
        float initialTemp = 0.8f;
        float finalTemp = 0.2f;
        
        // 温度半衰期等于棋盘宽度的步数
        float halfLifeSteps = static_cast<float>(boardSize);
        // 计算每步的衰减系数
        float decayFactor = std::pow(0.5f, 1.0f / halfLifeSteps);
        
        // 当前温度，从第r+1步开始使用
        float currentTemp = initialTemp;
        
        // 执行游戏直到结束
        while (game.getGameState() == -1) {
            std::vector<float> currentState = game.toTensor();
            gameStates.push_back(currentState);
            gameCurrentPlayers.push_back(game.getCurrentPlayer());

            mcts->reset();

            // 执行MCTS搜索并记录耗时
            auto search_start = std::chrono::high_resolution_clock::now();

            // 执行MCTS搜索
            for (int i = 0; i < config.numSimulations; ++i) {
                mcts->search(game, i == 0);
            }

            auto search_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> search_time = search_end - search_start;

            // printf("步数:%d, MCTS搜索耗时: %.2fms, 单次search用时: %.2fms\n", moveCount, search_time.count(), 1.0 * search_time.count() / config.numSimulations);

            // 获取MCTS策略
            std::vector<std::pair<std::pair<int, int>, float>> actionProbs;
            std::vector<float> flatPolicy(game.getBoardSize() * game.getBoardSize(), 0.0f);

            // 从MCTS获取所有合法动作的访问计数
            auto validMoves = game.getValidMoves();
            for (const auto& move : validMoves) {
                int x = move.first;
                int y = move.second;
                GameType tempGame = game;
                tempGame.makeMove(x, y);
                uint64_t hash = tempGame.getHash();

                int visits = mcts->getVisitCount(hash);
                if (visits > 0) {
                    actionProbs.push_back({{x, y}, static_cast<float>(visits)});
                }
            }

            // 计算总访问次数和生成策略
            float totalVisits = 0.0f;
            for (const auto& actionProb : actionProbs) {
                totalVisits += actionProb.second;
            }

            if (totalVisits < 1e-6) {
                float uniformProb = 1.0f / validMoves.size();
                for (const auto& move : validMoves) {
                    int x = move.first;
                    int y = move.second;
                    flatPolicy[x * game.getBoardSize() + y] = uniformProb;
                }
            } else {
                for (const auto& actionProb : actionProbs) {
                    int x = actionProb.first.first;
                    int y = actionProb.first.second;
                    float prob = actionProb.second / totalVisits;
                    flatPolicy[x * game.getBoardSize() + y] = prob;
                }
            }

            // 选择动作
            // KataGo风格的温度策略
            if (moveCount < randomMoveCount) {
                // 开局阶段使用温度T=1的策略（从pi分布中采样）
                std::vector<float> probs;
                std::vector<std::pair<int, int>> moves;

                for (const auto& actionProb : actionProbs) {
                    moves.push_back(actionProb.first);
                    // 温度为1，直接使用原始访问计数的比例
                    probs.push_back(actionProb.second);
                }

                float sum = 0.0f;
                for (float p : probs) {
                    sum += p;
                }

                if (sum > 0.0f && !moves.empty()) {
                    for (float& p : probs) {
                        p /= sum;
                    }

                    std::discrete_distribution<int> actionDist(probs.begin(), probs.end());
                    int selectedIdx = actionDist(rng);
                    auto selectedMove = moves[selectedIdx];
                    game.makeMove(selectedMove.first, selectedMove.second);
                } else {
                    game.pass();
                }
            } else {
                // 使用当前温度选择动作
                if (validMoves.empty()) {
                    game.pass();
                } else {
                    std::vector<float> probs;
                    std::vector<std::pair<int, int>> moves;

                    for (const auto& actionProb : actionProbs) {
                        moves.push_back(actionProb.first);
                        probs.push_back(std::pow(actionProb.second, 1.0f / currentTemp));
                    }

                    float sum = 0.0f;
                    for (float p : probs) {
                        sum += p;
                    }

                    if (sum > 0.0f) {
                        for (float& p : probs) {
                            p /= sum;
                        }

                        std::discrete_distribution<int> actionDist(probs.begin(), probs.end());
                        int selectedIdx = actionDist(rng);
                        auto selectedMove = moves[selectedIdx];
                        game.makeMove(selectedMove.first, selectedMove.second);
                    } else {
                        game.pass();
                    }
                }
            }

            gamePolicies.push_back(flatPolicy);
            moveCount++;
            
            // 更新温度（如果已经超过随机落子阶段）
            if (moveCount >= randomMoveCount) {
                currentTemp *= decayFactor;
                // 确保温度不低于最小值
                currentTemp = std::max(currentTemp, finalTemp);
            }
        }

        // 后续的数据处理不需要神经网络推理
        network->threadFinished();

        // 游戏结束，计算奖励
        int gameResult = game.getGameState();
        std::vector<float> rewards(gameStates.size());

        if (gameResult == 2) {  // 平局
            std::fill(rewards.begin(), rewards.end(), 0.0f);
        } else {
            for (size_t i = 0; i < gameStates.size(); ++i) {
                int playerI = gameCurrentPlayers[i];
                bool playerWon = ((gameResult == 0 && playerI == 0) || (gameResult == 1 && playerI == 1));
                rewards[i] = playerWon ? 1.0f : -1.0f;
            }
        }

        // 添加数据到训练集
        int samplesAdded = 0;
        {
            std::lock_guard<std::mutex> lock(dataMutex);
            for (size_t i = 0; i < gameStates.size(); ++i) {
                using SampleType = typename TrainingData<GameType>::Sample;
                SampleType sample(gameStates[i], gamePolicies[i], rewards[i]);

                if (config.useDataAugmentation) {
                    auto augmentedSamples = trainingData.augmentSample(sample);
                    for (const auto& augSample : augmentedSamples) {
                        trainingData.addSample(augSample);
                        samplesAdded++;
                    }
                } else {
                    trainingData.addSample(sample);
                    samplesAdded++;
                }
            }
        }

        return samplesAdded;
    }
};