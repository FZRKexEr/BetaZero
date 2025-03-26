#pragma once

#include <string>
#include <memory>
#include <vector>
#include <random>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>
#include <fstream> // 添加文件操作头文件
#include <filesystem> // 添加文件系统头文件
#include "NeuralNetworkMCTS.hpp"
#include "BatchNeuralNetwork.hpp"

// 模型评估器，支持模板化的棋盘游戏
template<typename GameType>
class ModelEvaluator {
public:
    struct EvaluationConfig {
        int numGames;          // 评估对局数量
        int numSimulations;    // 每步MCTS模拟次数
        int numThreads;        // 线程数
        bool displayProgress;  // 是否显示进度
        std::string outputDir; // 输出目录路径（更改名称使其更直观）

        EvaluationConfig() 
            : numGames(2), numSimulations(1000), numThreads(2), 
              displayProgress(true), outputDir("evaluations") {}
    };
    
    struct EvaluationResult {
        int newModelWins;      // 新模型获胜局数
        int newModelFirstWins; // 新模型执黑获胜局数
        int newModelFirst;     // 新模型执黑局数
        int newModelSecondWins;// 新模型执白获胜局数
        int newModelSecond;    // 新模型执白局数
        int oldModelWins;      // 旧模型获胜局数
        int draws;             // 平局数
        float winRate;         // 新模型胜率
        
        EvaluationResult() 
            : newModelWins(0), newModelFirstWins(0), newModelFirst(0),
              newModelSecondWins(0), newModelSecond(0), draws(0), winRate(0.0f) {}
    };
    
    // 构造函数
    ModelEvaluator(const std::string& newModelPath, 
                  const std::string& oldModelPath, 
                  const EvaluationConfig& config = EvaluationConfig())
        : config(config) {
        
        // 初始化随机数生成器
        std::random_device rd;
        
        // 加载模型
        newNetwork = std::make_shared<BatchNeuralNetwork>(
            newModelPath,
            GameType::BOARD_SIZE,
            3,  // 默认使用3个输入通道
            true,  // 默认使用GPU
            config.numThreads 
        );
        
        oldNetwork = std::make_shared<BatchNeuralNetwork>(
            oldModelPath,
            GameType::BOARD_SIZE,
            3,  // 默认使用3个输入通道
            true,  // 默认使用GPU
            config.numThreads
        );
        
        std::cout << "模型评估器初始化完成:" << std::endl;
        std::cout << "  - 新模型: " << newModelPath << std::endl;
        std::cout << "  - 旧模型: " << oldModelPath << std::endl;
        std::cout << "  - 评估对局数: " << config.numGames << std::endl;
        std::cout << "  - 每步模拟次数: " << config.numSimulations << std::endl;
        if (!config.outputDir.empty()) {
            std::cout << "  - 结果输出目录: " << config.outputDir << std::endl;
        }

    }
    // 执行评估
    EvaluationResult evaluate() {
        auto startTime = std::chrono::steady_clock::now();
        EvaluationResult result;
        
        std::cout << "开始评估..." << std::endl;
        std::cout << "使用 " << config.numThreads << " 个线程并行评估" << std::endl;
        
        // 线程安全相关
        std::mutex resultMutex;
        std::mutex consoleMutex;
        std::atomic<int> gamesCompleted{0};
        
        // 每个线程要处理的游戏数量
        int gamesPerThread = (config.numGames + config.numThreads - 1) / config.numThreads;
        
        // 创建线程
        std::vector<std::thread> threads;
        for (int t = 0; t < config.numThreads; ++t) {
            threads.push_back(std::thread([this, t, gamesPerThread, &result, &resultMutex, 
                                           &consoleMutex, &gamesCompleted]() {
                // 为每个线程创建独立的随机数生成器
                std::random_device rd;
                std::mt19937 threadRng(rd() + t);
                
                // 计算该线程负责的游戏范围
                int startGame = t * gamesPerThread;
                int endGame = std::min((t + 1) * gamesPerThread, config.numGames);
                
                for (int i = startGame; i < endGame; ++i) {
                    // 交替让新模型和旧模型先手
                    bool newModelFirst = (i % 2 == 0);
                    
                    // 执行游戏
                    int gameResult = playGame(newModelFirst, threadRng);
                    
                    // 线程安全地更新结果
                    {
                        std::lock_guard<std::mutex> lock(resultMutex);
                        if (newModelFirst) {
                            result.newModelFirst++;
                        } else {
                            result.newModelSecond++;
                        }
                        if (gameResult == 1) {
                            result.newModelWins++;
                            if (newModelFirst) {
                                result.newModelFirstWins++;
                            } else {
                                result.newModelSecondWins++;
                            }
                        } else if (gameResult == -1) {
                            result.oldModelWins++;
                        } else {
                            result.draws++;
                        }
                        
                        // 更新已完成游戏数
                        int completed = ++gamesCompleted;
                        
                        // 重新计算胜率
                        int totalGames = result.newModelWins + result.oldModelWins + result.draws;
                        result.winRate = (result.newModelWins + 0.5f * result.draws) / totalGames;
                        
                        // 更新进度显示
                        if (config.displayProgress) {
                            std::lock_guard<std::mutex> consoleLock(consoleMutex);
                            std::cout << "\r评估进度: " << completed << "/" << config.numGames
                                      << " - 新模型胜: " << result.newModelWins 
                                      << ", 旧模型胜: " << result.oldModelWins
                                      << ", 平局: " << result.draws
                                      << std::fixed << std::setprecision(2)
                                      << ", 执黑胜率: " << result.newModelFirstWins * 100.0f / result.newModelFirst << "%"
                                      << ", 执白胜率: " << result.newModelSecondWins * 100.0f / result.newModelSecond << "%"
                                      << ", 总胜率: "  
                                      << (result.winRate * 100.0f) << "%" << std::flush;
                        }
                    }
                }
            }));
        }
        
        // 等待所有线程完成
        for (auto& thread : threads) {
            thread.join();
        }
        
        auto endTime = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed = endTime - startTime;
        
        std::cout << std::endl;
        std::cout << "评估完成，耗时: " << elapsed.count() << " 秒" << std::endl;
        
        printResult(result);
        
        // 如果指定了输出目录，则将结果写入文件
        if (!config.outputDir.empty()) {
            writeResultToFile(result, elapsed.count());
        }
        
        return result;
    }
    
private:
    // 将结果写入文件
    void writeResultToFile(const EvaluationResult& result, double timeElapsed) const {
        // 确保输出目录存在
        std::filesystem::path outputDir;
        outputDir = config.outputDir; 
        
        // 创建目录（如果不存在）
        std::filesystem::create_directories(outputDir);
        
        // 生成基于时间戳的文件名
        auto now = std::chrono::system_clock::now();
        auto time_t_now = std::chrono::system_clock::to_time_t(now);
        std::tm* local_time = std::localtime(&time_t_now);
        
        std::ostringstream oss;
        oss << outputDir.string() << "/evaluation_" 
            << (local_time->tm_year + 1900) 
            << std::setw(2) << std::setfill('0') << (local_time->tm_mon + 1)
            << std::setw(2) << std::setfill('0') << local_time->tm_mday
            << "_"
            << std::setw(2) << std::setfill('0') << local_time->tm_hour
            << std::setw(2) << std::setfill('0') << local_time->tm_min
            << std::setw(2) << std::setfill('0') << local_time->tm_sec
            << ".json";
        
        std::string outputFilename = oss.str();
        
        // 打开输出文件
        std::ofstream outFile(outputFilename);
        if (!outFile) {
            std::cerr << "无法创建输出文件: " << outputFilename << std::endl;
            return;
        }
        
        // 生成JSON格式的结果数据
        outFile << "{\n";
        outFile << "  \"time_seconds\": " << timeElapsed << ",\n";
        outFile << "  \"total_games\": " << config.numGames << ",\n";
        outFile << "  \"new_model_wins\": " << result.newModelWins << ",\n";
        outFile << "  \"new_model_win_percent\": " << (result.newModelWins * 100.0f / config.numGames) << ",\n";
        outFile << "  \"old_model_wins\": " << result.oldModelWins << ",\n";
        outFile << "  \"old_model_win_percent\": " << (result.oldModelWins * 100.0f / config.numGames) << ",\n";
        outFile << "  \"draws\": " << result.draws << ",\n";
        outFile << "  \"draw_percent\": " << (result.draws * 100.0f / config.numGames) << ",\n";
        outFile << "  \"win_rate\": " << (result.winRate * 100.0f) << "\n";
        outFile << "}";
        
        std::cout << "结果已写入文件: " << outputFilename << std::endl;
    }
    
    // 执行一场对弈，返回结果（1: 新模型胜, -1: 旧模型胜, 0: 平局）
    int playGame(bool newModelFirst, std::mt19937& gameRng) {
        GameType game;

        // 创建MCTS实例
        auto newModelMCTS = std::make_shared<NeuralNetworkMCTS<GameType, BatchNeuralNetwork>>(newNetwork);
        auto oldModelMCTS = std::make_shared<NeuralNetworkMCTS<GameType, BatchNeuralNetwork>>(oldNetwork);
        
        // 当前使用的MCTS
        auto firstMCTS = newModelFirst ? newModelMCTS : oldModelMCTS;
        auto secondMCTS = newModelFirst ? oldModelMCTS : newModelMCTS;
        
        int moveCount = 0;
        
        while (game.getGameState() == -1) {
            // 确定当前玩家的MCTS
            auto currentMCTS = (game.getCurrentPlayer() == 0) ? firstMCTS : secondMCTS;
            
            // 执行MCTS搜索
            for (int i = 0; i < config.numSimulations; ++i) {
                currentMCTS->search(game, false); // 评估时不引入Dirichlet噪声
            }
            
            // 获取所有合法动作
            auto validMoves = game.getValidMoves();
            
            if (validMoves.empty()) {
                // 没有合法动作，跳过
                game.pass();
                continue;
            }
            
            // 收集每个动作的访问次数
            std::vector<std::pair<std::pair<int, int>, float>> actionCounts;
            for (const auto& move : validMoves) {
                int x = move.first;
                int y = move.second;
                
                GameType tempGame = game;
                tempGame.makeMove(x, y);
                uint64_t hash = tempGame.getHash();
                
                int visits = currentMCTS->getVisitCount(hash);
                if (visits > 0) {
                    actionCounts.push_back({{x, y}, static_cast<float>(visits)});
                }
            }
            
            std::pair<int, int> selectedMove;
            
            // 直接使用贪婪策略：选择访问次数最多的动作, 体现模型的真实能力
            auto bestAction = std::max_element(
                actionCounts.begin(),
                actionCounts.end(),
                [](const auto& a, const auto& b) { return a.second < b.second; }
            );
            
            if (bestAction != actionCounts.end()) {
                selectedMove = bestAction->first;
            } else if (!validMoves.empty()) {
                // 如果没有访问数据，随机选择一个合法动作
                // 这种情况不应该发生
                std::uniform_int_distribution<int> dist(0, validMoves.size() - 1);
                selectedMove = validMoves[dist(gameRng)];
            }

            // 执行选择的动作
            game.makeMove(selectedMove.first, selectedMove.second);
            moveCount++;
        }
        
        // 游戏结束，获取结果
        int gameResult = game.getGameState();
        
        // 转换游戏结果为评估结果
        if (gameResult == 2) {  // 平局
            return 0;
        } else if ((gameResult == 1 && newModelFirst) || (gameResult == 0 && !newModelFirst)) {
            // 新模型获胜（新模型执黑且黑胜，或新模型执白且白胜）
            return 1;
        } else {
            // 旧模型获胜
            return -1;
        }
    }
    
    // 打印评估结果
    void printResult(const EvaluationResult& result) const {
        std::cout << "评估结果:" << std::endl;
        std::cout << "  - 总对局数: " << config.numGames << std::endl;
        std::cout << "  - 新模型胜: " << result.newModelWins 
                  << " (" << std::fixed << std::setprecision(1)
                  << (result.newModelWins * 100.0f / config.numGames) << "%)" << std::endl;
        std::cout << "  - 旧模型胜: " << result.oldModelWins
                  << " (" << std::fixed << std::setprecision(1)
                  << (result.oldModelWins * 100.0f / config.numGames) << "%)" << std::endl;
        std::cout << "  - 平局: " << result.draws
                  << " (" << std::fixed << std::setprecision(1)
                  << (result.draws * 100.0f / config.numGames) << "%)" << std::endl;
        std::cout << "  - 新模型胜率: " << std::fixed << std::setprecision(2)
                  << (result.winRate * 100.0f) << "%" << std::endl;
    }
    
    // 配置
    EvaluationConfig config;
    
    // 神经网络
    std::shared_ptr<BatchNeuralNetwork> newNetwork;
    std::shared_ptr<BatchNeuralNetwork> oldNetwork;
};