#include <iostream>
#include <string>
#include <cctype>
#include <chrono>
#include <algorithm>
#include <thread>
#include <iomanip>  
#include "Othello.hpp"
#include "Gomoku.hpp"
#include "PureMCTS.hpp"
#include "NeuralNetworkMCTS.hpp"
#include "BatchNeuralNetwork.hpp"
#include <cassert>
#include <sstream>

// 使用三个全局变量来管理模型路径
std::string default_model_path = "./models/torchscript_20250330_221147.pt"; // 用于单模型模式，例如人机、纯MCTS vs NN
std::string new_model_path = "./models/torchscript_20250330_221147.pt";       // 用于模型比较，例如 NN vs NN
std::string old_model_path = "./models/torchscript_20250328_020207.pt";       // 用于模型比较

// Convert chess notation to x,y coordinates
template<typename GameType>
std::pair<int, int> chessToCoords(const std::string& notation) {
    // 检查输入长度，允许两位或三位（对于两位数的行号）
    if (notation.length() < 2 || notation.length() > 3)
        throw std::invalid_argument("Invalid notation length");

    char col = std::tolower(notation[0]);
    
    // 解析行号，可能是一位数或两位数
    int row;
    if (notation.length() == 2) {
        row = notation[1] - '0';
    } else {
        row = (notation[1] - '0') * 10 + (notation[2] - '0');
    }
    
    // 检查边界
    if (col < 'a' || col > ('a' + GameType::BOARD_SIZE - 1) || 
        row < 1 || row > GameType::BOARD_SIZE)
        throw std::invalid_argument("Invalid notation range");

    // 在五子棋中，行号从上到下增长，所以不需要翻转
    int row_idx = row - 1;
    int col_idx = col - 'a';

    return {row_idx, col_idx};
}

// Convert x,y coordinates to chess notation
template<typename GameType>
std::string coordsToChess(int x, int y) {
    // 确保坐标在有效范围内
    if (x < 0 || x >= GameType::BOARD_SIZE || y < 0 || y >= GameType::BOARD_SIZE) {
        return "invalid";
    }
    
    // x 是行号（从0开始），y 是列号（从0开始）
    char col = 'a' + y;
    // 行号从1开始
    int row = x + 1;
    
    std::ostringstream oss;
    oss << col << row;
    return oss.str();
}

// 将选择最佳移动函数改为模板函数
template<typename GameType, typename MCTSType>
std::pair<int, int> selectBestMove(const MCTSType& mcts, GameType& game) {
    auto validMoves = game.getValidMoves();
    if (validMoves.empty()) {
        return {-1, -1}; // 无有效移动，返回特殊值表示需要跳过回合
    }
    
    int maxVisits = -1;
    std::pair<int, int> bestMove = {-1, -1};
    
    // 找到访问次数最多的移动
    for (const auto& move : validMoves) {
        GameType nextState = game;
        if (nextState.makeMove(move.first, move.second)) {
            uint64_t nextHash = nextState.getHash();
            int visits = mcts.getVisitCount(nextHash);
            
            if (visits > maxVisits) {
                maxVisits = visits;
                bestMove = move;
            }
        }
    }
    
    return bestMove;
}

// 将打印移动统计信息函数改为模板函数
template<typename GameType, typename MCTSType>
void printMoveStatistics(const MCTSType& mcts, GameType& game) {
    auto validMoves = game.getValidMoves();
    if (validMoves.empty()) return;
    
    // 准备数据
    struct MoveInfo {
        std::string notation;
        int visits;
        float value;
        float probability; // 添加概率字段
    };
    std::vector<MoveInfo> moveInfos;
    
    // 计算总访问次数
    int totalVisits = 0;
    for (const auto& move : validMoves) {
        GameType nextState = game;
        if (nextState.makeMove(move.first, move.second)) {
            uint64_t nextHash = nextState.getHash();
            int visits = mcts.getVisitCount(nextHash);
            totalVisits += visits;
        }
    }
    
    // 收集每个移动的统计信息并计算概率
    for (const auto& move : validMoves) {
        GameType nextState = game;
        if (nextState.makeMove(move.first, move.second)) {
            uint64_t nextHash = nextState.getHash();
            int visits = mcts.getVisitCount(nextHash);
            float value = mcts.getValue(nextHash);
            float probability = totalVisits > 0 ? static_cast<float>(visits) / totalVisits : 0.0f;
            
            moveInfos.push_back({
                coordsToChess<GameType>(move.first, move.second),
                visits,
                value,
                probability
            });
        }
    }
    
    // 按访问次数排序（降序）
    std::sort(moveInfos.begin(), moveInfos.end(),
              [](const MoveInfo& a, const MoveInfo& b) { return a.visits > b.visits; });
    
    // 打印表头
    std::cout << "\n移动统计信息（只显示概率 > 0.1）:\n";
    std::cout << std::setw(6) << "移动" << std::setw(10) << "访问次数" << std::setw(12) << "价值" << std::setw(12) << "概率" << "\n";
    std::cout << std::string(40, '-') << "\n";
    
    // 打印每个移动的统计信息（只有概率大于0.1的）
    for (const auto& info : moveInfos) {
        if (info.probability > 0.1f) {
            std::cout << std::setw(6) << info.notation 
                      << std::setw(10) << info.visits 
                      << std::setw(12) << std::fixed << std::setprecision(4) << info.value
                      << std::setw(12) << std::fixed << std::setprecision(4) << info.probability << "\n";
        }
    }
    std::cout << std::endl;
}

// 获取用户输入的移动
template<typename GameType>
std::pair<int, int> getHumanMove(GameType& game) {
    auto validMoves = game.getValidMoves();

    if (validMoves.empty()) {
        return {-1, -1}; // 无合法移动
    }
    
    while (true) {
        std::cout << "Enter your move (e.g. 'e3') or 'pass': ";
        std::string input;
        std::cin >> input;
        
        if (input == "pass" || input == "Pass" || input == "PASS") {
            return {-1, -1};
        }
        
        try {
            auto move = chessToCoords<GameType>(input);
            
            // 检查是否是合法移动
            bool isValid = false;
            for (const auto& validMove : validMoves) {
                if (validMove.first == move.first && validMove.second == move.second) {
                    isValid = true;
                    break;
                }
            }
            
            if (isValid) {
                return move;
            } else {
                std::cout << "Invalid move! Please try again.\n";
            }
        } catch (const std::exception& e) {
            std::cout << "Invalid input! Please use format like 'e3'.\n";
        }
    }
}

// 修改runBenchmark为使用单一模板参数，针对不同类型的MCTS提供两个版本
// 版本1: 用于PureMCTS
template<typename GameType>
int runBenchmark(PureMCTS<GameType>& mcts, int targetDurationMs = 1000) {
    std::cout << "运行性能基准测试..." << std::endl;
    
    // 创建一个新的游戏实例用于基准测试
    GameType benchmarkGame;
    
    // 记录开始时间
    auto startTime = std::chrono::steady_clock::now();
    
    // 计数器
    int searchCount = 0;
    
    // 运行搜索，直到达到目标持续时间
    while (true) {
        mcts.search(benchmarkGame);
        searchCount++;
        
        // 检查是否达到目标时间
        auto currentTime = std::chrono::steady_clock::now();
        auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
            currentTime - startTime).count();
            
        if (elapsedMs >= targetDurationMs) {
            break;
        }
    }
    
    // 计算每秒搜索次数
    auto endTime = std::chrono::steady_clock::now();
    auto totalTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(
        endTime - startTime).count();
    
    // 将结果转换为每秒搜索次数
    int searchesPerSecond = static_cast<int>(searchCount * 1000.0 / totalTimeMs);
    
    std::cout << "基准测试结果: " << searchesPerSecond << " 次搜索/秒" << std::endl;
    
    return searchesPerSecond;
}

// 版本2: 用于NeuralNetworkMCTS
template<typename GameType>
int runBenchmark(NeuralNetworkMCTS<GameType, BatchNeuralNetwork>& mcts, int targetDurationMs = 1000) {
    std::cout << "运行神经网络MCTS性能基准测试..." << std::endl;
    
    // 创建一个新的游戏实例用于基准测试
    GameType benchmarkGame;
    
    // 记录开始时间
    auto startTime = std::chrono::steady_clock::now();
    
    // 计数器
    int searchCount = 0;
    
    // 运行搜索，直到达到目标持续时间
    while (true) {
        mcts.search(benchmarkGame);
        searchCount++;
        
        // 检查是否达到目标时间
        auto currentTime = std::chrono::steady_clock::now();
        auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
            currentTime - startTime).count();
            
        if (elapsedMs >= targetDurationMs) {
            break;
        }
    }
    
    // 计算每秒搜索次数
    auto endTime = std::chrono::steady_clock::now();
    auto totalTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(
        endTime - startTime).count();
    
    // 将结果转换为每秒搜索次数
    int searchesPerSecond = static_cast<int>(searchCount * 1000.0 / totalTimeMs);
    
    std::cout << "基准测试结果: " << searchesPerSecond << " 次搜索/秒" << std::endl;
    
    return searchesPerSecond;
}

// 新增：辅助函数用于加载神经网络模型
template<typename GameType>
std::shared_ptr<BatchNeuralNetwork> loadNetwork(const std::string& path, bool use_gpu = true, int batch_size = 1) {
    std::cout << "Loading neural network model from: " << path << std::endl;
    try {
        auto network = std::make_shared<BatchNeuralNetwork>(
            path,
            GameType::BOARD_SIZE,
            GameType::CHANNEL_SIZE, // 使用 GameType 的 CHANNEL_SIZE
            use_gpu,
            batch_size
        );
        // 可以在这里添加一些验证模型是否加载成功的逻辑（如果 BatchNeuralNetwork 支持的话）
        std::cout << "Model loaded successfully!" << std::endl;
        return network;
    } catch (const std::exception& e) {
        std::cerr << "Error loading model from " << path << ": " << e.what() << std::endl;
        // 可以选择抛出异常或返回 nullptr
        throw; // 或者 return nullptr;
    }
}

// 将computerVsComputer改为模板函数
template<typename GameType>
void computerVsComputer() {
    GameType game;
    bool gameRunning = true;
    
    // 创建两个MCTS引擎，使用不同的随机种子
    PureMCTS<GameType> mcts0(0.1f); // 白棋引擎
    PureMCTS<GameType> mcts1(0.1f); // 黑棋引擎
    
    // 运行基准测试，确定搜索次数
    int searchesPerSecond = runBenchmark(mcts0);
    
    // 设定思考时间系数(秒)
    const float thinkingTimeCoefficient = 1.0f;
    
    // 计算每步搜索次数 = 每秒搜索次数 * 思考时间系数
    const int numSearches = searchesPerSecond * thinkingTimeCoefficient;
    
    // 确保最小搜索次数
    const int minSearches = 100;
    const int actualSearches = std::max(numSearches, minSearches);
    
    // 每步之间的延迟（毫秒），便于观察游戏过程
    const int moveDelay = 0;
    
    std::cout << "Game - PureMCTS vs PureMCTS\n";
    std::cout << "Each engine will perform " << actualSearches << " searches per move\n";
    std::cout << "(Based on system performance of " << searchesPerSecond << " searches/second)\n\n";

    while (gameRunning) {
        // 打印当前棋盘状态
        game.printBoard();

        // 检查游戏状态
        int state = game.getGameState();
        if (state != -1) {
            if (state == 0) {
                std::cout << "White wins!\n";
            } else if (state == 1) {
                std::cout << "Black wins!\n";
            } else {
                std::cout << "Draw!\n";
            }
            break;
        }

        // 获取当前玩家
        int currentPlayer = game.getCurrentPlayer();
        std::cout << (currentPlayer == 1 ? "Black" : "White") << "'s turn\n";

        // 检查有效移动
        auto validMoves = game.getValidMoves();
        if (validMoves.empty()) {
            std::cout << "No valid moves. Turn passed automatically.\n";
            game.pass();
            std::this_thread::sleep_for(std::chrono::milliseconds(moveDelay));
            continue;
        }

        // 打印有效移动
        std::cout << "Valid moves: ";
        for (const auto& move : validMoves) {
            std::cout << coordsToChess<GameType>(move.first, move.second) << " ";
        }
        std::cout << "\n";

        // 选择MCTS引擎
        auto& currentMCTS = (currentPlayer == 0) ? mcts0 : mcts1;
        
        // 执行搜索
        std::cout << "Engine is thinking... ";
        auto startTime = std::chrono::steady_clock::now();
        
        for (int i = 0; i < actualSearches; ++i) {
            currentMCTS.search(game);
        }
        
        auto endTime = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
        std::cout << "done in " << duration << "ms\n";
        
        // 打印每个合法移动的Q值和N值
        printMoveStatistics<GameType>(currentMCTS, game);
        
        // 根据搜索结果选择最佳移动
        auto bestMove = selectBestMove<GameType>(currentMCTS, game);
        
        // 执行移动
        if (bestMove.first == -1 && bestMove.second == -1) {
            std::cout << "Engine passes its turn.\n";
            game.pass();
        } else {
            game.makeMove(bestMove.first, bestMove.second);
            std::cout << "Engine plays " << coordsToChess<GameType>(bestMove.first, bestMove.second) << "\n";
        }
        
        std::cout << "\n";
        
        // 添加延迟，使游戏过程更易观察
        std::this_thread::sleep_for(std::chrono::milliseconds(moveDelay));
    }
    
    // 游戏结束后显示最终棋盘
    game.printBoard();
    
    // 显示游戏统计信息
    int blackPieces = 0, whitePieces = 0;
    std::vector<float> tensor = game.toTensor();
    
    for (int i = 0; i < GameType::BOARD_SIZE; ++i) {
        for (int j = 0; j < GameType::BOARD_SIZE; ++j) {
            if (tensor[i*GameType::BOARD_SIZE + j] > 0.5) blackPieces++;
            if (tensor[GameType::BOARD_SIZE * GameType::BOARD_SIZE + i*GameType::BOARD_SIZE + j] > 0.5) whitePieces++;
        }
    }
    
    std::cout << "Game finished! Final score: Black " << blackPieces 
              << " - White " << whitePieces << "\n";
}

// 将humanVsComputer改为模板函数
template<typename GameType>
void humanVsComputer(bool humanPlaysBlack) {
    GameType game;
    bool gameRunning = true;
    
    // 使用辅助函数加载神经网络 - 使用 default_model_path
    auto network = loadNetwork<GameType>(default_model_path, true, 1);
    if (!network) {
        std::cerr << "加载神经网络模型失败：" << default_model_path << "。退出游戏。" << std::endl;
        return;
    }
    
    // 创建神经网络MCTS引擎
    NeuralNetworkMCTS<GameType, BatchNeuralNetwork> mcts(network, 5.0f);
    
    // 设定思考时间系数(秒) - 对人类玩家可以设置更长一些
    // 神经网络MCTS通常需要比纯MCTS少的搜索次数
    const float thinkingTimeCoefficient = 2.0f;
    
    // 计算每步搜索次数 - 对于NN-MCTS，使用固定次数更合适
    const int numSearches = 1000;  // 使用固定搜索次数，而不是基于性能测试
    
    std::cout << "游戏开始 - ";
    std::cout << (humanPlaysBlack ? "人类 (黑) vs 神经网络 (白)" : "神经网络 (黑) vs 人类 (白)");
    std::cout << "\n计算机将进行 " << numSearches << " 次搜索/步\n";
    std::cout << "使用神经网络模型: " << default_model_path << "\n\n";

    while (gameRunning) {
        // 打印当前棋盘状态
        game.printBoard();

        // 检查游戏状态
        int state = game.getGameState();
        if (state != -1) {
            if (state == 0) {
                std::cout << "白方胜利!\n";
            } else if (state == 1) {
                std::cout << "黑方胜利!\n";
            } else {
                std::cout << "平局!\n";
            }
            break;
        }

        // 获取当前玩家
        int currentPlayer = game.getCurrentPlayer();
        std::cout << (currentPlayer == 1 ? "黑方" : "白方") << "回合\n";
        
        // 检查有效移动
        auto validMoves = game.getValidMoves();
        if (validMoves.empty()) {
            std::cout << "无有效移动。自动跳过回合。\n";
            game.pass();
            continue;
        }

        // 打印有效移动
        std::cout << "有效移动: ";
        for (const auto& move : validMoves) {
            std::cout << coordsToChess<GameType>(move.first, move.second) << " ";
        }
        std::cout << "\n";

        // 确定当前是人类还是电脑的回合
        bool isHumanTurn = (humanPlaysBlack && currentPlayer == 1) || (!humanPlaysBlack && currentPlayer == 0);
        
        if (isHumanTurn) {
            // 人类回合
            auto move = getHumanMove<GameType>(game);
            
            if (move.first == -1 && move.second == -1) {
                std::cout << "您跳过了回合。\n";
                game.pass();
            } else {
                game.makeMove(move.first, move.second);
                std::cout << "您下在 " << coordsToChess<GameType>(move.first, move.second) << "\n";
            }
        } else {
            // 计算机回合
            std::cout << "神经网络AI思考中... ";
            auto startTime = std::chrono::steady_clock::now();
            
            for (int i = 0; i < numSearches; ++i) {
                mcts.search(game);
            }
            
            auto endTime = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
            std::cout << "用时 " << duration << "毫秒\n";
            
            // 打印每个合法移动的Q值和N值
            printMoveStatistics<GameType>(mcts, game);
            
            // 根据搜索结果选择最佳移动
            auto bestMove = selectBestMove<GameType>(mcts, game);
            
            // 执行移动
            if (bestMove.first == -1 && bestMove.second == -1) {
                std::cout << "AI跳过回合。\n";
                game.pass();
            } else {
                game.makeMove(bestMove.first, bestMove.second);
                std::cout << "AI下在 " << coordsToChess<GameType>(bestMove.first, bestMove.second) << "\n";
            }
        }
        
        std::cout << "\n";
    }
    
    // 游戏结束后显示最终棋盘
    game.printBoard();
    
    // 显示游戏统计信息
    int blackPieces = 0, whitePieces = 0;
    std::vector<float> tensor = game.toTensor();
    
    for (int i = 0; i < GameType::BOARD_SIZE; ++i) {
        for (int j = 0; j < GameType::BOARD_SIZE; ++j) {
            if (tensor[i*GameType::BOARD_SIZE + j] > 0.5) blackPieces++;
            if (tensor[GameType::BOARD_SIZE * GameType::BOARD_SIZE + i*GameType::BOARD_SIZE + j] > 0.5) whitePieces++;
        }
    }
    
    std::cout << "游戏结束! 最终比分: 黑 " << blackPieces 
              << " - 白 " << whitePieces << "\n";
}

// 将pureMctsVsNnMcts改为模板函数
template<typename GameType>
void pureMctsVsNnMcts(bool purePlaysBlack) {
    GameType game;
    bool gameRunning = true;
    
    // 创建纯MCTS引擎
    PureMCTS<GameType> pureMcts(0.1f);
    
    // 使用辅助函数加载神经网络 - 使用 default_model_path
    auto network = loadNetwork<GameType>(default_model_path, true, 1);
    if (!network) {
        std::cerr << "Failed to load neural network from " << default_model_path << ". Exiting." << std::endl;
        return;
    }
    
    // 创建神经网络MCTS引擎
    NeuralNetworkMCTS<GameType, BatchNeuralNetwork> nnMcts(network, 5.0f);
    
    // 运行基准测试，确定搜索次数
    int searchesPerSecond = runBenchmark(pureMcts);
    
    // 设定思考时间系数(秒)
    const float thinkingTimeCoefficient = 1.0f;
    
    // 计算每步搜索次数
    const int numSearches = 1000;
    
    // 确保最小搜索次数
    const int minSearches = 100;
    int actualSearches = std::max(numSearches, minSearches);
    
    // 每步之间的延迟（毫秒）
    const int moveDelay = 0;
    
    std::cout << "Othello Game - ";
    std::cout << (purePlaysBlack ? "PureMCTS (Black) vs NeuralNetworkMCTS (White)" 
                                : "NeuralNetworkMCTS (Black) vs PureMCTS (White)");
    std::cout << "\nEach engine will perform " << actualSearches << " searches per move\n";
    std::cout << "(Based on system performance of " << searchesPerSecond << " searches/second)\n\n";

    while (gameRunning) {
        // 打印当前棋盘状态
        game.printBoard();

        // 检查游戏状态
        int state = game.getGameState();
        if (state != -1) {
            if (state == 0) {
                std::cout << "White wins!\n";
            } else if (state == 1) {
                std::cout << "Black wins!\n";
            } else {
                std::cout << "Draw!\n";
            }
            break;
        }

        // 获取当前玩家
        int currentPlayer = game.getCurrentPlayer();
        std::cout << (currentPlayer == 1 ? "Black" : "White") << "'s turn\n";

        // 检查有效移动
        auto validMoves = game.getValidMoves();
        if (validMoves.empty()) {
            std::cout << "No valid moves. Turn passed automatically.\n";
            game.pass();
            std::this_thread::sleep_for(std::chrono::milliseconds(moveDelay));
            continue;
        }

        // 打印有效移动
        std::cout << "Valid moves: ";
        for (const auto& move : validMoves) {
            std::cout << coordsToChess<GameType>(move.first, move.second) << " ";
        }
        std::cout << "\n";

        // 确定当前是纯MCTS还是神经网络MCTS的回合
        bool isPureMctsTurn = (purePlaysBlack && currentPlayer == 1) || (!purePlaysBlack && currentPlayer == 0);
        
        // 显示当前引擎
        std::string currentEngine = isPureMctsTurn ? "PureMCTS" : "NeuralNetworkMCTS";

        if (isPureMctsTurn) {
            actualSearches = 10000;
        } else {
            actualSearches = 1000;
        }

        std::cout << currentEngine << " is thinking... ";
        
        auto startTime = std::chrono::steady_clock::now();
        std::pair<int, int> bestMove;
        
        // 在 pureMctsVsNnMcts 函数中使用模板函数
        if (isPureMctsTurn) {
            // PureMCTS回合
            for (int i = 0; i < actualSearches; ++i) {
                pureMcts.search(game);
            }
            
            // 使用模板函数打印统计信息
            printMoveStatistics<GameType>(pureMcts, game);
            
            // 使用模板函数选择最佳移动
            bestMove = selectBestMove<GameType>(pureMcts, game);
        } else {
            // NeuralNetworkMCTS回合
            for (int i = 0; i < actualSearches; ++i) {
                nnMcts.search(game);
            }
            
            // 同样使用模板函数处理神经网络MCTS
            printMoveStatistics<GameType>(nnMcts, game);
            bestMove = selectBestMove<GameType>(nnMcts, game);
        }
        
        auto endTime = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
        std::cout << "done in " << duration << "ms\n";
        
        // 执行移动
        if (bestMove.first == -1 && bestMove.second == -1) {
            std::cout << currentEngine << " passes its turn.\n";
            game.pass();
        } else {
            game.makeMove(bestMove.first, bestMove.second);
            std::cout << currentEngine << " plays " << coordsToChess<GameType>(bestMove.first, bestMove.second) << "\n";
        }
        
        std::cout << "\n";
        
        // 添加延迟，使游戏过程更易观察
        std::this_thread::sleep_for(std::chrono::milliseconds(moveDelay));
    }
    
    // 游戏结束后显示最终棋盘
    game.printBoard();
    
    // 显示游戏统计信息
    int blackPieces = 0, whitePieces = 0;
    std::vector<float> tensor = game.toTensor();
    
    for (int i = 0; i < GameType::BOARD_SIZE; ++i) {
        for (int j = 0; j < GameType::BOARD_SIZE; ++j) {
            if (tensor[i*GameType::BOARD_SIZE + j] > 0.5) blackPieces++;
            if (tensor[GameType::BOARD_SIZE * GameType::BOARD_SIZE + i*GameType::BOARD_SIZE + j] > 0.5) whitePieces++;
        }
    }
    
    std::cout << "Game finished! Final score: Black " << blackPieces 
              << " - White " << whitePieces << "\n";
}

// 将NNvsNN改为模板函数
template<typename GameType>
int NNvsNN(std::string model_path1, std::string model_path0) {
    GameType game;
    bool gameRunning = true;
    
    // 使用辅助函数加载神经网络
    auto network1 = loadNetwork<GameType>(model_path1, true, 1);
    auto network0 = loadNetwork<GameType>(model_path0, true, 1);

    if (!network1 || !network0) {
         std::cerr << "Failed to load one or both neural networks. Exiting." << std::endl;
         // 根据需要返回错误代码或抛出异常
         return -1; // 或者 throw std::runtime_error("Failed to load networks");
    }
    
    // 创建神经网络MCTS引擎
    NeuralNetworkMCTS<GameType, BatchNeuralNetwork> nnMcts1(network1, 5.0f);
    NeuralNetworkMCTS<GameType, BatchNeuralNetwork> nnMcts0(network0, 5.0f);
    
    // 运行基准测试，确定搜索次数
    int searchesPerSecond = 1000;
    
    // 设定思考时间系数(秒)
    const float thinkingTimeCoefficient = 1.0f;
    
    // 计算每步搜索次数
    const int numSearches = static_cast<int>(searchesPerSecond * thinkingTimeCoefficient);
    
    // 确保最小搜索次数
    const int minSearches = 100;
    const int actualSearches = std::max(numSearches, minSearches);
    
    // 每步之间的延迟（毫秒）
    const int moveDelay = 0;
    
    std::cout << "Othello Game - ";
    std::cout << "NeuralNetworkMCTS (Black) vs NeuralNetworkMCTS (White)";
    std::cout << "\nEach engine will perform " << actualSearches << " searches per move\n";
    std::cout << "(Based on system performance of " << searchesPerSecond << " searches/second)\n\n";

    while (gameRunning) {
        // 打印当前棋盘状态
        game.printBoard();

        // 检查游戏状态
        int state = game.getGameState();
        if (state != -1) {
            if (state == 0) {
                std::cout << "White wins!\n";
            } else if (state == 1) {
                std::cout << "Black wins!\n";
            } else {
                std::cout << "Draw!\n";
            }
            break;
        }

        // 获取当前玩家
        int currentPlayer = game.getCurrentPlayer();
        std::cout << (currentPlayer == 1 ? "Black" : "White") << "'s turn\n";

        // 检查有效移动
        auto validMoves = game.getValidMoves();
        if (validMoves.empty()) {
            std::cout << "No valid moves. Turn passed automatically.\n";
            game.pass();
            std::this_thread::sleep_for(std::chrono::milliseconds(moveDelay));
            continue;
        }

        // 打印有效移动
        std::cout << "Valid moves: ";
        for (const auto& move : validMoves) {
            std::cout << coordsToChess<GameType>(move.first, move.second) << " ";
        }
        std::cout << "\n";
        
        // 显示当前引擎
        std::string currentEngine = currentPlayer == 1 ? "NeuralNetworkMCTS 1" : "NeuralNetworkMCTS 0";
        std::cout << currentEngine << " is thinking... ";
        
        auto startTime = std::chrono::steady_clock::now();
        std::pair<int, int> bestMove;
        
        // 在 pureMctsVsNnMcts 函数中使用模板函数
        if (currentPlayer == 1) {
            // NN1 
            for (int i = 0; i < actualSearches; ++i) {
                nnMcts1.search(game);
            }
            
            // 使用模板函数打印统计信息
            printMoveStatistics<GameType>(nnMcts1, game);
            
            // 使用模板函数选择最佳移动
            bestMove = selectBestMove<GameType>(nnMcts1, game);
        } else {
            // NN0
            for (int i = 0; i < actualSearches; ++i) {
                nnMcts0.search(game);
            }
            
            // 同样使用模板函数处理神经网络MCTS
            printMoveStatistics<GameType>(nnMcts0, game);
            bestMove = selectBestMove<GameType>(nnMcts0, game);
        }
        
        auto endTime = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
        std::cout << "done in " << duration << "ms\n";
        
        // 执行移动
        if (bestMove.first == -1 && bestMove.second == -1) {
            std::cout << currentEngine << " passes its turn.\n";
            game.pass();
        } else {
            game.makeMove(bestMove.first, bestMove.second);
            std::cout << currentEngine << " plays " << coordsToChess<GameType>(bestMove.first, bestMove.second) << "\n";
        }
        
        std::cout << "\n";
        
        // 添加延迟，使游戏过程更易观察
        std::this_thread::sleep_for(std::chrono::milliseconds(moveDelay));
    }
    
    // 游戏结束后显示最终棋盘
    game.printBoard();
    
    // 显示游戏统计信息
    int blackPieces = 0, whitePieces = 0;
    std::vector<float> tensor = game.toTensor();
    
    for (int i = 0; i < GameType::BOARD_SIZE; ++i) {
        for (int j = 0; j < GameType::BOARD_SIZE; ++j) {
            if (tensor[i*GameType::BOARD_SIZE + j] > 0.5) blackPieces++;
            if (tensor[GameType::BOARD_SIZE * GameType::BOARD_SIZE + i*GameType::BOARD_SIZE + j] > 0.5) whitePieces++;
        }
    }
    
    std::cout << "Game finished! Final score: Black " << blackPieces 
              << " - White " << whitePieces << "\n";
     
    if (blackPieces == whitePieces) {
        return 2;
    }
    return blackPieces > whitePieces ? 1 : 0; 
}

void NNvsNN_white_and_black() {
    // 不再在此函数内部定义路径，直接使用全局变量
    std::string output;

    std::cout << "Comparing New Model (Black): " << new_model_path
              << " vs Old Model (White): " << old_model_path << std::endl;
    int result = NNvsNN<Othello>(new_model_path, old_model_path); // 新黑 vs 旧白
    if (result == 1) {
        output += "New model (Black) wins!\n";
    } else if (result == 0) {
        output += "Old model (White) wins!\n";
    } else {
        output += "Draw!\n";
    }

    std::cout << "Comparing Old Model (Black): " << old_model_path
              << " vs New Model (White): " << new_model_path << std::endl;
    result = NNvsNN<Othello>(old_model_path, new_model_path); // 旧黑 vs 新白
    if (result == 0) { // 注意这里判断条件反了，因为现在是 newModel 执白
        output += "New model (White) wins!\n";
    } else if (result == 1) {
        output += "Old model (Black) wins!\n";
    } else {
        output += "Draw!\n";
    }
    std::cout << output << std::endl;
}

// 添加一个专门的benchmark测试函数
template<typename GameType>
void runBenchmarkTest() {
    std::cout << "\n==== 性能基准测试 ====\n";
    
    // 创建一个MCTS引擎用于测试
    PureMCTS<GameType> mcts(0.1f);
    
    // 运行多个时间段的测试
    std::vector<int> testDurations = {100, 500, 1000, 2000}; // 毫秒
    
    std::cout << "测试不同时间段的搜索性能：\n";
    for (int duration : testDurations) {
        int searches = runBenchmark(mcts, duration);
        std::cout << "测试时长 " << duration << "ms: " 
                  << searches << " 次搜索/秒"
                  << " (总计 " << (searches * duration / 1000) << " 次搜索)\n";
    }
    
    // 测试神经网络性能（如果选择包含神经网络）
    std::cout << "\n是否测试神经网络性能？(y/n): ";
    char choice;
    std::cin >> choice;
    
    if (choice == 'y' || choice == 'Y') {
        try {
            auto network = loadNetwork<GameType>(default_model_path, true, 1);
            if (network) {
                NeuralNetworkMCTS<GameType, BatchNeuralNetwork> nnMcts(network, 5.0f);
                std::cout << "\n神经网络MCTS性能测试：\n";
                for (int duration : testDurations) {
                    int searches = runBenchmark(nnMcts, duration);
                    std::cout << "测试时长 " << duration << "ms: " 
                              << searches << " 次搜索/秒"
                              << " (总计 " << (searches * duration / 1000) << " 次搜索)\n";
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "加载神经网络时出错: " << e.what() << std::endl;
        }
    }
}

// 修改main函数，添加游戏类型选择
int main() {
    // 首先选择游戏类型
    std::cout << "==== BetaZero游戏选择 ====\n";
    std::cout << "1. 黑白棋\n";
    std::cout << "2. 五子棋\n";
    std::cout << "3. 运行性能基准测试\n";  // 新增选项
    std::cout << "请选择: ";
    
    int gameChoice;
    std::cin >> gameChoice;
    
    if (gameChoice == 3) {  // 处理基准测试选项
        std::cout << "\n选择要测试的游戏类型：\n";
        std::cout << "1. 黑白棋\n";
        std::cout << "2. 五子棋\n";
        std::cout << "请选择: ";
        
        int benchmarkChoice;
        std::cin >> benchmarkChoice;
        
        if (benchmarkChoice == 1) {
            runBenchmarkTest<Othello>();
        } else if (benchmarkChoice == 2) {
            runBenchmarkTest<Gomoku>();
        } else {
            std::cout << "无效的选择。退出...\n";
        }
        return 0;
    }
    
    // 然后选择游戏模式
    std::cout << "\n==== Game Modes ====\n";
    std::cout << "1. PureMCTS vs PureMCTS\n";
    std::cout << "2. Human (Black) vs NeuralNetworkMCTS (White)\n";
    std::cout << "3. Human (White) vs NeuralNetworkMCTS (Black)\n";
    std::cout << "4. PureMCTS (Black) vs NeuralNetworkMCTS (White)\n";
    std::cout << "5. NeuralNetworkMCTS (Black) vs PureMCTS (White)\n";
    std::cout << "6. NeuralNetworkMCTS (Black) vs NeuralNetworkMCTS (White)\n";
    std::cout << "Please select a mode: ";
    
    int modeChoice;
    std::cin >> modeChoice;
    
    // 根据选择执行相应的游戏
    try { // 添加 try-catch 块以捕获可能的模型加载错误
        if (gameChoice == 1) {  // Othello
            switch (modeChoice) {
                case 1:
                    computerVsComputer<Othello>();
                    break;
                case 2:
                    humanVsComputer<Othello>(true);
                    break;
                case 3:
                    humanVsComputer<Othello>(false);
                    break;
                case 4:
                    pureMctsVsNnMcts<Othello>(true);
                    break;
                case 5:
                    pureMctsVsNnMcts<Othello>(false);
                    break;
                case 6:
                    NNvsNN_white_and_black();
                    break;
                default:
                    std::cout << "Invalid mode choice. Exiting...\n";
                    break;
            }
        } else if (gameChoice == 2) {  // Gomoku
            switch (modeChoice) {
                case 1:
                    computerVsComputer<Gomoku>();
                    break;
                case 2:
                    humanVsComputer<Gomoku>(true);
                    break;
                case 3:
                    humanVsComputer<Gomoku>(false);
                    break;
                case 4:
                    pureMctsVsNnMcts<Gomoku>(true);
                    break;
                case 5:
                    pureMctsVsNnMcts<Gomoku>(false);
                    break;
                case 6:
                    // NNvsNN_white_and_black 目前不适用于 Gomoku
                    // 你可以调用 NNvsNN<Gomoku> 一次，或者修改 NNvsNN_white_and_black
                    std::cout << "Warning: Mode 6 (NNvsNN comparison) currently uses Othello scoring logic." << std::endl;
                    std::cout << "Running NNvsNN game for Gomoku using new_model_path and old_model_path." << std::endl;
                    // 使用全局变量 new_model_path 和 old_model_path
                    NNvsNN<Gomoku>(new_model_path, old_model_path);
                    // 或者提示用户此模式不完全兼容
                    // std::cout << "Mode 6 is primarily designed for Othello score comparison and may not be fully applicable to Gomoku." << std::endl;
                    break;
                default:
                    std::cout << "Invalid mode choice. Exiting...\n";
                    break;
            }
        } else {
            std::cout << "Invalid game choice. Exiting...\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1; // 返回非零表示错误
    }
    
    return 0;
}