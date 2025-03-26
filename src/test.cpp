#include <iostream>
#include <string>
#include <cctype>
#include <chrono>
#include <algorithm>
#include <thread>
#include <iomanip>  // 用于格式化输出
#include "Othello.hpp"
#include "PureMCTS.hpp"
#include "NeuralNetworkMCTS.hpp"
#include "NeuralNetwork.hpp"
#include "BatchNeuralNetwork.hpp"
#include <cassert>

std::string model_path = "./models/torchscript_model_20250326_210702.pt";

// Convert chess notation to x,y coordinates
std::pair<int, int> chessToCoords(const std::string& notation) {
    if (notation.length() != 2)
        throw std::invalid_argument("Invalid notation length");

    char col = std::tolower(notation[0]);
    char row = notation[1];

    // 使用 Othello::BOARD_SIZE 作为边界检查的依据
    if (col < 'a' || col > ('a' + Othello::BOARD_SIZE - 1) || 
        row < '1' || row > ('1' + Othello::BOARD_SIZE - 1))
        throw std::invalid_argument("Invalid notation range");

    int row_idx = row - '1';
    int col_idx = col - 'a';

    return {row_idx, col_idx};
}

// Convert x,y coordinates to chess notation
std::string coordsToChess(int x, int y) {
    // x 是行号，y 是列号
    char col = 'a' + y;
    char row = '1' + x;
    return std::string(1, col) + std::string(1, row);
}

// 将选择最佳移动函数改为模板函数
template<typename MCTSType>
std::pair<int, int> selectBestMove(const MCTSType& mcts, Othello& game) {
    auto validMoves = game.getValidMoves();
    if (validMoves.empty()) {
        return {-1, -1}; // 无有效移动，返回特殊值表示需要跳过回合
    }
    
    int maxVisits = -1;
    std::pair<int, int> bestMove = {-1, -1};
    
    // 找到访问次数最多的移动
    for (const auto& move : validMoves) {
        Othello nextState = game;
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
template<typename MCTSType>
void printMoveStatistics(const MCTSType& mcts, Othello& game) {
    auto validMoves = game.getValidMoves();
    if (validMoves.empty()) return;
    
    // 准备数据
    struct MoveInfo {
        std::string notation;
        int visits;
        float value;
    };
    std::vector<MoveInfo> moveInfos;
    
    // 收集每个移动的统计信息
    for (const auto& move : validMoves) {
        Othello nextState = game;
        if (nextState.makeMove(move.first, move.second)) {
            uint64_t nextHash = nextState.getHash();
            int visits = mcts.getVisitCount(nextHash);
            float value = mcts.getValue(nextHash);
            
            moveInfos.push_back({
                coordsToChess(move.first, move.second),
                visits,
                value
            });
        }
    }
    
    // 按访问次数排序（降序）
    std::sort(moveInfos.begin(), moveInfos.end(),
              [](const MoveInfo& a, const MoveInfo& b) { return a.visits > b.visits; });
    
    // 打印表头
    std::cout << "\nMove Statistics:\n";
    std::cout << std::setw(6) << "Move" << std::setw(10) << "N (visits)" << std::setw(12) << "Q (value)" << "\n";
    std::cout << std::string(30, '-') << "\n";
    
    // 打印每个移动的统计信息
    for (const auto& info : moveInfos) {
        std::cout << std::setw(6) << info.notation 
                  << std::setw(10) << info.visits 
                  << std::setw(12) << std::fixed << std::setprecision(4) << info.value << "\n";
    }
    std::cout << std::endl;
}

// 获取用户输入的移动
std::pair<int, int> getHumanMove(Othello& game) {
    auto backUp = game;
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
            auto move = chessToCoords(input);
            
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

// 新增：性能基准测试，计算每秒可执行的搜索次数
int runBenchmark(PureMCTS<Othello>& mcts, int targetDurationMs = 1000) {
    std::cout << "Running performance benchmark..." << std::endl;
    
    // 创建一个新的游戏实例用于基准测试
    Othello benchmarkGame;
    
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
    
    std::cout << "Benchmark result: " << searchesPerSecond << " searches per second" << std::endl;
    
    return searchesPerSecond;
}

// 修改机器对机器对弈函数
void computerVsComputer() {
    Othello othello;
    bool gameRunning = true;
    
    // 创建两个MCTS引擎，使用不同的随机种子
    PureMCTS<Othello> mcts0(0.1f); // 白棋引擎
    PureMCTS<Othello> mcts1(0.1f); // 黑棋引擎
    
    
    // 运行基准测试，确定搜索次数
    int searchesPerSecond = runBenchmark(mcts0);
    
    // 设定思考时间系数(秒)
    const float thinkingTimeCoefficient = 1.0f;
    
    // 计算每步搜索次数 = 每秒搜索次数 * 思考时间系数
    const int numSearches = 1000;
    
    // 确保最小搜索次数
    const int minSearches = 100;
    const int actualSearches = std::max(numSearches, minSearches);
    
    // 每步之间的延迟（毫秒），便于观察游戏过程
    const int moveDelay = 0;
    
    std::cout << "Othello Game - PureMCTS vs PureMCTS\n";
    std::cout << "Each engine will perform " << actualSearches << " searches per move\n";
    std::cout << "(Based on system performance of " << searchesPerSecond << " searches/second)\n\n";

    while (gameRunning) {
        // 打印当前棋盘状态
        othello.printBoard();

        // 检查游戏状态
        int state = othello.getGameState();
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
        int currentPlayer = othello.getCurrentPlayer();
        std::cout << (currentPlayer == 1 ? "Black" : "White") << "'s turn\n";

        // 检查有效移动
        auto validMoves = othello.getValidMoves();
        if (validMoves.empty()) {
            std::cout << "No valid moves. Turn passed automatically.\n";
            othello.pass();
            std::this_thread::sleep_for(std::chrono::milliseconds(moveDelay));
            continue;
        }

        // 打印有效移动
        std::cout << "Valid moves: ";
        for (const auto& move : validMoves) {
            std::cout << coordsToChess(move.first, move.second) << " ";
        }
        std::cout << "\n";

        // 选择MCTS引擎
        auto& currentMCTS = (currentPlayer == 0) ? mcts0 : mcts1;
        
        // 执行搜索
        std::cout << "Engine is thinking... ";
        auto startTime = std::chrono::steady_clock::now();
        
        for (int i = 0; i < actualSearches; ++i) {
            currentMCTS.search(othello);
        }
        
        auto endTime = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
        std::cout << "done in " << duration << "ms\n";
        
        // 打印每个合法移动的Q值和N值
        printMoveStatistics(currentMCTS, othello);
        
        // 根据搜索结果选择最佳移动
        auto bestMove = selectBestMove(currentMCTS, othello);
        
        // 执行移动
        if (bestMove.first == -1 && bestMove.second == -1) {
            std::cout << "Engine passes its turn.\n";
            othello.pass();
        } else {
            othello.makeMove(bestMove.first, bestMove.second);
            std::cout << "Engine plays " << coordsToChess(bestMove.first, bestMove.second) << "\n";
        }
        
        std::cout << "\n";
        
        // 添加延迟，使游戏过程更易观察
        std::this_thread::sleep_for(std::chrono::milliseconds(moveDelay));
    }
    
    // 游戏结束后显示最终棋盘
    othello.printBoard();
    
    // 显示游戏统计信息
    int blackPieces = 0, whitePieces = 0;
    std::vector<float> tensor = othello.toTensor();
    
    for (int i = 0; i < Othello::BOARD_SIZE; ++i) {
        for (int j = 0; j < Othello::BOARD_SIZE; ++j) {
            if (tensor[i*Othello::BOARD_SIZE + j] > 0.5) blackPieces++;
            if (tensor[Othello::BOARD_SIZE * Othello::BOARD_SIZE + i*Othello::BOARD_SIZE + j] > 0.5) whitePieces++;
        }
    }
    
    std::cout << "Game finished! Final score: Black " << blackPieces 
              << " - White " << whitePieces << "\n";
}

// 修改人机对弈函数
void humanVsComputer(bool humanPlaysBlack) {
    Othello othello;
    bool gameRunning = true;
    
    // 创建一个MCTS引擎
    PureMCTS<Othello> mcts(0.1f);
    
    // 运行基准测试，确定搜索次数
    int searchesPerSecond = runBenchmark(mcts);
    
    // 设定思考时间系数(秒) - 对人类玩家可以设置更长一些
    const float thinkingTimeCoefficient = 3.0f;
    
    // 计算每步搜索次数
    const int numSearches = static_cast<int>(searchesPerSecond * thinkingTimeCoefficient);
    
    // 确保最小搜索次数
    const int minSearches = 1000;
    const int actualSearches = std::max(numSearches, minSearches);
    
    std::cout << "Othello Game - ";
    std::cout << (humanPlaysBlack ? "Human (Black) vs Computer (White)" : "Computer (Black) vs Human (White)");
    std::cout << "\nComputer will perform " << actualSearches << " searches per move\n";
    std::cout << "(Based on system performance of " << searchesPerSecond << " searches/second)\n\n";

    while (gameRunning) {
        // 打印当前棋盘状态
        othello.printBoard();

        // 检查游戏状态
        int state = othello.getGameState();
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
        int currentPlayer = othello.getCurrentPlayer();
        std::cout << (currentPlayer == 1 ? "Black" : "White") << "'s turn\n";
        
        // 检查有效移动
        auto validMoves = othello.getValidMoves();
        if (validMoves.empty()) {
            std::cout << "No valid moves. Turn passed automatically.\n";
            othello.pass();
            continue;
        }

        // 打印有效移动
        std::cout << "Valid moves: ";
        for (const auto& move : validMoves) {
            std::cout << coordsToChess(move.first, move.second) << " ";
        }
        std::cout << "\n";

        // 确定当前是人类还是电脑的回合
        bool isHumanTurn = (humanPlaysBlack && currentPlayer == 1) || (!humanPlaysBlack && currentPlayer == 0);
        
        if (isHumanTurn) {
            // 人类回合
            auto move = getHumanMove(othello);
            
            if (move.first == -1 && move.second == -1) {
                std::cout << "You passed your turn.\n";
                othello.pass();
            } else {
                othello.makeMove(move.first, move.second);
                std::cout << "You played " << coordsToChess(move.first, move.second) << "\n";
            }
        } else {
            // 计算机回合
            std::cout << "Computer is thinking... ";
            auto startTime = std::chrono::steady_clock::now();
            
            for (int i = 0; i < actualSearches; ++i) {
                mcts.search(othello);
            }
            
            auto endTime = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
            std::cout << "done in " << duration << "ms\n";
            
            // 打印每个合法移动的Q值和N值
            printMoveStatistics(mcts, othello);
            
            // 根据搜索结果选择最佳移动
            auto bestMove = selectBestMove(mcts, othello);
            
            // 执行移动
            if (bestMove.first == -1 && bestMove.second == -1) {
                std::cout << "Computer passes its turn.\n";
                othello.pass();
            } else {
                othello.makeMove(bestMove.first, bestMove.second);
                std::cout << "Computer plays " << coordsToChess(bestMove.first, bestMove.second) << "\n";
            }
        }
        
        std::cout << "\n";
    }
    
    // 游戏结束后显示最终棋盘
    othello.printBoard();
    
    // 显示游戏统计信息
    int blackPieces = 0, whitePieces = 0;
    std::vector<float> tensor = othello.toTensor();
    
    for (int i = 0; i < Othello::BOARD_SIZE; ++i) {
        for (int j = 0; j < Othello::BOARD_SIZE; ++j) {
            if (tensor[i*Othello::BOARD_SIZE + j] > 0.5) blackPieces++;
            if (tensor[Othello::BOARD_SIZE * Othello::BOARD_SIZE + i*Othello::BOARD_SIZE + j] > 0.5) whitePieces++;
        }
    }
    
    std::cout << "Game finished! Final score: Black " << blackPieces 
              << " - White " << whitePieces << "\n";
}

// 添加在 computerVsComputer 函数后面

// 纯MCTS与神经网络MCTS的对抗
void pureMctsVsNnMcts(bool purePlaysBlack) {
    Othello othello;
    bool gameRunning = true;
    
    // 创建纯MCTS引擎
    PureMCTS<Othello> pureMcts(0.1f);
    
    // 创建神经网络并加载模型
    std::cout << "Loading neural network model from: " << model_path << std::endl;
    
    // 创建神经网络模型
    auto network = std::make_shared<BatchNeuralNetwork>(
        model_path,
        Othello::BOARD_SIZE,
        3, // 输入通道：黑棋、白棋、当前玩家
        true, // 使用GPU
        1 // batch_size
    );
    
    // 创建神经网络MCTS引擎
    NeuralNetworkMCTS<Othello, BatchNeuralNetwork> nnMcts(network, 5.0f);
    
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
        othello.printBoard();

        // 检查游戏状态
        int state = othello.getGameState();
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
        int currentPlayer = othello.getCurrentPlayer();
        std::cout << (currentPlayer == 1 ? "Black" : "White") << "'s turn\n";

        // 检查有效移动
        auto validMoves = othello.getValidMoves();
        if (validMoves.empty()) {
            std::cout << "No valid moves. Turn passed automatically.\n";
            othello.pass();
            std::this_thread::sleep_for(std::chrono::milliseconds(moveDelay));
            continue;
        }

        // 打印有效移动
        std::cout << "Valid moves: ";
        for (const auto& move : validMoves) {
            std::cout << coordsToChess(move.first, move.second) << " ";
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
                pureMcts.search(othello);
            }
            
            // 使用模板函数打印统计信息
            printMoveStatistics(pureMcts, othello);
            
            // 使用模板函数选择最佳移动
            bestMove = selectBestMove(pureMcts, othello);
        } else {
            // NeuralNetworkMCTS回合
            for (int i = 0; i < actualSearches; ++i) {
                nnMcts.search(othello);
            }
            
            // 同样使用模板函数处理神经网络MCTS
            printMoveStatistics(nnMcts, othello);
            bestMove = selectBestMove(nnMcts, othello);
        }
        
        auto endTime = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
        std::cout << "done in " << duration << "ms\n";
        
        // 执行移动
        if (bestMove.first == -1 && bestMove.second == -1) {
            std::cout << currentEngine << " passes its turn.\n";
            othello.pass();
        } else {
            othello.makeMove(bestMove.first, bestMove.second);
            std::cout << currentEngine << " plays " << coordsToChess(bestMove.first, bestMove.second) << "\n";
        }
        
        std::cout << "\n";
        
        // 添加延迟，使游戏过程更易观察
        std::this_thread::sleep_for(std::chrono::milliseconds(moveDelay));
    }
    
    // 游戏结束后显示最终棋盘
    othello.printBoard();
    
    // 显示游戏统计信息
    int blackPieces = 0, whitePieces = 0;
    std::vector<float> tensor = othello.toTensor();
    
    for (int i = 0; i < Othello::BOARD_SIZE; ++i) {
        for (int j = 0; j < Othello::BOARD_SIZE; ++j) {
            if (tensor[i*Othello::BOARD_SIZE + j] > 0.5) blackPieces++;
            if (tensor[Othello::BOARD_SIZE * Othello::BOARD_SIZE + i*Othello::BOARD_SIZE + j] > 0.5) whitePieces++;
        }
    }
    
    std::cout << "Game finished! Final score: Black " << blackPieces 
              << " - White " << whitePieces << "\n";
}

// 纯MCTS与神经网络MCTS的对抗
int NNvsNN(std::string model_path1, std::string model_path0) {
    Othello othello;
    bool gameRunning = true;
    
    // 创建神经网络并加载模型
    std::cout << "Loading neural network model from: " << model_path1 << std::endl;
    std::cout << "Loading neural network model from: " << model_path0 << std::endl;

    
    // 创建神经网络模型
    auto network1 = std::make_shared<BatchNeuralNetwork>(
        model_path1,
        Othello::BOARD_SIZE,
        3, // 输入通道：黑棋、白棋、当前玩家
        true, // 使用GPU
        1 // batch_size
    );
    // 创建神经网络模型
    auto network0 = std::make_shared<BatchNeuralNetwork>(
        model_path0,
        Othello::BOARD_SIZE,
        3, // 输入通道：黑棋、白棋、当前玩家
        true, // 使用GPU
        1 // batch_size
    );
    
    // 创建神经网络MCTS引擎
    NeuralNetworkMCTS<Othello, BatchNeuralNetwork> nnMcts1(network1, 5.0f);
    NeuralNetworkMCTS<Othello, BatchNeuralNetwork> nnMcts0(network0, 5.0f);
    
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
        othello.printBoard();

        // 检查游戏状态
        int state = othello.getGameState();
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
        int currentPlayer = othello.getCurrentPlayer();
        std::cout << (currentPlayer == 1 ? "Black" : "White") << "'s turn\n";

        // 检查有效移动
        auto validMoves = othello.getValidMoves();
        if (validMoves.empty()) {
            std::cout << "No valid moves. Turn passed automatically.\n";
            othello.pass();
            std::this_thread::sleep_for(std::chrono::milliseconds(moveDelay));
            continue;
        }

        // 打印有效移动
        std::cout << "Valid moves: ";
        for (const auto& move : validMoves) {
            std::cout << coordsToChess(move.first, move.second) << " ";
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
                nnMcts1.search(othello);
            }
            
            // 使用模板函数打印统计信息
            printMoveStatistics(nnMcts1, othello);
            
            // 使用模板函数选择最佳移动
            bestMove = selectBestMove(nnMcts1, othello);
        } else {
            // NN0
            for (int i = 0; i < actualSearches; ++i) {
                nnMcts0.search(othello);
            }
            
            // 同样使用模板函数处理神经网络MCTS
            printMoveStatistics(nnMcts0, othello);
            bestMove = selectBestMove(nnMcts0, othello);
        }
        
        auto endTime = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
        std::cout << "done in " << duration << "ms\n";
        
        // 执行移动
        if (bestMove.first == -1 && bestMove.second == -1) {
            std::cout << currentEngine << " passes its turn.\n";
            othello.pass();
        } else {
            othello.makeMove(bestMove.first, bestMove.second);
            std::cout << currentEngine << " plays " << coordsToChess(bestMove.first, bestMove.second) << "\n";
        }
        
        std::cout << "\n";
        
        // 添加延迟，使游戏过程更易观察
        std::this_thread::sleep_for(std::chrono::milliseconds(moveDelay));
    }
    
    // 游戏结束后显示最终棋盘
    othello.printBoard();
    
    // 显示游戏统计信息
    int blackPieces = 0, whitePieces = 0;
    std::vector<float> tensor = othello.toTensor();
    
    for (int i = 0; i < Othello::BOARD_SIZE; ++i) {
        for (int j = 0; j < Othello::BOARD_SIZE; ++j) {
            if (tensor[i*Othello::BOARD_SIZE + j] > 0.5) blackPieces++;
            if (tensor[Othello::BOARD_SIZE * Othello::BOARD_SIZE + i*Othello::BOARD_SIZE + j] > 0.5) whitePieces++;
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
    std::string newModel = "/home/liznb/Desktop/BetaZero/models/torchscript_model_20250326_210702.pt";
    std::string oldModel = "/home/liznb/Desktop/BetaZero/old_models/torchscript_model_20250325_190045.pt";
    std::string output;
  
    int result = NNvsNN(newModel, oldModel);
    if (result == 1) {
        output += "newModel black win!\n";
    } else if (result == 0) {
        output += "oldModel white win!\n";
    } else {
        output += "draw!\n";
    }
    result = NNvsNN(oldModel, newModel); 
    if (result == 0) {
        output += "newModel white win!\n";
    } else if (result == 1) {
        output += "oldModel black win!\n";
    } else {
        output += "draw!\n";
    }
    std::cout << output << std::endl;
}

// 修改main函数，添加新的菜单选项
int main() {
    // 菜单选择
    std::cout << "==== BetaZero Othello ====\n";
    std::cout << "1. PureMCTS vs PureMCTS\n";
    std::cout << "2. Human (Black) vs PureMCTS (White)\n";
    std::cout << "3. Human (White) vs PureMCTS (Black)\n";
    std::cout << "4. PureMCTS (Black) vs NeuralNetworkMCTS (White)\n";
    std::cout << "5. NeuralNetworkMCTS (Black) vs PureMCTS (White)\n";
    std::cout << "6. NeuralNetworkMCTS (Black) vs NeuralNetworkMCTS (White)\n";
    std::cout << "Please select a mode: ";
    
    int choice;
    std::cin >> choice;
    
    switch (choice) {
        case 1:
            computerVsComputer();
            break;
        case 2:
            humanVsComputer(true);  // 人类执黑
            break;
        case 3:
            humanVsComputer(false); // 人类执白
            break;
        case 4:
            pureMctsVsNnMcts(true); // 纯MCTS执黑
            break;
        case 5:
            pureMctsVsNnMcts(false); // 纯MCTS执白
            break;
        case 6:
            NNvsNN_white_and_black(); // 神经网络对抗
            break;
        default:
            std::cout << "Invalid choice. Exiting...\n";
            break;
    }
    
    return 0;
}