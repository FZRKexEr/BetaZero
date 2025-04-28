#pragma once

#include "BoardGame.h"
#include <random>   
#include <iostream> 
#include <sstream>  

class Gomoku : public BoardGame<Gomoku> {
public:
    static constexpr int BOARD_SIZE = 15;
    static constexpr int CHANNEL_SIZE = 8;

    Gomoku();

    bool makeMove(int x, int y) override;

    std::vector<std::pair<int, int>> getValidMoves() override;

    void reset() override;
    void pass() override;

    int getGameState() override;
    int getCurrentPlayer() const override;

    uint64_t getHash() const override;
    std::vector<float> toTensor() override;

    void printBoard() const override;
    std::string getBoardString() const override;

private:
    int board[BOARD_SIZE][BOARD_SIZE]; // -1 未落子, 0 白棋, 1 黑棋

    int currentPlayer;
    uint64_t currentHash;

    // 棋盘状态 -2=未缓存, -1=未结束, 0=白棋胜, 1=黑棋胜, 2=平局
    int gameState;

    // 历史落子记录，最多保存最近5手
    static constexpr int MAX_HISTORY = CHANNEL_SIZE - 3;

    std::vector<std::pair<int, int>> moveHistory;

    static std::array<uint64_t, BOARD_SIZE * BOARD_SIZE> zobristBlack;
    static std::array<uint64_t, BOARD_SIZE * BOARD_SIZE> zobristWhite;
    static uint64_t zobristPlayer;
    static bool zobristInitialized;
    static void initZobrist();

    static inline int getIndex(int x, int y) { return x * BOARD_SIZE + y; }
    static inline std::pair<int, int> getCoordinates(int index) { return {index / BOARD_SIZE, index % BOARD_SIZE}; }

    static const std::array<int, 8> DIRECTION_OFFSETS_X;
    static const std::array<int, 8> DIRECTION_OFFSETS_Y;
};

bool Gomoku::zobristInitialized = false;
std::array<uint64_t, Gomoku::BOARD_SIZE * Gomoku::BOARD_SIZE> Gomoku::zobristBlack;
std::array<uint64_t, Gomoku::BOARD_SIZE * Gomoku::BOARD_SIZE> Gomoku::zobristWhite;
uint64_t Gomoku::zobristPlayer;

// 方向偏移量：左、左上、上、右上、右、右下、下、左下
const std::array<int, 8> Gomoku::DIRECTION_OFFSETS_X = {-1, -1, 0, 1, 1, 1, 0, -1};
const std::array<int, 8> Gomoku::DIRECTION_OFFSETS_Y = {0, -1, -1, -1, 0, 1, 1, 1};

void Gomoku::initZobrist() {
    if (zobristInitialized) {
        return;
    }

    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> dist;

    for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; i++) {
        zobristBlack[i] = dist(gen);
        zobristWhite[i] = dist(gen);
    }

    zobristPlayer = dist(gen);

    zobristInitialized = true;
}

Gomoku::Gomoku() {
    initZobrist();
    reset();
}

void Gomoku::reset() {
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            board[i][j] = -1;
        }
    }

    // 黑棋先行
    currentPlayer = 1;

    // 计算初始棋盘哈希值
    currentHash = zobristPlayer;

    // 清空GameState缓存
    gameState = -2;

    // 清空历史落子记录
    moveHistory.clear();
}

bool Gomoku::makeMove(int x, int y) {
    if (x < 0 || x >= BOARD_SIZE || y < 0 || y >= BOARD_SIZE || board[x][y] != -1) {
        return false;
    }

    // 更新棋盘状态
    board[x][y] = currentPlayer;
    currentHash ^= (currentPlayer == 1 ? zobristBlack[getIndex(x, y)] : zobristWhite[getIndex(x, y)]);

    // 记录历史落子
    moveHistory.push_back({x, y});
    if (moveHistory.size() > MAX_HISTORY) {
        moveHistory.erase(moveHistory.begin());
    }

    // 切换玩家
    currentPlayer ^= 1;
    currentHash ^= zobristPlayer;

    // 清空GameState缓存
    gameState = -2;

    return true;
}

std::vector<std::pair<int, int>> Gomoku::getValidMoves() {
    std::vector<std::pair<int, int>> validMoves;
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            if (board[i][j] == -1) {
                validMoves.push_back({i, j});
            }
        }
    }
    return validMoves;
}

void Gomoku::pass() {
    currentPlayer ^= 1;
    currentHash ^= zobristPlayer;
}

int Gomoku::getGameState() {
    if (gameState != -2) {
        return gameState;
    }

    // 如果没有历史记录，游戏刚开始
    if (moveHistory.empty()) {
        return gameState = -1; // 游戏未结束
    }

    bool hasEmptyCell = false;

    // 检查整个棋盘以寻找五子连线
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            if (board[i][j] == -1) {
                hasEmptyCell = true;
                continue;
            }

            int player = board[i][j]; // 0 或 1
            
            // 只检查四个方向：左、左上、上、右上
            // 将当前位置视为五子连珠的末端，向反方向检查
            for (int d = 0; d < 4; ++d) { 
                int count = 1; // 包含当前落子
                
                // 向一个方向检查四个位置
                for (int k = 1; k < 5; ++k) {
                    int x = i + k * DIRECTION_OFFSETS_X[d];
                    int y = j + k * DIRECTION_OFFSETS_Y[d];
                    if (x < 0 || x >= BOARD_SIZE || y < 0 || y >= BOARD_SIZE || board[x][y] != player) {
                        break;
                    }
                    count++;
                }

                if (count >= 5) {
                    // player 获胜 (0 代表白棋胜, 1 代表黑棋胜)
                    return gameState = player;
                }
            }
        }
    }
   
    if (!hasEmptyCell) {
        return gameState = 2; // 平局
    }

    return gameState = -1;
}

int Gomoku::getCurrentPlayer() const {
    return currentPlayer;
}

uint64_t Gomoku::getHash() const {
    return currentHash;
}

std::vector<float> Gomoku::toTensor() {
    int boardArea = BOARD_SIZE * BOARD_SIZE;
    std::vector<float> tensor(CHANNEL_SIZE * boardArea, 0.0f);
    
    // 通道0：黑棋位置 (1)
    // 通道1：白棋位置 (0)
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            int index = getIndex(i, j);
            if (board[i][j] == 1) {  // 黑棋
                tensor[index] = 1.0f;
            } else if (board[i][j] == 0) {  // 白棋
                tensor[boardArea + index] = 1.0f;
            }
        }
    }
    
    for (int i = 0; i < boardArea; i++) {
        tensor[2 * boardArea + i] = (currentPlayer == 1) ? 1.0f : 0.0f;
    }
    
    for (int i = 0; i < (int)moveHistory.size(); i++) {
        const auto& move = moveHistory[(int)moveHistory.size() - 1 - i];
        int moveIndex = getIndex(move.first, move.second);
        tensor[(3 + i) * boardArea + moveIndex] = 1.0f;
    }
    
    return tensor;
}

void Gomoku::printBoard() const {
    std::cout << getBoardString() << std::endl;
}

std::string Gomoku::getBoardString() const {
    std::ostringstream oss;
    oss << "   "; // 增加一个空格，为两位数的行号预留空间
    for (int i = 0; i < BOARD_SIZE; i++) {
        oss << char('a' + i) << ' ';
    }
    oss << std::endl;
    
    for (int i = 0; i < BOARD_SIZE; i++) {
        if (i < 9) {
            oss << " " << (i + 1) << " "; // 单位数前面加一个空格
        } else {
            oss << (i + 1) << " "; // 两位数正常显示
        }
        for (int j = 0; j < BOARD_SIZE; j++) {
            oss << (board[i][j] == 1 ? "X " : (board[i][j] == 0 ? "O " : ". "));
        }
        oss << std::endl;
    }
    return oss.str();
}   