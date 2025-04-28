#pragma once

#include "BoardGame.h"
#include <array>
#include <iostream>
#include <sstream>
#include <random>
#include <cassert>

class Othello : public BoardGame<Othello> {
public:        

    // 棋盘大小是静态常量，用来给 MCTS 推导出棋盘大小，便于编译时优化
    // 同时也传递给了基类
    // 因为使用了uint64_t存储棋盘，所以棋盘大小必须是小于等于8， 实际上只能是 6 或者 8
    static constexpr int BOARD_SIZE = 8; 
    static constexpr int CHANNEL_SIZE = 3;

    Othello();    

    bool makeMove(int x, int y) override;
    std::vector<std::pair<int, int>> getValidMoves() override; // 最耗时
    void reset() override;
    void pass() override;

    int getGameState() override;
    int getCurrentPlayer() const override;
    
    uint64_t getHash() const override;
    std::vector<float> toTensor() override;

    void printBoard() const override;
    std::string getBoardString() const override;

private:
    // 棋盘表示
    uint64_t blackBoard;  // 1表示黑棋(玩家1)
    uint64_t whiteBoard;  // 1表示白棋(玩家0)
    
    // 对局状态
    int currentPlayer;    // 当前玩家，0=白棋，1=黑棋
    uint64_t currentHash; // 当前状态的Zobrist哈希值

    // GameState缓存
    int gameState; // -2=未缓存, -1=未结束, 0=白棋胜, 1=黑棋胜, 2=平局

    // 合法Move的缓存
    uint64_t blackValidMoves;  
    uint64_t whiteValidMoves;
    bool isBlackValidMovesCached; // 缓存标志
    bool isWhiteValidMovesCached; // 缓存标志
    
    // 全局Zobrist哈希表, 不占用Othello对象的内存, 且保持hash一致性，仅初始化一次
    static std::array<uint64_t, BOARD_SIZE * BOARD_SIZE> zobristBlack; // 黑棋哈希值
    static std::array<uint64_t, BOARD_SIZE * BOARD_SIZE> zobristWhite; // 白棋哈希值
    static uint64_t zobristPlayer;                // 玩家哈希值
    static bool zobristInitialized;               // 初始化标志, 确保只初始化一次
    static void initZobrist();                    // 初始化Zobrist哈希表

    // Othello 特有的函数
    // 获取所有需要翻转的位置, 这是整个Othello类时间复杂度瓶颈
    std::pair<uint64_t, uint64_t> getFlips(int index, int player) const; 
    
    // 位运算辅助函数
    static inline int getIndex(int x, int y) { return x * BOARD_SIZE + y; }
    static inline std::pair<int, int> getCoordinates(int index) { return {index / BOARD_SIZE, index % BOARD_SIZE}; }

    // 设置和获取位的辅助函数
    static inline bool getBit(uint64_t board, int index) { return (board & (1ULL << index)) != 0; }
    static inline uint64_t setBit(uint64_t board, int index) { return board | (1ULL << index); }
    static inline uint64_t clearBit(uint64_t board, int index) { return board & ~(1ULL << index); }

    // 操作棋盘
    inline int getPiece(int index) const;
    inline void setPiece(int index, int player);   

    // 方向偏移量
    static const std::array<int, 8> DIRECTION_OFFSETS_X;
    static const std::array<int, 8> DIRECTION_OFFSETS_Y;
};

/*
一些设计
1. public 里面允许传(x, y), private 里面只能用 index
2. 一个Othello对象的空间开销只能是常数级别的
3. private 里面异常情况可以少考虑，因为可以相信从public传进来的参数是合法的, 但是public里面要考虑异常情况
*/


// 初始化静态成员变量
bool Othello::zobristInitialized = false;
std::array<uint64_t, Othello::BOARD_SIZE * Othello::BOARD_SIZE> Othello::zobristBlack;
std::array<uint64_t, Othello::BOARD_SIZE * Othello::BOARD_SIZE> Othello::zobristWhite;
uint64_t Othello::zobristPlayer;

// 方向偏移量：左、左上、上、右上、右、右下、下、左下
const std::array<int, 8> Othello::DIRECTION_OFFSETS_X = {-1, -1, 0, 1, 1, 1, 0, -1};
const std::array<int, 8> Othello::DIRECTION_OFFSETS_Y = {0, -1, -1, -1, 0, 1, 1, 1};

// 按需初始化Zobrist表
void Othello::initZobrist() {
    if (zobristInitialized) {
        return;
    }
    
    // 创建随机数生成器
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> dist;
    
    // 为每个位置的每种颜色生成随机数
    for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; i++) {
        zobristBlack[i] = dist(gen);
        zobristWhite[i] = dist(gen);
    }
    
    // 为玩家生成随机数
    zobristPlayer = dist(gen);
    
    zobristInitialized = true;
}

Othello::Othello() {
    // 确保Zobrist表已初始化
    initZobrist();
    
    // 初始化/重置棋盘
    reset();
}

void Othello::reset() {
    // 清空棋盘
    blackBoard = 0;
    whiteBoard = 0;
    
    // 动态计算棋盘中心位置的四个格子
    // 在任何大小的棋盘上，初始棋子都应该放在中心的2x2区域内
    int center = BOARD_SIZE / 2;
    
    // 计算四个中心格子的索引
    int topLeft = getIndex(center - 1, center - 1);     // 左上
    int topRight = getIndex(center - 1, center);        // 右上
    int bottomLeft = getIndex(center, center - 1);      // 左下
    int bottomRight = getIndex(center, center);         // 右下
    
    // 按照黑白棋标准规则放置初始棋子
    // 白子在左上和右下
    setPiece(topLeft, 0);    // 白棋
    setPiece(bottomRight, 0); // 白棋
    
    // 黑子在右上和左下
    setPiece(topRight, 1);   // 黑棋
    setPiece(bottomLeft, 1);  // 黑棋
    
    // 黑棋先行
    currentPlayer = 1;
    
    // 计算初始棋盘哈希值
    currentHash = zobristBlack[topLeft] ^ zobristBlack[topRight] ^ zobristWhite[bottomLeft] ^ zobristWhite[bottomRight];
    
    // 黑棋先行，所以添加玩家哈希
    currentHash ^= zobristPlayer;

    // 清空黑白合法着法缓存
    isBlackValidMovesCached = false;
    isWhiteValidMovesCached = false;
    blackValidMoves = 0;
    whiteValidMoves = 0;

    // 清空GameState缓存
    gameState = -2; 
}

// 获取 index 位置的棋子颜色：1=黑棋，0=白棋，-1=空格
inline int Othello::getPiece(int index) const {
    if (getBit(blackBoard, index)) {
        return 1;
    } else if (getBit(whiteBoard, index)) {
        return 0;
    } else {
        return -1;
    }
}

// 在 index 位置放置棋子
inline void Othello::setPiece(int index, int player) {
    if (player == 1) {
        blackBoard = setBit(blackBoard, index);
    } else {
        whiteBoard = setBit(whiteBoard, index);
    }
}

std::pair<uint64_t, uint64_t> Othello::getFlips(int index, int player) const {
    uint64_t flips = 0;      // 保存需要flips的位置
    uint64_t flipsHash = 0;  // 保存flips位置的哈希值

    auto [x, y] = getCoordinates(index);

    // 遍历8个方向
    for (int i = 0; i < 8; i++) {
        int offsetX = DIRECTION_OFFSETS_X[i];
        int offsetY = DIRECTION_OFFSETS_Y[i];

        uint64_t mask = 0;
        uint64_t updateHash = 0;
                
        for (int j = 1; j < BOARD_SIZE + 1; j++) {
            int newX = x + offsetX * j; 
            int newY = y + offsetY * j;
            int newIndex = getIndex(newX, newY);
            
            // 越界或者空格
            if (newX < 0 || newX >= BOARD_SIZE || newY < 0 || newY >= BOARD_SIZE || getPiece(newIndex) == -1) { 
                mask = 0; 
                updateHash = 0;
                break;
            }
            // 遇到同色棋子
            if (getPiece(newIndex) == player) {
                break;
            }
            mask = setBit(mask, newIndex);
            updateHash ^= (zobristWhite[newIndex] ^ zobristBlack[newIndex]);
        }
        flips |= mask; 
        flipsHash ^= updateHash;
    }
    
    return {flips, flipsHash};
}

bool Othello::makeMove(int x, int y) {
    int index = getIndex(x, y);

    // 无效位置
    if (x < 0 || x >= BOARD_SIZE || y < 0 || y >= BOARD_SIZE || getPiece(index) != -1) {
        return false;
    }
    
    auto [flips, flipsHash] = getFlips(index, currentPlayer);

    // 无翻转位置，非法
    if (flips == 0) {
        return false;
    }

    // 更新新放置棋子的哈希值 - 添加这一行
    setPiece(index, currentPlayer);
    currentHash ^= (currentPlayer == 1 ? zobristBlack[index] : zobristWhite[index]);
    
    // 快速翻转棋子, 更新哈希值
    blackBoard ^= flips;
    whiteBoard ^= flips;
    currentHash ^= flipsHash;
    
    // 切换玩家, 更新哈希值
    currentPlayer ^= 1;
    currentHash ^= zobristPlayer;
    
    // 清空合法着法缓存
    isBlackValidMovesCached = false;
    isWhiteValidMovesCached = false;
    blackValidMoves = 0;
    whiteValidMoves = 0;

    // 清空GameState缓存
    gameState = -2;

    return true;
}

// 获取当前落子玩家的合法着法
std::vector<std::pair<int, int>> Othello::getValidMoves() {
    if ((getCurrentPlayer() == 1 && isBlackValidMovesCached) || (getCurrentPlayer() == 0 && isWhiteValidMovesCached)) {
        std::vector<std::pair<int, int>> moves;
        uint64_t mask = (getCurrentPlayer() == 1) ? blackValidMoves : whiteValidMoves;
        moves.reserve(__builtin_popcountll(mask));
        while (mask) {
            int index = __builtin_ctzll(mask);
            moves.push_back(getCoordinates(index));
            mask &= (mask - 1);
        }
        return moves; // NRVO 机制消除返回开销，不需要担心返回值的拷贝。不需要显式调用 std::move()
    }
    
    uint64_t mask = 0;
    for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; i++) {
        if (getPiece(i) != -1) {
            continue;
        }
        auto [flips, _] = getFlips(i, currentPlayer);
        if (flips != 0) {
            mask = setBit(mask, i);
        }
    }
    
    if (getCurrentPlayer() == 1) {
        blackValidMoves = mask;
        isBlackValidMovesCached = true;
    } else {
        whiteValidMoves = mask;
        isWhiteValidMovesCached = true;
    }
    
    return getValidMoves();
}

void Othello::pass() {
    currentPlayer ^= 1;
    currentHash ^= zobristPlayer;
    // 两个缓存都不需要清除，因为和当前执子玩家无关
}

int Othello::getGameState() {
    // -1=未结束, 0=白棋，1=黑棋，2=平局
    if (gameState != -2) {
        return gameState;
    }

    int blackCount = __builtin_popcountll(blackBoard);
    int whiteCount = __builtin_popcountll(whiteBoard);
    int emptyCount = BOARD_SIZE * BOARD_SIZE - blackCount - whiteCount;
    
    if (emptyCount == 0) { // 棋盘已满
        return gameState = (blackCount > whiteCount ? 1 : blackCount < whiteCount ? 0 : 2);
    }
    auto currentValidMoves = getValidMoves();
    pass();
    auto opponentValidMoves = getValidMoves();
    pass();
    if (currentValidMoves.empty() && opponentValidMoves.empty()) { // 双方都无合法着法
        return gameState = (blackCount > whiteCount ? 1 : blackCount < whiteCount ? 0 : 2);
    }
    return gameState = -1;
}

int Othello::getCurrentPlayer() const {
    return currentPlayer;
}

uint64_t Othello::getHash() const {
    return currentHash;
}

// 三层结构，第一层黑子位置，第二层白子位置, 第三层当前玩家
std::vector<float> Othello::toTensor() {
    int boardArea = BOARD_SIZE * BOARD_SIZE; 
    // 三个通道，每个通道8*8
    std::vector<float> tensor(3 * boardArea, 0.0f);
    
    // 第一通道：黑棋位置
    // 第二通道：白棋位置
    // 第三通道：当前玩家平面：若当前玩家为黑（currentPlayer==1），全1；否则全0。
    for (int i = 0; i < boardArea; i++) {
        // 第一层：黑棋
        tensor[i] = getBit(blackBoard, i) ? 1.0f : 0.0f;
        // 第二层：白棋
        tensor[boardArea + i] = getBit(whiteBoard, i) ? 1.0f : 0.0f;
        // 第三层：当前玩家层（此处设为黑棋使用全1，否则全0）
        tensor[2 * boardArea + i] = (currentPlayer == 1) ? 1.0f : 0.0f;
    }
    
    return tensor;
}

void Othello::printBoard() const {
    std::cout << getBoardString() << std::endl;
}

std::string Othello::getBoardString() const {
    std::ostringstream oss;
    oss << "  ";
    for (int i = 0; i < BOARD_SIZE; i++) {
        oss << char('a' + i) << ' ';
    }
    oss << std::endl;
    
    for (int i = 0; i < BOARD_SIZE; i++) {
        oss << i + 1 << ' ';
        for (int j = 0; j < BOARD_SIZE; j++) {
            int piece = getPiece(getIndex(i, j));
            if (piece == 1) {
                oss << "X ";
            } else if (piece == 0) {
                oss << "O ";
            } else {
                oss << ". ";
            }
        }
        oss << std::endl;
    }
    
    return oss.str();
}