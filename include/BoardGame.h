#pragma once

#include <vector>
#include <string>
#include <utility>
#include <cstdint>

// 使用 CRTP 模式的基类，从派生类中推导出 BoardSize
template<typename Derived>
class BoardGame {
public:    
    // 虚析构函数
    virtual ~BoardGame() = default;

    // CRTP 通过派生类获取棋盘大小
    static constexpr int getBoardSize() { return Derived::BOARD_SIZE; }

    // 核心游戏逻辑
    virtual bool makeMove(int x, int y) = 0;  // 尝试在(x,y)位置落子
    virtual std::vector<std::pair<int, int>> getValidMoves() = 0;  // 获取所有合法着法, 这里不用const，因为允许记忆化
    virtual void reset() = 0;  // 重置游戏到初始状态
    virtual void pass() = 0;   // 停一手
    
    // 游戏状态查询
    virtual int getGameState() = 0;  // -1=进行中, 0/1=对应玩家胜利, 2=平局, 不用const，因为会调用 getValidMoves
    virtual int getCurrentPlayer() const = 0;  // 获取当前玩家(0或1)
    
    // 用于MCTS和神经网络
    virtual uint64_t getHash() const = 0;  // 获取当前棋盘状态的哈希值
    virtual std::vector<float> toTensor() const = 0;  // 转换为神经网络输入格式
    
    // 调试和展示
    virtual void printBoard() const = 0;  // 打印当前棋盘状态
    virtual std::string getBoardString() const = 0;  // 获取棋盘的字符串表示
};
