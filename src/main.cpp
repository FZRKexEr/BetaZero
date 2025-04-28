#include <iostream>
#include <string>
#include <filesystem>
#include "Othello.hpp"
#include "Gomoku.hpp"
#include "SelfPlay.hpp"

void printUsage(const char* programName) {
    std::cerr << "用法:" << std::endl;
    std::cerr << "  " << programName << " --selfplay <模型路径>" << std::endl;
}

template <typename GameType>
int runSelfPlay(const std::string& modelPath) {
    try {
        // 检查模型路径是否存在
        if (!std::filesystem::exists(modelPath)) {
            std::cerr << "错误: 模型文件不存在: " << modelPath << std::endl;
            return 1;
        }
        
        // 打印基本信息
        std::cout << "BetaZero - 自我对弈程序" << std::endl;
        std::cout << "游戏类型: " << typeid(GameType).name() << std::endl;
        
        // 配置自我对弈参数
        typename SelfPlay<GameType>::SelfPlayConfig config;
        config.numGames = 128;        // 对弈局数
        config.numSimulations = 1000; // 每步MCTS模拟次数
        config.outputDir = "./data";  // 数据输出目录
        config.useDataAugmentation = true; // 是否使用数据增强
        
        // 创建自我对弈实例
        SelfPlay<GameType> selfPlay(modelPath, config);
        
        // 开始自我对弈
        std::cout << "开始自我对弈，将生成 " << config.numGames << " 局对弈数据..." << std::endl;
        selfPlay.run();
        
        // 保存生成的训练数据
        selfPlay.saveData();
        
        std::cout << "自我对弈已完成。" << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3 || std::string(argv[1]) != "--selfplay") {
        printUsage(argv[0]);
        return 1;
    }

    // 获取模型路径
    std::string modelPath = argv[2];
    
    return runSelfPlay<Gomoku>(modelPath);  // 五子棋
}