#include <iostream>
#include <string>
#include <filesystem>
#include "Othello.hpp"
#include "SelfPlay.hpp"
#include "NeuralNetwork.hpp"
#include "ModelEvaluator.hpp"

// 在printUsage函数中添加
void printUsage(const char* programName) {
    std::cerr << "用法:" << std::endl;
    std::cerr << "  " << programName << " --selfplay <模型路径>" << std::endl;
    std::cerr << "  " << programName << " --evaluation <新模型路径> <旧模型路径> [--output <输出目录>]" << std::endl;
}

int main(int argc, char* argv[]) {
    // 检查命令行参数
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }

    std::string mode = argv[1];

    try {
        // 自我对弈模式
        if (mode == "--selfplay") {
            if (argc != 3) {
                std::cerr << "错误: 自我对弈模式需要指定模型路径" << std::endl;
                printUsage(argv[0]);
                return 1;
            }

            std::string modelPath = argv[2];
            
            // 检查模型路径是否存在
            if (!std::filesystem::exists(modelPath)) {
                std::cerr << "错误: 模型文件不存在: " << modelPath << std::endl;
                return 1;
            }
            
            // 打印基本信息
            std::cout << "BetaZero - 自我对弈程序" << std::endl;
            std::cout << "正在加载模型: " << modelPath << std::endl;
            
            // 配置自我对弈参数
            SelfPlay<Othello>::SelfPlayConfig config;
            config.numGames = 256;        // 对弈局数
            config.numSimulations = 1000; // 每步MCTS模拟次数
            config.outputDir = "./data";  // 数据输出目录
            config.useDataAugmentation = true; // 是否使用数据增强
            
            // 创建自我对弈实例
            SelfPlay<Othello> selfPlay(modelPath, config);
            
            // 开始自我对弈
            std::cout << "开始自我对弈，将生成 " << config.numGames << " 局对弈数据..." << std::endl;
            selfPlay.run();
            
            // 保存生成的训练数据
            selfPlay.saveData();
            
            std::cout << "自我对弈已完成。" << std::endl;
        }
        // 模型评估模式
        else if (mode == "--evaluation") {
            if (argc < 4) {
                std::cerr << "错误: 评估模式需要指定新旧两个模型路径" << std::endl;
                printUsage(argv[0]);
                return 1;
            }
        
            std::string newModelPath = argv[2];
            std::string oldModelPath = argv[3];

            // 检查模型路径是否存在
            if (!std::filesystem::exists(newModelPath)) {
                std::cerr << "错误: 新模型文件不存在: " << newModelPath << std::endl;
                return 1;
            }

            if (!std::filesystem::exists(oldModelPath)) {
                std::cerr << "错误: 旧模型文件不存在: " << oldModelPath << std::endl;
                return 1;
            }

            // 打印基本信息
            std::cout << "BetaZero - 模型评估程序" << std::endl;
            std::cout << "正在加载新模型: " << newModelPath << std::endl;
            std::cout << "正在加载旧模型: " << oldModelPath << std::endl;

            // 配置评估参数
            ModelEvaluator<Othello>::EvaluationConfig config;

            // 检查是否提供了输出目录参数
            for (int i = 4; i < argc; i++) {
                std::string arg = argv[i];
                if (arg == "--output" && i + 1 < argc) {
                    config.outputDir = argv[i+1];
                    i++; // 跳过下一个参数
                }
            }

            // 创建评估器实例
            ModelEvaluator<Othello> evaluator(newModelPath, oldModelPath, config);

            // 开始评估
            std::cout << "开始模型评估..." << std::endl;
            auto result = evaluator.evaluate();
        }
        else {
            std::cerr << "错误: 未知模式 " << mode << std::endl;
            printUsage(argv[0]);
            return 1;
        }
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }
}