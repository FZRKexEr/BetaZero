#pragma once

#include <torch/script.h>
#include <mutex>
#include <string>
#include <vector>
#include <utility>
#include <iostream>
#include <torch/torch.h>
#include <cassert>


// 基于LibTorch的神经网络实现，支持线程安全
class NeuralNetwork {
private:
    torch::jit::script::Module model;
    std::mutex predict_mutex;  // 用于保证线程安全
    int board_size;            // 棋盘大小
    int input_channels;        // 输入通道数
    bool using_gpu;            // 是否使用GPU
    int predict_count = 0;     // 用于 benchmark
    
public:
    // 从文件加载模型
    NeuralNetwork(const std::string& model_path, int board_size, int input_channels = 3, bool use_gpu = true);
    
    // 析构函数
    virtual ~NeuralNetwork() = default;
    
    // 线程安全的推理方法
    std::pair<float, std::vector<float>> predict(const std::vector<float>& state);
};

NeuralNetwork::NeuralNetwork(const std::string& model_path, int board_size, int input_channels, bool use_gpu) 
    : board_size(board_size), input_channels(input_channels) {
    
    // 加载TorchScript模型
    try {
        std::cout << "尝试从 " << model_path << " 加载模型..." << std::endl;
        model = torch::jit::load(model_path);
        std::cout << "模型加载成功!" << std::endl;
        
        // 设置为评估模式
        model.eval();
        
        // 检查是否可以实际使用GPU
        using_gpu = use_gpu && torch::cuda::is_available();
        
        // 如果可以使用GPU，将模型移至GPU
        if (using_gpu) {
            model.to(torch::kCUDA);
            std::cout << "神经网络模型已加载到GPU" << std::endl;
        } else {
            if (use_gpu && !torch::cuda::is_available()) {
                std::cout << "警告: 请求使用GPU但不可用，将使用CPU" << std::endl;
            } else {
                std::cout << "神经网络模型已加载到CPU" << std::endl;
            }
        }
    } catch (const c10::Error& e) {
        std::cerr << "加载神经网络模型出错: " << e.what() << std::endl;
        std::cerr << "请确保您已将模型转换为TorchScript格式" << std::endl;
        throw;  // 重新抛出异常
    }
}

std::pair<float, std::vector<float>> NeuralNetwork::predict(const std::vector<float>& state) {
    std::lock_guard<std::mutex> lock(predict_mutex);  // 保证线程安全
    
    // 验证输入数据
    assert(state.size() == input_channels * board_size * board_size && "输入数据大小错误");
    
    // 创建输入张量
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    auto input_tensor = torch::from_blob(
        const_cast<float*>(state.data()), 
        {1, input_channels, board_size, board_size},
        options
    ).clone();
    
    // 如果使用GPU，移至GPU (使用缓存的标志)
    if (using_gpu) {
        input_tensor = input_tensor.to(torch::kCUDA);
    }
    
    // 执行推理，禁用梯度计算
    torch::NoGradGuard no_grad;
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input_tensor);
    
    auto output = model.forward(inputs);
    
    // 解析输出元组 (value, policy)
    auto output_tuple = output.toTuple();
    auto value_tensor = output_tuple->elements()[0].toTensor().cpu();
    auto policy_tensor = output_tuple->elements()[1].toTensor().cpu();
    
    // 提取价值
    float value = value_tensor.item<float>();
    
    // 提取策略向量 - 优化预分配大小
    std::vector<float> policy(board_size * board_size);
    auto policy_accessor = policy_tensor.flatten();
    
    // 复制张量数据到vector
    std::memcpy(policy.data(), policy_accessor.data_ptr<float>(), 
               policy.size() * sizeof(float));
    
    predict_count++;  // 用于 benchmark
    return {value, policy};
}