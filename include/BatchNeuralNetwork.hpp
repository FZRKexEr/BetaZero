#pragma once

#include <torch/script.h>
#include <torch/torch.h>
#include <mutex>
#include <string>
#include <vector>
#include <queue>
#include <future>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <utility>
#include <iostream>
#include <cassert>

class BatchNeuralNetwork {
private:
    struct BatchItem {
        std::vector<float> input;
        std::promise<std::pair<float, std::vector<float>>> promise;
    };

    torch::jit::script::Module model;
    int board_size;
    int input_channels;
    bool using_gpu;
    std::atomic<int> active_threads{0};  // 活跃线程计数
    int predict_count = 0;
    
    std::mutex queue_mutex;
    std::condition_variable cv_queue;
    std::condition_variable cv_batch;
    std::queue<BatchItem> request_queue;
    std::atomic<bool> should_stop{false};
    std::thread worker_thread;
    
    // 工作线程函数
    void processBatches();

public:
    BatchNeuralNetwork(const std::string& model_path, int board_size, 
                       int input_channels, bool use_gpu, int batch_size);
    
    ~BatchNeuralNetwork();
    
    // 对外接口，与原NeuralNetwork接口兼容
    std::pair<float, std::vector<float>> predict(const std::vector<float>& state);
    
    // 线程活跃状态管理
    void threadFinished();
    int getActiveThreads() const { return active_threads; }
};


// batch_size 最好等于线程数,不要多不要少
BatchNeuralNetwork::BatchNeuralNetwork(const std::string& model_path, int board_size, 
                                       int input_channels, bool use_gpu, int batch_size)
    : board_size(board_size), input_channels(input_channels) {
    
    try {
        std::cout << "尝试从 " << model_path << " 加载模型..." << std::endl;
        model = torch::jit::load(model_path);
        std::cout << "模型加载成功!" << std::endl;
        
        model.eval();
        
        using_gpu = use_gpu && torch::cuda::is_available();
        active_threads = batch_size;
        
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
        
        // 启动工作线程
        worker_thread = std::thread(&BatchNeuralNetwork::processBatches, this);
        
    } catch (const c10::Error& e) {
        std::cerr << "加载神经网络模型出错: " << e.what() << std::endl;
        std::cerr << "请确保您已将模型转换为TorchScript格式" << std::endl;
        throw;
    }
}

void BatchNeuralNetwork::threadFinished() {
    active_threads--;
    cv_batch.notify_all();
}

BatchNeuralNetwork::~BatchNeuralNetwork() {
    // 停止工作线程
    should_stop = true;
    active_threads = 0;
    cv_queue.notify_one();
    cv_batch.notify_all();
    
    if (worker_thread.joinable()) {
        worker_thread.join();
    }
}

std::pair<float, std::vector<float>> BatchNeuralNetwork::predict(const std::vector<float>& state) {
    // 创建承诺/期望对象
    std::promise<std::pair<float, std::vector<float>>> promise;
    auto future = promise.get_future();
    
    // 创建请求项并加入队列
    BatchItem item{state, std::move(promise)};
    
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        request_queue.push(std::move(item));
    }
    
    // 通知工作线程有新请求
    cv_queue.notify_one();
    cv_batch.notify_all();
    
    predict_count++;

    // 等待结果返回
    return future.get();
}

void BatchNeuralNetwork::processBatches() {
    while (!should_stop) {
        std::vector<BatchItem> batch;
        
        // 等待队列中有足够的请求
        {
            std::unique_lock<std::mutex> lock(queue_mutex);

            cv_queue.wait(lock, [this] { // 空队列就阻塞
                return !request_queue.empty() || should_stop;
            });
            
            // 收集批处理项目
            cv_batch.wait(lock, [this] { // 没有满就阻塞
                return request_queue.size() >= active_threads.load() || should_stop;
            });

            assert((int)request_queue.size() == active_threads.load());

            // 如果已经满足条件，说明 active_threads 此时不会变
            batch.reserve(request_queue.size());
            
            int batch_to_process = request_queue.size();

            for (size_t i = 0; i < batch_to_process; ++i) {
                batch.push_back(std::move(request_queue.front()));
                request_queue.pop();
            }
        }
        
        if (batch.empty()) {
            continue;
        }
        
        // 执行批处理推理
        try {
            // 准备批处理输入张量
            torch::NoGradGuard no_grad;
            torch::Tensor batch_tensor = torch::zeros({static_cast<long>(batch.size()), 
                                                      input_channels, 
                                                      board_size, 
                                                      board_size});
            
            // 填充批处理张量
            for (size_t i = 0; i < batch.size(); ++i) {
                auto options = torch::TensorOptions().dtype(torch::kFloat32);
                auto input_tensor = torch::from_blob(
                    const_cast<float*>(batch[i].input.data()),
                    {1, input_channels, board_size, board_size},
                    options
                ).clone();
                
                batch_tensor[i] = input_tensor[0];
            }
            
            // 移动到GPU (如果需要)
            if (using_gpu) {
                batch_tensor = batch_tensor.to(torch::kCUDA);
            }
            
            // 执行批量推理
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(batch_tensor);
            
            auto output = model.forward(inputs);
            
            // 解析输出元组 (value, policy)
            auto output_tuple = output.toTuple();
            auto value_tensor = output_tuple->elements()[0].toTensor().cpu();
            auto policy_tensor = output_tuple->elements()[1].toTensor().cpu();
            
            // 分发结果到各个promise
            for (size_t i = 0; i < batch.size(); ++i) {
                // 提取价值
                float value = value_tensor[i].item<float>();
                
                // 提取策略向量
                std::vector<float> policy(board_size * board_size);
                auto policy_accessor = policy_tensor[i].flatten();
                
                // 复制张量数据到vector
                std::memcpy(policy.data(), policy_accessor.data_ptr<float>(), 
                           policy.size() * sizeof(float));
                
                // 设置结果
                batch[i].promise.set_value({value, policy});
            }
        } catch (const std::exception& e) {
            // 发生错误时，设置所有promise为异常
            for (auto& item : batch) {
                try {
                    item.promise.set_exception(std::current_exception());
                } catch (...) {
                    // 忽略已设置的promise
                }
            }
        }
    }
}