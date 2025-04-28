#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <cstdint>
#include <memory>
#include <array>

// 训练数据结构，支持模板化的棋盘游戏
template<typename GameType>
class TrainingData {
public:
    // 游戏样本结构
    struct Sample {
        std::vector<float> state;  // 状态向量
        std::vector<float> policy; // 策略向量
        float value;               // 价值

        Sample() : value(0.0f) {}
        Sample(const std::vector<float>& s, const std::vector<float>& p, float v)
            : state(s), policy(p), value(v) {}
    };

    // 构造函数
    TrainingData() {}

    // 添加训练样本
    void addSample(const Sample& sample) {
        samples.push_back(sample);
    }

    void addSample(const std::vector<float>& state, const std::vector<float>& policy, float value) {
        samples.emplace_back(state, policy, value);
    }

    // 数据增强：黑白对称和旋转翻转, 数据增强后的样本数量为16倍
    std::vector<Sample> augmentSample(const Sample& sample) const;

    // 保存数据到文件
    bool saveToFile(const std::string& filename) const;

    // 获取样本数量
    size_t size() const { return samples.size(); }

    // 获取所有样本
    const std::vector<Sample>& getSamples() const { return samples; }

    // 清空数据
    void clear() { samples.clear(); }

private:
    // 样本集合
    std::vector<Sample> samples;

    // 棋盘大小和输入通道数
    static constexpr int BOARD_SIZE = GameType::BOARD_SIZE;
    static constexpr int INPUT_CHANNELS = GameType::CHANNEL_SIZE; // 输入通道数

    // 将棋盘状态进行旋转和翻转变换
    std::vector<float> rotateState(const std::vector<float>& state, int rotationType) const;

    // 将策略向量进行旋转和翻转变换
    std::vector<float> rotatePolicy(const std::vector<float>& policy, int rotationType) const;

    // 将黑白棋子互换
    std::vector<float> swapColors(const std::vector<float>& state) const;
};

// 实现部分

template<typename GameType>
std::vector<typename TrainingData<GameType>::Sample> TrainingData<GameType>::augmentSample(const Sample& sample) const {
    std::vector<Sample> augmentedSamples;

    // 原始样本
    augmentedSamples.push_back(sample);

    // 颜色反转（黑白对称）
    std::vector<float> invertedState = swapColors(sample.state);

    // 价值不能取反
    augmentedSamples.emplace_back(invertedState, sample.policy, sample.value);

    // 旋转和翻转变换（8种变换：原始、旋转90°、180°、270°，以及它们的水平翻转）
    for (int rotationType = 1; rotationType < 8; ++rotationType) {
        std::vector<float> rotatedState = rotateState(sample.state, rotationType);
        std::vector<float> rotatedPolicy = rotatePolicy(sample.policy, rotationType);
        augmentedSamples.emplace_back(rotatedState, rotatedPolicy, sample.value);

        // 对颜色反转的样本也应用同样的旋转变换
        std::vector<float> rotatedInvertedState = rotateState(invertedState, rotationType);
        augmentedSamples.emplace_back(rotatedInvertedState, rotatedPolicy, sample.value);
    }

    return augmentedSamples;
}

template<typename GameType>
std::vector<float> TrainingData<GameType>::swapColors(const std::vector<float>& state) const {
    std::vector<float> swapped = state;
    const int plane_size = BOARD_SIZE * BOARD_SIZE;
    // 交换第一通道（黑子）和第二通道（白子）
    for (int i = 0; i < plane_size; ++i) {
        std::swap(swapped[i], swapped[i + plane_size]);
    }
    // 反转第三通道（当前玩家标记）
    // 1变成0，0变成1
    for (int i = 0; i < plane_size; ++i) {
        swapped[i + 2 * plane_size] = 1.0f - swapped[i + 2 * plane_size];
    }
    return swapped;
}

template<typename GameType>
std::vector<float> TrainingData<GameType>::rotateState(const std::vector<float>& state, int rotationType) const {
    std::vector<float> rotated(state.size());

    for (int c = 0; c < INPUT_CHANNELS; ++c) {
        for (int i = 0; i < BOARD_SIZE; ++i) {
            for (int j = 0; j < BOARD_SIZE; ++j) {
                int srcIdx = c * BOARD_SIZE * BOARD_SIZE + i * BOARD_SIZE + j;
                int dstI, dstJ;

                switch (rotationType) {
                    case 1: // 旋转90度
                        dstI = j;
                        dstJ = BOARD_SIZE - 1 - i;
                        break;
                    case 2: // 旋转180度
                        dstI = BOARD_SIZE - 1 - i;
                        dstJ = BOARD_SIZE - 1 - j;
                        break;
                    case 3: // 旋转270度
                        dstI = BOARD_SIZE - 1 - j;
                        dstJ = i;
                        break;
                    case 4: // 水平翻转
                        dstI = i;
                        dstJ = BOARD_SIZE - 1 - j;
                        break;
                    case 5: // 水平翻转后旋转90度
                        dstI = BOARD_SIZE - 1 - j;
                        dstJ = BOARD_SIZE - 1 - i;
                        break;
                    case 6: // 水平翻转后旋转180度（等同于垂直翻转）
                        dstI = BOARD_SIZE - 1 - i;
                        dstJ = j;
                        break;
                    case 7: // 水平翻转后旋转270度
                        dstI = j;
                        dstJ = i;
                        break;
                    default:
                        dstI = i;
                        dstJ = j;
                }

                int dstIdx = c * BOARD_SIZE * BOARD_SIZE + dstI * BOARD_SIZE + dstJ;
                rotated[dstIdx] = state[srcIdx];
            }
        }
    }

    return rotated;
}

template<typename GameType>
std::vector<float> TrainingData<GameType>::rotatePolicy(const std::vector<float>& policy, int rotationType) const {
    std::vector<float> rotated(policy.size());

    for (int i = 0; i < BOARD_SIZE; ++i) {
        for (int j = 0; j < BOARD_SIZE; ++j) {
            int srcIdx = i * BOARD_SIZE + j;
            int dstI, dstJ;

            switch (rotationType) {
                case 1: // 旋转90度
                    dstI = j;
                    dstJ = BOARD_SIZE - 1 - i;
                    break;
                case 2: // 旋转180度
                    dstI = BOARD_SIZE - 1 - i;
                    dstJ = BOARD_SIZE - 1 - j;
                    break;
                case 3: // 旋转270度
                    dstI = BOARD_SIZE - 1 - j;
                    dstJ = i;
                    break;
                case 4: // 水平翻转
                    dstI = i;
                    dstJ = BOARD_SIZE - 1 - j;
                    break;
                case 5: // 水平翻转后旋转90度
                    dstI = BOARD_SIZE - 1 - j;
                    dstJ = BOARD_SIZE - 1 - i;
                    break;
                case 6: // 水平翻转后旋转180度（等同于垂直翻转）
                    dstI = BOARD_SIZE - 1 - i;
                    dstJ = j;
                    break;
                case 7: // 水平翻转后旋转270度
                    dstI = j;
                    dstJ = i;
                    break;
                default:
                    dstI = i;
                    dstJ = j;
            }

            int dstIdx = dstI * BOARD_SIZE + dstJ;
            rotated[dstIdx] = policy[srcIdx];
        }
    }

    return rotated;
}

template<typename GameType>
bool TrainingData<GameType>::saveToFile(const std::string& filename) const {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    // 写入样本数量
    uint32_t numSamples = static_cast<uint32_t>(samples.size());
    file.write(reinterpret_cast<const char*>(&numSamples), sizeof(numSamples));

    // 写入第一个样本的元数据以确定格式
    if (!samples.empty()) {
        uint32_t stateSize = static_cast<uint32_t>(samples[0].state.size());
        uint32_t policySize = static_cast<uint32_t>(samples[0].policy.size());
        file.write(reinterpret_cast<const char*>(&stateSize), sizeof(stateSize));
        file.write(reinterpret_cast<const char*>(&policySize), sizeof(policySize));
    }

    // 写入每个样本
    for (const auto& sample : samples) {
        // 写入状态向量
        for (const auto& value : sample.state) {
            file.write(reinterpret_cast<const char*>(&value), sizeof(float));
        }

        // 写入策略向量
        for (const auto& value : sample.policy) {
            file.write(reinterpret_cast<const char*>(&value), sizeof(float));
        }

        // 写入价值
        file.write(reinterpret_cast<const char*>(&sample.value), sizeof(float));
    }

    return true;
}
 