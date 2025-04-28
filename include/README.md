# cpp写的自我对弈部分

include/BatchNeuralNetwork.hpp: 批量进行推理的神经网络类, 收集目前所有活跃线程的推理请求，批量处理。

在极少数情况下，出现过卡住的情况。我仔细检查了也没发现问题在哪。这种情况很罕见，自我对弈了几十个小时才出现一次。

include/BoardGame.h: 基础的棋盘类

include/Gomoku.hpp : 五子棋类。速度还行。应该还能再快。但是我尝试优化的那一次, 后面怎么都有bug，或者速度变慢了许多，就没管了，这里的速度应该不是瓶颈。

include/MCTS.hpp : 蒙特卡洛树搜索基类

include/NeuralNetworkMCTS.hpp : 不使用批量推理的神经网络蒙特卡洛树搜索

include/Othello.hpp : 黑白棋类

include/PureMCTS.hpp : 纯蒙特卡洛树搜索

include/SelfPlay.hpp : 自我对弈类

include/TrainingData.hpp : 训练数据类, 包含数据增强。会生成x16个数据增强的训练数据（包含原始数据）。

## 自我对弈的细节

可以重点关注 selfplay.hpp, 自我对弈的前期需要手动引入随机性，来保证局面的多样性。狄利克雷噪声，温度策略等, 这部分还挺重要。

自我对弈的瓶颈还是在GPU推理上（或者说调度上），在cpu开128线程做自我对弈，cpu占用也不会高。