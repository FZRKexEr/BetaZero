import os
import yaml
import torch

# 默认配置
DEFAULT_CONFIG = {
    # 训练参数
    "iterations": 500,           # 训练迭代次数
    "batch_size": 1024,          # 批次大小
    "samples_per_iteration": 120000,  # 每次迭代训练的样本数
    "memory_size": 2000000,      # 内存缓冲区大小（样本数）
    
    # 优化器参数
    "optimizer": "sgd",          # 优化器类型
    "learning_rate": 0.02,       # 学习率(2e-5 * 1000)
    "momentum": 0.9,             # 动量
    "max_grad_norm": 1.0,        # 梯度裁剪阈值
    "warmup_steps": 3000,        # 预热步数, batch为单位, 相当于前3M(30代)个样本进行预热
    
    # 模型参数
    "board_size": 15,            # 棋盘大小
    "input_channels": 8,         # 输入通道数
    "num_channels": 128,         # 网络通道数
    "num_res_blocks": 8,         # 残差块数量
    
    # 路径设置
    "model_dir": "./models",     # 模型保存目录
    "data_dir": "./data",        # 数据目录
    "log_dir": "./logs",         # 日志目录
    
    # 硬件设置
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_workers": 2,            # 数据加载线程数
    
    # 其他设置
    "skip_first_selfplay": True,  # 是否跳过第一次自我对弈
}

class Config:
    def __init__(self, config_path=None):
        """加载配置，优先使用配置文件，否则使用默认配置"""
        self.config = DEFAULT_CONFIG.copy()
       
        # 创建必要的目录
        for directory in ["model_dir", "data_dir", "log_dir"]:
            os.makedirs(self.config[directory], exist_ok=True)
    
    def __getitem__(self, key):
        return self.config[key]
    
    def __setitem__(self, key, value):
        self.config[key] = value
