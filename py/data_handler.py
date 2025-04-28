import os
import glob
import time
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import deque

class TrainingData(Dataset):
    """自我对弈数据集"""
    def __init__(self, examples):
        self.states = [state for state, _, _ in examples]
        self.policies = [policy for _, policy, _ in examples]
        self.values = [value for _, _, value in examples]
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return self.states[idx], self.policies[idx], self.values[idx]

class DataHandler:
    def __init__(self, config):
        self.config = config
        self.memory_size = config["memory_size"]
        self.memory = deque(maxlen=self.memory_size)
        print(f"初始化内存大小为 {self.memory_size}")
    
    def create_dataloader(self):
        """创建数据加载器"""
        dataset = TrainingData(list(self.memory))
        return DataLoader(
            dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=self.config["num_workers"],
            pin_memory=True
        )
    
    def load_existing_data(self):
        """加载现有数据，按时间顺序加载，填满固定大小的deque"""
        print(f"加载现有数据从 {self.config['data_dir']}...")
        
        # 按修改时间排序（从旧到新）
        data_files = sorted(
            glob.glob(os.path.join(self.config["data_dir"], "selfplay_*.data")),
            key=os.path.getmtime,
            reverse=False
        )
        
        if not data_files:
            print("未找到数据文件，内存为空。")
            return
        
        loaded_samples_count = 0
        # 从旧到新加载所有文件，deque会自动移除最旧的数据
        for file_path in data_files:
            examples = self._read_data_file(file_path)
            if examples:
                self.memory.extend(examples)
                loaded_samples_count += len(examples)
                print(f"从 {os.path.basename(file_path)} 加载了 {len(examples)} 个样本，当前内存大小: {len(self.memory)}/{self.memory.maxlen}")
        
        print(f"数据加载完成, 尝试加载了 {loaded_samples_count} 个样本，内存最终包含 {len(self.memory)} 个样本")
   
    def add_new_data(self, data_file):
        """添加新数据，旧数据自动从左侧被移除"""
        if not data_file or not os.path.exists(data_file):
            raise ValueError(f"数据文件不存在或无效: {data_file}")
        
        # 读取新数据文件
        new_examples = self._read_data_file(data_file)
        if not new_examples:
            print(f"警告: 数据文件 {data_file} 中没有有效样本。")
            return False
        
        self.memory.extend(new_examples)
        added_count = len(new_examples)
        
        print(f"添加了 {added_count} 个新样本，内存现在有 {len(self.memory)} 个样本 (容量: {self.memory.maxlen})")
        
        return True
    
    def _read_data_file(self, file_path):
        """读取数据文件"""
        try:
            with open(file_path, 'rb') as f:
                # 读取样本数量
                num_samples = int.from_bytes(f.read(4), byteorder='little')
                
                # 读取状态大小和策略大小
                state_size = int.from_bytes(f.read(4), byteorder='little')
                policy_size = int.from_bytes(f.read(4), byteorder='little')
                
                # 计算每个样本的总字节数
                bytes_per_sample = 4 * (state_size + policy_size + 1)  # +1 是价值
                
                # 预计算游戏参数
                board_size = self.config["board_size"]
                input_channels = self.config["input_channels"]
                
                # 一次性读取所有样本数据
                all_data = np.frombuffer(f.read(bytes_per_sample * num_samples), dtype=np.float32)
                
                # 快速创建样本列表
                examples = []
                for i in range(num_samples):
                    # 计算当前样本在大数组中的索引
                    start_idx = i * (state_size + policy_size + 1)
                    
                    # 提取当前样本的数据
                    state_end = start_idx + state_size
                    policy_end = state_end + policy_size
                    
                    state = all_data[start_idx:state_end].reshape(input_channels, board_size, board_size)
                    policy = all_data[state_end:policy_end]
                    value = all_data[policy_end:policy_end+1]
                    
                    # 添加到样本列表
                    examples.append((
                        torch.FloatTensor(state.copy()),
                        torch.FloatTensor(policy.copy()),
                        torch.FloatTensor(value.copy())
                    ))
            
            print(f"从 {file_path} 读取了 {len(examples)} 个样本")
            return examples
            
        except Exception as e:
            print(f"读取数据文件错误: {e}")
            return []
    
    def get_buffer_stats(self):
        """获取内存统计信息"""
        if self.memory is None or not self.memory:
            return {
                "total_samples": 0,
                "buffer_capacity": self.memory_size if hasattr(self, 'memory_size') else 0,
                "buffer_utilization": 0,
            }
        
        utilization = len(self.memory) / self.memory.maxlen if self.memory.maxlen > 0 else 0
        
        return {
            "total_samples": len(self.memory),
            "buffer_capacity": self.memory.maxlen,
            "buffer_utilization": utilization,
        } 