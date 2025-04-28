import os
import glob
import torch
import numpy as np

def find_latest_model(model_dir, prefix="pytorch_"):
    """查找最新的模型文件"""
    pattern = os.path.join(model_dir, f"{prefix}*.pt")
    model_files = glob.glob(pattern)
    if not model_files:
        return None
    return max(model_files, key=os.path.getmtime)

def set_random_seed(seed=42):
    """设置随机种子以确保可重复性"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def save_torchscript_model(model, model_dir, timestamp, input_channels, board_size, device):
    """保存TorchScript模型"""
    model.net.eval()
    
    # 创建示例输入
    example_input = torch.rand(
        1, 
        input_channels, 
        board_size, 
        board_size
    ).to(device)
    
    # 追踪并保存模型
    torchscript_path = os.path.join(model_dir, f"torchscript_{timestamp}.pt")
    traced_script_module = torch.jit.trace(model.net, example_input)
    traced_script_module.save(torchscript_path)
    
    print(f"TorchScript模型已保存: {torchscript_path}")
    return torchscript_path 