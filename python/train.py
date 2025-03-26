import os
import sys
import time
import json
import glob
import random
import datetime
import argparse
import subprocess
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# 添加 TensorBoard 支持
from torch.utils.tensorboard import SummaryWriter

# 导入神经网络定义
from neural_net import Net, GameModel

# 修改TrainingData类，预先转换数据格式

class TrainingData(Dataset):
    """自我对弈数据集，直接使用已预处理的张量数据"""
    def __init__(self, examples, device=None):
        # 直接使用已转换的张量
        self.states = [state for state, _, _ in examples]
        self.policies = [policy for _, policy, _ in examples]
        self.values = [value for _, _, value in examples]
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return self.states[idx], self.policies[idx], self.values[idx]

def find_latest_model(model_dir, prefix):
    """查找最新的prefix模型文件"""
    pattern = os.path.join(model_dir, f"{prefix}*.pt")
    model_files = glob.glob(pattern)
    if not model_files:
        return None
    # 按照修改时间排序，返回最新的
    latest_model = max(model_files, key=os.path.getmtime)
    return latest_model

def save_model_torchscript(model, filepath):
    """保存模型为TorchScript格式"""
    model.net.eval()
    
    # 创建示例输入并移动到正确的设备上
    example_input = torch.rand(1, 3, 8, 8).to(model.device)  # [batch=1, channels=3, height=8, width=8]
    
    traced_script_module = torch.jit.trace(model.net, example_input)
    
    traced_script_module.save(filepath)
    print(f"模型已保存为TorchScript格式: {filepath}")

def save_model(model, timestamp, model_dir, is_best=False):
    """保存模型（同时保存PyTorch和TorchScript格式）"""
    prefix = "best_" if is_best else ""
    
    # 保存PyTorch格式（用于训练）
    pytorch_path = os.path.join(model_dir, f"{prefix}pytorch_model_{timestamp}.pt")
    model.save(pytorch_path)
    print(f"保存{'最佳' if is_best else ''}PyTorch模型: {pytorch_path}")
    
    # 保存TorchScript格式（用于自我对弈和评估）
    torchscript_path = os.path.join(model_dir, f"{prefix}torchscript_model_{timestamp}.pt")
    save_model_torchscript(model, torchscript_path)
    print(f"保存{'最佳' if is_best else ''}TorchScript模型: {torchscript_path}")
    
    return pytorch_path, torchscript_path

def read_selfplay_data(file_path):
    """读取自我对弈数据，直接返回PyTorch张量格式"""
    examples = []
    
    try:
        with open(file_path, 'rb') as f:
            # 读取样本数量
            num_samples = int.from_bytes(f.read(4), byteorder='little')
            
            # 读取状态大小和策略大小
            state_size = int.from_bytes(f.read(4), byteorder='little')
            policy_size = int.from_bytes(f.read(4), byteorder='little')
            
            print(f"读取数据文件: {file_path}")
            print(f"样本数量: {num_samples}, 状态大小: {state_size}, 策略大小: {policy_size}")
            
            # 读取每个样本并直接转换为张量
            for _ in range(num_samples):
                # 读取状态
                state = np.zeros(state_size, dtype=np.float32)
                for i in range(state_size):
                    state[i] = np.frombuffer(f.read(4), dtype=np.float32)[0]
                
                # 直接将状态重塑为 [channels, height, width] 格式
                state = state.reshape(3, 8, 8)
                
                # 读取策略
                policy = np.zeros(policy_size, dtype=np.float32)
                for i in range(policy_size):
                    policy[i] = np.frombuffer(f.read(4), dtype=np.float32)[0]
                
                # 读取价值
                value = np.frombuffer(f.read(4), dtype=np.float32)[0]
                
                # 直接转换为PyTorch张量
                state_tensor = torch.FloatTensor(state)
                policy_tensor = torch.FloatTensor(policy)
                value_tensor = torch.FloatTensor([value])
                
                examples.append((state_tensor, policy_tensor, value_tensor))
        
        print(f"成功读取 {len(examples)} 个样本")
        return examples
    
    except Exception as e:
        print(f"读取数据文件出错: {e}")
        return []

def load_existing_data(data_dir, memory, max_size):
    """从data目录加载现有的自我对弈数据到内存"""
    print(f"尝试从{data_dir}加载现有自我对弈数据...")
    
    # 找到所有数据文件
    data_files = glob.glob(os.path.join(data_dir, "selfplay_*.data"))
    if not data_files:
        print("未找到任何数据文件")
        return
    
    # 按修改时间从新到旧排序
    data_files.sort(key=os.path.getmtime, reverse=True)
    print(f"找到{len(data_files)}个数据文件")
    
    # 逐个加载数据文件
    total_samples = 0
    for file_path in data_files:
        # 如果内存已满，停止加载
        if len(memory) >= max_size:
            print(f"内存已满，已加载{total_samples}个样本")
            break
        
        # 读取数据文件
        examples = read_selfplay_data(file_path)
        if examples:
            # 计算还能添加多少样本
            space_left = max_size - len(memory)
            
            # 如果空间不足，只添加部分样本
            if len(examples) > space_left:
                memory.extend(examples[:space_left])
                print(f"内存已满，从{file_path}加载了{space_left}个样本")
                total_samples += space_left
                break
            else:
                memory.extend(examples)
                total_samples += len(examples)
                print(f"从{file_path}加载了{len(examples)}个样本")
    
    print(f"共加载了{total_samples}个样本，当前内存中有{len(memory)}个样本")

def run_selfplay(model_path):
    """运行自我对弈程序"""
    print(f"开始自我对弈，使用模型: {model_path}")
    cmd = ["./bin/BetaZero", "--selfplay", model_path]
    try:
        subprocess.run(cmd, check=True)
        # 找到最新生成的数据文件
        data_dir = "./data"
        data_files = glob.glob(os.path.join(data_dir, "selfplay_*.data"))
        if data_files:
            latest_data = max(data_files, key=os.path.getmtime)
            return latest_data
        else:
            print("自我对弈完成，但未找到生成的数据文件")
            return None
    except subprocess.CalledProcessError as e:
        print(f"自我对弈程序执行失败: {e}")
        return None

def train_network(model, memory, args, writer, global_step):
    """训练神经网络 - 优化版 (增加详细计时)"""
    print(f"开始训练神经网络，使用 {len(memory)} 个样本")
    
    dataset = TrainingData(memory)
    
    # 减少工作线程数以避免打开过多文件
    num_workers = min(4, args.num_workers)
    print(f"使用 {num_workers} 个数据加载工作线程 (原始设置: {args.num_workers})")
    
    # 改进DataLoader配置，减少资源消耗
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,  # 禁用持久工作进程以避免资源泄漏
        prefetch_factor=2  # 减少预取因子以降低资源使用
    )
    
    # 训练模型
    total_loss = 0
    total_batches = 0
    
    # 记录当前学习率
    writer.add_scalar('Training/learning_rate', model.current_lr, global_step)
    
    try:
        for epoch in range(args.epochs):
            # 添加epoch计时
            epoch_start_time = time.time()
            epoch_loss = 0
            batch_count = 0
            
            # 用于记录各轮次的梯度和损失细节
            epoch_grad_norm = 0
            epoch_policy_loss = 0
            epoch_value_loss = 0
            epoch_zero_gradients = 0  # 记录接近于零的梯度数量
            
            for states, policies, values in dataloader:
                # 添加批次计时
                batch_start_time = time.time()
                
                # 直接传递整批数据进行训练
                loss = model.train(states, policies, values)
                
                # 记录策略和价值损失
                epoch_policy_loss += model.policy_loss
                epoch_value_loss += model.value_loss
                
                # 计算梯度范数并分析梯度消失情况
                total_norm = 0
                total_params = 0
                near_zero_grads = 0
                for p in model.net.parameters():
                    if p.grad is not None:
                        total_params += p.numel()
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                        
                        # 统计接近于零的梯度数量
                        near_zero = (p.grad.abs() < 1e-6).sum().item()
                        near_zero_grads += near_zero
                
                total_norm = total_norm ** 0.5
                epoch_grad_norm += total_norm
                
                # 计算梯度接近零的比例
                zero_grad_ratio = near_zero_grads / (total_params + 1e-8)
                epoch_zero_gradients += zero_grad_ratio
                
                # 计算批次耗时
                batch_time = time.time() - batch_start_time
                
                epoch_loss += loss
                batch_count += 1
                
                # 记录批次损失和梯度
                writer.add_scalar('Loss/batch', loss, global_step)
                writer.add_scalar('Loss/policy_batch', model.policy_loss, global_step)
                writer.add_scalar('Loss/value_batch', model.value_loss, global_step)
                writer.add_scalar('Gradients/norm', total_norm, global_step)
                writer.add_scalar('Gradients/zero_ratio', zero_grad_ratio, global_step)
                
                global_step += 1
            
            # 计算并记录每个epoch的耗时
            epoch_time = time.time() - epoch_start_time
            
            # 计算平均损失
            avg_epoch_loss = epoch_loss / batch_count
            avg_policy_loss = epoch_policy_loss / batch_count
            avg_value_loss = epoch_value_loss / batch_count
            avg_grad_norm = epoch_grad_norm / batch_count
            avg_zero_grad_ratio = epoch_zero_gradients / batch_count
            
            # 记录每个epoch的平均损失和梯度信息
            writer.add_scalar('Loss/epoch', avg_epoch_loss, global_step)
            writer.add_scalar('Loss/policy_epoch', avg_policy_loss, global_step)
            writer.add_scalar('Loss/value_epoch', avg_value_loss, global_step)
            writer.add_scalar('Loss/policy_value_ratio', avg_policy_loss / (avg_value_loss + 1e-8), global_step)
            writer.add_scalar('Gradients/epoch_avg', avg_grad_norm, global_step)
            writer.add_scalar('Gradients/loss_ratio', avg_grad_norm / (avg_epoch_loss + 1e-8), global_step)
            writer.add_scalar('Gradients/zero_ratio_epoch', avg_zero_grad_ratio, global_step)
            
            total_loss += epoch_loss
            total_batches += batch_count
            
            # 计算平均批次时间和吞吐量
            avg_batch_time = epoch_time / batch_count
            examples_per_second = (batch_count * args.batch_size) / epoch_time
            
            print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_epoch_loss:.4f}, 耗时: {epoch_time:.2f}秒")
            print(f"  - 平均批次时间: {avg_batch_time*1000:.2f}毫秒, 吞吐量: {examples_per_second:.1f}样本/秒")
            print(f"  - 梯度范数: {avg_grad_norm:.6f}, 接近零梯度比例: {avg_zero_grad_ratio:.2%}")
            
            # 在每个epoch结束时添加
            for name, param in model.net.named_parameters():
                writer.add_histogram(f'Weights/{name}', param.data, global_step)
                if param.grad is not None:
                    writer.add_histogram(f'Gradients/{name}', param.grad, global_step)
            
            # 保存最后一轮的梯度信息供主函数使用
            if epoch == args.epochs - 1:
                train_network.last_grad_norm = avg_grad_norm
                train_network.last_zero_grad_ratio = avg_zero_grad_ratio
                
                # 添加特别表格展示梯度分布
                writer.add_figure('Gradients/distribution', 
                                 create_gradient_figure(model.net), 
                                 global_step)
        
        avg_loss = total_loss / total_batches
        print(f"训练完成，平均损失: {avg_loss:.4f}")
        return avg_loss, global_step
        
    finally:
        # 确保数据加载器资源被释放
        print("清理数据加载器资源...")
        del dataloader
        import gc
        gc.collect()  # 强制垃圾回收
        torch.cuda.empty_cache()  # 清空CUDA缓存

# 添加创建梯度分布图表的函数
def create_gradient_figure(net):
    """创建梯度分布图表"""
    try:
        import matplotlib
        matplotlib.use('Agg')  # 非交互式后端
        
        # 使用Ubuntu系统自带字体
        matplotlib.rcParams['font.family'] = 'DejaVu Sans'
        
        import matplotlib.pyplot as plt
        import numpy as np
        
        # 收集梯度数据
        all_grads = []
        layer_names = []
        
        for name, param in net.named_parameters():
            if param.grad is not None:
                # 将梯度数据转换为numpy数组
                grad_data = param.grad.data.cpu().numpy().flatten()
                all_grads.append(grad_data)
                # 简化层名称只保留最后一部分
                layer_names.append(name.split('.')[-1])
        
        # 创建图表
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # 绘制所有梯度的直方图
        all_grads_flat = np.concatenate(all_grads)
        axes[0].hist(all_grads_flat, bins=50, alpha=0.7)
        
        # 只使用英文标签
        axes[0].set_title('Gradient Distribution of All Layers')
        axes[0].set_xlabel('Gradient Value')
        axes[0].set_ylabel('Frequency')
        axes[0].grid(True, linestyle='--', alpha=0.7)
        
        # 计算每层的梯度统计
        zero_ratios = []
        for grads in all_grads:
            zero_ratio = (np.abs(grads) < 1e-6).mean() * 100
            zero_ratios.append(zero_ratio)
        
        # 绘制每层的零梯度比例条形图
        bars = axes[1].bar(range(len(layer_names)), zero_ratios, alpha=0.7)
        
        # 只使用英文标签
        axes[1].set_title('Near-Zero Gradient Ratio by Layer')
        axes[1].set_ylabel('Near-Zero Gradient Percentage (%)')
        axes[1].set_xticks(range(len(layer_names)))
        axes[1].set_xticklabels(layer_names, rotation=90)
        axes[1].grid(True, linestyle='--', alpha=0.7)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%', ha='center', va='bottom', rotation=0)
        
        plt.tight_layout()
        return fig
    except Exception as e:
        # 如果无法创建图表，则创建一个简单的错误消息图
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, f"Failed to create gradient chart: {str(e)}", 
                horizontalalignment='center', verticalalignment='center')
        plt.tight_layout()
        return plt.gcf()

def main(args):
    # 创建必要的目录
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)
    
    # 设置资源限制监控
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    print(f"系统文件句柄限制: 软限制={soft}, 硬限制={hard}")
    
    # 如果软限制太低，尝试提高它
    if soft < 4096 and hard > 4096:
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (4096, hard))
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            print(f"已提高文件句柄限制: 软限制={soft}, 硬限制={hard}")
        except Exception as e:
            print(f"无法提高文件句柄限制: {e}")
    
    # 为TensorBoard日志创建带时间戳的子目录
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.model_dir, 'logs', current_time)
    os.makedirs(log_dir, exist_ok=True)
    
    # 初始化TensorBoard，使用带时间戳的日志目录
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard日志将保存到: {log_dir}")
    global_step = 0
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    args.device = device
    print(f"使用设备: {device}")
    writer.add_text('Training/device', str(device), 0)
    
    # 初始化模型
    model = GameModel(board_size=8, input_channels=3, device=device, learning_rate=args.lr)
    
    # 只查找最新的普通模型
    latest_script_path = find_latest_model(args.model_dir, prefix="torchscript_model_")
    
    if latest_script_path:
        print(f"找到最新模型: {latest_script_path}")
        # 提取完整时间戳
        filename = os.path.basename(latest_script_path)
        parts = filename.split('_')
        timestamp = f"{parts[2]}_{parts[3].split('.')[0]}"
        
        # 加载最新PyTorch模型
        latest_pytorch_path = os.path.join(args.model_dir, f"pytorch_model_{timestamp}.pt")
        print(f"加载最新PyTorch模型: {latest_pytorch_path}")
        model.load(latest_pytorch_path)
        writer.add_text('Training/model_loaded', latest_pytorch_path, 0)
    else:
        # 如果没有找到任何模型，初始化新模型
        print("未找到任何模型，初始化新模型")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        _, latest_script_path = save_model(model, timestamp, args.model_dir, is_best=False)
        writer.add_text('Training/new_model_created', latest_script_path, 0)
    
    # 初始化内存 (保留所有样本直到达到最大限制)
    memory = deque(maxlen=args.memory_size)

    # 加载现有的自我对弈数据
    load_existing_data(args.data_dir, memory, args.memory_size)
    # 删除内存大小记录
    writer.add_text('Training/initial_data_loaded', f"初始加载了 {len(memory)} 个样本", 0)
    
    # 记录训练超参数
    hyperparams = {
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'epochs': args.epochs,
        'optimizer': 'Adam',
        'weight_decay': 1e-4,
        'gradient_clip': 1.0
    }
    writer.add_text('Hyperparameters', str(hyperparams), 0)
    
    # 用于追踪梯度和零梯度比例趋势
    grad_norms = []
    zero_grad_ratios = []
    iterations_x = []
    
    # 训练迭代
    for iteration in range(args.iterations):
        # 每次迭代前强制执行垃圾回收
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        # 记录迭代开始时间
        iter_start_time = time.time()
        
        print(f"\n----- 迭代 {iteration+1}/{args.iterations} -----")
        writer.add_text('Training/iteration', f"迭代 {iteration+1}/{args.iterations}", iteration)
        
        # 判断是否跳过第一次自我对弈
        if iteration == 0 and args.skip_first_selfplay:
            print("跳过第一次自我对弈，直接使用已加载的数据进行训练")
            writer.add_text('Training/info', "跳过第一次自我对弈", iteration)
        else:
            # 记录自我对弈开始时间
            selfplay_start_time = time.time()
            
            # 使用最新模型进行自我对弈
            print(f"使用模型进行自我对弈: {latest_script_path}")
            latest_data_file = run_selfplay(latest_script_path)
            
            # 计算自我对弈耗时
            selfplay_time = time.time() - selfplay_start_time
            print(f"自我对弈耗时: {selfplay_time:.2f}秒")
            # 删除自我对弈时间记录
            
            if not latest_data_file:
                print("无法生成自我对弈数据，跳过本次迭代")
                writer.add_text('Training/error', "无法生成自我对弈数据", iteration)
                continue
            
            # 读取数据放入内存
            data_load_start = time.time()
            new_examples = read_selfplay_data(latest_data_file)
            data_load_time = time.time() - data_load_start
            print(f"数据加载耗时: {data_load_time:.2f}秒")
            
            if not new_examples:
                print("读取自我对弈数据失败，跳过本次迭代")
                writer.add_text('Training/error', "读取数据失败", iteration)
                continue
            
            # 添加新样本到内存
            memory.extend(new_examples)
            print(f"内存中当前样本数: {len(memory)}")
            # 删除内存大小记录
        
        # 记录训练开始时间
        train_start_time = time.time()
        
        # 训练神经网络
        train_loss, global_step = train_network(model, list(memory), args, writer, global_step)
        
        # 计算训练耗时
        train_time = time.time() - train_start_time
        print(f"神经网络训练耗时: {train_time:.2f}秒")
        
        # 显式执行一次清理
        gc.collect()
        torch.cuda.empty_cache()
        
        # 记录每次迭代的平均损失
        writer.add_scalar('Loss/iteration', train_loss, iteration)
        
        # 保存模型
        save_start_time = time.time()
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        _, latest_script_path = save_model(model, timestamp, args.model_dir, is_best=False)
        save_time = time.time() - save_start_time
        print(f"模型保存耗时: {save_time:.2f}秒")
        
        # 计算整个迭代的总耗时
        iter_total_time = time.time() - iter_start_time
        print(f"迭代 {iteration+1} 总耗时: {iter_total_time:.2f}秒 ({iter_total_time/60:.2f}分钟)")
        # 删除迭代时间记录
        
        print(f"训练完成，模型已更新: {latest_script_path}")
        
        # 收集当前迭代的梯度统计信息
        if hasattr(train_network, 'last_grad_norm') and hasattr(train_network, 'last_zero_grad_ratio'):
            grad_norm = getattr(train_network, 'last_grad_norm')
            zero_grad_ratio = getattr(train_network, 'last_zero_grad_ratio')
            grad_norms.append(grad_norm)
            zero_grad_ratios.append(zero_grad_ratio)
            iterations_x.append(iteration)
            
            # 记录到TensorBoard
            writer.add_scalar('Gradients/iteration_norm', grad_norm, iteration)
            writer.add_scalar('Gradients/iteration_zero_ratio', zero_grad_ratio, iteration)
    
    # 训练完成后，添加最终的梯度统计图表
    # 创建更详细的梯度和零梯度比例图表
    if grad_norms and zero_grad_ratios:
        # 记录最终梯度状态
        final_grad_norm = grad_norms[-1]
        final_zero_ratio = zero_grad_ratios[-1]
        writer.add_scalar('Training/final_grad_norm', final_grad_norm, 0)
        writer.add_scalar('Training/final_zero_grad_ratio', final_zero_ratio, 0)
        writer.add_text('Training/gradient_summary', 
                       f"最终梯度范数: {final_grad_norm:.6f}, 零梯度比例: {final_zero_ratio:.2%}", 0)
        
        # 记录训练过程中梯度变化趋势
        writer.add_text('Training/gradient_trend', 
                       f"梯度范数趋势: {'上升' if grad_norms[-1] > grad_norms[0] else '下降'}, " +
                       f"零梯度比例趋势: {'上升' if zero_grad_ratios[-1] > zero_grad_ratios[0] else '下降'}", 0)
    
    # 关闭TensorBoard writer
    writer.close()
    print("训练完成，TensorBoard日志已保存")

# 在参数解析部分添加新选项
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BetaZero 训练脚本")
    parser.add_argument("--iterations", type=int, default=500, help="训练迭代次数")
    parser.add_argument("--memory_size", type=int, default=250000, help="记忆库大小")
    parser.add_argument("--epochs", type=int, default=3, help="每次迭代的训练轮数")
    parser.add_argument("--batch_size", type=int, default=1024, help="训练批次大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载器工作线程数")
    parser.add_argument("--no_gpu", action="store_false", dest="use_gpu", help="禁用GPU使用")
    parser.add_argument("--skip_first_selfplay", action="store_true", help="跳过第一次迭代的自我对弈，直接使用现有数据")
    parser.add_argument("--model_dir", type=str, default="./models", help="模型保存目录")
    parser.add_argument("--data_dir", type=str, default="./data", help="自我对弈数据目录")
    parser.add_argument("--eval_dir", type=str, default="./evaluations", help="评估结果目录")
    
    args = parser.parse_args()
    main(args)