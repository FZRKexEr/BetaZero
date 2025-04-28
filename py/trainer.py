import os
import time
import torch
import datetime
import subprocess
import glob
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import save_torchscript_model

class Trainer:
    def __init__(self, model, data_handler, config):
        self.model = model
        self.data_handler = data_handler
        self.config = config
        
        # 初始化TensorBoard
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(config["log_dir"], current_time)
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.global_step = 0
        
        # 添加计数器，用于跟踪已训练样本数
        self.samples_trained = 0
    
    def run_selfplay(self, model_path):
        """运行自我对弈"""
        print(f"开始自我对弈，使用模型: {model_path}")
        try:
            subprocess.run(["./bin/BetaZero", "--selfplay", model_path], check=True)
            # 查找最新生成的数据文件
            data_files = glob.glob(os.path.join(self.config["data_dir"], "selfplay_*.data"))
            if data_files:
                return max(data_files, key=os.path.getmtime)
        except Exception as e:
            print(f"自我对弈执行失败: {e}")
        return None
    
    
    def train(self):
        """训练主循环"""
        # 加载现有数据
        self.data_handler.load_existing_data()
        
        for iteration in range(self.config["iterations"]):
            print(f"\n===== 迭代 {iteration+1}/{self.config['iterations']} =====")
            iteration_start_time = time.time() # 记录迭代开始时间
            
            selfplay_duration = 0 # 初始化自我对弈耗时
            # 第一次迭代是否跳过自我对弈
            if iteration > 0 or not self.config["skip_first_selfplay"]:
                # 运行自我对弈
                latest_model_path = self._get_latest_model_path()
                if latest_model_path:
                    selfplay_start_time = time.time() # 记录自我对弈开始时间
                    data_file = self.run_selfplay(latest_model_path)
                    selfplay_end_time = time.time() # 记录自我对弈结束时间
                    selfplay_duration = selfplay_end_time - selfplay_start_time # 计算自我对弈耗时
                    print(f"自我对弈耗时: {selfplay_duration:.2f}秒") # 打印自我对弈耗时
                    self.data_handler.add_new_data(data_file)
                else:
                    print("未找到模型文件，跳过自我对弈。")
            else:
                print("根据配置跳过第一次自我对弈。") # 添加提示
            
            # 训练模型
            dataloader = self.data_handler.create_dataloader()
            training_start_time = time.time() # 记录训练开始时间
            self._train_by_samples(dataloader, iteration)
            training_end_time = time.time() # 记录训练结束时间
            training_duration = training_end_time - training_start_time # 计算训练耗时
            print(f"模型训练耗时: {training_duration:.2f}秒") # 打印训练耗时
            
            # 保存模型
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self._save_model(timestamp)
            
            iteration_end_time = time.time() # 记录迭代结束时间
            iteration_duration = iteration_end_time - iteration_start_time # 计算迭代总耗时
            print(f"迭代 {iteration+1} 总耗时: {iteration_duration:.2f}秒") # 打印迭代总耗时
            
            # 记录耗时到TensorBoard (使用 iteration 作为 x 轴)
            self.writer.add_scalar('Time/selfplay_duration_sec', selfplay_duration, iteration)
            self.writer.add_scalar('Time/training_duration_sec', training_duration, iteration)
            self.writer.add_scalar('Time/iteration_duration_sec', iteration_duration, iteration)
    
    def _train_by_samples(self, dataloader, iteration):
        total_samples_target = self.config["samples_per_iteration"]
        print(f"开始训练，目标训练样本数: {total_samples_target}")
        
        samples_this_iteration = 0
        dataset_passes = 0
        
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        batch_count = 0
        
        start_time = time.time()
        dataloader_iter = iter(dataloader)
        

        buffer_stats = self.data_handler.get_buffer_stats()
        self.writer.add_scalar('Memory/current_samples', buffer_stats["total_samples"], self.global_step)
        self.writer.add_scalar('Memory/buffer_capacity', buffer_stats["buffer_capacity"], self.global_step)
        self.writer.add_scalar('Memory/buffer_utilization', buffer_stats["buffer_utilization"], self.global_step)
        
        # 创建进度条
        pbar = tqdm(total=total_samples_target, desc=f"训练迭代 {iteration+1}", 
                   unit="样本", unit_scale=True, ncols=100, 
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        # 记录训练速度的变量
        last_time_check = time.time()
        last_samples_check = samples_this_iteration
        samples_per_sec = 0
        
        while samples_this_iteration < total_samples_target:
            try:
                # 尝试获取下一个批次
                states, policies, values = next(dataloader_iter)
            except StopIteration:
                # 如果遍历完整个数据集，重新创建迭代器
                dataloader_iter = iter(dataloader)
                dataset_passes += 1
                states, policies, values = next(dataloader_iter)
            
            # 训练一个批次
            loss = self.model.train(states, policies, values)
            
            # 更新样本计数
            batch_size = states.size(0)
            samples_this_iteration += batch_size
            self.samples_trained += batch_size
            
            # 记录损失
            total_loss += loss
            total_policy_loss += self.model.policy_loss
            total_value_loss += self.model.value_loss
            batch_count += 1
            
            # 记录每个批次的损失
            self.writer.add_scalar('Loss/batch', loss, self.global_step)
            self.writer.add_scalar('Loss/policy_batch', self.model.policy_loss, self.global_step)
            self.writer.add_scalar('Loss/value_batch', self.model.value_loss, self.global_step)
            self.global_step += 1
            
            # 更新进度条
            pbar.update(batch_size)
            
            if samples_this_iteration % 5000 <= batch_size:
                current_time = time.time()
                time_diff = current_time - last_time_check
                samples_diff = samples_this_iteration - last_samples_check
                
                if time_diff > 0:
                    samples_per_sec = samples_diff / time_diff
                
                last_time_check = current_time
                last_samples_check = samples_this_iteration
                
                avg_loss = total_loss / batch_count if batch_count > 0 else 0
                avg_policy_loss = total_policy_loss / batch_count if batch_count > 0 else 0
                avg_value_loss = total_value_loss / batch_count if batch_count > 0 else 0
                
                # 更新进度条描述
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'p_loss': f'{avg_policy_loss:.4f}',
                    'v_loss': f'{avg_value_loss:.4f}',
                    '样本/秒': f'{samples_per_sec:.1f}'
                })
                
        # 关闭进度条
        pbar.close()
        
        # 计算平均损失
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        avg_policy_loss = total_policy_loss / batch_count if batch_count > 0 else 0
        avg_value_loss = total_value_loss / batch_count if batch_count > 0 else 0
        
        # 记录每次迭代的平均损失
        self.writer.add_scalar('Loss/iteration', avg_loss, self.global_step)
        self.writer.add_scalar('Loss/policy', avg_policy_loss, self.global_step)
        self.writer.add_scalar('Loss/value', avg_value_loss, self.global_step)
        
        # 计算梯度统计
        grad_stats = self._compute_gradient_stats()
        self.writer.add_scalar('Gradients/norm', grad_stats['avg_norm'], self.global_step)
        self.writer.add_scalar('Gradients/max', grad_stats['grad_max'], self.global_step)
        self.writer.add_scalar('Gradients/mean', grad_stats['grad_mean'], self.global_step)
        self.writer.add_scalar('Gradients/zero_ratio', grad_stats['zero_ratio'], self.global_step)
        self.writer.add_scalar('Gradients/exploding_ratio', grad_stats['exploding_ratio'], self.global_step)
        
        # 训练总结
        elapsed_time = time.time() - start_time
        overall_samples_per_sec = samples_this_iteration / elapsed_time if elapsed_time > 0 else 0
        
        print(f"\n训练完成，共训练 {samples_this_iteration} 个样本，完成 {dataset_passes} 次数据集遍历")
        print(f"总耗时: {elapsed_time:.2f}秒，平均损失: {avg_loss:.4f}")
        print(f"策略损失: {avg_policy_loss:.4f}, 价值损失: {avg_value_loss:.4f}")
        print(f"平均训练速度: {overall_samples_per_sec:.1f} 样本/秒")
        
        # 添加梯度直方图
        for name, param in self.model.net.named_parameters():
            self.writer.add_histogram(f'Gradients/{name}', param.grad, self.global_step)
    
    def _compute_gradient_stats(self):
        """计算更全面的梯度统计信息"""
        # 使用PyTorch内置函数计算总体梯度范数
        total_norm = torch.nn.utils.clip_grad_norm_(
            self.model.net.parameters(), float('inf'), norm_type=2
        )
        
        # 收集详细梯度信息
        grads_abs = []
        grad_max = 0
        grad_min = float('inf')
        grad_mean = 0
        total_params = 0
        zero_grad_count = 0
        exploding_grad_count = 0
        
        for p in self.model.net.parameters():
            if p.grad is not None:
                grad_tensor = p.grad.abs()
                flat_grad = grad_tensor.view(-1)
                
                # 计算基础统计信息
                num_params = flat_grad.numel()
                total_params += num_params
                grad_max = max(grad_max, flat_grad.max().item())
                grad_min = min(grad_min, flat_grad.min().item()) if flat_grad.min().item() > 0 else grad_min
                grad_mean += flat_grad.sum().item()
                
                # 统计梯度消失/爆炸
                zero_grad_count += (flat_grad < 1e-6).sum().item()
                exploding_grad_count += (flat_grad > 10).sum().item()
                
                # 收集所有梯度用于进一步分析
                grads_abs.extend(flat_grad.tolist())
        
        if total_params > 0:
            grad_mean /= total_params
            zero_ratio = zero_grad_count / total_params
            exploding_ratio = exploding_grad_count / total_params
        else:
            grad_mean = 0
            zero_ratio = 0
            exploding_ratio = 0
        
        return {
            'avg_norm': total_norm.item(),
            'grad_max': grad_max,
            'grad_min': grad_min if grad_min != float('inf') else 0,
            'grad_mean': grad_mean,
            'zero_ratio': zero_ratio,
            'exploding_ratio': exploding_ratio
        }
    
    def _get_latest_model_path(self):
        """获取最新模型路径"""
        model_files = glob.glob(os.path.join(self.config["model_dir"], "torchscript_*.pt"))
        if not model_files:
            return None
        return max(model_files, key=os.path.getmtime)

   

    def _save_model(self, timestamp):
        """保存模型"""
        # 保存PyTorch模型
        pytorch_path = os.path.join(self.config["model_dir"], f"pytorch_{timestamp}.pt")
        self.model.save(pytorch_path)
        
        # 保存TorchScript模型
        self._save_torchscript_model(timestamp)
        
        print(f"模型已保存: {pytorch_path}")
    
    def _save_torchscript_model(self, timestamp):
        """保存TorchScript模型"""
        save_torchscript_model(
            model=self.model,
            model_dir=self.config["model_dir"],
            timestamp=timestamp,
            input_channels=self.config["input_channels"],
            board_size=self.config["board_size"],
            device=self.config["device"]
        ) 