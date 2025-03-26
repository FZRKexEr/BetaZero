import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        # 使用较小的卷积核和更合理的初始化
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
        # 添加PreLU激活函数，更现代的激活函数可以帮助缓解梯度消失
        self.activation = nn.PReLU(num_parameters=channels)
        
        # 添加SE注意力模块
        self.se = SELayer(channels)
        
        # 改进初始化方法
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.conv2.weight)  # 零初始化最后一层，有助于训练初期稳定性
    
    def forward(self, x):
        residual = x
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.se(x)  # 应用SE注意力
        x += residual
        x = self.activation(x)
        return x

# Squeeze-and-Excitation模块，提高网络表达能力
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class Net(nn.Module):
    def __init__(self, board_size, input_channels, num_channels=96, num_res_blocks=4):
        super(Net, self).__init__()
        self.board_size = board_size
        self.num_res_blocks = num_res_blocks
        
        # 输入层，使用更大的卷积核捕获更广范围的特征
        self.conv_input = nn.Conv2d(input_channels, num_channels, 5, padding=2)
        self.bn_input = nn.BatchNorm2d(num_channels)
        self.activation = nn.PReLU(num_parameters=num_channels)
        
        # 残差层
        self.res_blocks = nn.ModuleList([ResidualBlock(num_channels) for _ in range(num_res_blocks)])
        
        # 策略头 - 输出每个位置的着子概率
        self.conv_policy = nn.Conv2d(num_channels, 32, 1)  # 1x1卷积
        self.bn_policy = nn.BatchNorm2d(32)
        self.fc_policy = nn.Linear(32 * board_size * board_size, board_size * board_size)
        
        # 价值头 - 输出局面估值
        self.conv_value = nn.Conv2d(num_channels, 32, 1)  # 1x1卷积
        self.bn_value = nn.BatchNorm2d(32)
        self.fc_value1 = nn.Linear(32 * board_size * board_size, 256)
        self.fc_value2 = nn.Linear(256, 1)
        
        # 模型初始化
        self._init_weights()
    
    def _init_weights(self):
        # 输入层使用He初始化
        nn.init.kaiming_normal_(self.conv_input.weight, mode='fan_out', nonlinearity='relu')
        
        # 策略头初始化
        nn.init.kaiming_normal_(self.conv_policy.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc_policy.weight, mode='fan_out', nonlinearity='linear')
        nn.init.zeros_(self.fc_policy.bias)  # 零初始化偏置
        
        # 价值头初始化
        nn.init.kaiming_normal_(self.conv_value.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc_value1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.fc_value1.bias)
        # 价值头最后一层使用较小的权重初始化
        nn.init.uniform_(self.fc_value2.weight, -0.003, 0.003)
        nn.init.zeros_(self.fc_value2.bias)
    
    def forward(self, x):
        # 输入处理
        x = self.activation(self.bn_input(self.conv_input(x)))

        # 残差层
        for block in self.res_blocks:
            x = block(x)

        # 策略头
        policy = F.relu(self.bn_policy(self.conv_policy(x)))
        policy = policy.contiguous().view(-1, 32 * self.board_size * self.board_size)
        policy = self.fc_policy(policy)
        policy = F.softmax(policy, dim=1)  # 输出概率分布

        # 价值头
        value = F.relu(self.bn_value(self.conv_value(x)))
        value = value.contiguous().view(-1, 32 * self.board_size * self.board_size)
        value = F.relu(self.fc_value1(value))
        value = torch.tanh(self.fc_value2(value))  # 输出范围为[-1, 1]

        return value, policy

class GameModel:
    def __init__(self, board_size, input_channels, device='cuda', learning_rate=0.003):
        self.board_size = board_size
        self.device = device
        
        # 如果使用GPU，启用CUDNN优化
        if self.device == 'cuda':
            torch.backends.cudnn.benchmark = True
        
        # 初始化小型网络
        self.net = Net(board_size=board_size, 
                       input_channels=input_channels,
                       num_channels=96,
                       num_res_blocks=4).to(self.device)
        
        # 使用Adam优化器，调整学习率和weight_decay
        self.optimizer = optim.Adam(self.net.parameters(), 
                                   lr=learning_rate, 
                                   betas=(0.9, 0.999),
                                   eps=1e-8,
                                   weight_decay=1e-5)  # 减小weight_decay避免过度正则化
        
        # 混合精度训练
        self.scaler = GradScaler()
        
        # 添加策略损失和价值损失属性
        self.policy_loss = 0.0
        self.value_loss = 0.0
        
        # 记录当前学习率
        self.current_lr = learning_rate

    def train(self, states, policies, values):
        """优化后的训练方法，直接接收张量批次"""
        self.net.train()
        
        # 确保数据在正确设备上
        states = states.to(self.device)
        policies = policies.to(self.device) 
        values = values.to(self.device)
        
        batch_size = states.size(0)
        
        # 混合精度训练
        self.optimizer.zero_grad()
        
        # 使用autocast
        with autocast(device_type='cuda' if self.device == 'cuda' else 'cpu'):
            value_pred, pi_pred = self.net(states)
            
            # 添加更大的epsilon防止数值不稳定
            epsilon = 1e-7
            
            # 计算损失
            value_loss = F.mse_loss(value_pred, values)
            
            # 更安全的交叉熵计算
            log_pi_pred = torch.log(pi_pred + epsilon)
            policy_loss = -torch.sum(policies * log_pi_pred) / batch_size
            
            # 平衡策略和价值头，稍微增加策略损失权重
            loss = value_loss + 1.2 * policy_loss
            
            # 保存损失值作为类属性，供TensorBoard记录
            self.value_loss = value_loss.item()
            self.policy_loss = policy_loss.item()

        # 使用scaler进行反向传播
        self.scaler.scale(loss).backward()
        
        # 添加梯度裁剪，使用更大的阈值
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=2.0)
        
        # 执行优化步骤
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        # 更新学习率状态
        self.current_lr = self.optimizer.param_groups[0]['lr']
    
        return loss.item()
    
    def set_learning_rate(self, new_lr):
        """手动设置新的学习率"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        self.current_lr = new_lr
        print(f"学习率已手动设置为: {new_lr}")
   
    def save(self, filepath):
        torch.save({
            'net': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'current_lr': self.current_lr
        }, filepath)
        print(f"Model saved to: {filepath}")
    
    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.net.load_state_dict(checkpoint['net'])
        
        # 加载优化器状态
        if 'optimizer' in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                print(f"Optimizer state loaded successfully")
            except Exception as e:
                print(f"Warning: Could not load optimizer state: {e}")
        
        # 恢复学习率
        if 'current_lr' in checkpoint:
            self.current_lr = checkpoint['current_lr']
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.current_lr
            print(f"Current learning rate: {self.current_lr}")
        
        # 重置梯度缩放器
        self.scaler = GradScaler() 