import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        # 使用3x3卷积核，保持空间维度不变
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
        # 使用PreLU激活函数
        self.activation = nn.PReLU(num_parameters=channels)
        
        # SE注意力模块，用于捕捉全局依赖关系
        self.se = SELayer(channels)
        
        # 初始化
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x):
        identity = x
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.se(x)  # 应用SE注意力
        x += identity
        x = self.activation(x)
        return x

class SELayer(nn.Module):
    """Squeeze-and-Excitation注意力模块"""
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
    def __init__(self, board_size=15, input_channels=8, num_channels=128, num_res_blocks=8):
        super(Net, self).__init__()
        self.board_size = board_size
        self.num_res_blocks = num_res_blocks
        
        # 输入层：使用5x5卷积捕获更大的感受野
        self.conv_input = nn.Conv2d(input_channels, num_channels, 5, padding=2)
        self.bn_input = nn.BatchNorm2d(num_channels)
        self.activation = nn.PReLU(num_parameters=num_channels)
        
        # 残差层
        self.res_blocks = nn.ModuleList([ResidualBlock(num_channels) for _ in range(num_res_blocks)])
        
        # 策略头
        self.conv_policy = nn.Conv2d(num_channels, 32, 1)
        self.bn_policy = nn.BatchNorm2d(32)
        self.fc_policy = nn.Linear(32 * board_size * board_size, board_size * board_size)
        
        # 价值头
        self.conv_value = nn.Conv2d(num_channels, 32, 1)
        self.bn_value = nn.BatchNorm2d(32)
        self.fc_value1 = nn.Linear(32 * board_size * board_size, 256)
        self.fc_value2 = nn.Linear(256, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        # 输入层初始化
        nn.init.kaiming_normal_(self.conv_input.weight, mode='fan_out', nonlinearity='relu')
        
        # 策略头初始化
        nn.init.kaiming_normal_(self.conv_policy.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc_policy.weight, mode='fan_out', nonlinearity='linear')
        nn.init.zeros_(self.fc_policy.bias)
        
        # 价值头初始化
        nn.init.kaiming_normal_(self.conv_value.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc_value1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.fc_value1.bias)
        nn.init.uniform_(self.fc_value2.weight, -0.003, 0.003)
        nn.init.zeros_(self.fc_value2.bias)
    
    def forward(self, x):
        # 输入处理
        x = self.activation(self.bn_input(self.conv_input(x)))
        
        # 残差层处理
        for block in self.res_blocks:
            x = block(x)
        
        # 策略头
        policy = F.relu(self.bn_policy(self.conv_policy(x)))
        policy = policy.view(-1, 32 * self.board_size * self.board_size)
        policy = self.fc_policy(policy)
        policy = F.softmax(policy, dim=1)
        
        # 价值头
        value = F.relu(self.bn_value(self.conv_value(x)))
        value = value.view(-1, 32 * self.board_size * self.board_size)
        value = F.relu(self.fc_value1(value))
        value = torch.tanh(self.fc_value2(value))
        
        return value, policy

class GameModel:
    def __init__(self, board_size=8, input_channels=3, device='cuda', learning_rate=0.06144, warmup_steps=1000):
        self.board_size = board_size
        self.device = device
        self.target_lr = learning_rate
        self.warmup_steps = warmup_steps
        self.current_step = 0
        
        # 启用CUDNN优化
        if self.device == 'cuda':
            torch.backends.cudnn.benchmark = True
        
        # 初始化网络 - 使用8层残差块
        self.net = Net(board_size=board_size, 
                      input_channels=input_channels,
                      num_channels=128,
                      num_res_blocks=8   # 增加到8层
                      ).to(self.device)
        
        # 切换到SGD，初始学习率先设置为0或一个很小的值，将在预热阶段增加
        initial_lr = 0.0 if warmup_steps > 0 else learning_rate
        self.optimizer = optim.SGD(
            self.net.parameters(),
            lr=initial_lr,
            momentum=0.9,
            nesterov=False
        )
        
        # 记录当前学习率
        self.current_lr = initial_lr
        
        # 梯度裁剪阈值
        self.max_grad_norm = 1.0  # 保持和之前一样的梯度裁剪阈值
        
        # 记录损失
        self.policy_loss = 0.0
        self.value_loss = 0.0

    def _adjust_learning_rate(self):
        """根据当前步数调整学习率（预热）"""
        if self.warmup_steps > 0 and self.current_step < self.warmup_steps:
            # 线性预热
            lr = self.target_lr * (self.current_step / self.warmup_steps)
        else:
            # 预热结束后使用目标学习率
            lr = self.target_lr
            # 如果之后需要实现其他调度器（如余弦退火），可以在这里添加逻辑

        # 更新优化器中的学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.current_lr = lr

    def train(self, states, policies, values):
        """训练一个批次"""
        self.net.train()
        
        # 确保数据在正确设备上
        states = states.to(self.device)
        policies = policies.to(self.device)
        values = values.to(self.device)
        
        batch_size = states.size(0)
        
        # 在优化器步骤之前调整学习率
        self.current_step += 1
        self._adjust_learning_rate()
        
        # 清零梯度
        self.optimizer.zero_grad()
        
        value_pred, policy_pred = self.net(states)
        
        # 计算损失
        value_loss = F.mse_loss(value_pred, values)
        policy_loss = -torch.sum(policies * torch.log(policy_pred + 1e-7)) / batch_size
        
        # 总损失
        loss = value_loss + policy_loss
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            self.net.parameters(), 
            max_norm=self.max_grad_norm
        )
        
        # 优化器步进
        self.optimizer.step()
        
        # 保存损失值用于记录
        self.value_loss = value_loss.item()
        self.policy_loss = policy_loss.item()
        
        return loss.item()
    
    def save(self, filepath):
        """保存模型"""
        torch.save({
            'net': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'current_lr': self.current_lr,
            'current_step': self.current_step,
            'target_lr': self.target_lr,
            'warmup_steps': self.warmup_steps
        }, filepath)
    
    def load(self, filepath):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.net.load_state_dict(checkpoint['net'])
        
        # 加载优化器状态时，要确保学习率被正确设置
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        # 恢复训练状态
        self.current_step = checkpoint.get('current_step', 0)
        self.target_lr = checkpoint.get('target_lr', self.target_lr)
        self.warmup_steps = checkpoint.get('warmup_steps', self.warmup_steps)
        
        # 根据加载的步数重新计算并设置当前学习率
        self._adjust_learning_rate()
        
        # 更新记录的当前学习率 (虽然_adjust_learning_rate会做，这里显式再做一次确保一致)
        self.current_lr = self.optimizer.param_groups[0]['lr']
