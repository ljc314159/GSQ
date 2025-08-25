import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random

class RunningNormalizer:
    """运行时的标准化器，用于稳定训练"""
    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon
        self.mean = 0.0
        self.var = 1.0
        self.count = 0
    
    def update(self, x):
        """更新标准化参数"""
        if isinstance(x, (list, np.ndarray)):
            x = np.array(x)
            for val in x:
                self._update_single(val)
        else:
            self._update_single(x)
    
    def _update_single(self, x):
        """更新单个值"""
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.var += delta * delta2
    
    def normalize(self, x):
        """标准化值"""
        if self.count < 2:
            return x
        std = np.sqrt(self.var / (self.count - 1)) + self.epsilon
        return (x - self.mean) / std
    
    def reset(self):
        """重置标准化器"""
        self.mean = 0.0
        self.var = 1.0
        self.count = 0

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(ActorCritic, self).__init__()
        
        # Actor网络 (策略网络)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic网络 (价值网络)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        action_probs = self.actor(state)
        value = self.critic(state)
        return action_probs, value

class PPOMemory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.action_probs = []
        self.values = []
        self.dones = []
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.action_probs.clear()
        self.values.clear()
        self.dones.clear()
    
    def __len__(self):
        return len(self.states)

class PPOAgent:
    def __init__(self, state_dim=4, action_dim=2, lr=3e-5, gamma=0.99, 
                 epsilon=0.15, epochs=6, batch_size=64, gae_lambda=0.95):
        """
        初始化PPO代理 - 优化超参数，提高训练稳定性和收敛性
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr  # 降低学习率，从1e-4到3e-5，提高稳定性
        self.gamma = gamma
        self.epsilon = epsilon  # 减少裁剪参数，从0.2到0.15，提高稳定性
        self.epochs = epochs  # 减少训练轮数，从10到6，避免过拟合
        self.batch_size = batch_size  # 减小批次大小，从128到64，提高梯度估计准确性
        self.gae_lambda = gae_lambda
        
        # 网络结构 - 简化网络，提高稳定性
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),  # 减少隐藏层，从256到128
            nn.ReLU(),
            nn.Dropout(0.1),  # 增加dropout，从0.05到0.1，提高泛化能力
            nn.Linear(128, 64),  # 减少隐藏层，从128到64
            nn.ReLU(),
            nn.Dropout(0.1),  # 增加dropout
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),  # 减少隐藏层，从256到128
            nn.ReLU(),
            nn.Dropout(0.1),  # 增加dropout
            nn.Linear(128, 64),  # 减少隐藏层，从128到64
            nn.ReLU(),
            nn.Dropout(0.1),  # 增加dropout
            nn.Linear(64, 1)
        )
        
        # 优化器 - 优化设置，提高稳定性
        self.optimizer = optim.AdamW([  # 使用AdamW优化器，提高稳定性
            {'params': self.actor.parameters(), 'lr': lr},
            {'params': self.critic.parameters(), 'lr': lr}
        ], weight_decay=1e-4,  # 增加权重衰减，从1e-5到1e-4，提高正则化
          betas=(0.9, 0.999),  # 优化beta参数
          eps=1e-8)  # 优化epsilon参数
        
        # 内存 - 优化内存管理
        self.memory = PPOMemory()
        self.min_memory_size = 64  # 减少最小内存大小，从128到64，提高训练频率
        
        # 训练稳定性机制 - 优化
        self.recent_rewards = []
        self.reward_history = []  # 新增奖励历史记录
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(  # 使用自适应学习率调度
            self.optimizer, mode='max', factor=0.9, patience=100,  # 增加耐心值，从50到100
            min_lr=1e-6
        )
        
        # 奖励标准化 - 优化
        self.reward_normalizer = RunningNormalizer()
        self.value_normalizer = RunningNormalizer()
        
        # 新增：训练稳定性参数
        self.clip_grad_norm = 0.5  # 梯度裁剪参数
        self.value_loss_coef = 0.5  # 价值损失系数
        self.entropy_coef = 0.02  # 增加熵正则化系数，从0.01到0.02，鼓励探索
        self.max_grad_norm = 0.5  # 最大梯度范数
        
        # 新增：奖励平滑参数
        self.reward_smoothing_window = 100  # 增加奖励平滑窗口，从50到100，提高稳定性
        self.smoothed_rewards = []  # 平滑后的奖励
        
        # 新增：早停机制
        self.best_reward = float('-inf')
        self.patience_counter = 0
        self.patience = 300  # 增加早停耐心值，从200到300，避免过早停止

    def select_action(self, state):
        """选择动作 - 优化版本，提高稳定性"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # 获取动作概率
        with torch.no_grad():
            action_probs = self.actor(state_tensor)
            value = self.critic(state_tensor)
        
        # 采样动作
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        
        return action.item(), action_probs[0][action.item()].item(), value.item()
    
    def store_transition(self, state, action, reward, next_state, action_prob, value, done):
        """存储经验 - 优化版本"""
        # 奖励预处理：限制极端值
        reward = np.clip(reward, -10.0, 10.0)
        
        self.memory.states.append(state)
        self.memory.actions.append(action)
        self.memory.rewards.append(reward)
        self.memory.next_states.append(next_state)
        self.memory.action_probs.append(action_prob)
        self.memory.values.append(value)
        self.memory.dones.append(done)
    
    def smooth_rewards(self, rewards):
        """奖励平滑，减少噪声"""
        if len(rewards) < self.reward_smoothing_window:
            return rewards
        
        smoothed = []
        for i in range(len(rewards)):
            start_idx = max(0, i - self.reward_smoothing_window // 2)
            end_idx = min(len(rewards), i + self.reward_smoothing_window // 2 + 1)
            window_rewards = rewards[start_idx:end_idx]
            smoothed.append(np.mean(window_rewards))
        
        return smoothed
    
    def compute_gae(self, rewards, values, next_states, dones):
        """计算广义优势估计 - 优化版本，提高稳定性"""
        advantages = []
        gae = 0
        
        # 使用平滑后的奖励
        smoothed_rewards = self.smooth_rewards(rewards)
        
        # 奖励标准化 - 使用更稳定的方法
        if len(smoothed_rewards) > 1:
            # 确保使用正确的数据类型
            if isinstance(smoothed_rewards, torch.Tensor):
                reward_mean = torch.mean(smoothed_rewards).item()
                reward_std = torch.std(smoothed_rewards).item()
            else:
                reward_mean = np.mean(smoothed_rewards)
                reward_std = np.std(smoothed_rewards)
            
            if reward_std > 1e-8:
                if isinstance(smoothed_rewards, torch.Tensor):
                    normalized_rewards = [(r.item() - reward_mean) / reward_std for r in smoothed_rewards]
                else:
                    normalized_rewards = [(r - reward_mean) / reward_std for r in smoothed_rewards]
            else:
                if isinstance(smoothed_rewards, torch.Tensor):
                    normalized_rewards = [r.item() for r in smoothed_rewards]
                else:
                    normalized_rewards = smoothed_rewards
        else:
            if isinstance(smoothed_rewards, torch.Tensor):
                normalized_rewards = [r.item() for r in smoothed_rewards]
            else:
                normalized_rewards = smoothed_rewards
        
        for i in reversed(range(len(normalized_rewards))):
            if i == len(normalized_rewards) - 1:
                # 最后一步，使用下一个状态的价值
                with torch.no_grad():
                    next_value = self.critic(next_states[i].unsqueeze(0)).item()
            else:
                next_value = values[i + 1].item()
            
            delta = normalized_rewards[i] + self.gamma * next_value * (1 - dones[i].item()) - values[i].item()
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[i].item()) * gae
            advantages.insert(0, gae)
        
        return torch.FloatTensor(advantages)
    
    def update(self):
        """更新策略 - 优化版本，提高稳定性"""
        if len(self.memory.states) < self.min_memory_size:
            return
        
        # 获取数据
        states = torch.FloatTensor(self.memory.states)
        actions = torch.LongTensor(self.memory.actions)
        rewards = torch.FloatTensor(self.memory.rewards)
        next_states = torch.FloatTensor(self.memory.next_states)
        action_probs = torch.FloatTensor(self.memory.action_probs)
        values = torch.FloatTensor(self.memory.values)
        dones = torch.FloatTensor(self.memory.dones)
        
        # 计算优势函数
        advantages = self.compute_gae(rewards, values, next_states, dones)
        returns = advantages + values
        
        # 标准化优势 - 使用更稳定的方法
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            # 限制优势值的范围，防止极端值
            advantages = torch.clamp(advantages, -5.0, 5.0)  # 从-10.0,10.0改为-5.0,5.0
        
        # 标准化回报 - 新增，提高训练稳定性
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            returns = torch.clamp(returns, -5.0, 5.0)
        
        # 记录训练前的损失
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        # 多轮训练
        for epoch in range(self.epochs):
            # 随机打乱数据
            indices = torch.randperm(len(states))
            
            for start_idx in range(0, len(states), self.batch_size):
                end_idx = start_idx + self.batch_size
                batch_indices = indices[start_idx:end_idx]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_probs = action_probs[batch_indices]
                
                # 前向传播
                new_action_probs = self.actor(batch_states)
                new_values = self.critic(batch_states).squeeze()
                
                # 计算策略损失
                action_dist = torch.distributions.Categorical(new_action_probs)
                new_probs = action_dist.probs.gather(1, batch_actions.unsqueeze(1)).squeeze()
                
                # 改进的比率计算，防止除零
                ratio = new_probs / (batch_old_probs + 1e-8)
                ratio = torch.clamp(ratio, 0.0, 5.0)  # 限制比率范围，从10.0改为5.0，提高稳定性
                
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 计算价值损失 - 使用Huber损失，提高稳定性
                value_loss = F.huber_loss(new_values, batch_returns, delta=0.5)  # 从1.0改为0.5，提高稳定性
                
                # 计算熵损失（鼓励探索）
                entropy = action_dist.entropy().mean()
                
                # 总损失 - 使用可配置的系数
                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
                
                # 检查损失是否有效
                if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 1000:
                    print(f"Warning: Invalid loss detected: {loss.item()}, skipping update")
                    continue
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪 - 使用更严格的裁剪
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                
                # 检查梯度是否包含NaN
                if torch.isnan(loss) or torch.isinf(loss):
                    print("Warning: Loss contains NaN or Inf, skipping update")
                    continue
                
                self.optimizer.step()
                
                # 累积损失
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
        
        # 计算平均损失
        num_updates = self.epochs * (len(states) // self.batch_size + 1)
        avg_policy_loss = total_policy_loss / num_updates
        avg_value_loss = total_value_loss / num_updates
        avg_entropy = total_entropy / num_updates
        
        # 记录平均奖励
        avg_reward = np.mean(self.memory.rewards)
        self.recent_rewards.append(avg_reward)
        self.reward_history.append(avg_reward)
        
        # 奖励平滑
        if len(self.reward_history) >= self.reward_smoothing_window:
            smoothed_reward = np.mean(self.reward_history[-self.reward_smoothing_window:])
            self.smoothed_rewards.append(smoothed_reward)
        
        # 早停检查
        if len(self.smoothed_rewards) > 0:
            current_reward = self.smoothed_rewards[-1]
            if current_reward > self.best_reward:
                self.best_reward = current_reward
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # 早停
            if self.patience_counter >= self.patience:
                print(f"Early stopping triggered after {self.patience} updates without improvement")
                # 可以在这里保存最佳模型
        
        # 学习率调度 - 基于平滑后的奖励
        if len(self.smoothed_rewards) > 0:
            self.scheduler.step(self.smoothed_rewards[-1])
        
        # 限制历史记录长度
        if len(self.recent_rewards) > 200:
            self.recent_rewards = self.recent_rewards[-200:]
        if len(self.reward_history) > 500:
            self.reward_history = self.reward_history[-500:]
        if len(self.smoothed_rewards) > 200:
            self.smoothed_rewards = self.smoothed_rewards[-200:]
        
        # 打印训练统计
        if len(self.recent_rewards) % 50 == 0:
            print(f"Training Stats - Policy Loss: {avg_policy_loss:.4f}, "
                  f"Value Loss: {avg_value_loss:.4f}, Entropy: {avg_entropy:.4f}, "
                  f"Avg Reward: {avg_reward:.4f}")
        
        # 清空内存
        self.memory.clear()
    
    def get_training_stats(self):
        """获取训练统计信息"""
        if not self.smoothed_rewards:
            return {}
        
        return {
            'recent_rewards': self.recent_rewards[-50:] if self.recent_rewards else [],
            'smoothed_rewards': self.smoothed_rewards[-50:] if self.smoothed_rewards else [],
            'best_reward': self.best_reward,
            'patience_counter': self.patience_counter,
            'current_lr': self.optimizer.param_groups[0]['lr']
        }
    
    def save(self, path='ppo_model.pth'):
        """保存模型"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_reward': self.best_reward,
            'training_stats': self.get_training_stats()
        }, path)
    
    def load(self, path='ppo_model.pth'):
        """加载模型"""
        checkpoint = torch.load(path, map_location='cpu')
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'best_reward' in checkpoint:
            self.best_reward = checkpoint['best_reward'] 