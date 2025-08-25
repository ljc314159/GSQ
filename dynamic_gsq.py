import numpy as np
import random
from config import SystemConfig
from ppo_agent import PPOAgent
import torch

class DynamicGSQ:
    def __init__(self):
        # PPO代理初始化 - 使用新的优化参数
        self.ppo_agent = PPOAgent(
            state_dim=4, 
            action_dim=2,
            lr=1e-4,  # 提高学习率，从3e-5到1e-4，加快收敛
            epochs=10,  # 增加训练轮数，从4到10，提高收敛性
            batch_size=128,  # 增大批次大小，从64到128，提高稳定性
            gae_lambda=0.95,
            epsilon=0.2  # 增加裁剪参数，从0.15到0.2，提高稳定性
        )
        
        # GSQ使用控制参数 - 优化参数
        self.gsq_usage_threshold = 0.15  # GSQ使用概率阈值 - 从0.2降低到0.15
        self.min_interval = 1  # 最小GSQ间隔
        
        # 奖励连续性检测器
        self.reward_history = []
        self.continuity_threshold = 0.08  # 连续性阈值 - 从0.1降低到0.08
        
        # 新增：训练规模控制
        self.training_episodes = 0
        self.max_training_episodes = 1000  # 最大训练episode数
        self.episode_rewards = []
        self.convergence_check_interval = 50  # 每50个episode检查一次收敛
        
        # 新增：训练稳定性参数
        self.reward_smoothing_window = 50  # 奖励平滑窗口
        self.convergence_threshold = 0.01  # 收敛阈值
        self.stability_window = 100  # 稳定性检查窗口


    

    
    def get_state_index(self, worker):
        """将工人状态转为状态向量（能力、任务间隔、波动、能力误差）"""
        ability = float(worker.ability)
        interval = float(worker.tasks_since_last_gsq)
        variance = float(worker.ability_variance)
        # 添加能力误差作为新的状态特征
        ability_error = abs(worker.ability - worker.true_ability)
        return np.array([ability, interval, variance, ability_error], dtype=np.float32)

    def choose_action(self, state_vec):
        """用PPOAgent选择动作（0=不发GSQ, 1=发GSQ）"""
        action, action_prob, value = self.ppo_agent.select_action(state_vec)
        
        # 添加GSQ使用控制逻辑
        interval = state_vec[1]  # 任务间隔
        ability_error = state_vec[3]  # 能力误差
        
        # 如果间隔太短，强制不发布GSQ
        if interval < self.min_interval:
            return 0, action_prob, value
        
        # 动态调整GSQ使用阈值：能力误差越大，越容易触发GSQ
        dynamic_threshold = self.gsq_usage_threshold
        if ability_error > 0.3:  # 能力误差较大
            dynamic_threshold = 0.1  # 降低阈值，更容易触发GSQ
        elif ability_error > 0.2:  # 能力误差中等
            dynamic_threshold = 0.15
        elif ability_error > 0.1:  # 能力误差较小
            dynamic_threshold = 0.2
        else:  # 能力误差很小
            dynamic_threshold = 0.25  # 提高阈值，减少不必要的GSQ
        
        # 如果RL建议发布GSQ，但概率不够高，则不发布
        if action == 1 and action_prob < dynamic_threshold:
            return 0, action_prob, value
            
        return action, action_prob, value

    def step(self, state, action, reward, next_state, done, action_prob, value):
        """存储transition并训练PPOAgent"""
        self.ppo_agent.store_transition(state, action, reward, next_state, action_prob, value, done)
        self.ppo_agent.update()
    
    def add_reward_noise(self, reward, noise_scale=0.02):
        """添加小的随机噪声到奖励，减少离散性"""
        noise = np.random.normal(0, noise_scale)
        return reward + noise
    
    def calculate_reward(self, worker, action, new_ability):
        """平衡的奖励函数 - 保持连续性同时提供有效学习信号"""
        # 核心指标：能力误差改善
        ability_error_before = abs(worker.ability - worker.true_ability)
        ability_error_after = abs(new_ability - worker.true_ability)
        error_improvement = ability_error_before - ability_error_after
        
        # 1. 基础奖励：能力改善（增加奖励系数，更强调误差改善）
        if error_improvement > 0:
            base_reward = 2.0 * error_improvement  # 大幅增加奖励系数
        else:
            base_reward = 0.5 * error_improvement  # 增加惩罚，但保持合理
        
        # 2. GSQ使用奖励（基于效果）
        if action == 1:  # 发布GSQ
            if error_improvement > 0:
                gsq_reward = 0.5 * error_improvement  # 增加GSQ有效时的奖励
            else:
                gsq_reward = -0.05  # 轻微惩罚GSQ无效的情况
        else:
            gsq_reward = 0.0
        
        # 3. 能力接近度奖励（鼓励接近真实能力）
        current_error = ability_error_after
        proximity_reward = 0.2 * (1.0 - current_error)  # 增加接近度奖励权重
        
        # 4. 能力稳定性奖励（新增）
        if hasattr(worker, 'ability_variance'):
            stability_reward = 0.1 * (1.0 - min(1.0, worker.ability_variance / 0.1))
        else:
            stability_reward = 0.0
        
        # 5. 组合奖励
        total_reward = base_reward + gsq_reward + proximity_reward + stability_reward
        
        # 6. 限制奖励范围，但允许更大的变化
        return np.clip(total_reward, -0.5, 1.5)
    
    def decompose_reward(self, worker, action, new_ability):
        """分解奖励函数，返回各个组成部分，用于分析和调试"""
        # 能力误差改善
        ability_error_before = abs(worker.ability - worker.true_ability)
        ability_error_after = abs(new_ability - worker.true_ability)
        error_improvement = ability_error_before - ability_error_after
        
        # 1. 基础能力改善奖励
        improvement_scale = 20.0
        base_ability_reward = 1.5 * np.tanh(improvement_scale * error_improvement)
        
        # 2. 奖励shaping组件
        # 2.1 能力误差接近度奖励
        error_proximity = 1.0 / (1.0 + 10.0 * ability_error_after)
        proximity_reward = 0.3 * error_proximity
        
        # 2.2 能力稳定性奖励
        if hasattr(worker, 'ability_variance'):
            stability_reward = 0.2 * np.exp(-10.0 * worker.ability_variance)
        else:
            stability_reward = 0.0
        
        # 2.3 能力改善趋势奖励
        if hasattr(worker, 'ability_history') and len(worker.ability_history) >= 3:
            recent_improvements = []
            for i in range(1, min(4, len(worker.ability_history))):
                prev_error = abs(worker.ability_history[-i] - worker.true_ability)
                curr_error = abs(worker.ability_history[-(i-1)] - worker.true_ability)
                recent_improvements.append(prev_error - curr_error)
            
            if recent_improvements:
                trend_reward = 0.1 * np.mean(recent_improvements)
            else:
                trend_reward = 0.0
        else:
            trend_reward = 0.0
        
        # 3. GSQ使用奖励
        if action == 1:
            if error_improvement > 0:
                gsq_effectiveness = min(1.0, error_improvement / 0.05)
                gsq_reward = 0.2 * gsq_effectiveness
            else:
                gsq_reward = -0.05 * min(1.0, abs(error_improvement) / 0.05)
            
            if hasattr(worker, 'gsq_count'):
                if worker.gsq_count <= 10:
                    frequency_bonus = 0.1 * (1.0 - worker.gsq_count / 10.0)
                elif worker.gsq_count <= 20:
                    frequency_bonus = 0.0
                else:
                    frequency_bonus = -0.05 * (worker.gsq_count - 20) / 10.0
            else:
                frequency_bonus = 0.0
            
            gsq_total_reward = gsq_reward + frequency_bonus
        else:
            gsq_total_reward = 0.0
        
        # 4. 环境状态奖励
        if hasattr(worker, 'tasks_since_last_gsq'):
            interval = worker.tasks_since_last_gsq
            if interval < 2:
                interval_reward = -0.1 * (2 - interval)
            elif interval > 15:
                interval_reward = -0.1 * (interval - 15) / 10.0
            else:
                interval_reward = 0.0
        else:
            interval_reward = 0.0
        
        if hasattr(worker, 'ability_variance'):
            confidence_reward = 0.1 * (1.0 - min(1.0, worker.ability_variance / 0.1))
        else:
            confidence_reward = 0.0
        
        # 5. 组合奖励
        total_reward = (
            0.4 * base_ability_reward +
            0.15 * proximity_reward +
            0.1 * stability_reward +
            0.05 * trend_reward +
            0.2 * gsq_total_reward +
            0.05 * interval_reward +
            0.05 * confidence_reward
        )
        
        # 6. 最终平滑奖励
        smoothed_reward = 2.0 * (1.0 / (1.0 + np.exp(-total_reward)) - 0.5)
        final_reward = np.clip(smoothed_reward, -1.0, 2.0)
        
        return {
            'base_ability': base_ability_reward,
            'proximity': proximity_reward,
            'stability': stability_reward,
            'trend': trend_reward,
            'gsq_total': gsq_total_reward,
            'interval': interval_reward,
            'confidence': confidence_reward,
            'total_before_smooth': total_reward,
            'final_reward': final_reward,
            'error_improvement': error_improvement,
            'ability_error_after': ability_error_after
        }
    
    def potential_function(self, worker):
        """奖励shaping的潜在函数，用于改善奖励信号"""
        # 基于当前状态的潜在值
        potential = 0.0
        
        # 1. 能力误差潜在（误差越小，潜在值越高）
        if hasattr(worker, 'ability') and hasattr(worker, 'true_ability'):
            error = abs(worker.ability - worker.true_ability)
            potential += 1.0 / (1.0 + 5.0 * error)
        
        # 2. 能力稳定性潜在（方差越小，潜在值越高）
        if hasattr(worker, 'ability_variance'):
            potential += 0.5 * np.exp(-5.0 * worker.ability_variance)
        
        # 3. 任务间隔潜在（合理间隔，潜在值较高）
        if hasattr(worker, 'tasks_since_last_gsq'):
            interval = worker.tasks_since_last_gsq
            if 3 <= interval <= 10:
                potential += 0.3
            elif interval > 15:
                potential -= 0.1 * (interval - 15) / 10.0
        
        # 4. GSQ使用频率潜在（适度使用，潜在值较高）
        if hasattr(worker, 'gsq_count'):
            if worker.gsq_count <= 15:
                potential += 0.2 * (1.0 - worker.gsq_count / 15.0)
            else:
                potential -= 0.1 * (worker.gsq_count - 15) / 10.0
        
        return potential
    
    def calculate_reward_with_shaping(self, worker, action, new_ability):
        """使用奖励shaping计算奖励"""
        # 计算当前状态和下一状态的潜在值
        current_potential = self.potential_function(worker)
        
        # 创建临时worker来模拟下一状态
        temp_worker = type('TempWorker', (), {
            'ability': new_ability,
            'true_ability': worker.true_ability,
            'ability_variance': getattr(worker, 'ability_variance', 0.0),
            'tasks_since_last_gsq': 0 if action == 1 else getattr(worker, 'tasks_since_last_gsq', 0) + 1,
            'gsq_count': getattr(worker, 'gsq_count', 0) + (1 if action == 1 else 0)
        })()
        
        next_potential = self.potential_function(temp_worker)
        
        # 计算shaping奖励
        shaping_reward = next_potential - current_potential
        
        # 计算基础奖励
        base_reward = self.calculate_reward(worker, action, new_ability)
        
        # 组合奖励（shaping权重较小）
        total_reward = base_reward + 0.1 * shaping_reward
        
        return np.clip(total_reward, -1.0, 2.0)



    def check_reward_continuity(self, reward):
        """检查奖励的连续性"""
        self.reward_history.append(reward)
        if len(self.reward_history) > 100:
            self.reward_history.pop(0)
        
        if len(self.reward_history) >= 2:
            # 计算最近奖励的变化
            recent_changes = []
            for i in range(1, min(10, len(self.reward_history))):
                change = abs(self.reward_history[-i] - self.reward_history[-(i+1)])
                recent_changes.append(change)
            
            # 计算连续性指标
            avg_change = np.mean(recent_changes)
            max_change = np.max(recent_changes)
            
            # 检测离散跳变
            discrete_jumps = sum(1 for change in recent_changes if change > self.continuity_threshold)
            continuity_ratio = 1.0 - (discrete_jumps / len(recent_changes))
            
            return {
                'avg_change': avg_change,
                'max_change': max_change,
                'discrete_jumps': discrete_jumps,
                'continuity_ratio': continuity_ratio,
                'is_continuous': continuity_ratio > 0.8
            }
        
        return None