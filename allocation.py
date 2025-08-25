import random
import numpy as np
from config import SystemConfig

class TaskAllocator:
    @staticmethod
    def allocate_tasks(workers, tasks):
        """任务分配：按能力排序工人，按奖励排序任务，轮流分配"""
        # 过滤出普通任务
        normal_tasks = [t for t in tasks.values() if not t.is_gsq]
        if not normal_tasks:
            return []
        
        # 按奖励降序排序任务
        sorted_tasks = sorted(normal_tasks, key=lambda x: x.reward, reverse=True)
        
        # 按能力降序排序工人
        sorted_workers = sorted(workers.values(), key=lambda w: w.ability, reverse=True)
        
        # 分组分配任务
        batch_size = SystemConfig().GSQ_TASK_BATCH_SIZE
        allocated = []
        
        for i in range(0, len(sorted_tasks), batch_size):
            task_group = sorted_tasks[i:i+batch_size]
            worker_group = sorted_workers[i:i+batch_size]
            
            for task in task_group:
                for worker in worker_group:
                    if worker.add_task(task):
                        allocated.append((worker.id, task.id))
        
        return allocated
    
    @staticmethod
    def allocate_tasks_with_priority(workers, tasks, priority_weights=None):
        """基于优先级的任务分配"""
        if priority_weights is None:
            priority_weights = {'reward': 0.4, 'difficulty': 0.3, 'deadline': 0.3}
        
        normal_tasks = [t for t in tasks.values() if not t.is_gsq]
        if not normal_tasks:
            return []
        
        # 计算任务优先级分数
        for task in normal_tasks:
            task.priority_score = (
                priority_weights['reward'] * task.reward +
                priority_weights['difficulty'] * (1.0 - task.difficulty) +
                priority_weights['deadline'] * (1.0 / (task.deadline + 1))
            )
        
        # 按优先级排序
        sorted_tasks = sorted(normal_tasks, key=lambda x: x.priority_score, reverse=True)
        sorted_workers = sorted(workers.values(), key=lambda w: w.ability, reverse=True)
        
        return TaskAllocator._assign_tasks_to_workers(sorted_tasks, sorted_workers)
    
    @staticmethod
    def allocate_tasks_load_balanced(workers, tasks):
        """负载均衡的任务分配"""
        normal_tasks = [t for t in tasks.values() if not t.is_gsq]
        if not normal_tasks:
            return []
        
        # 按当前负载排序工人（负载低的优先）
        sorted_workers = sorted(workers.values(), key=lambda w: len(w.tasks))
        sorted_tasks = sorted(normal_tasks, key=lambda x: x.reward, reverse=True)
        
        return TaskAllocator._assign_tasks_to_workers(sorted_tasks, sorted_workers)
    
    @staticmethod
    def _assign_tasks_to_workers(tasks, workers):
        """通用的任务分配逻辑"""
        allocated = []
        worker_idx = 0
        
        for task in tasks:
            # 轮询分配工人
            assigned = False
            attempts = 0
            max_attempts = len(workers)
            
            while not assigned and attempts < max_attempts:
                worker = workers[worker_idx % len(workers)]
                if worker.add_task(task):
                    allocated.append((worker.id, task.id))
                    assigned = True
                worker_idx += 1
                attempts += 1
        
        return allocated

class RewardAllocator:
    @staticmethod
    def allocate_rewards(task, platform=None):
        """
        分配任务奖励
        
        Args:
            task: 任务对象
            platform: 平台对象（用于更新工人奖励记录）
        """
        if task.is_gsq:
            # GSQ任务使用固定奖励
            allocations = {wid: task.reward for wid in task.worker_answers.keys()}
        else:
            # 普通任务按最终权重分配奖励
            total_reward = task.reward
            allocations = {}
            for worker_id, weight in task.final_weights.items():
                worker_reward = weight * total_reward
                allocations[worker_id] = worker_reward
                
                # 更新工人的普通任务奖励记录
                if platform and worker_id in platform.workers:
                    platform.workers[worker_id].add_normal_reward(worker_reward)
        
        return allocations
    
    @staticmethod
    def allocate_rewards_with_bonus(task, platform=None, bonus_factor=0.1):
        """带奖励加成的分配"""
        base_allocations = RewardAllocator.allocate_rewards(task, platform)
        
        if task.is_gsq:
            return base_allocations
        
        # 计算质量奖励
        total_quality = sum(task.final_weights.values())
        if total_quality > 0:
            quality_bonus = task.reward * bonus_factor
            for worker_id in base_allocations:
                quality_ratio = task.final_weights[worker_id] / total_quality
                base_allocations[worker_id] += quality_ratio * quality_bonus
        
        return base_allocations
    
    @staticmethod
    def allocate_rewards_fairness_adjusted(task, platform=None, fairness_weight=0.5):
        """考虑公平性的奖励分配"""
        if task.is_gsq:
            return RewardAllocator.allocate_rewards(task, platform)
        
        # 基础分配
        base_allocations = RewardAllocator.allocate_rewards(task, platform)
        
        # 计算公平性调整
        weights = list(task.final_weights.values())
        mean_weight = np.mean(weights)
        std_weight = np.std(weights) if len(weights) > 1 else 0
        
        adjusted_allocations = {}
        for worker_id, base_reward in base_allocations.items():
            worker_weight = task.final_weights[worker_id]
            
            # 公平性调整：权重接近平均值的工人获得更多奖励
            if std_weight > 0:
                fairness_score = 1.0 - abs(worker_weight - mean_weight) / std_weight
                fairness_score = max(0.1, min(2.0, fairness_score))  # 限制调整范围
            else:
                fairness_score = 1.0
            
            # 混合基础分配和公平性调整
            adjusted_reward = (1 - fairness_weight) * base_reward + \
                            fairness_weight * base_reward * fairness_score
            
            adjusted_allocations[worker_id] = adjusted_reward
        
        return adjusted_allocations

class DynamicRewardAdjuster:
    """动态奖励调整器"""
    
    def __init__(self):
        self.worker_performance_history = {}
        self.task_completion_rates = {}
    
    def adjust_reward_based_on_performance(self, task, worker_id, base_reward):
        """基于工人历史表现调整奖励"""
        if worker_id not in self.worker_performance_history:
            return base_reward
        
        # 计算工人的平均表现
        performance = np.mean(self.worker_performance_history[worker_id])
        
        # 根据表现调整奖励（表现好的工人获得更多奖励）
        adjustment_factor = 0.8 + 0.4 * performance  # 0.8-1.2倍调整
        return base_reward * adjustment_factor
    
    def update_performance_history(self, worker_id, performance_score):
        """更新工人表现历史"""
        if worker_id not in self.worker_performance_history:
            self.worker_performance_history[worker_id] = []
        
        self.worker_performance_history[worker_id].append(performance_score)
        
        # 保持历史记录在合理范围内
        if len(self.worker_performance_history[worker_id]) > 50:
            self.worker_performance_history[worker_id] = \
                self.worker_performance_history[worker_id][-50:]

class GSQRewardAllocator:
    """GSQ奖励分配器"""
    
    def __init__(self, platform):
        """
        初始化GSQ奖励分配器
        
        Args:
            platform: 平台对象
        """
        self.platform = platform
        self.gsq_reward_pool = 0.0
    
    def collect_platform_fee(self, task_reward):
        """
        从普通任务中收集平台费用
        
        Args:
            task_reward: 任务原始奖励
        """
        config = SystemConfig()
        fee = task_reward * config.PLATFORM_FEE_RATE
        self.gsq_reward_pool += fee
        return fee
    
    def allocate_gsq_reward(self, worker_id):
        """
        为工人分配GSQ奖励
        
        Args:
            worker_id: 工人ID
            
        Returns:
            float: GSQ奖励值
        """
        worker = self.platform.workers[worker_id]
        
        # 获取工人最近3次普通任务的平均奖励
        avg_normal_reward = worker.get_average_normal_reward()
        
        # GSQ奖励在平均奖励的±10%范围内随机选择
        gsq_reward = random.uniform(
            avg_normal_reward * 0.9,  # 下限：平均奖励的90%
            avg_normal_reward * 1.1   # 上限：平均奖励的110%
        )
        
        # 确保GSQ奖励在合理范围内
        gsq_reward = max(0.5, min(2.0, gsq_reward))
        
        return gsq_reward
    
    def get_reward_pool_balance(self):
        """
        获取GSQ奖励池余额
        
        Returns:
            float: 奖励池余额
        """
        return self.gsq_reward_pool
    
    def can_allocate_gsq(self, worker_id):
        """
        检查是否可以为工人分配GSQ任务
        
        Args:
            worker_id: 工人ID
            
        Returns:
            bool: 是否可以分配
        """
        worker = self.platform.workers[worker_id]
        
        # 检查工人是否有足够的普通任务历史
        if len(worker.recent_normal_rewards) < 1:
            return False
        
        # 检查奖励池是否有足够余额
        estimated_gsq_reward = self.allocate_gsq_reward(worker_id)
        return self.gsq_reward_pool >= estimated_gsq_reward 