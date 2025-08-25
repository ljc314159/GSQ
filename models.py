import random
import numpy as np
from config import SystemConfig

class Worker:
    """
    工作者类：表示平台上的一个工作者
    
    属性：
        id: 工作者唯一标识符
        true_ability: 工作者的真实能力值（系统内部使用，工作者本人不知道）
        ability: 当前估计的能力值
        ability_history: 能力值历史记录列表
        assigned_tasks: 当前分配的任务列表
        last_gsq_task_id: 最后一次GSQ任务ID
        ability_variance: 能力值方差（用于衡量能力估计的不确定性）
        tasks_since_last_gsq: 自上次GSQ任务后完成的任务数量
        total_reward: 累计获得的总奖励
        has_initial_gsq: 是否已经完成过初始GSQ任务
        recent_normal_abilities: 最近普通任务的能力表现记录
        gsq_count: 完成的GSQ任务数量
    """
    def __init__(self, worker_id, true_ability):
        """
        初始化工作者
        
        Args:
            worker_id: 工作者ID
            true_ability: 真实能力值
        """
        self.id = worker_id
        self.true_ability = true_ability
        self.ability = 0.5  # 初始能力值设为0.5（中等水平）
        self.ability_history = []
        self.assigned_tasks = []
        self.last_gsq_task_id = None
        self.ability_variance = 0.0
        self.tasks_since_last_gsq = 0
        self.total_reward = 0.0
        self.has_initial_gsq = False
        self.recent_normal_abilities = []
        self.gsq_count = 0
        self.recent_normal_rewards = []  # 最近普通任务的奖励记录
        
    def update_ability(self, new_ability):
        """
        更新工作者的能力值
        
        Args:
            new_ability: 新的能力值
        """
        # 将当前能力值保存到历史记录中
        self.ability_history.append(self.ability)
        self.ability = new_ability
        
        # 计算最近3次能力值的方差，用于衡量能力估计的不确定性
        if len(self.ability_history) >= 3:
            recent_abilities = self.ability_history[-3:]
            self.ability_variance = np.std(recent_abilities)
            # 如果方差太小，添加一些随机噪声避免过拟合
            if self.ability_variance < 0.01:
                self.ability_variance = 0.01 + np.random.uniform(0, 0.02)
        
    def add_normal_ability(self, normal_ability):
        """
        添加普通任务的能力表现记录
        
        Args:
            normal_ability: 普通任务中表现出的能力值
        """
        self.ability_history.append(normal_ability)
        
        # 计算能力值方差
        if len(self.ability_history) >= 3:
            recent_abilities = self.ability_history[-3:]
            self.ability_variance = np.std(recent_abilities)
            if self.ability_variance < 0.01:
                self.ability_variance = 0.01 + np.random.uniform(0, 0.02)
        elif len(self.ability_history) > 0:
            # 如果历史记录不足3个，基于平均值和噪声计算方差
            base_ability = np.mean(self.ability_history)
            noise = np.random.normal(0, 0.1)
            self.ability_variance = abs(noise) + 0.01
        
    def update_ability_with_gsq(self, gsq_ability, normal_abilities):
        """
        基于GSQ任务和普通任务表现更新能力值
        
        Args:
            gsq_ability: GSQ任务中表现出的能力值
            normal_abilities: 普通任务能力值列表
        """
        # 如果有普通任务记录，使用加权平均更新能力值
        if normal_abilities:
            avg_normal_ability = np.mean(normal_abilities)
            
            # 自适应权重调整：基于普通任务的一致性
            normal_variance = np.var(normal_abilities) if len(normal_abilities) > 1 else 0.1
            
            # 如果普通任务表现一致性好，增加其权重；否则更依赖GSQ
            if normal_variance < 0.05:  # 一致性很好
                gsq_weight = 0.7
                normal_weight = 0.3
            elif normal_variance < 0.1:  # 一致性较好
                gsq_weight = 0.75
                normal_weight = 0.25
            else:  # 一致性较差，更依赖GSQ
                gsq_weight = 0.85
                normal_weight = 0.15
            
            new_ability = gsq_weight * gsq_ability + normal_weight * avg_normal_ability
        else:
            # 如果没有普通任务记录，直接使用GSQ能力值
            new_ability = gsq_ability
        
        # 更新能力值历史记录
        self.ability_history.append(self.ability)
        self.ability = new_ability
        
        # 改进的能力值方差计算
        if len(self.ability_history) >= 3:
            recent_abilities = self.ability_history[-3:]
            self.ability_variance = np.std(recent_abilities)
            # 减少最小方差限制，允许更精确的方差估计
            if self.ability_variance < 0.005:
                self.ability_variance = 0.005 + np.random.uniform(0, 0.01)
        elif len(self.ability_history) > 0:
            # 如果历史记录不足3个，基于当前能力和历史平均值计算方差
            base_ability = np.mean(self.ability_history)
            self.ability_variance = abs(self.ability - base_ability) + 0.01
        
        # 清空最近的普通任务能力记录
        self.recent_normal_abilities = []
        
    def add_task(self, task):
        """
        为工作者分配任务
        
        Args:
            task: 要分配的任务对象
            
        Returns:
            bool: 分配是否成功
        """
        # 检查是否超过最大并发任务数限制
        if len(self.assigned_tasks) < SystemConfig().MAX_CONCURRENT_TASKS:
            self.assigned_tasks.append(task)
            return True
        return False
        
    def complete_task(self, task):
        """
        完成指定任务
        
        Args:
            task: 要完成的任务对象
            
        Returns:
            bool: 任务完成是否成功
        """
        if task in self.assigned_tasks:
            self.assigned_tasks.remove(task)
            # 如果不是GSQ任务，增加普通任务计数
            if not task.is_gsq:
                self.tasks_since_last_gsq += 1
            return True
        return False
    
    def add_normal_reward(self, reward):
        """
        添加普通任务奖励记录
        
        Args:
            reward: 普通任务获得的奖励
        """
        self.recent_normal_rewards.append(reward)
        # 只保留最近3次普通任务的奖励记录
        if len(self.recent_normal_rewards) > 3:
            self.recent_normal_rewards = self.recent_normal_rewards[-3:]
    
    def get_average_normal_reward(self):
        """
        获取最近3次普通任务的平均奖励
        
        Returns:
            float: 平均奖励，如果没有记录则返回默认值
        """
        if not self.recent_normal_rewards:
            return 1.0  # 默认奖励值
        return np.mean(self.recent_normal_rewards)

class Task:
    """
    任务类：表示平台上的一个任务
    
    属性：
        id: 任务唯一标识符
        true_value: 任务的真实值（标准答案）
        reward: 任务奖励
        is_gsq: 是否为GSQ任务
        worker_answers: 工作者答案字典 {worker_id: (measurement, ability)}
        aggregated_value: 聚合后的值
        final_weights: 最终权重
    """
    def __init__(self, task_id, true_value, reward, is_gsq=False):
        """
        初始化任务
        
        Args:
            task_id: 任务ID
            true_value: 真实值
            reward: 奖励
            is_gsq: 是否为GSQ任务
        """
        self.id = task_id
        self.true_value = true_value
        self.reward = reward
        self.is_gsq = is_gsq
        self.worker_answers = {}
        self.aggregated_value = None
        self.final_weights = None
        
    def add_answer(self, worker_id, measurement, ability):
        """
        添加工作者的答案
        
        Args:
            worker_id: 工作者ID
            measurement: 工作者的测量值
            ability: 工作者当前的能力值
        """
        self.worker_answers[worker_id] = (measurement, ability)
        
    def calculate_worker_error(self, worker_id, ground_truth):
        """
        计算工作者的误差
        
        Args:
            worker_id: 工作者ID
            ground_truth: 真实值
            
        Returns:
            float: 误差绝对值
        """
        measurement, _ = self.worker_answers[worker_id]
        return abs(measurement - ground_truth)

class Platform:
    """
    平台类：管理工作者和任务的核心平台
    
    属性：
        gsq_reward_pool: GSQ奖励池（从普通任务中抽取的平台费用）
        task_counter: 任务计数器
        workers: 工作者字典 {worker_id: Worker}
        tasks: 任务字典 {task_id: Task}
    """
    def __init__(self):
        """初始化平台"""
        self.gsq_reward_pool = 0.0
        self.task_counter = 0
        self.workers = {}
        self.tasks = {}
        
    def register_worker(self, true_ability):
        """
        注册新工作者
        
        Args:
            true_ability: 工作者的真实能力值
            
        Returns:
            str: 新工作者的ID
        """
        worker_id = f"W{len(self.workers)+1}"
        self.workers[worker_id] = Worker(worker_id, true_ability)
        return worker_id
        
    def publish_task(self, true_value, reward, is_gsq=False):
        """
        发布新任务
        
        Args:
            true_value: 任务真实值
            reward: 任务奖励
            is_gsq: 是否为GSQ任务
            
        Returns:
            str: 新任务的ID
        """
        task_id = f"T{self.task_counter}"
        self.task_counter += 1
        
        config = SystemConfig()
        
        # 普通任务需要扣除平台费用，费用进入GSQ奖励池
        if not is_gsq:
            reward_after_fee = reward * (1 - config.PLATFORM_FEE_RATE)
            self.gsq_reward_pool += reward * config.PLATFORM_FEE_RATE
        else:
            # GSQ任务不需要扣除费用
            reward_after_fee = reward
        
        # 创建任务对象并添加到任务字典中
        task = Task(task_id, true_value, reward_after_fee, is_gsq)
        self.tasks[task_id] = task
        return task_id
        
    def assign_gsq_to_worker(self, worker_id):
        """
        为指定工作者分配GSQ任务
        
        Args:
            worker_id: 工作者ID
            
        Returns:
            str: GSQ任务的ID
        """
        config = SystemConfig()
        worker = self.workers[worker_id]
        
        # 根据工人最近3次普通任务的平均奖励确定GSQ奖励
        avg_normal_reward = worker.get_average_normal_reward()
        
        # GSQ奖励在平均奖励的±10%范围内随机选择
        gsq_reward = random.uniform(
            avg_normal_reward * 0.9,  # 下限：平均奖励的90%
            avg_normal_reward * 1.1   # 上限：平均奖励的110%
        )
        
        # 确保GSQ奖励在合理范围内
        gsq_reward = max(0.5, min(2.0, gsq_reward))
        
        # 创建GSQ任务（真实值随机生成）
        gsq_task_id = self.publish_task(true_value=random.uniform(0, 100), 
                                       reward=gsq_reward, 
                                       is_gsq=True)
        
        # 将GSQ任务分配给工作者
        worker.add_task(self.tasks[gsq_task_id])
        worker.last_gsq_task_id = gsq_task_id
        worker.tasks_since_last_gsq = 0  # 重置普通任务计数
        
        return gsq_task_id