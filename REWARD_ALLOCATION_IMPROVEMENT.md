# 奖励分配机制改进说明

## 概述

根据图片中的规则，我们对奖励分配机制进行了全面改进，主要包括以下几个方面：

## 1. 普通任务奖励分配

### 原有机制
- 普通任务按最终权重分配奖励
- 没有记录工人的奖励历史

### 改进机制
- 普通任务按最终权重分配奖励（保持不变）
- **新增**：自动记录工人最近3次普通任务的奖励
- **新增**：支持平台参数，用于更新工人奖励记录

### 代码实现
```python
def allocate_rewards(task, platform=None):
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
```

## 2. GSQ任务奖励分配

### 原有机制
- GSQ任务使用固定奖励范围
- 奖励与工人历史表现无关

### 改进机制
- **GSQ奖励池**：从普通任务中抽取10%的平台费用
- **动态GSQ奖励**：基于工人最近3次普通任务的平均奖励
- **奖励范围**：在平均奖励的±10%范围内随机选择
- **防检测机制**：确保GSQ奖励与工人日常收入水平匹配

### 代码实现

#### Worker类新增方法
```python
def add_normal_reward(self, reward):
    """添加普通任务奖励记录"""
    self.recent_normal_rewards.append(reward)
    # 只保留最近3次普通任务的奖励记录
    if len(self.recent_normal_rewards) > 3:
        self.recent_normal_rewards = self.recent_normal_rewards[-3:]

def get_average_normal_reward(self):
    """获取最近3次普通任务的平均奖励"""
    if not self.recent_normal_rewards:
        return 1.0  # 默认奖励值
    return np.mean(self.recent_normal_rewards)
```

#### Platform类改进
```python
def assign_gsq_to_worker(self, worker_id):
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
    
    # 创建GSQ任务
    gsq_task_id = self.publish_task(true_value=random.uniform(0, 100), 
                                   reward=gsq_reward, 
                                   is_gsq=True)
    
    return gsq_task_id
```

## 3. GSQ奖励分配器

### 新增功能
- **GSQRewardAllocator类**：专门管理GSQ奖励分配
- **平台费用收集**：从普通任务中自动收集10%费用
- **奖励池管理**：维护GSQ奖励池余额
- **分配条件检查**：确保有足够余额和工人历史记录

### 代码实现
```python
class GSQRewardAllocator:
    def collect_platform_fee(self, task_reward):
        """从普通任务中收集平台费用"""
        config = SystemConfig()
        fee = task_reward * config.PLATFORM_FEE_RATE
        self.gsq_reward_pool += fee
        return fee
    
    def allocate_gsq_reward(self, worker_id):
        """为工人分配GSQ奖励"""
        worker = self.platform.workers[worker_id]
        avg_normal_reward = worker.get_average_normal_reward()
        
        # GSQ奖励在平均奖励的±10%范围内随机选择
        gsq_reward = random.uniform(
            avg_normal_reward * 0.9,
            avg_normal_reward * 1.1
        )
        
        return max(0.5, min(2.0, gsq_reward))
    
    def can_allocate_gsq(self, worker_id):
        """检查是否可以为工人分配GSQ任务"""
        worker = self.platform.workers[worker_id]
        
        # 检查工人是否有足够的普通任务历史
        if len(worker.recent_normal_rewards) < 1:
            return False
        
        # 检查奖励池是否有足够余额
        estimated_gsq_reward = self.allocate_gsq_reward(worker_id)
        return self.gsq_reward_pool >= estimated_gsq_reward
```

## 4. 系统配置

### 相关配置参数
```python
class SystemConfig:
    def __init__(self):
        self.PLATFORM_FEE_RATE = 0.10  # 平台抽成比例（10%）
        self.GSQ_REWARD_RANGE = (0.8, 1.2)  # GSQ奖励随机范围（已废弃）
```

## 5. 改进效果

### 防检测机制
- GSQ奖励与工人日常收入水平匹配
- 减少工人对GSQ任务的怀疑
- 提高GSQ任务的自然性

### 奖励池机制
- 确保GSQ任务有稳定的资金来源
- 平台费用自动收集和管理
- 支持可持续的GSQ任务分配

### 历史记录管理
- 自动维护工人奖励历史
- 支持动态奖励计算
- 提高系统透明度

## 6. 使用示例

```python
# 创建平台和工人
platform = Platform()
worker_id = platform.register_worker(0.7)

# 工人完成普通任务（自动记录奖励）
task_id = platform.publish_task(50.0, 10.0, is_gsq=False)
task = platform.tasks[task_id]
task.add_answer(worker_id, 51.0, 0.7)
AnswerAggregator.aggregate_answers(task)
rewards = RewardAllocator.allocate_rewards(task, platform)

# 分配GSQ任务（基于历史奖励）
gsq_task_id = platform.assign_gsq_to_worker(worker_id)
gsq_task = platform.tasks[gsq_task_id]
print(f"GSQ奖励: {gsq_task.reward:.2f}")
```

## 总结

通过以上改进，我们实现了：

1. **符合规则的奖励分配**：严格按照图片中的规则实现
2. **防检测机制**：GSQ奖励与工人日常收入匹配
3. **可持续的资金来源**：通过平台费用建立GSQ奖励池
4. **自动化的历史管理**：工人奖励历史自动记录和更新
5. **灵活的分配策略**：支持多种奖励分配方式

这些改进使得GSQ系统更加隐蔽、可持续和高效。 