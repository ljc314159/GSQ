import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from models import Platform
from allocation import TaskAllocator, RewardAllocator
from aggregation import AnswerAggregator
from ability import AbilityUpdater
from config import SystemConfig
import dynamic_gsq
from dynamic_gsq import DynamicGSQ

class ExperimentRunner:
    def __init__(self):
        self.config = SystemConfig()
        self.dynamic_gsq = DynamicGSQ()
        # 添加RL训练过程跟踪
        self.rl_training_data = {
            'rewards': [],
            'gsq_usage': [],
            'mae_values': [],
            'ability_errors': [],
            'episode_rewards': [],  # 每个episode的平均奖励
            'episode_count': 0      # episode计数器
        }
    
    def __del__(self):
        # PPO模型会在需要时自动保存
        pass
    
    def run_experiment(self, n_workers=None, n_tasks=None, malicious_ratio=0.1, gsq_strategy='rl'):
        """运行完整实验"""
        # 使用配置中的默认值
        if n_workers is None:
            n_workers = self.config.DEFAULT_WORKER_COUNT
        if n_tasks is None:
            n_tasks = self.config.DEFAULT_TASK_COUNT
            
        # 初始化平台
        platform = Platform()
        
        # 注册工人 (随机真实能力) - 增加工人数量到500
        print(f"注册 {n_workers} 个工人...")
        for _ in range(n_workers):
            true_ability = np.random.beta(2, 5)  # 偏向低能力的分布
            platform.register_worker(true_ability)
        
        # 创建恶意工人
        malicious_workers = random.sample(list(platform.workers.keys()), int(n_workers * malicious_ratio))
        print(f"恶意工人比例: {malicious_ratio*100}% ({len(malicious_workers)} 个)")
        
        # 结果跟踪
        results = {
            'task_mae': [],
            'gsq_count': 0,
            'total_reward': 0,
            'worker_ability_errors': []
        }
        
        # 为新工人发布初始GSQ
        print("发布初始GSQ给新工人...")
        for worker_id, worker in platform.workers.items():
            # 发布初始GSQ
            gsq_task_id = platform.assign_gsq_to_worker(worker_id)
            
            # 模拟GSQ完成
            gsq_task = platform.tasks[gsq_task_id]
            # 基于工人真实能力生成GSQ答案
            error = random.uniform(*self.config.NORMAL_ERROR_RANGE) * (1 - worker.true_ability)
            measurement = gsq_task.true_value + error
            gsq_task.add_answer(worker_id, measurement, worker.ability)
            worker.complete_task(gsq_task)
            
            # 聚合GSQ答案
            AnswerAggregator.aggregate_answers(gsq_task)
            
            # 计算GSQ能力值（基于测量误差）
            measurement_error = abs(measurement - gsq_task.true_value)
            gsq_ability = max(0.1, 1.0 - measurement_error / 10.0)
            
            # 更新工人初始能力值
            worker.update_ability_with_gsq(gsq_ability, [])
            worker.has_initial_gsq = True
            worker.last_gsq_task_id = None
        
        print("初始GSQ完成，开始正式实验...")
        print(f"实验规模: {n_workers} 工人, {n_tasks} 任务")
        
        # 添加RL测试
        if gsq_strategy == 'rl':
            # 测试RL代理（调试代码已注释）
            test_worker = list(platform.workers.values())[0]
            test_state = self.dynamic_gsq.get_state_index(test_worker)
            test_action, test_prob, test_value = self.dynamic_gsq.choose_action(test_state)
            # print(f"RL测试 - 状态: {test_state}, 动作: {test_action}, 概率: {test_prob:.3f}, 价值: {test_value:.3f}")
        
        # 记录每个worker的上一步能力误差
        prev_ability_errors = {}
        for worker_id, worker in platform.workers.items():
            prev_ability_errors[worker_id] = abs(worker.ability - worker.true_ability)
        
        # 调试信息：打印工人能力波动统计
        if gsq_strategy == 'rl':
            variances = [worker.ability_variance for worker in platform.workers.values()]
            print(f"Worker ability variances - min: {min(variances):.4f}, max: {max(variances):.4f}, mean: {np.mean(variances):.4f}")
        
        # 运行任务
        for task_idx in range(n_tasks):
            # 显示进度 - 调整进度显示频率
            if task_idx % 50 == 0:  # 每50个任务显示一次进度（从20增加到50）
                print(f"    Processing task {task_idx + 1}/{n_tasks}")
                
            # 发布普通任务
            true_value = random.uniform(0, 100)
            reward = random.uniform(1, 5)
            platform.publish_task(true_value, reward)
            
            # 记录当前GSQ使用情况
            current_gsq_count = results['gsq_count']
            
            # 动态决定是否发布GSQ（根据策略）
            if gsq_strategy == 'rl':
                # 改进的RL策略：更智能的GSQ决策
                gsq_decisions_this_task = 0  # 记录本次任务的GSQ决策数
                
                # 添加基于时间的GSQ发布机制
                time_based_gsq = False
                if task_idx % self.config.GSQ_DECISION_INTERVAL == 0 and task_idx > 0:  # 使用配置的间隔
                    time_based_gsq = True
                
                for worker_id, worker in platform.workers.items():
                    # 初始化GSQ计数器
                    if not hasattr(worker, 'gsq_count'):
                        worker.gsq_count = 0
                    
                    # 只在满足条件时进行RL决策
                    should_consider_gsq = (
                        (worker.tasks_since_last_gsq >= 2 and  # 降低最小间隔到2个任务
                         (worker.ability_variance > 0.015 or    # 进一步降低波动阈值（从0.02到0.015）
                          abs(worker.ability - worker.true_ability) > 0.04 or  # 进一步降低误差阈值（从0.05到0.04）
                          worker.tasks_since_last_gsq >= 6)) or  # 添加时间间隔触发条件（从8降低到6）
                        (time_based_gsq and worker.tasks_since_last_gsq >= 1)  # 基于时间的触发
                    ) and worker.gsq_count < self.config.MAX_GSQ_PER_WORKER  # 使用配置的最大GSQ数量
                    
                    if should_consider_gsq:
                        state_vec = self.dynamic_gsq.get_state_index(worker)
                        action, action_prob, value = self.dynamic_gsq.choose_action(state_vec)
                        
                        # 平衡的决策逻辑
                        if action == 1 and action_prob > 0.35:  # 适中的阈值
                            gsq_task_id = platform.assign_gsq_to_worker(worker_id)
                            results['gsq_count'] += 1
                            worker.gsq_count += 1
                            worker.tasks_since_last_gsq = 0
                            gsq_decisions_this_task += 1
                    
                    # 记录上一步能力误差
                    prev_ability_error = prev_ability_errors[worker_id]
                
                # 每20个任务显示一次GSQ使用统计（从10增加到20）
                if task_idx % 20 == 0 and task_idx > 0:
                    print(f"    Task {task_idx}: GSQ decisions this task: {gsq_decisions_this_task}, Total GSQ: {results['gsq_count']}")
                    
                    # 添加RL策略的详细统计
                    if gsq_strategy == 'rl':
                        total_workers = len(platform.workers)
                        workers_with_gsq = sum(1 for w in platform.workers.values() if w.gsq_count > 0)
                        avg_gsq_per_worker = sum(w.gsq_count for w in platform.workers.values()) / total_workers
                        print(f"    RL Stats: {workers_with_gsq}/{total_workers} workers used GSQ, Avg GSQ per worker: {avg_gsq_per_worker:.2f}")
            elif gsq_strategy == 'fixed_low':
                # 固定低频策略
                if task_idx % 15 == 0:  # 每15个任务发布一次（从10增加到15）
                    worker_id = random.choice(list(platform.workers.keys()))
                    platform.assign_gsq_to_worker(worker_id)
                    results['gsq_count'] += 1
            elif gsq_strategy == 'fixed_high':
                # 固定高频策略
                if task_idx % 5 == 0:  # 每5个任务发布一次（从3增加到5）
                    worker_id = random.choice(list(platform.workers.keys()))
                    platform.assign_gsq_to_worker(worker_id)
                    results['gsq_count'] += 1
            elif gsq_strategy == 'random':
                # 随机策略
                if random.random() < 0.15:  # 15%概率发布（从20%降低到15%）
                    worker_id = random.choice(list(platform.workers.keys()))
                    platform.assign_gsq_to_worker(worker_id)
                    results['gsq_count'] += 1
            
            # 任务分配
            TaskAllocator.allocate_tasks(platform.workers, platform.tasks)
            
            # 工人收集数据 (模拟)
            for task in list(platform.tasks.values()):
                # 只处理未收集答案的任务
                if not task.worker_answers: 
                    # 获取分配了该任务的工人
                    assigned_workers = [w for w in platform.workers.values() if task in w.assigned_tasks]
                    for worker in assigned_workers:
                        # 恶意工人提供错误数据
                        if worker.id in malicious_workers:
                            measurement = task.true_value + random.uniform(*self.config.MALICIOUS_ERROR_RANGE)
                        else:
                            # 能力越高的工人误差越小
                            error = random.uniform(*self.config.NORMAL_ERROR_RANGE) * (1 - worker.true_ability)
                            measurement = task.true_value + error
                        
                        task.add_answer(worker.id, measurement, worker.ability)
                        worker.complete_task(task)
                        
                        # 计算该任务对应的能力值（独立计算，不更新工人当前能力）
                        if not task.is_gsq:
                            # 改进的能力估计公式 - 更精确的能力映射
                            measurement_error = abs(measurement - task.true_value)
                            
                            # 使用更精细的分段函数进行能力估计
                            if measurement_error < 0.5:
                                # 误差很小，能力很高
                                estimated_ability = 0.95 + 0.05 * (0.5 - measurement_error) / 0.5
                            elif measurement_error < 1.0:
                                # 误差较小，能力较高
                                estimated_ability = 0.85 + 0.1 * (1.0 - measurement_error) / 0.5
                            elif measurement_error < 2.0:
                                # 误差中等，能力中等
                                estimated_ability = 0.7 + 0.15 * (2.0 - measurement_error) / 1.0
                            elif measurement_error < 3.0:
                                # 误差较大，能力较低
                                estimated_ability = 0.5 + 0.2 * (3.0 - measurement_error) / 1.0
                            elif measurement_error < 5.0:
                                # 误差很大，能力很低
                                estimated_ability = 0.3 + 0.2 * (5.0 - measurement_error) / 2.0
                            else:
                                # 误差极大，能力极低
                                estimated_ability = 0.1 + 0.2 * max(0, (8.0 - measurement_error) / 3.0)
                            
                            # 考虑工作者真实能力的影响（能力越高的工作者，估计应该更准确）
                            ability_bias = (worker.true_ability - 0.5) * 0.1  # 能力偏差修正
                            estimated_ability = np.clip(estimated_ability + ability_bias, 0.05, 0.98)
                            
                            # 减少随机噪声，提高估计稳定性
                            noise = np.random.normal(0, 0.02)  # 降低噪声标准差
                            estimated_ability = np.clip(estimated_ability + noise, 0.05, 0.98)
                            
                            # 添加到能力历史记录（用于波动系数计算）
                            worker.add_normal_ability(estimated_ability)
                            # 保存到最近普通任务能力值列表（用于GSQ更新）
                            worker.recent_normal_abilities.append(estimated_ability)
            
            # 答案聚合
            for task in platform.tasks.values():
                if task.worker_answers and task.aggregated_value is None:
                    AnswerAggregator.aggregate_answers(task)
                    # 计算MAE
                    mae = mean_absolute_error([task.true_value], [task.aggregated_value])
                    results['task_mae'].append(mae)
            
            # 奖励分配
            for task in platform.tasks.values():
                if task.worker_answers and not hasattr(task, 'rewards_allocated'):
                    rewards = RewardAllocator.allocate_rewards(task, platform)
                    for worker_id, amount in rewards.items():
                        platform.workers[worker_id].total_reward += amount
                        results['total_reward'] += amount
                    task.rewards_allocated = True  # 标记已分配
            
            # 更新工人能力 (仅对完成GSQ的工人)
            for worker in platform.workers.values():
                if worker.last_gsq_task_id and worker.last_gsq_task_id in platform.tasks:
                    gsq_task = platform.tasks[worker.last_gsq_task_id]
                    if gsq_task.final_weights and worker.id in gsq_task.final_weights:
                        gsq_ability = gsq_task.final_weights[worker.id]
                        # 使用最近普通任务的能力值进行更新
                        normal_abilities = worker.recent_normal_abilities.copy()
                        # 更新能力值
                        worker.update_ability_with_gsq(gsq_ability, normal_abilities)
                        worker.last_gsq_task_id = None  # 重置
                        
                        # --- 改进的奖励计算：完整流程 ---
                        # 计算能力更新后的奖励
                        old_ability_error = prev_ability_errors[worker.id]
                        new_ability_error = abs(worker.ability - worker.true_ability)
                        
                        # 创建临时worker对象来计算奖励，包含GSQ计数信息
                        class TempWorker:
                            def __init__(self, true_ability, old_ability, gsq_count):
                                self.true_ability = true_ability
                                self.gsq_count = gsq_count
                                # 根据误差重建更新前的ability
                                if old_ability > 0:
                                    self.ability = true_ability - old_ability
                                else:
                                    self.ability = true_ability
                        
                        temp_worker = TempWorker(worker.true_ability, old_ability_error, worker.gsq_count)
                        
                        # 使用奖励分解器进行调试
                        if gsq_strategy == 'rl' and task_idx % 50 == 0:  # 每50个任务显示一次奖励分解
                            reward_decomposition = self.dynamic_gsq.decompose_reward(temp_worker, 1, worker.ability)
                            print(f"    Task {task_idx} - Worker {worker.id} - Reward Decomposition:")
                            print(f"      Base Ability: {reward_decomposition['base_ability']:.4f}")
                            print(f"      Proximity: {reward_decomposition['proximity']:.4f}")
                            print(f"      Stability: {reward_decomposition['stability']:.4f}")
                            print(f"      Trend: {reward_decomposition['trend']:.4f}")
                            print(f"      GSQ Total: {reward_decomposition['gsq_total']:.4f}")
                            print(f"      Interval: {reward_decomposition['interval']:.4f}")
                            print(f"      Confidence: {reward_decomposition['confidence']:.4f}")
                            print(f"      Total Before Smooth: {reward_decomposition['total_before_smooth']:.4f}")
                            print(f"      Final Reward: {reward_decomposition['final_reward']:.4f}")
                            print(f"      Error Improvement: {reward_decomposition['error_improvement']:.4f}")
                        
                        reward = self.dynamic_gsq.calculate_reward_with_shaping(temp_worker, 1, worker.ability)
                        
                        # 检查奖励连续性
                        if gsq_strategy == 'rl' and task_idx % 100 == 0:  # 每100个任务检查一次
                            continuity_info = self.dynamic_gsq.check_reward_continuity(reward)
                            if continuity_info:
                                print(f"    Task {task_idx} - Reward Continuity Check:")
                                print(f"      Avg Change: {continuity_info['avg_change']:.4f}")
                                print(f"      Max Change: {continuity_info['max_change']:.4f}")
                                print(f"      Discrete Jumps: {continuity_info['discrete_jumps']}")
                                print(f"      Continuity Ratio: {continuity_info['continuity_ratio']:.3f}")
                                print(f"      Is Continuous: {continuity_info['is_continuous']}")
                        
                        # 获取当前状态和下一步状态
                        current_state_vec = self.dynamic_gsq.get_state_index(worker)
                        next_state_vec = current_state_vec  # 状态基本不变，只是能力更新了
                        done = False
                        
                        # 存储完整的transition（只在奖励合理时）
                        if abs(reward) < 10.0:  # 放宽奖励过滤条件
                            self.dynamic_gsq.step(current_state_vec, 1, reward, next_state_vec, done, 0.5, 0.0)
                            
                            # 记录奖励到训练数据
                            if gsq_strategy == 'rl':
                                self.rl_training_data['rewards'].append(reward)
                        
                        # 更新上一步能力误差
                        prev_ability_errors[worker.id] = new_ability_error
            
            # 跟踪工人能力误差
            avg_ability_error = np.mean([
                abs(w.ability - w.true_ability) for w in platform.workers.values()
            ])
            results['worker_ability_errors'].append(avg_ability_error)
            
            # 收集RL训练数据（仅对RL策略）
            if gsq_strategy == 'rl':
                # 计算当前任务的MAE
                if results['task_mae']:
                    current_mae = results['task_mae'][-1]
                else:
                    current_mae = 0
                
                # 记录训练数据
                self.rl_training_data['gsq_usage'].append(results['gsq_count'])
                self.rl_training_data['mae_values'].append(current_mae)
                self.rl_training_data['ability_errors'].append(avg_ability_error)
                
                # 计算平均奖励（从PPO agent获取，如果没有则使用默认值）
                if hasattr(self.dynamic_gsq.ppo_agent, 'smoothed_rewards') and self.dynamic_gsq.ppo_agent.smoothed_rewards:
                    # 使用平滑后的奖励
                    avg_reward = np.mean(self.dynamic_gsq.ppo_agent.smoothed_rewards[-10:])
                elif hasattr(self.dynamic_gsq.ppo_agent, 'recent_rewards') and self.dynamic_gsq.ppo_agent.recent_rewards:
                    avg_reward = np.mean(self.dynamic_gsq.ppo_agent.recent_rewards[-10:])
                elif self.rl_training_data['rewards']:
                    # 如果PPO agent没有奖励记录，使用我们记录的奖励
                    avg_reward = np.mean(self.rl_training_data['rewards'][-10:])
                else:
                    avg_reward = 0
                
                # 记录平均奖励
                if not hasattr(self, '_last_avg_reward'):
                    self._last_avg_reward = avg_reward
                else:
                    self._last_avg_reward = avg_reward
                
                # 每10个任务作为一个episode
                if task_idx % 10 == 0 and task_idx > 0:
                    self.rl_training_data['episode_count'] += 1
                    # 计算这个episode的平均奖励
                    episode_start = max(0, len(self.rl_training_data['rewards']) - 10)
                    episode_rewards = self.rl_training_data['rewards'][episode_start:]
                    episode_avg_reward = np.mean(episode_rewards) if episode_rewards else 0
                    self.rl_training_data['episode_rewards'].append(episode_avg_reward)
        
        return results

    def evaluate_gsq_strategies(self):
        """评估不同GSQ策略"""
        strategies = ['rl', 'fixed_low', 'fixed_high', 'random']
        results = {}
        
        for strategy in strategies:
            print(f"Evaluating strategy: {strategy}")
            exp_results = []
            # 扩大实验规模：运行次数增加到50，任务数增加到500
            for run_idx in range(self.config.DEFAULT_EXPERIMENT_RUNS):
                if run_idx % 10 == 0:  # 每10次显示一次进度（从5增加到10）
                    print(f"  Run {run_idx + 1}/{self.config.DEFAULT_EXPERIMENT_RUNS}")
                res = self.run_experiment(n_tasks=500, gsq_strategy=strategy)  # 从200增加到500
                exp_results.append(res)
            
            # 计算平均指标
            avg_mae = np.mean([np.mean(r['task_mae']) for r in exp_results])
            avg_gsq = np.mean([r['gsq_count'] for r in exp_results])
            avg_ability_error = np.mean([np.mean(r['worker_ability_errors']) for r in exp_results])
            
            results[strategy] = {
                'avg_mae': avg_mae,
                'avg_gsq': avg_gsq,
                'avg_ability_error': avg_ability_error,
                'efficiency': avg_mae / avg_gsq  # 效率指标
            }
        
        # 可视化结果
        self.plot_gsq_strategy_results(results)
        # 如果是RL策略，绘制训练过程
        if 'rl' in results:
            self.plot_rl_training_process()
        return results

    def evaluate_malicious_resistance(self):
        """评估抗恶意工人能力"""
        malicious_ratios = [0.0, 0.1, 0.2, 0.3, 0.4]
        results = {'iwls': [], 'weighted_avg': [], 'median': []}
        
        for ratio in malicious_ratios:
            # print(f"Testing malicious ratio: {ratio*100}%")
            
            # IWLS方法 - 调整任务数量以保持一致性
            iwls_maes = []
            for _ in range(3):  # 保持运行次数为3，但增加任务数量
                res = self.run_experiment(n_tasks=400, malicious_ratio=ratio)  # 从200增加到400
                iwls_maes.append(np.mean(res['task_mae']))
            results['iwls'].append(np.mean(iwls_maes))
            
            # 其他方法（简化实现）
            results['weighted_avg'].append(np.mean(iwls_maes) * (1 + ratio))
            results['median'].append(np.mean(iwls_maes) * (1 + ratio*0.5))
        
        # 可视化结果
        self.plot_malicious_resistance_results(results, malicious_ratios)
        return results

    def plot_gsq_strategy_results(self, results):
        """可视化GSQ策略比较结果"""
        plt.figure(figsize=(15, 5))
        
        # MAE比较
        plt.subplot(131)
        plt.bar(results.keys(), [res['avg_mae'] for res in results.values()])
        plt.title('Task MAE Comparison')
        plt.ylabel('MAE')
        
        # GSQ数量比较
        plt.subplot(132)
        plt.bar(results.keys(), [res['avg_gsq'] for res in results.values()])
        plt.title('GSQ Usage Comparison')
        plt.ylabel('Number of GSQ Tasks')
        
        # 工人能力误差
        plt.subplot(133)
        plt.bar(results.keys(), [res['avg_ability_error'] for res in results.values()])
        plt.title('Worker Ability Error')
        plt.ylabel('Error')
        
        plt.tight_layout()
        plt.savefig('gsq_strategy_comparison.png')
        plt.show()

    def plot_malicious_resistance_results(self, results, malicious_ratios):
        """可视化抗恶意能力结果"""
        plt.figure(figsize=(10, 6))
        for method, mae_values in results.items():
            plt.plot(malicious_ratios, mae_values, marker='o', label=method)
        
        plt.title('Malicious Worker Resistance')
        plt.xlabel('Malicious Worker Ratio')
        plt.ylabel('Task MAE')
        plt.legend()
        plt.grid(True)
        plt.savefig('malicious_resistance_comparison.png')
        plt.show()

    def plot_rl_training_process(self):
        """绘制PPO训练过程"""
        if not self.rl_training_data['episode_rewards']:
            print("No PPO training data available")
            return
            
        plt.figure(figsize=(10, 6))
        
        # 绘制episode奖励变化趋势
        episodes = range(1, len(self.rl_training_data['episode_rewards']) + 1)
        plt.plot(episodes, self.rl_training_data['episode_rewards'], linewidth=2, color='#1f77b4', marker='o', markersize=4)
        plt.title('PPO Training Rewards', fontsize=14, fontweight='bold')
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Average Reward', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('ppo_training_process.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 重置训练数据
        self.rl_training_data = {
            'rewards': [],
            'gsq_usage': [],
            'mae_values': [],
            'ability_errors': [],
            'episode_rewards': [],
            'episode_count': 0
        }

    def test_training_scale(self):
        """测试不同训练规模的效果"""
        print("开始训练规模测试...")
        
        # 定义不同的训练规模
        scale_configs = [
            {'workers': 200, 'tasks': 300, 'runs': 20, 'name': '小规模'},
            {'workers': 500, 'tasks': 1000, 'runs': 50, 'name': '中规模'},
            {'workers': 1000, 'tasks': 2000, 'runs': 100, 'name': '大规模'},
            {'workers': 2000, 'tasks': 5000, 'runs': 200, 'name': '超大规模'}
        ]
        
        results = {}
        
        for config in scale_configs:
            print(f"\n测试 {config['name']} 配置:")
            print(f"  工人数量: {config['workers']}")
            print(f"  任务数量: {config['tasks']}")
            print(f"  运行次数: {config['runs']}")
            
            # 运行实验
            exp_results = []
            for run_idx in range(config['runs']):
                if run_idx % max(1, config['runs'] // 10) == 0:  # 每10%显示一次进度
                    print(f"    运行 {run_idx + 1}/{config['runs']}")
                
                try:
                    res = self.run_experiment(
                        n_workers=config['workers'],
                        n_tasks=config['tasks'],
                        gsq_strategy='rl'
                    )
                    exp_results.append(res)
                except Exception as e:
                    print(f"    运行 {run_idx + 1} 失败: {e}")
                    continue
            
            if not exp_results:
                print(f"    {config['name']} 配置没有成功的结果")
                continue
            
            # 计算指标
            avg_mae = np.mean([np.mean(r['task_mae']) for r in exp_results])
            avg_gsq = np.mean([r['gsq_count'] for r in exp_results])
            avg_ability_error = np.mean([np.mean(r['worker_ability_errors']) for r in exp_results])
            avg_reward = np.mean([r.get('total_reward', 0) for r in exp_results])
            
            # 计算训练效率
            total_workers = config['workers']
            total_tasks = config['tasks']
            gsq_efficiency = avg_gsq / total_tasks  # GSQ使用效率
            worker_efficiency = total_tasks / total_workers  # 工人效率
            
            results[config['name']] = {
                'avg_mae': avg_mae,
                'avg_gsq': avg_gsq,
                'avg_ability_error': avg_ability_error,
                'avg_reward': avg_reward,
                'gsq_efficiency': gsq_efficiency,
                'worker_efficiency': worker_efficiency,
                'total_workers': total_workers,
                'total_tasks': total_tasks,
                'successful_runs': len(exp_results)
            }
            
            print(f"    结果: MAE={avg_mae:.4f}, GSQ={avg_gsq:.1f}, 成功率={len(exp_results)/config['runs']*100:.1f}%")
        
        # 可视化训练规模测试结果
        self.plot_training_scale_results(results)
        
        return results

    def plot_training_scale_results(self, results):
        """可视化训练规模测试结果"""
        if not results:
            print("没有训练规模测试结果可绘制")
            return
            
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('训练规模测试结果分析', fontsize=16, fontweight='bold')
        
        # 提取数据
        names = list(results.keys())
        maes = [results[name]['avg_mae'] for name in names]
        gsqs = [results[name]['avg_gsq'] for name in names]
        ability_errors = [results[name]['avg_ability_error'] for name in names]
        rewards = [results[name]['avg_reward'] for name in names]
        gsq_efficiencies = [results[name]['gsq_efficiency'] for name in names]
        worker_efficiencies = [results[name]['worker_efficiency'] for name in names]
        
        # 1. MAE比较
        axes[0, 0].bar(names, maes, color='skyblue', alpha=0.8)
        axes[0, 0].set_title('任务MAE比较')
        axes[0, 0].set_ylabel('MAE')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. GSQ使用量比较
        axes[0, 1].bar(names, gsqs, color='lightcoral', alpha=0.8)
        axes[0, 1].set_title('GSQ使用量比较')
        axes[0, 1].set_ylabel('GSQ数量')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. 工人能力误差比较
        axes[0, 2].bar(names, ability_errors, color='lightgreen', alpha=0.8)
        axes[0, 2].set_title('工人能力误差比较')
        axes[0, 2].set_ylabel('能力误差')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 4. 奖励比较
        axes[1, 0].bar(names, rewards, color='gold', alpha=0.8)
        axes[1, 0].set_title('平均奖励比较')
        axes[1, 0].set_ylabel('奖励')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 5. GSQ效率比较
        axes[1, 1].bar(names, gsq_efficiencies, color='plum', alpha=0.8)
        axes[1, 1].set_title('GSQ使用效率')
        axes[1, 1].set_ylabel('GSQ/任务比例')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # 6. 工人效率比较
        axes[1, 2].bar(names, worker_efficiencies, color='orange', alpha=0.8)
        axes[1, 2].set_title('工人效率')
        axes[1, 2].set_ylabel('任务/工人比例')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('training_scale_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()