"""
训练规模配置文件
专门管理动态GSQ系统的训练规模参数
"""

class TrainingScaleConfig:
    """训练规模配置类"""
    
    def __init__(self):
        # 基础训练规模配置
        self.BASE_WORKER_COUNT = 200
        self.BASE_TASK_COUNT = 300
        self.BASE_EXPERIMENT_RUNS = 20
        
        # 中等训练规模配置
        self.MEDIUM_WORKER_COUNT = 500
        self.MEDIUM_TASK_COUNT = 1000
        self.MEDIUM_EXPERIMENT_RUNS = 50
        
        # 大规模训练配置
        self.LARGE_WORKER_COUNT = 1000
        self.LARGE_TASK_COUNT = 2000
        self.LARGE_EXPERIMENT_RUNS = 100
        
        # 超大规模训练配置
        self.EXTRA_LARGE_WORKER_COUNT = 2000
        self.EXTRA_LARGE_TASK_COUNT = 5000
        self.EXTRA_LARGE_EXPERIMENT_RUNS = 200
        
        # RL训练参数
        self.RL_LEARNING_RATE = 3e-5  # 降低学习率，从3e-5到3e-5，提高稳定性
        self.RL_BATCH_SIZE = 64  # 保持批次大小64，平衡稳定性和效率
        self.RL_EPOCHS = 6  # 减少训练轮数，从4到6，避免过拟合
        self.RL_MEMORY_SIZE = 64  # 保持内存大小64，提高训练频率
        self.RL_UPDATE_FREQUENCY = 8  # 减少更新频率，从10到8，提高训练频率
        
        # 训练稳定性参数
        self.CONVERGENCE_THRESHOLD = 0.01  # 降低收敛阈值，从0.015到0.01，提高精度
        self.REWARD_SMOOTHING_WINDOW = 100  # 增加平滑窗口，从20到100，提高稳定性
        self.EARLY_STOPPING_PATIENCE = 150  # 增加早停耐心值，从100到150，避免过早停止
        
        # 性能监控参数
        self.PROGRESS_UPDATE_INTERVAL = 50
        self.METRICS_SAVE_INTERVAL = 100
        self.MODEL_SAVE_INTERVAL = 500
        
        # 资源管理参数
        self.MAX_MEMORY_USAGE = 0.8  # 最大内存使用率
        self.BATCH_PROCESSING_SIZE = 100  # 批处理大小
        self.PARALLEL_WORKERS = 4  # 并行工作进程数

class ScaleTestConfig:
    """规模测试配置类"""
    
    def __init__(self):
        # 测试配置列表
        self.TEST_CONFIGS = [
            {
                'name': '小规模测试',
                'workers': 200,
                'tasks': 300,
                'runs': 20,
                'description': '基础功能验证'
            },
            {
                'name': '中规模测试',
                'workers': 500,
                'tasks': 1000,
                'runs': 50,
                'description': '性能基准测试'
            },
            {
                'name': '大规模测试',
                'workers': 1000,
                'tasks': 2000,
                'runs': 100,
                'description': '压力测试'
            },
            {
                'name': '超大规模测试',
                'workers': 2000,
                'tasks': 5000,
                'runs': 200,
                'description': '极限测试'
            }
        ]
        
        # 测试评估指标
        self.EVALUATION_METRICS = [
            'avg_mae',           # 平均MAE
            'avg_gsq',           # 平均GSQ使用量
            'avg_ability_error', # 平均能力误差
            'avg_reward',        # 平均奖励
            'gsq_efficiency',    # GSQ使用效率
            'worker_efficiency', # 工人效率
            'training_time',     # 训练时间
            'memory_usage',      # 内存使用
            'convergence_rate'   # 收敛率
        ]
        
        # 性能基准
        self.PERFORMANCE_BENCHMARKS = {
            'mae_threshold': 0.1,      # MAE阈值
            'gsq_efficiency_min': 0.05, # GSQ效率最小值
            'convergence_time_max': 1000, # 最大收敛时间
            'memory_usage_max': 0.8    # 最大内存使用率
        }

class OptimizationConfig:
    """优化配置类"""
    
    def __init__(self):
        # 网络结构优化
        self.NETWORK_ARCHITECTURES = {
            'small': {'hidden_layers': [64, 32], 'dropout': 0.1},
            'medium': {'hidden_layers': [128, 64], 'dropout': 0.15},
            'large': {'hidden_layers': [256, 128, 64], 'dropout': 0.2}
        }
        
        # 学习率调度
        self.LEARNING_RATE_SCHEDULES = {
            'constant': {'lr': 3e-5, 'decay': 1.0},
            'step': {'lr': 5e-5, 'decay': 0.9, 'step_size': 500},
            'exponential': {'lr': 1e-4, 'decay': 0.95}
        }
        
        # 正则化参数
        self.REGULARIZATION = {
            'weight_decay': 1e-4,
            'dropout_rate': 0.15,
            'gradient_clip': 1.0,
            'batch_norm': True
        }
        
        # 训练策略
        self.TRAINING_STRATEGIES = {
            'standard': {'epochs': 4, 'batch_size': 64},
            'aggressive': {'epochs': 6, 'batch_size': 128},
            'conservative': {'epochs': 2, 'batch_size': 32}
        }

# 创建配置实例
training_config = TrainingScaleConfig()
scale_test_config = ScaleTestConfig()
optimization_config = OptimizationConfig()

def get_optimal_config(scale_level='medium'):
    """根据规模级别获取最优配置"""
    if scale_level == 'small':
        return {
            'workers': training_config.BASE_WORKER_COUNT,
            'tasks': training_config.BASE_TASK_COUNT,
            'runs': training_config.BASE_EXPERIMENT_RUNS,
            'rl_config': {
                'lr': training_config.RL_LEARNING_RATE,
                'batch_size': 32,
                'epochs': 2
            }
        }
    elif scale_level == 'medium':
        return {
            'workers': training_config.MEDIUM_WORKER_COUNT,
            'tasks': training_config.MEDIUM_TASK_COUNT,
            'runs': training_config.MEDIUM_EXPERIMENT_RUNS,
            'rl_config': {
                'lr': training_config.RL_LEARNING_RATE,
                'batch_size': training_config.RL_BATCH_SIZE,
                'epochs': training_config.RL_EPOCHS
            }
        }
    elif scale_level == 'large':
        return {
            'workers': training_config.LARGE_WORKER_COUNT,
            'tasks': training_config.LARGE_TASK_COUNT,
            'runs': training_config.LARGE_EXPERIMENT_RUNS,
            'rl_config': {
                'lr': training_config.RL_LEARNING_RATE * 0.8,
                'batch_size': training_config.RL_BATCH_SIZE * 2,
                'epochs': training_config.RL_EPOCHS + 1
            }
        }
    elif scale_level == 'extra_large':
        return {
            'workers': training_config.EXTRA_LARGE_WORKER_COUNT,
            'tasks': training_config.EXTRA_LARGE_TASK_COUNT,
            'runs': training_config.EXTRA_LARGE_EXPERIMENT_RUNS,
            'rl_config': {
                'lr': training_config.RL_LEARNING_RATE * 0.6,
                'batch_size': training_config.RL_BATCH_SIZE * 4,
                'epochs': training_config.RL_EPOCHS + 2
            }
        }
    else:
        raise ValueError(f"不支持的规模级别: {scale_level}")

if __name__ == "__main__":
    # 测试配置
    print("训练规模配置测试:")
    print(f"小规模配置: {get_optimal_config('small')}")
    print(f"中规模配置: {get_optimal_config('medium')}")
    print(f"大规模配置: {get_optimal_config('large')}")
    print(f"超大规模配置: {get_optimal_config('extra_large')}")
