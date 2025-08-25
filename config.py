class SystemConfig:
    def __init__(self):
        # 系统参数配置
        self.PLATFORM_FEE_RATE = 0.10  # 平台抽成比例
        self.MAX_CONCURRENT_TASKS = 3  # 每人最多同时任务数 - 从2增加到3
        self.GSQ_TASK_BATCH_SIZE = 5  # 每次分配任务给几个工人 - 从3增加到5
        self.GSQ_REWARD_RANGE = (0.8, 1.2)  # GSQ奖励随机范围
        self.CONVERGENCE_THRESHOLD = 0.015  # 收敛阈值从2%降低到1.5%
        self.ABILITY_UPDATE_WEIGHTS = (0.8, 0.2)  # 能力更新权重 - 增加GSQ权重
        self.WORKER_ABILITY_WINDOW = 5  # 能力波动计算窗口大小 - 从3增加到5
        self.GSQ_COST_PENALTY_FACTOR = 0.08  # GSQ成本惩罚因子 - 从0.1降低到0.08
        self.MALICIOUS_ERROR_RANGE = (-15, 15)  # 恶意工人误差范围 - 从(-10,10)扩大到(-15,15)
        self.NORMAL_ERROR_RANGE = (-8, 8)  # 正常工人误差范围 - 从(-5,5)扩大到(-8,8)
        
        # 新增训练规模配置
        self.DEFAULT_WORKER_COUNT = 500  # 默认工人数量 - 从200增加到500
        self.DEFAULT_TASK_COUNT = 1000   # 默认任务数量 - 从500增加到1000
        self.DEFAULT_EXPERIMENT_RUNS = 50  # 默认实验运行次数 - 从30增加到50
        self.RL_TRAINING_EPISODES = 100  # RL训练episode数量
        self.GSQ_DECISION_INTERVAL = 3   # GSQ决策间隔 - 从6降低到3
        self.MAX_GSQ_PER_WORKER = 50    # 每个工人最大GSQ数量 - 从20增加到50
        
        # 新增PPO优化配置
        self.PPO_LEARNING_RATE = 1e-4   # PPO学习率 - 从3e-5提高到1e-4
        self.PPO_BATCH_SIZE = 128       # PPO批次大小 - 从64增加到128
        self.PPO_EPOCHS = 10            # PPO训练轮数 - 从4增加到10
        self.PPO_EPSILON = 0.2          # PPO裁剪参数 - 从0.15增加到0.2
        self.PPO_MEMORY_SIZE = 128      # PPO内存大小 - 从64增加到128
        self.PPO_GAE_LAMBDA = 0.95      # PPO GAE参数
        self.PPO_ENTROPY_COEF = 0.01    # PPO熵正则化系数 - 从0.02减少到0.01
        self.PPO_VALUE_COEF = 0.5       # PPO价值损失系数
        self.PPO_GRAD_CLIP = 0.5        # PPO梯度裁剪参数