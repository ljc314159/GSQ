"""
PPO超参数配置文件
专门管理PPO训练的超参数，提高训练稳定性和收敛性
"""

class PPOConfig:
    """PPO训练配置类"""
    
    def __init__(self):
        # 基础超参数 - 已优化
        self.LEARNING_RATE = 3e-5  # 学习率：降低到3e-5提高稳定性
        self.GAMMA = 0.99  # 折扣因子：保持0.99
        self.EPSILON = 0.15  # PPO裁剪参数：降低到0.15提高稳定性
        self.EPOCHS = 6  # 训练轮数：减少到6避免过拟合
        self.BATCH_SIZE = 64  # 批次大小：保持64平衡稳定性和效率
        self.GAE_LAMBDA = 0.95  # GAE参数：保持0.95
        
        # 网络结构参数 - 已优化
        self.HIDDEN_DIM = 128  # 隐藏层维度：从256减少到128
        self.DROPOUT_RATE = 0.1  # Dropout率：从0.05增加到0.1
        self.NETWORK_LAYERS = 2  # 网络层数：从3减少到2
        
        # 优化器参数 - 已优化
        self.WEIGHT_DECAY = 1e-4  # 权重衰减：从1e-5增加到1e-4
        self.BETA1 = 0.9  # Adam优化器beta1
        self.BETA2 = 0.999  # Adam优化器beta2
        self.EPS = 1e-8  # Adam优化器epsilon
        
        # 训练稳定性参数 - 已优化
        self.CLIP_GRAD_NORM = 0.5  # 梯度裁剪范数
        self.VALUE_LOSS_COEF = 0.5  # 价值损失系数
        self.ENTROPY_COEF = 0.02  # 熵正则化系数：从0.01增加到0.02
        self.MAX_GRAD_NORM = 0.5  # 最大梯度范数
        
        # 奖励处理参数 - 已优化
        self.REWARD_SMOOTHING_WINDOW = 100  # 奖励平滑窗口：从50增加到100
        self.ADVANTAGE_CLIP_RANGE = (-5.0, 5.0)  # 优势值裁剪范围：从(-10,10)改为(-5,5)
        self.RETURN_CLIP_RANGE = (-5.0, 5.0)  # 回报值裁剪范围：新增
        self.RATIO_CLIP_RANGE = (0.0, 5.0)  # 比率裁剪范围：从(0,10)改为(0,5)
        
        # 学习率调度参数 - 已优化
        self.LR_SCHEDULER_FACTOR = 0.9  # 学习率衰减因子：从0.8增加到0.9
        self.LR_SCHEDULER_PATIENCE = 100  # 学习率调度耐心值：从50增加到100
        self.MIN_LR = 1e-6  # 最小学习率
        
        # 早停参数 - 已优化
        self.EARLY_STOPPING_PATIENCE = 300  # 早停耐心值：从200增加到300
        
        # 内存管理参数 - 已优化
        self.MIN_MEMORY_SIZE = 64  # 最小内存大小：从128减少到64
        self.UPDATE_FREQUENCY = 8  # 更新频率：从10减少到8
        
        # 损失函数参数 - 已优化
        self.HUBER_DELTA = 0.5  # Huber损失delta：从1.0减少到0.5
        
        # 收敛检测参数 - 已优化
        self.CONVERGENCE_THRESHOLD = 0.01  # 收敛阈值：从0.015减少到0.01
        
    def get_optimized_config(self):
        """获取优化后的配置字典"""
        return {
            'learning_rate': self.LEARNING_RATE,
            'gamma': self.GAMMA,
            'epsilon': self.EPSILON,
            'epochs': self.EPOCHS,
            'batch_size': self.BATCH_SIZE,
            'gae_lambda': self.GAE_LAMBDA,
            'hidden_dim': self.HIDDEN_DIM,
            'dropout_rate': self.DROPOUT_RATE,
            'network_layers': self.NETWORK_LAYERS,
            'weight_decay': self.WEIGHT_DECAY,
            'clip_grad_norm': self.CLIP_GRAD_NORM,
            'value_loss_coef': self.VALUE_LOSS_COEF,
            'entropy_coef': self.ENTROPY_COEF,
            'reward_smoothing_window': self.REWARD_SMOOTHING_WINDOW,
            'advantage_clip_range': self.ADVANTAGE_CLIP_RANGE,
            'return_clip_range': self.RETURN_CLIP_RANGE,
            'ratio_clip_range': self.RATIO_CLIP_RANGE,
            'lr_scheduler_factor': self.LR_SCHEDULER_FACTOR,
            'lr_scheduler_patience': self.LR_SCHEDULER_PATIENCE,
            'early_stopping_patience': self.EARLY_STOPPING_PATIENCE,
            'min_memory_size': self.MIN_MEMORY_SIZE,
            'update_frequency': self.UPDATE_FREQUENCY,
            'huber_delta': self.HUBER_DELTA,
            'convergence_threshold': self.CONVERGENCE_THRESHOLD
        }
    
    def print_optimization_summary(self):
        """打印优化总结"""
        print("=" * 60)
        print("PPO超参数优化总结")
        print("=" * 60)
        print("主要优化方向：")
        print("1. 降低学习率：3e-5 (提高训练稳定性)")
        print("2. 简化网络结构：128->64 (减少过拟合)")
        print("3. 增加正则化：Dropout 0.1, Weight Decay 1e-4")
        print("4. 优化奖励处理：平滑窗口100, 裁剪范围±5.0")
        print("5. 改进训练策略：6 epochs, 64 batch size")
        print("6. 增强稳定性：梯度裁剪0.5, 早停耐心300")
        print("=" * 60)
        print("预期效果：")
        print("- 奖励曲线更平滑")
        print("- 训练过程更稳定")
        print("- 收敛速度适中")
        print("- 泛化能力增强")
        print("=" * 60)

class PPOOptimizationTips:
    """PPO优化建议类"""
    
    @staticmethod
    def get_training_tips():
        """获取训练建议"""
        tips = [
            "1. 如果奖励仍然不稳定，可以进一步降低学习率到1e-5",
            "2. 如果收敛太慢，可以增加学习率到5e-5",
            "3. 如果过拟合严重，可以增加Dropout到0.15",
            "4. 如果训练不稳定，可以减少batch_size到32",
            "5. 如果探索不足，可以增加entropy_coef到0.03",
            "6. 如果价值估计不准确，可以增加value_loss_coef到0.8",
            "7. 如果梯度爆炸，可以减少clip_grad_norm到0.3",
            "8. 如果奖励范围过大，可以调整reward_clip_range"
        ]
        return tips
    
    @staticmethod
    def get_monitoring_metrics():
        """获取监控指标"""
        metrics = [
            "训练损失 (Policy Loss, Value Loss, Entropy)",
            "奖励统计 (平均奖励, 奖励标准差, 平滑奖励)",
            "梯度统计 (梯度范数, 梯度裁剪次数)",
            "网络统计 (权重范数, 激活值分布)",
            "收敛指标 (奖励趋势, 损失下降率)"
        ]
        return metrics

# 创建配置实例
ppo_config = PPOConfig()

if __name__ == "__main__":
    # 打印优化总结
    ppo_config.print_optimization_summary()
    
    # 打印训练建议
    print("\n训练建议：")
    for tip in PPOOptimizationTips.get_training_tips():
        print(tip)
    
    # 打印监控指标
    print("\n监控指标：")
    for metric in PPOOptimizationTips.get_monitoring_metrics():
        print(f"- {metric}")
