# 动态GSQ系统 - 训练规模配置指南

## 概述

本文档详细说明了如何配置和使用动态GSQ系统的训练规模参数，以支持从小规模到超大规模的训练需求。

## 主要改进

### 1. 训练规模大幅提升

- **工人数量**: 从200增加到500（默认），支持最大2000
- **任务数量**: 从500增加到1000（默认），支持最大5000
- **实验运行次数**: 从30增加到50（默认），支持最大200
- **RL训练参数**: 批次大小从32增加到64，训练轮数从2增加到4

### 2. 新增配置系统

- `training_config.py`: 专门的训练规模配置文件
- `config.py`: 增强的系统配置参数
- 支持多种规模级别的自动配置

### 3. 性能优化

- 网络结构优化：增加隐藏层大小，添加Dropout
- 训练稳定性提升：降低学习率，增加权重衰减
- 内存管理优化：增大内存容量，优化批处理

## 配置参数详解

### 基础配置 (config.py)

```python
class SystemConfig:
    # 工人和任务数量
    DEFAULT_WORKER_COUNT = 500      # 默认工人数量
    DEFAULT_TASK_COUNT = 1000       # 默认任务数量
    DEFAULT_EXPERIMENT_RUNS = 50    # 默认实验运行次数
    
    # 系统参数优化
    MAX_CONCURRENT_TASKS = 3        # 每人最多同时任务数
    GSQ_TASK_BATCH_SIZE = 5         # 每次分配任务给几个工人
    CONVERGENCE_THRESHOLD = 0.015   # 收敛阈值
    WORKER_ABILITY_WINDOW = 5       # 能力波动计算窗口
```

### 训练规模配置 (training_config.py)

```python
class TrainingScaleConfig:
    # 不同规模级别
    BASE_WORKER_COUNT = 200         # 小规模
    MEDIUM_WORKER_COUNT = 500       # 中规模
    LARGE_WORKER_COUNT = 1000       # 大规模
    EXTRA_LARGE_WORKER_COUNT = 2000 # 超大规模
    
    # RL训练参数
    RL_LEARNING_RATE = 3e-5         # 学习率
    RL_BATCH_SIZE = 64              # 批次大小
    RL_EPOCHS = 4                   # 训练轮数
    RL_MEMORY_SIZE = 64             # 内存大小
```

## 使用方法

### 1. 基础使用

```python
from experiment import ExperimentRunner

# 使用默认配置（500工人，1000任务，50次运行）
experiment_runner = ExperimentRunner()
results = experiment_runner.evaluate_gsq_strategies()
```

### 2. 自定义规模

```python
# 自定义工人和任务数量
results = experiment_runner.run_experiment(
    n_workers=1000,    # 1000个工人
    n_tasks=2000,      # 2000个任务
    gsq_strategy='rl'
)
```

### 3. 规模测试

```python
# 运行完整的规模测试
scale_results = experiment_runner.test_training_scale()
```

### 4. 使用预定义配置

```python
from training_config import get_optimal_config

# 获取大规模配置
large_config = get_optimal_config('large')
print(f"工人数量: {large_config['workers']}")
print(f"任务数量: {large_config['tasks']}")
print(f"运行次数: {large_config['runs']}")
```

## 规模级别说明

### 小规模 (Small)
- **工人数量**: 200
- **任务数量**: 300
- **运行次数**: 20
- **用途**: 基础功能验证，快速测试

### 中规模 (Medium)
- **工人数量**: 500
- **任务数量**: 1000
- **运行次数**: 50
- **用途**: 性能基准测试，推荐配置

### 大规模 (Large)
- **工人数量**: 1000
- **任务数量**: 2000
- **运行次数**: 100
- **用途**: 压力测试，性能评估

### 超大规模 (Extra Large)
- **工人数量**: 2000
- **任务数量**: 5000
- **运行次数**: 200
- **用途**: 极限测试，生产环境验证

## 性能优化建议

### 1. 内存管理

- 监控内存使用率，保持在80%以下
- 使用批处理减少内存峰值
- 定期清理不需要的数据

### 2. 训练稳定性

- 使用较低的学习率（3e-5）
- 增加训练轮数（4轮）
- 添加Dropout防止过拟合

### 3. 收敛监控

- 设置合理的收敛阈值（0.015）
- 监控奖励变化趋势
- 使用早停机制避免过拟合

## 运行示例

### 完整实验流程

```bash
# 运行主程序
python main.py

# 输出示例:
# ============================================================
# 动态GSQ系统 - 大规模训练实验
# ============================================================
# 默认配置:
#   工人数量: 500
#   任务数量: 1000
#   实验运行次数: 50
#   RL训练episode数: 100
# ============================================================
```

### 单独运行规模测试

```python
from experiment import ExperimentRunner

runner = ExperimentRunner()
scale_results = runner.test_training_scale()

# 查看结果
for scale_name, result in scale_results.items():
    print(f"{scale_name}: MAE={result['avg_mae']:.4f}")
```

## 注意事项

### 1. 硬件要求

- **内存**: 建议8GB以上
- **CPU**: 多核处理器，支持并行计算
- **存储**: 足够的磁盘空间存储结果

### 2. 时间成本

- 小规模: 约10-30分钟
- 中规模: 约1-3小时
- 大规模: 约3-8小时
- 超大规模: 约8-24小时

### 3. 结果分析

- 关注MAE、GSQ使用量、工人能力误差等关键指标
- 比较不同规模下的性能表现
- 分析训练收敛性和稳定性

## 故障排除

### 常见问题

1. **内存不足**: 减少批次大小或工人数量
2. **训练不收敛**: 降低学习率，增加训练轮数
3. **性能下降**: 检查网络结构，调整正则化参数

### 调试建议

- 启用详细日志输出
- 监控关键指标变化
- 使用小规模配置进行快速验证

## 更新日志

- **v2.0**: 大幅提升训练规模，新增配置系统
- **v1.5**: 优化RL训练参数，提升稳定性
- **v1.0**: 基础功能实现

## 联系方式

如有问题或建议，请通过以下方式联系：
- 项目仓库: [GitHub链接]
- 问题反馈: [Issues链接]
- 文档更新: [Wiki链接]
