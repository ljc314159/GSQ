#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练规模配置测试脚本
用于快速验证新的训练规模配置是否正常工作
"""

import time
import sys
import traceback
from config import SystemConfig
from training_config import get_optimal_config, training_config, scale_test_config

def test_config_loading():
    """测试配置加载"""
    print("=" * 50)
    print("测试配置加载...")
    
    try:
        # 测试基础配置
        config = SystemConfig()
        print(f"✓ 基础配置加载成功")
        print(f"  默认工人数量: {config.DEFAULT_WORKER_COUNT}")
        print(f"  默认任务数量: {config.DEFAULT_TASK_COUNT}")
        print(f"  默认实验运行次数: {config.DEFAULT_EXPERIMENT_RUNS}")
        
        # 测试训练规模配置
        print(f"\n✓ 训练规模配置加载成功")
        print(f"  小规模工人数: {training_config.BASE_WORKER_COUNT}")
        print(f"  中规模工人数: {training_config.MEDIUM_WORKER_COUNT}")
        print(f"  大规模工人数: {training_config.LARGE_WORKER_COUNT}")
        print(f"  超大规模工人数: {training_config.EXTRA_LARGE_WORKER_COUNT}")
        
        return True
        
    except Exception as e:
        print(f"✗ 配置加载失败: {e}")
        traceback.print_exc()
        return False

def test_optimal_configs():
    """测试最优配置获取"""
    print("\n" + "=" * 50)
    print("测试最优配置获取...")
    
    try:
        scale_levels = ['small', 'medium', 'large', 'extra_large']
        
        for level in scale_levels:
            config = get_optimal_config(level)
            print(f"✓ {level} 配置:")
            print(f"  工人数量: {config['workers']}")
            print(f"  任务数量: {config['tasks']}")
            print(f"  运行次数: {config['runs']}")
            print(f"  RL配置: {config['rl_config']}")
        
        return True
        
    except Exception as e:
        print(f"✗ 最优配置获取失败: {e}")
        traceback.print_exc()
        return False

def test_scale_test_config():
    """测试规模测试配置"""
    print("\n" + "=" * 50)
    print("测试规模测试配置...")
    
    try:
        print(f"✓ 测试配置数量: {len(scale_test_config.TEST_CONFIGS)}")
        
        for config in scale_test_config.TEST_CONFIGS:
            print(f"  {config['name']}: {config['workers']}工人, {config['tasks']}任务, {config['runs']}次运行")
            print(f"    描述: {config['description']}")
        
        print(f"\n✓ 评估指标数量: {len(scale_test_config.EVALUATION_METRICS)}")
        print(f"  指标: {', '.join(scale_test_config.EVALUATION_METRICS)}")
        
        print(f"\n✓ 性能基准:")
        for metric, value in scale_test_config.PERFORMANCE_BENCHMARKS.items():
            print(f"  {metric}: {value}")
        
        return True
        
    except Exception as e:
        print(f"✗ 规模测试配置失败: {e}")
        traceback.print_exc()
        return False

def test_optimization_config():
    """测试优化配置"""
    print("\n" + "=" * 50)
    print("测试优化配置...")
    
    try:
        print(f"✓ 网络架构配置:")
        for arch_name, arch_config in optimization_config.NETWORK_ARCHITECTURES.items():
            print(f"  {arch_name}: {arch_config}")
        
        print(f"\n✓ 学习率调度配置:")
        for lr_name, lr_config in optimization_config.LEARNING_RATE_SCHEDULES.items():
            print(f"  {lr_name}: {lr_config}")
        
        print(f"\n✓ 正则化配置:")
        for reg_name, reg_value in optimization_config.REGULARIZATION.items():
            print(f"  {reg_name}: {reg_value}")
        
        print(f"\n✓ 训练策略配置:")
        for strategy_name, strategy_config in optimization_config.TRAINING_STRATEGIES.items():
            print(f"  {strategy_name}: {strategy_config}")
        
        return True
        
    except Exception as e:
        print(f"✗ 优化配置失败: {e}")
        traceback.print_exc()
        return False

def performance_benchmark():
    """性能基准测试"""
    print("\n" + "=" * 50)
    print("性能基准测试...")
    
    try:
        import numpy as np
        
        # 测试数组操作性能
        start_time = time.time()
        large_array = np.random.rand(1000, 1000)
        result = np.dot(large_array, large_array.T)
        array_time = time.time() - start_time
        print(f"✓ 1000x1000矩阵乘法: {array_time:.4f}秒")
        
        # 测试内存使用
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        print(f"✓ 当前内存使用: {memory_info.rss / 1024 / 1024:.2f} MB")
        
        return True
        
    except ImportError as e:
        print(f"⚠ 跳过性能测试: {e}")
        return True
    except Exception as e:
        print(f"✗ 性能测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("动态GSQ系统 - 训练规模配置测试")
    print("=" * 60)
    
    tests = [
        ("配置加载测试", test_config_loading),
        ("最优配置测试", test_optimal_configs),
        ("规模测试配置", test_scale_test_config),
        ("优化配置测试", test_optimization_config),
        ("性能基准测试", performance_benchmark)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name} 通过")
            else:
                print(f"✗ {test_name} 失败")
        except Exception as e:
            print(f"✗ {test_name} 异常: {e}")
    
    print("\n" + "=" * 60)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！训练规模配置正常工作。")
        print("\n下一步:")
        print("1. 运行 python main.py 开始完整实验")
        print("2. 或运行 python -c \"from experiment import ExperimentRunner; runner = ExperimentRunner(); runner.test_training_scale()\" 进行规模测试")
        return 0
    else:
        print("❌ 部分测试失败，请检查配置。")
        return 1

if __name__ == "__main__":
    sys.exit(main())
