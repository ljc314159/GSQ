"""
PPO优化配置测试脚本
"""

import torch
import numpy as np
from ppo_agent import PPOAgent
from ppo_config import ppo_config

def test_ppo_optimization():
    """测试PPO优化配置"""
    print("测试PPO优化配置...")
    
    # 创建优化后的PPO代理
    agent = PPOAgent(
        state_dim=4,
        action_dim=2,
        lr=ppo_config.LEARNING_RATE,
        gamma=ppo_config.GAMMA,
        epsilon=ppo_config.EPSILON,
        epochs=ppo_config.EPOCHS,
        batch_size=ppo_config.BATCH_SIZE,
        gae_lambda=ppo_config.GAE_LAMBDA
    )
    
    print(f"✓ PPO代理创建成功")
    print(f"  - 学习率: {agent.lr}")
    print(f"  - 裁剪参数: {agent.epsilon}")
    print(f"  - 训练轮数: {agent.epochs}")
    print(f"  - 批次大小: {agent.batch_size}")
    
    # 测试网络前向传播
    test_state = torch.randn(1, 4)
    with torch.no_grad():
        action_probs, value = agent.actor(test_state), agent.critic(test_state)
    
    print(f"✓ 网络前向传播成功")
    print(f"  - 动作概率形状: {action_probs.shape}")
    print(f"  - 价值输出形状: {value.shape}")
    
    # 测试动作选择
    action, prob, value = agent.select_action(test_state)
    print(f"✓ 动作选择成功")
    print(f"  - 选择的动作: {action}")
    print(f"  - 动作概率: {prob:.4f}")
    
    print("\nPPO优化配置测试完成！")
    return True

if __name__ == "__main__":
    test_ppo_optimization()
