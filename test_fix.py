"""
测试PPO代理修复的简单脚本
"""

import torch
import numpy as np
from ppo_agent import PPOAgent

def test_ppo_fix():
    """测试PPO代理修复"""
    print("测试PPO代理修复...")
    
    try:
        # 创建PPO代理
        agent = PPOAgent()
        print("✓ PPO代理创建成功")
        
        # 测试GAE计算
        print("测试GAE计算...")
        rewards = torch.randn(10)
        values = torch.randn(10)
        next_states = torch.randn(10, 4)
        dones = torch.zeros(10)
        
        advantages = agent.compute_gae(rewards, values, next_states, dones)
        print(f"✓ GAE计算成功，优势值形状: {advantages.shape}")
        
        # 测试内存存储
        print("测试内存存储...")
        for i in range(20):
            state = torch.randn(4)
            action = np.random.randint(0, 2)
            reward = np.random.normal(0, 1)
            next_state = torch.randn(4)
            action_prob = np.random.random()
            value = np.random.normal(0, 1)
            done = False
            
            agent.store_transition(state, action, reward, next_state, action_prob, value, done)
        
        print(f"✓ 内存存储成功，内存大小: {len(agent.memory)}")
        
        # 测试训练更新（如果内存足够）
        if len(agent.memory) >= agent.min_memory_size:
            print("测试训练更新...")
            agent.update()
            print("✓ 训练更新成功")
        
        print("\n所有测试通过！PPO代理修复成功！")
        return True
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_ppo_fix()
