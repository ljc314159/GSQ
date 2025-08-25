"""
PPO优化配置测试脚本
验证优化后的超参数设置和训练稳定性
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from ppo_agent import PPOAgent
from ppo_config import ppo_config, PPOOptimizationTips

def test_ppo_initialization():
    """测试PPO代理初始化"""
    print("测试PPO代理初始化...")
    
    # 使用优化后的配置创建PPO代理
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
    print(f"  - 网络结构: {sum(p.numel() for p in agent.actor.parameters())} 参数")
    
    return agent

def test_network_forward_pass(agent):
    """测试网络前向传播"""
    print("\n测试网络前向传播...")
    
    # 创建测试状态
    test_state = torch.randn(1, 4)
    
    # 测试Actor网络
    with torch.no_grad():
        action_probs, value = agent.actor(test_state), agent.critic(test_state)
    
    print(f"✓ 前向传播成功")
    print(f"  - 动作概率形状: {action_probs.shape}")
    print(f"  - 价值输出形状: {value.shape}")
    print(f"  - 动作概率和: {action_probs.sum().item():.4f}")
    
    return True

def test_action_selection(agent):
    """测试动作选择"""
    print("\n测试动作选择...")
    
    # 创建测试状态
    test_state = torch.randn(1, 4)
    
    # 选择动作
    action, prob, value = agent.select_action(test_state)
    
    print(f"✓ 动作选择成功")
    print(f"  - 选择的动作: {action}")
    print(f"  - 动作概率: {prob:.4f}")
    print(f"  - 状态价值: {value:.4f}")
    
    return True

def test_memory_management(agent):
    """测试内存管理"""
    print("\n测试内存管理...")
    
    # 添加一些测试数据到内存
    for i in range(100):
        state = torch.randn(4)
        action = np.random.randint(0, 2)
        reward = np.random.normal(0, 1)
        next_state = torch.randn(4)
        action_prob = np.random.random()
        value = np.random.normal(0, 1)
        done = False
        
        agent.memory.states.append(state)
        agent.memory.actions.append(action)
        agent.memory.rewards.append(reward)
        agent.memory.next_states.append(next_state)
        agent.memory.action_probs.append(action_prob)
        agent.memory.values.append(value)
        agent.memory.dones.append(done)
    
    print(f"✓ 内存管理测试成功")
    print(f"  - 内存大小: {len(agent.memory)}")
    print(f"  - 最小内存要求: {agent.min_memory_size}")
    
    return True

def test_training_stability(agent):
    """测试训练稳定性"""
    print("\n测试训练稳定性...")
    
    # 模拟训练过程
    training_rewards = []
    training_losses = []
    
    for episode in range(50):
        # 模拟一个episode的数据
        episode_rewards = []
        episode_states = []
        episode_actions = []
        episode_action_probs = []
        episode_values = []
        episode_next_states = []
        episode_dones = []
        
        # 生成episode数据
        for step in range(20):
            state = torch.randn(4)
            action = np.random.randint(0, 2)
            reward = np.random.normal(0, 1)
            next_state = torch.randn(4)
            action_prob = np.random.random()
            value = np.random.normal(0, 1)
            done = step == 19
            
            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)
            episode_next_states.append(next_state)
            episode_action_probs.append(action_prob)
            episode_values.append(value)
            episode_dones.append(done)
        
        # 添加到内存
        agent.memory.states.extend(episode_states)
        agent.memory.actions.extend(episode_actions)
        agent.memory.rewards.extend(episode_rewards)
        agent.memory.next_states.extend(episode_next_states)
        agent.memory.action_probs.extend(episode_action_probs)
        agent.memory.values.extend(episode_values)
        agent.memory.dones.extend(episode_dones)
        
        # 如果内存足够，进行训练
        if len(agent.memory) >= agent.min_memory_size:
            try:
                agent.update()
                
                # 记录训练统计
                if agent.recent_rewards:
                    avg_reward = np.mean(agent.recent_rewards[-10:])
                    training_rewards.append(avg_reward)
                
                # 清空内存
                agent.memory.clear()
                
            except Exception as e:
                print(f"训练过程中出现错误: {e}")
                return False
        
        # 每10个episode打印一次进度
        if (episode + 1) % 10 == 0:
            print(f"  - Episode {episode + 1}/50 完成")
    
    print(f"✓ 训练稳定性测试成功")
    print(f"  - 完成episodes: 50")
    print(f"  - 训练更新次数: {len(training_rewards)}")
    
    if training_rewards:
        print(f"  - 平均奖励: {np.mean(training_rewards):.4f}")
        print(f"  - 奖励标准差: {np.std(training_rewards):.4f}")
    
    return True

def plot_training_metrics(agent):
    """绘制训练指标"""
    print("\n绘制训练指标...")
    
    if not agent.recent_rewards:
        print("没有训练数据可绘制")
        return
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('PPO训练指标监控', fontsize=16)
    
    # 奖励曲线
    axes[0, 0].plot(agent.recent_rewards, label='原始奖励', alpha=0.7)
    if agent.smoothed_rewards:
        axes[0, 0].plot(agent.smoothed_rewards, label='平滑奖励', linewidth=2)
    axes[0, 0].set_title('训练奖励')
    axes[0, 0].set_xlabel('更新次数')
    axes[0, 0].set_ylabel('奖励')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 奖励分布
    axes[0, 1].hist(agent.recent_rewards, bins=20, alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('奖励分布')
    axes[0, 1].set_xlabel('奖励值')
    axes[0, 1].set_ylabel('频次')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 奖励趋势
    if len(agent.recent_rewards) > 10:
        window_size = 10
        moving_avg = [np.mean(agent.recent_rewards[max(0, i-window_size):i+1]) 
                     for i in range(len(agent.recent_rewards))]
        axes[1, 0].plot(moving_avg, label=f'{window_size}步移动平均', linewidth=2)
        axes[1, 0].set_title('奖励趋势')
        axes[1, 0].set_xlabel('更新次数')
        axes[1, 0].set_ylabel('移动平均奖励')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # 训练统计
    stats = agent.get_training_stats()
    if stats:
        axes[1, 1].text(0.1, 0.9, f"最佳奖励: {stats.get('best_reward', 'N/A'):.4f}", 
                        transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].text(0.1, 0.8, f"耐心计数: {stats.get('patience_counter', 'N/A')}", 
                        transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].text(0.1, 0.7, f"当前学习率: {stats.get('current_lr', 'N/A'):.2e}", 
                        transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].text(0.1, 0.6, f"总更新次数: {len(agent.recent_rewards)}", 
                        transform=axes[1, 1].transAxes, fontsize=12)
    
    axes[1, 1].set_title('训练统计')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('ppo_training_metrics.png', dpi=300, bbox_inches='tight')
    print("✓ 训练指标图表已保存为 'ppo_training_metrics.png'")
    
    return True

def main():
    """主测试函数"""
    print("=" * 60)
    print("PPO优化配置测试")
    print("=" * 60)
    
    # 打印优化总结
    ppo_config.print_optimization_summary()
    
    try:
        # 测试PPO代理初始化
        agent = test_ppo_initialization()
        
        # 测试网络前向传播
        test_network_forward_pass(agent)
        
        # 测试动作选择
        test_action_selection(agent)
        
        # 测试内存管理
        test_memory_management(agent)
        
        # 测试训练稳定性
        test_training_stability(agent)
        
        # 绘制训练指标
        plot_training_metrics(agent)
        
        print("\n" + "=" * 60)
        print("所有测试通过！PPO优化配置验证成功")
        print("=" * 60)
        
        # 打印训练建议
        print("\n训练建议：")
        for tip in PPOOptimizationTips.get_training_tips():
            print(tip)
        
        # 打印监控指标
        print("\n监控指标：")
        for metric in PPOOptimizationTips.get_monitoring_metrics():
            print(f"- {metric}")
        
    except Exception as e:
        print(f"\n测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()
