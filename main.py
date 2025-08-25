from experiment import ExperimentRunner
from config import SystemConfig

def main():
    # 初始化实验运行器
    experiment_runner = ExperimentRunner()
    config = SystemConfig()
    
    print("=" * 60)
    print("动态GSQ系统 - 大规模训练实验")
    print("=" * 60)
    print(f"默认配置:")
    print(f"  工人数量: {config.DEFAULT_WORKER_COUNT}")
    print(f"  任务数量: {config.DEFAULT_TASK_COUNT}")
    print(f"  实验运行次数: {config.DEFAULT_EXPERIMENT_RUNS}")
    print(f"  RL训练episode数: {config.RL_TRAINING_EPISODES}")
    print("=" * 60)

    # 实验1：比较不同GSQ策略（大规模）
    print("\n运行GSQ策略评估（大规模）...")
    gsq_results = experiment_runner.evaluate_gsq_strategies()
    print("\nGSQ策略结果:")
    for strategy, res in gsq_results.items():
        print(f"{strategy}: MAE={res['avg_mae']:.4f}, GSQ={res['avg_gsq']}, Ability Error={res['avg_ability_error']:.4f}")

    # 实验2：评估抗恶意工人能力（大规模）
    print("\n运行抗恶意工人能力评估（大规模）...")
    malicious_results = experiment_runner.evaluate_malicious_resistance()
    print("\n抗恶意能力结果:")
    print(malicious_results)

    # 实验3：训练规模测试
    print("\n运行训练规模测试...")
    scale_test_results = experiment_runner.test_training_scale()
    print("\n训练规模测试结果:")
    print(scale_test_results)

    print("\n所有实验完成！结果已保存到PNG文件。")
    print("=" * 60)

if __name__ == "__main__":
    main()