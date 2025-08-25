import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd

class ConfidenceIntervalPlotter:
    def __init__(self, style='seaborn-v0_8'):
        """初始化绘图器"""
        plt.style.use(style)
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    def plot_with_confidence_interval(self, data_dict, x_label='Steps', y_label='Value', 
                                    title='Confidence Interval Plot', confidence_level=0.95,
                                    save_path=None, show_plot=True):
        """
        绘制带置信区间的折线图
        
        Args:
            data_dict: 字典，格式为 {'label1': [data1], 'label2': [data2], ...}
                      每个data是一个列表的列表，包含多次实验的数据
            x_label: x轴标签
            y_label: y轴标签
            title: 图表标题
            confidence_level: 置信水平 (0.95表示95%置信区间)
            save_path: 保存路径
            show_plot: 是否显示图表
        """
        plt.figure(figsize=(12, 8))
        
        for i, (label, data_list) in enumerate(data_dict.items()):
            # 确保数据是numpy数组
            data_array = np.array(data_list)
            
            # 计算每个时间点的统计量
            mean_values = np.mean(data_array, axis=0)
            std_values = np.std(data_array, axis=0)
            
            # 计算置信区间
            n_experiments = data_array.shape[0]
            t_value = stats.t.ppf((1 + confidence_level) / 2, n_experiments - 1)
            margin_of_error = t_value * std_values / np.sqrt(n_experiments)
            
            # 创建x轴数据
            x = np.arange(len(mean_values))
            
            # 绘制均值线
            plt.plot(x, mean_values, label=label, color=self.colors[i % len(self.colors)], 
                    linewidth=2, marker='o', markersize=4)
            
            # 绘制置信区间
            plt.fill_between(x, mean_values - margin_of_error, mean_values + margin_of_error,
                           alpha=0.3, color=self.colors[i % len(self.colors)])
        
        plt.xlabel(x_label, fontsize=12)
        plt.ylabel(y_label, fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_training_metrics_with_ci(self, training_data, save_path=None):
        """
        专门用于绘制训练指标的置信区间图
        
        Args:
            training_data: 包含训练数据的字典
            save_path: 保存路径
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Training Metrics with Confidence Intervals', fontsize=16, fontweight='bold')
        
        # 1. 奖励变化
        if 'rewards' in training_data:
            self._plot_single_metric(axes[0, 0], training_data['rewards'], 
                                   'Training Steps', 'Average Reward', 'Reward Progression')
        
        # 2. MAE变化
        if 'mae_values' in training_data:
            self._plot_single_metric(axes[0, 1], training_data['mae_values'], 
                                   'Tasks', 'MAE', 'Task MAE Progression')
        
        # 3. 能力误差变化
        if 'ability_errors' in training_data:
            self._plot_single_metric(axes[1, 0], training_data['ability_errors'], 
                                   'Tasks', 'Ability Error', 'Worker Ability Error Progression')
        
        # 4. GSQ使用量变化
        if 'gsq_usage' in training_data:
            self._plot_single_metric(axes[1, 1], training_data['gsq_usage'], 
                                   'Tasks', 'GSQ Count', 'GSQ Usage Progression')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _plot_single_metric(self, ax, data_list, x_label, y_label, title):
        """绘制单个指标的置信区间"""
        data_array = np.array(data_list)
        mean_values = np.mean(data_array, axis=0)
        std_values = np.std(data_array, axis=0)
        
        # 计算置信区间
        n_experiments = data_array.shape[0]
        t_value = stats.t.ppf(0.975, n_experiments - 1)  # 95%置信区间
        margin_of_error = t_value * std_values / np.sqrt(n_experiments)
        
        x = np.arange(len(mean_values))
        
        ax.plot(x, mean_values, color='#1f77b4', linewidth=2, marker='o', markersize=3)
        ax.fill_between(x, mean_values - margin_of_error, mean_values + margin_of_error,
                       alpha=0.3, color='#1f77b4')
        
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    
    def plot_strategy_comparison_with_ci(self, strategy_results, save_path=None):
        """
        绘制策略比较的置信区间图
        
        Args:
            strategy_results: 策略结果字典
            save_path: 保存路径
        """
        metrics = ['avg_mae', 'avg_gsq', 'avg_ability_error']
        metric_names = ['Average MAE', 'Average GSQ Count', 'Average Ability Error']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Strategy Comparison with Confidence Intervals', fontsize=16, fontweight='bold')
        
        for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            ax = axes[i]
            
            # 提取数据
            labels = []
            data_lists = []
            
            for strategy, results in strategy_results.items():
                if metric in results:
                    labels.append(strategy)
                    data_lists.append(results[metric])
            
            if data_lists:
                # 计算置信区间
                data_array = np.array(data_lists)
                mean_values = np.mean(data_array, axis=0)
                std_values = np.std(data_array, axis=0)
                
                n_strategies = data_array.shape[0]
                t_value = stats.t.ppf(0.975, n_strategies - 1)
                margin_of_error = t_value * std_values / np.sqrt(n_strategies)
                
                x = np.arange(len(labels))
                
                bars = ax.bar(x, mean_values, yerr=margin_of_error, capsize=5, 
                             color=self.colors[:len(labels)], alpha=0.7)
                
                ax.set_xlabel('Strategy')
                ax.set_ylabel(metric_name)
                ax.set_title(f'{metric_name} by Strategy')
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=45)
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

 