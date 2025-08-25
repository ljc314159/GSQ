import numpy as np
from config import SystemConfig

class AnswerAggregator:
    @staticmethod
    def aggregate_answers(task):
        """IWLS算法聚合答案"""
        measurements = []
        initial_weights = []
        worker_ids = []
        
        # 收集测量值和初始权重
        for wid, (measurement, ability) in task.worker_answers.items():
            measurements.append(measurement)
            initial_weights.append(ability)
            worker_ids.append(wid)
        
        # 初始归一化权重
        weights = np.array(initial_weights)
        weights_sum = weights.sum()
        if weights_sum == 0:
            weights = np.ones(len(weights)) / len(weights)  # 避免全零
        else:
            weights /= weights_sum
        
        # 初始聚合值
        aggregated_value = np.dot(measurements, weights)
        
        # 迭代过程
        prev_weights = weights.copy()
        converged = False
        iteration = 0
        max_iterations = 100
        config = SystemConfig()
        
        while not converged and iteration < max_iterations:
            # 计算误差
            errors = [abs(m - aggregated_value) for m in measurements]
            
            # 更新权重 (误差小权重高)
            new_weights = 1.0 / (np.array(errors) + 1e-6)  # 避免除以零
            
            # 归一化
            new_weights_sum = new_weights.sum()
            if new_weights_sum == 0:
                new_weights = np.ones(len(new_weights)) / len(new_weights)
            else:
                new_weights /= new_weights_sum
            
            # 更新聚合值
            new_aggregated_value = np.dot(measurements, new_weights)
            
            # 检查收敛: 权重变化率
            weight_changes = np.abs(new_weights - prev_weights) / (prev_weights + 1e-6)  # 避免除以零
            max_change = np.max(weight_changes)
            
            if max_change < config.CONVERGENCE_THRESHOLD:
                converged = True
            else:
                prev_weights = new_weights
                weights = new_weights
                aggregated_value = new_aggregated_value
                iteration += 1
        
        # 保存最终结果
        task.aggregated_value = aggregated_value
        task.final_weights = dict(zip(worker_ids, weights))
        return aggregated_value, weights
    
    @staticmethod
    def aggregate_answers_weighted_average(task):
        """简单加权平均聚合"""
        measurements = []
        weights = []
        worker_ids = []
        
        for wid, (measurement, ability) in task.worker_answers.items():
            measurements.append(measurement)
            weights.append(ability)
            worker_ids.append(wid)
        
        weights = np.array(weights)
        weights_sum = weights.sum()
        if weights_sum == 0:
            weights = np.ones(len(weights)) / len(weights)
        else:
            weights /= weights_sum
        
        aggregated_value = np.dot(measurements, weights)
        
        task.aggregated_value = aggregated_value
        task.final_weights = dict(zip(worker_ids, weights))
        return aggregated_value, weights
    
    @staticmethod
    def aggregate_answers_median(task):
        """中位数聚合（对异常值鲁棒）"""
        measurements = []
        worker_ids = []
        
        for wid, (measurement, ability) in task.worker_answers.items():
            measurements.append(measurement)
            worker_ids.append(wid)
        
        aggregated_value = np.median(measurements)
        
        # 计算每个工人答案与中位数的接近程度作为权重
        distances = [abs(m - aggregated_value) for m in measurements]
        max_distance = max(distances) if distances else 1.0
        
        # 距离越小权重越大
        weights = [(max_distance - d) / max_distance for d in distances]
        weights = np.array(weights)
        weights_sum = weights.sum()
        if weights_sum > 0:
            weights /= weights_sum
        else:
            weights = np.ones(len(weights)) / len(weights)
        
        task.aggregated_value = aggregated_value
        task.final_weights = dict(zip(worker_ids, weights))
        return aggregated_value, weights
    
    @staticmethod
    def aggregate_answers_robust(task, outlier_threshold=2.0):
        """鲁棒聚合（去除异常值）"""
        measurements = []
        abilities = []
        worker_ids = []
        
        for wid, (measurement, ability) in task.worker_answers.items():
            measurements.append(measurement)
            abilities.append(ability)
            worker_ids.append(wid)
        
        measurements = np.array(measurements)
        abilities = np.array(abilities)
        
        # 计算中位数和MAD（中位数绝对偏差）
        median_val = np.median(measurements)
        mad = np.median(np.abs(measurements - median_val))
        
        # 识别异常值
        outlier_mask = np.abs(measurements - median_val) <= outlier_threshold * mad
        
        # 只使用非异常值进行聚合
        valid_measurements = measurements[outlier_mask]
        valid_abilities = abilities[outlier_mask]
        valid_worker_ids = [worker_ids[i] for i in range(len(worker_ids)) if outlier_mask[i]]
        
        if len(valid_measurements) == 0:
            # 如果没有有效值，使用所有值
            valid_measurements = measurements
            valid_abilities = abilities
            valid_worker_ids = worker_ids
        
        # 加权平均
        weights = valid_abilities / valid_abilities.sum() if valid_abilities.sum() > 0 else \
                 np.ones(len(valid_abilities)) / len(valid_abilities)
        
        aggregated_value = np.dot(valid_measurements, weights)
        
        # 为所有工人分配权重（异常值权重为0）
        final_weights = np.zeros(len(worker_ids))
        for i, wid in enumerate(worker_ids):
            if wid in valid_worker_ids:
                final_weights[i] = weights[valid_worker_ids.index(wid)]
        
        task.aggregated_value = aggregated_value
        task.final_weights = dict(zip(worker_ids, final_weights))
        return aggregated_value, final_weights

class ConsensusAggregator:
    """共识聚合器"""
    
    @staticmethod
    def aggregate_with_consensus(task, consensus_threshold=0.8):
        """基于共识的聚合"""
        measurements = []
        abilities = []
        worker_ids = []
        
        for wid, (measurement, ability) in task.worker_answers.items():
            measurements.append(measurement)
            abilities.append(ability)
            worker_ids.append(wid)
        
        measurements = np.array(measurements)
        abilities = np.array(abilities)
        
        # 计算所有答案对之间的相似度
        n_workers = len(measurements)
        consensus_scores = np.zeros(n_workers)
        
        for i in range(n_workers):
            similarities = []
            for j in range(n_workers):
                if i != j:
                    # 计算答案相似度（基于距离）
                    distance = abs(measurements[i] - measurements[j])
                    similarity = 1.0 / (1.0 + distance)
                    similarities.append(similarity)
            
            # 平均相似度作为共识分数
            consensus_scores[i] = np.mean(similarities) if similarities else 0
        
        # 选择高共识的工人
        high_consensus_mask = consensus_scores >= consensus_threshold
        
        if np.sum(high_consensus_mask) == 0:
            # 如果没有高共识工人，降低阈值
            high_consensus_mask = consensus_scores >= np.median(consensus_scores)
        
        # 使用高共识工人的答案进行聚合
        valid_measurements = measurements[high_consensus_mask]
        valid_abilities = abilities[high_consensus_mask]
        valid_worker_ids = [worker_ids[i] for i in range(len(worker_ids)) if high_consensus_mask[i]]
        
        # 加权平均
        weights = valid_abilities / valid_abilities.sum() if valid_abilities.sum() > 0 else \
                 np.ones(len(valid_abilities)) / len(valid_abilities)
        
        aggregated_value = np.dot(valid_measurements, weights)
        
        # 为所有工人分配权重
        final_weights = np.zeros(len(worker_ids))
        for i, wid in enumerate(worker_ids):
            if wid in valid_worker_ids:
                final_weights[i] = weights[valid_worker_ids.index(wid)]
        
        task.aggregated_value = aggregated_value
        task.final_weights = dict(zip(worker_ids, final_weights))
        return aggregated_value, final_weights

class QualityBasedAggregator:
    """基于质量的聚合器"""
    
    @staticmethod
    def aggregate_with_quality_estimation(task, quality_window=10):
        """基于质量估计的聚合"""
        measurements = []
        abilities = []
        worker_ids = []
        
        for wid, (measurement, ability) in task.worker_answers.items():
            measurements.append(measurement)
            abilities.append(ability)
            worker_ids.append(wid)
        
        measurements = np.array(measurements)
        abilities = np.array(abilities)
        
        # 估计每个工人的质量（基于历史表现）
        quality_scores = QualityBasedAggregator._estimate_worker_quality(worker_ids, quality_window)
        
        # 结合能力和质量分数
        combined_weights = abilities * quality_scores
        weights = combined_weights / combined_weights.sum() if combined_weights.sum() > 0 else \
                 np.ones(len(combined_weights)) / len(combined_weights)
        
        aggregated_value = np.dot(measurements, weights)
        
        task.aggregated_value = aggregated_value
        task.final_weights = dict(zip(worker_ids, weights))
        return aggregated_value, weights
    
    @staticmethod
    def _estimate_worker_quality(worker_ids, window_size):
        """估计工人质量分数（简化版本）"""
        # 这里可以接入更复杂的质量估计逻辑
        # 目前使用随机质量分数作为示例
        np.random.seed(hash(tuple(worker_ids)) % 10000)  # 确保可重现
        quality_scores = np.random.uniform(0.5, 1.0, len(worker_ids))
        return quality_scores

class AdaptiveAggregator:
    """自适应聚合器"""
    
    def __init__(self):
        self.aggregation_history = []
        self.performance_metrics = {}
    
    def aggregate_adaptively(self, task):
        """自适应选择最佳聚合方法"""
        measurements = []
        abilities = []
        worker_ids = []
        
        for wid, (measurement, ability) in task.worker_answers.items():
            measurements.append(measurement)
            abilities.append(ability)
            worker_ids.append(wid)
        
        # 根据答案分布特征选择聚合方法
        measurements = np.array(measurements)
        
        # 计算答案的变异系数
        cv = np.std(measurements) / np.mean(measurements) if np.mean(measurements) != 0 else 0
        
        # 根据变异系数选择聚合方法
        if cv < 0.1:  # 低变异，使用简单加权平均
            return AnswerAggregator.aggregate_answers_weighted_average(task)
        elif cv < 0.3:  # 中等变异，使用IWLS
            return AnswerAggregator.aggregate_answers(task)
        elif cv < 0.5:  # 高变异，使用中位数聚合
            return AnswerAggregator.aggregate_answers_median(task)
        else:  # 极高变异，使用鲁棒聚合
            return AnswerAggregator.aggregate_answers_robust(task)
    
    def update_performance(self, method_name, performance_score):
        """更新聚合方法性能"""
        if method_name not in self.performance_metrics:
            self.performance_metrics[method_name] = []
        
        self.performance_metrics[method_name].append(performance_score)
        
        # 保持历史记录在合理范围内
        if len(self.performance_metrics[method_name]) > 100:
            self.performance_metrics[method_name] = \
                self.performance_metrics[method_name][-100:] 