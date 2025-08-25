import numpy as np
from config import SystemConfig

class AbilityUpdater:
    @staticmethod
    def update_worker_ability(worker, gsq_ability, normal_abilities):
        """更新工人能力值（兼容旧接口）"""
        worker.update_ability_with_gsq(gsq_ability, normal_abilities)