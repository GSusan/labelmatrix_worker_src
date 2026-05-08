# -*- coding: utf-8 -*-
"""
引擎模块
"""

from .base_engine import BaseEngine, TrainResult, PredictResult
from .ultralytics_engine import UltralyticsEngine
from .labelmatrix_trainer import LabelMatrixTrainer
from .remote_sensing_predictor import RemoteSensingPredictor

__all__ = [
    'BaseEngine',
    'TrainResult',
    'PredictResult',
    'UltralyticsEngine',
    'LabelMatrixTrainer',
    'RemoteSensingPredictor',
    'create_engine'
]


def create_engine(config: dict) -> BaseEngine:
    """
    根据配置创建引擎

    Args:
        config: 配置字典，应包含:
            - task_type: 任务类型 (detect, segment, classify, obb)
            - mode: 模式 (train, predict, resume)
            - predict: 预测配置 (可选)

    Returns:
        引擎实例

    Raises:
        ValueError: 不支持的任务类型

    Note:
        - 对于训练任务，优先使用LabelMatrixTrainer以获得完整的状态监控功能
        - 对于大幅面遥感影像预测（配置中包含tile_size），使用RemoteSensingPredictor
        - 对于其他推理任务，使用UltralyticsEngine
    """
    task_type = config.get('task_type', 'detect')
    mode = config.get('mode', 'train')
    predict_config = config.get('predict', {})

    # 检查是否为支持的任务类型
    supported_tasks = set()
    if hasattr(UltralyticsEngine, 'SUPPORTED_TASKS'):
        supported_tasks.update(UltralyticsEngine.SUPPORTED_TASKS)
    if hasattr(LabelMatrixTrainer, 'SUPPORTED_TASKS'):
        supported_tasks.update(LabelMatrixTrainer.SUPPORTED_TASKS)
    if hasattr(RemoteSensingPredictor, 'SUPPORTED_TASKS'):
        supported_tasks.update(RemoteSensingPredictor.SUPPORTED_TASKS)

    if task_type not in supported_tasks:
        raise ValueError(f"Unsupported task type: {task_type}. Supported: {sorted(supported_tasks)}")

    # 对于训练任务，使用LabelMatrixTrainer以获得完整的状态监控
    if mode in ('train', 'resume') and hasattr(LabelMatrixTrainer, 'SUPPORTED_TASKS'):
        if task_type in LabelMatrixTrainer.SUPPORTED_TASKS:
            return LabelMatrixTrainer(config)

    # 对于推理任务，检查是否需要使用RemoteSensingPredictor
    if mode == 'predict' and task_type in RemoteSensingPredictor.SUPPORTED_TASKS:
        # 检查是否显式启用遥感大幅面预测器
        use_rs_predictor = predict_config.get('use_rs_predictor', False)

        # 或者如果配置中指定了有效的tile_size（大于0），也使用RemoteSensingPredictor
        tile_size = predict_config.get('tile_size', 0)
        if use_rs_predictor or (isinstance(tile_size, (int, float)) and tile_size > 0):
            return RemoteSensingPredictor(config)

    # 对于其他推理任务，使用UltralyticsEngine
    if task_type in UltralyticsEngine.SUPPORTED_TASKS:
        return UltralyticsEngine(config)

    raise ValueError(f"Unsupported task type: {task_type}")
