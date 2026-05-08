# -*- coding: utf-8 -*-
"""
引擎相关异常
"""

from .base import LabelMatrixException


class EngineError(LabelMatrixException):
    """引擎基础错误"""
    pass


class ModelLoadError(EngineError):
    """模型加载错误"""
    pass


class DatasetError(EngineError):
    """数据集错误"""
    pass


class TrainingError(EngineError):
    """训练错误"""
    pass


class GPUMemoryError(EngineError):
    """GPU内存相关错误"""
    pass
