# -*- coding: utf-8 -*-
"""
遥感预测相关异常
"""

from .base import LabelMatrixException


class RemoteSensingError(LabelMatrixException):
    """遥感预测基础异常"""
    pass


class ImageReadError(RemoteSensingError):
    """影像读取失败"""

    def __init__(self, message: str, image_path: str = None):
        self.image_path = image_path
        super().__init__(message)


class InvalidGeoReferenceError(RemoteSensingError):
    """无效的地理参考信息"""

    def __init__(self, message: str):
        super().__init__(message)


class TileProcessingError(RemoteSensingError):
    """分块处理失败"""

    def __init__(self, message: str, tile_id: int = None):
        self.tile_id = tile_id
        super().__init__(message)


class MergeError(RemoteSensingError):
    """结果合并失败"""

    def __init__(self, message: str):
        super().__init__(message)


class ModelLoadError(RemoteSensingError):
    """模型加载失败"""

    def __init__(self, message: str, model_path: str = None):
        self.model_path = model_path
        super().__init__(message)
