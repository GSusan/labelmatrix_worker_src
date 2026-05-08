# -*- coding: utf-8 -*-
"""
数据集相关异常
"""

from .base import LabelMatrixException


class DatasetValidationError(LabelMatrixException):
    """数据集验证失败异常"""
    pass


class DatasetConversionError(LabelMatrixException):
    """数据集转换失败异常"""
    pass


class GeoJSONFormatError(DatasetConversionError):
    """GeoJSON格式错误异常"""
    pass


class MissingCategoriesError(DatasetConversionError):
    """缺少类别信息异常"""
    pass


class UnsupportedTaskError(DatasetConversionError):
    """不支持的task类型异常"""
    pass


class InvalidGeometryError(GeoJSONFormatError):
    """几何形状无效异常"""
    pass
