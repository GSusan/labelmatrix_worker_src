# -*- coding: utf-8 -*-
"""
资源相关异常
"""

from .base import LabelMatrixException


class ResourceError(LabelMatrixException):
    """资源错误"""
    pass


class GPUNotAvailableError(ResourceError):
    """GPU不可用错误"""
    pass


class InsufficientGPUMemoryError(ResourceError):
    """GPU显存不足错误"""
    pass


class DiskSpaceError(ResourceError):
    """磁盘空间不足错误"""
    pass
