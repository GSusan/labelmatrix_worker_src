# -*- coding: utf-8 -*-
"""
配置相关异常
"""

from .base import LabelMatrixException


class ConfigFileError(LabelMatrixException):
    """配置文件错误"""
    pass


class ConfigValidationError(LabelMatrixException):
    """配置验证错误"""

    def __init__(self, message: str, errors: list = None):
        super().__init__(message)
        self.errors = errors or []

    def to_dict(self) -> dict:
        result = super().to_dict()
        result['errors'] = self.errors
        return result


class MissingRequiredFieldError(ConfigValidationError):
    """缺少必填字段"""
    pass


class InvalidFieldValueError(ConfigValidationError):
    """无效的字段值"""
    pass
