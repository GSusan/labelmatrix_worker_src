# -*- coding: utf-8 -*-
"""
LabelMatrix 异常模块

包含所有自定义异常类
"""

from .base import LabelMatrixException
from .config_errors import (
    ConfigFileError,
    ConfigValidationError,
    MissingRequiredFieldError,
    InvalidFieldValueError
)
from .engine_errors import (
    EngineError,
    ModelLoadError,
    DatasetError,
    TrainingError,
    GPUMemoryError
)
from .resource_errors import (
    ResourceError,
    GPUNotAvailableError,
    InsufficientGPUMemoryError,
    DiskSpaceError
)
from .remote_sensing_errors import (
    RemoteSensingError,
    ImageReadError,
    InvalidGeoReferenceError,
    TileProcessingError,
    MergeError,
    ModelLoadError as RSModelLoadError
)
from .dataset_errors import (
    DatasetValidationError,
    DatasetConversionError,
    GeoJSONFormatError,
    MissingCategoriesError
)

__all__ = [
    'LabelMatrixException',
    'ConfigFileError',
    'ConfigValidationError',
    'MissingRequiredFieldError',
    'InvalidFieldValueError',
    'EngineError',
    'ModelLoadError',
    'DatasetError',
    'TrainingError',
    'GPUMemoryError',
    'ResourceError',
    'GPUNotAvailableError',
    'InsufficientGPUMemoryError',
    'DiskSpaceError',
    'RemoteSensingError',
    'ImageReadError',
    'InvalidGeoReferenceError',
    'TileProcessingError',
    'MergeError',
    'RSModelLoadError',
    'DatasetValidationError',
    'DatasetConversionError',
    'GeoJSONFormatError',
    'MissingCategoriesError'
]
