# -*- coding: utf-8 -*-
"""
LabelMatrix - 遥感影像智能处理平台后端模块

提供基于Ultralytics YOLO的深度学习训练与推理功能。
"""

__version__ = '1.0.0'
__author__ = 'LabelMatrix Team'

from .exceptions import *
from .config import *
from .state import *
from .utils import *
from .engines import *
from .server import *
from .handlers import *

__all__ = [
    # Exceptions
    'LabelMatrixException',
    'ConfigFileError',
    'ConfigValidationError',
    'EngineError',
    'ResourceError',

    # Config
    'ConfigParser',
    'ConfigValidator',

    # State
    'StateManager',

    # Utils
    'setup_logger',
    'GPUMemoryChecker',
    'MetadataBuilder',

    # Engines
    'BaseEngine',
    'TrainResult',
    'PredictResult',
    'UltralyticsEngine',
    'create_engine',

    # Server
    'WorkerServer',

    # Handlers
    'TileProcessor',
    'ResultConverter',
]
