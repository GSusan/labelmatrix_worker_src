# -*- coding: utf-8 -*-
"""
配置数据类定义
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List


@dataclass
class HardwareConfig:
    """硬件配置"""
    device: str = 'cuda:0'
    workers: int = 8
    min_memory_mb: Optional[int] = None


@dataclass
class HyperparametersConfig:
    """超参数配置"""
    epochs: int = 100
    batch: int = 16
    lr0: float = 0.001
    optimizer: str = 'Adam'
    weight_decay: float = 0.0005
    imgsz: int = 640
    amp: bool = True


@dataclass
class PredictConfig:
    """推理配置"""
    source: str = ''
    save_dir: str = ''
    conf_thres: float = 0.5
    iou_thres: float = 0.45
    augment: bool = False
    half: bool = True
    save_format: str = 'geojson'
    tile_size: int = 512
    tile_overlap: float = 0.1


@dataclass
class BaseConfig:
    """基础配置"""
    task_id: str
    task_type: str
    model_architecture: str
    output_dir: str
    mode: str = 'train'


@dataclass
class TrainConfig(BaseConfig):
    """训练配置"""
    data_config: str = ''
    resume_from: str = ''
    hyperparameters: HyperparametersConfig = field(default_factory=HyperparametersConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)


@dataclass
class PredictConfigFull(BaseConfig):
    """完整推理配置"""
    model_path: str = ''
    predict: PredictConfig = field(default_factory=PredictConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
