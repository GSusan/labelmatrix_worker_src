# -*- coding: utf-8 -*-
"""
配置验证器
"""

from typing import Dict, Any, List
from ..exceptions.config_errors import ConfigValidationError


class ConfigValidator:
    """配置验证器"""

    VALID_TASK_TYPES = ['detect', 'segment', 'classify', 'semantic_seg', 'obb']
    VALID_MODES = ['train', 'predict', 'resume']
    VALID_OPTIMIZERS = ['SGD', 'Adam', 'AdamW', 'RMSProp', 'auto']
    VALID_FORMATS = ['geojson', 'tiff', 'json']

    def __init__(self):
        self.errors: List[str] = []

    def validate(self, config: Dict[str, Any]) -> bool:
        """
        验证配置

        Args:
            config: 配置字典

        Returns:
            验证是否通过

        Raises:
            ConfigValidationError: 验证失败时抛出
        """
        self.errors = []

        # 必填字段验证
        self._validate_required_fields(config)

        # 任务类型验证
        if 'task_type' in config:
            self._validate_task_type(config['task_type'])

        # 模式验证
        if 'mode' in config:
            self._validate_mode(config['mode'])

        # 硬件配置验证
        if 'hardware' in config:
            self._validate_hardware(config['hardware'])

        # 训练参数验证
        if 'hyperparameters' in config:
            self._validate_hyperparameters(config['hyperparameters'])

        # 推理参数验证
        if 'predict' in config:
            self._validate_predict_config(config['predict'])

        if self.errors:
            raise ConfigValidationError(
                f"Configuration validation failed with {len(self.errors)} error(s)",
                self.errors
            )

        return True

    def _validate_required_fields(self, config: Dict[str, Any]):
        """验证必填字段"""
        required_fields = ['task_id', 'task_type', 'output_dir']

        for field in required_fields:
            if field not in config or not config[field]:
                self.errors.append(f"Missing required field: {field}")

    def _validate_task_type(self, task_type: str):
        """验证任务类型"""
        if task_type not in self.VALID_TASK_TYPES:
            self.errors.append(
                f"Invalid task_type '{task_type}'. "
                f"Must be one of {self.VALID_TASK_TYPES}"
            )

    def _validate_mode(self, mode: str):
        """验证运行模式"""
        if mode not in self.VALID_MODES:
            self.errors.append(
                f"Invalid mode '{mode}'. "
                f"Must be one of {self.VALID_MODES}"
            )

    def _validate_hardware(self, hardware: Dict[str, Any]):
        """验证硬件配置"""
        if 'device' in hardware:
            device = hardware['device']
            if not (device == 'cpu' or device.startswith('cuda:')):
                self.errors.append(
                    f"Invalid device '{device}'. "
                    f"Must be 'cpu' or 'cuda:N'"
                )

        if 'workers' in hardware:
            workers = hardware['workers']
            if not isinstance(workers, int) or workers < 0:
                self.errors.append("workers must be a non-negative integer")

        if 'min_memory_mb' in hardware:
            min_mem = hardware['min_memory_mb']
            if not isinstance(min_mem, int) or min_mem < 0:
                self.errors.append("min_memory_mb must be a non-negative integer")

    def _validate_hyperparameters(self, hyperparams: Dict[str, Any]):
        """验证超参数"""
        numeric_fields = {
            'epochs': (1, 100000),
            'batch': (1, 1024),
            'lr0': (1e-6, 10.0),
            'weight_decay': (0.0, 1.0),
            'imgsz': (32, 8192)
        }

        for field, (min_val, max_val) in numeric_fields.items():
            if field in hyperparams:
                value = hyperparams[field]
                if not isinstance(value, (int, float)):
                    self.errors.append(f"{field} must be numeric")
                elif not (min_val <= value <= max_val):
                    self.errors.append(
                        f"{field} must be between {min_val} and {max_val}, got {value}"
                    )

        if 'optimizer' in hyperparams:
            optimizer = hyperparams['optimizer']
            if optimizer not in self.VALID_OPTIMIZERS:
                self.errors.append(
                    f"Invalid optimizer '{optimizer}'. "
                    f"Must be one of {self.VALID_OPTIMIZERS}"
                )

        if 'amp' in hyperparams:
            if not isinstance(hyperparams['amp'], bool):
                self.errors.append("amp must be a boolean value")

    def _validate_predict_config(self, predict: Dict[str, Any]):
        """验证推理配置"""
        if 'source' in predict and not predict['source']:
            self.errors.append("predict.source is required for predict mode")

        if 'conf_thres' in predict:
            conf = predict['conf_thres']
            if not isinstance(conf, (int, float)) or not (0 <= conf <= 1):
                self.errors.append("predict.conf_thres must be between 0 and 1")

        if 'iou_thres' in predict:
            iou = predict['iou_thres']
            if not isinstance(iou, (int, float)) or not (0 <= iou <= 1):
                self.errors.append("predict.iou_thres must be between 0 and 1")

        if 'tile_size' in predict:
            size = predict['tile_size']
            if not isinstance(size, int) or size < 32:
                self.errors.append("predict.tile_size must be an integer >= 32")

        if 'tile_overlap' in predict:
            overlap = predict['tile_overlap']
            if not isinstance(overlap, (int, float)) or not (0 <= overlap < 1):
                self.errors.append("predict.tile_overlap must be between 0 and 1")

        if 'save_format' in predict:
            fmt = predict['save_format']
            if fmt not in self.VALID_FORMATS:
                self.errors.append(
                    f"Invalid save_format '{fmt}'. Must be one of {self.VALID_FORMATS}"
                )
