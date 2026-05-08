# -*- coding: utf-8 -*-
"""
配置解析器
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from .validator import ConfigValidator
from ..exceptions.config_errors import ConfigFileError


class ConfigParser:
    """配置文件解析器"""

    def __init__(self, config_path: str):
        """
        初始化配置解析器

        Args:
            config_path: YAML配置文件路径
        """
        self.config_path = Path(config_path)
        self.config: Optional[Dict[str, Any]] = None
        self.validator = ConfigValidator()

    def parse(self) -> Dict[str, Any]:
        """
        解析配置文件

        Returns:
            配置字典

        Raises:
            ConfigFileError: 配置文件不存在或格式错误
            ConfigValidationError: 配置验证失败
        """
        if not self.config_path.exists():
            raise ConfigFileError(
                f"Config file not found: {self.config_path}"
            )

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigFileError(f"Invalid YAML format: {e}")
        except Exception as e:
            raise ConfigFileError(f"Failed to read config file: {e}")

        # 验证配置
        self.validator.validate(self.config)

        # 转换路径为绝对路径
        self._normalize_paths()

        # 设置默认值
        self._set_defaults()

        return self.config

    def _normalize_paths(self):
        """将所有路径转换为绝对路径"""
        path_fields = [
            'data_config',
            'output_dir',
            'resume_from',
            'model_path'
        ]

        for field in path_fields:
            if field in self.config and self.config[field]:
                path = Path(self.config[field])
                if not path.is_absolute():
                    self.config[field] = str(path.resolve())

        # 处理predict.source路径
        if 'predict' in self.config:
            predict = self.config['predict']
            if 'source' in predict and predict['source']:
                source = Path(predict['source'])
                if not source.is_absolute():
                    predict['source'] = str(source.resolve())

            if 'save_dir' in predict and predict['save_dir']:
                save_dir = Path(predict['save_dir'])
                if not save_dir.is_absolute():
                    predict['save_dir'] = str(save_dir.resolve())

    def _set_defaults(self):
        """设置默认值"""
        # 模式默认值
        if 'mode' not in self.config:
            self.config['mode'] = 'train'

        # 硬件配置默认值
        if 'hardware' not in self.config:
            self.config['hardware'] = {}

        hardware = self.config['hardware']
        if 'device' not in hardware:
            hardware['device'] = 'cuda:0'
        if 'workers' not in hardware:
            hardware['workers'] = 8

        # 训练参数默认值
        if 'hyperparameters' not in self.config:
            self.config['hyperparameters'] = {}

        hyperparams = self.config['hyperparameters']
        if 'epochs' not in hyperparams:
            hyperparams['epochs'] = 100
        if 'batch' not in hyperparams:
            hyperparams['batch'] = 16
        if 'lr0' not in hyperparams:
            hyperparams['lr0'] = 0.001
        if 'optimizer' not in hyperparams:
            hyperparams['optimizer'] = 'Adam'
        if 'weight_decay' not in hyperparams:
            hyperparams['weight_decay'] = 0.0005
        if 'imgsz' not in hyperparams:
            hyperparams['imgsz'] = 640
        if 'amp' not in hyperparams:
            hyperparams['amp'] = True

        # 推理参数默认值
        if 'predict' not in self.config:
            self.config['predict'] = {}

        predict = self.config['predict']
        if 'conf_thres' not in predict:
            predict['conf_thres'] = 0.5
        if 'iou_thres' not in predict:
            predict['iou_thres'] = 0.45
        if 'augment' not in predict:
            predict['augment'] = False
        if 'half' not in predict:
            predict['half'] = True
        if 'save_format' not in predict:
            predict['save_format'] = 'geojson'

        # 注意：tile_size 和 tile_overlap 不再设置默认值
        # 只有在显式配置或使用遥感预测器时才需要这些参数

    def get_task_type(self) -> str:
        """获取任务类型"""
        return self.config.get('task_type', 'detect')

    def get_mode(self) -> str:
        """获取运行模式"""
        return self.config.get('mode', 'train')

    def get_task_id(self) -> str:
        """获取任务ID"""
        return self.config.get('task_id', '')

    def get_output_dir(self) -> Path:
        """获取输出目录"""
        return Path(self.config.get('output_dir', ''))
