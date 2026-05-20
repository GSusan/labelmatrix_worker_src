# -*- coding: utf-8 -*-
"""
数据集验证器 - 验证数据集是否符合YOLO格式要求
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class DatasetValidator:
    """数据集格式验证器"""

    def __init__(
        self,
        data_config_path: str,
        train_val_split: float = 0.8,
        random_seed: int = 42,
        verbose_logging: bool = False
    ):
        """
        初始化验证器

        Args:
            data_config_path: 数据集配置文件路径 (data.yaml)
            train_val_split: 训练集比例，默认0.8（80%训练，20%验证）
            random_seed: 随机种子，默认42
            verbose_logging: 是否使用详细日志模式，默认False
        """
        self.data_config_path = Path(data_config_path)
        self.dataset_root = self.data_config_path.parent
        self.train_val_split = train_val_split
        self.random_seed = random_seed
        self.verbose_logging = verbose_logging

    def validate(self) -> bool:
        """
        验证数据集格式是否符合YOLO要求

        Returns:
            bool: True表示格式正确，False表示需要转换
        """
        # 检查 data.yaml 是否存在
        if not self.data_config_path.exists():
            logger.warning(f"data.yaml not found: {self.data_config_path}")
            return False

        # 检查必需目录（YOLO格式要求）
        required_dirs = ['labels', 'labels/train2017', 'labels/val2017', 'images']
        for dir_path in required_dirs:
            full_path = self.dataset_root / dir_path
            if not full_path.exists():
                logger.debug(f"Required directory not found: {dir_path}")
                return False

        # 检查是否有标注文件
        label_files = list(self.dataset_root.glob('labels/**/*.txt'))
        if not label_files:
            logger.debug("No label files found")
            return False

        logger.info(f"Dataset validation passed: {self.dataset_root}")
        return True

    def get_converter(self):
        """
        获取转换器实例

        Returns:
            GeoJSONToYOLOConverter: 转换器实例
        """
        from labelmatrix.utils.geojson_to_yolo import GeoJSONToYOLOConverter
        return GeoJSONToYOLOConverter(
            str(self.dataset_root),
            train_val_split=self.train_val_split,
            random_seed=self.random_seed,
            verbose_logging=self.verbose_logging
        )
