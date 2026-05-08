# -*- coding: utf-8 -*-
"""
任务转换器基类和具体实现
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


class BaseTaskConverter(ABC):
    """任务转换器抽象基类"""

    def __init__(
        self,
        categories: dict,
        source_config_data: dict,
        verbose_logging: bool = False
    ):
        """
        初始化转换器

        Args:
            categories: 类别ID到类别名称的映射
            source_config_data: 源data.yaml配置数据
            verbose_logging: 是否使用详细日志模式
        """
        self.categories = categories
        self._source_config_data = source_config_data
        self.verbose_logging = verbose_logging

    @abstractmethod
    def convert_feature(
        self,
        feature: dict,
        img_width: int,
        img_height: int
    ) -> Optional[str]:
        """
        转换单个feature为YOLO格式行

        Args:
            feature: GeoJSON feature对象
            img_width: 图像宽度
            img_height: 图像高度

        Returns:
            Optional[str]: YOLO格式的标注行，如果转换失败返回None
        """
        pass

    def normalize_coordinate(
        self,
        x: float,
        y: float,
        img_width: int,
        img_height: int
    ) -> Tuple[float, float]:
        """
        归一化单个坐标点

        Args:
            x: 原始x坐标
            y: 原始y坐标
            img_width: 图像宽度
            img_height: 图像高度

        Returns:
            Tuple[float, float]: 归一化后的坐标 (x, y)
        """
        return (x / img_width, y / img_height)

    def normalize_coordinates(
        self,
        coordinates: List[List[float]],
        img_width: int,
        img_height: int
    ) -> List[Tuple[float, float]]:
        """
        归一化坐标列表

        Args:
            coordinates: 原始坐标列表 [[x, y, z], ...]
            img_width: 图像宽度
            img_height: 图像高度

        Returns:
            List[Tuple[float, float]]: 归一化后的坐标列表 [(x, y), ...]
        """
        normalized = []
        for coord in coordinates:
            x = coord[0] / img_width
            y = coord[1] / img_height
            normalized.append((x, y))
        return normalized

    def get_yolo_class_id(self, class_id: int) -> int:
        """
        获取YOLO格式的class_id（处理从1开始的索引）

        Args:
            class_id: 原始class_id

        Returns:
            int: YOLO格式的class_id（从0开始）
        """
        names = self._source_config_data.get('names')
        needs_conversion = False

        if isinstance(names, dict):
            keys = [int(k) for k in names.keys()]
            if keys and min(keys) == 1:
                needs_conversion = True
        elif isinstance(names, list):
            needs_conversion = True

        return class_id - 1 if needs_conversion and class_id > 0 else class_id
