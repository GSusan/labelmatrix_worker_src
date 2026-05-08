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


class SegmentConverter(BaseTaskConverter):
    """分割任务转换器"""

    def convert_feature(
        self,
        feature: dict,
        img_width: int,
        img_height: int
    ) -> Optional[str]:
        """
        转换分割任务的feature

        Args:
            feature: GeoJSON feature对象
            img_width: 图像宽度
            img_height: 图像高度

        Returns:
            Optional[str]: YOLO分割格式的标注行
        """
        props = feature.get('properties', {})
        class_id = props.get('class_id')
        geometry = feature.get('geometry', {})

        if class_id is None:
            if self.verbose_logging:
                logger.debug("Feature missing class_id")
            return None

        if geometry.get('type') != 'Polygon':
            if self.verbose_logging:
                logger.debug(f"Invalid geometry type: {geometry.get('type')}")
            return None

        coordinates = geometry.get('coordinates', [])
        if not coordinates:
            if self.verbose_logging:
                logger.debug("Empty coordinates")
            return None

        # 转换多边形坐标
        normalized_coords = self.normalize_coordinates(
            coordinates[0], img_width, img_height
        )

        # 转换class_id
        yolo_class_id = self.get_yolo_class_id(class_id)

        # YOLO格式: class_id x1 y1 x2 y2 ... xn yn
        line = f"{yolo_class_id} " + " ".join(
            f"{x:.6f} {y:.6f}" for x, y in normalized_coords
        )
        return line


class DetectConverter(BaseTaskConverter):
    """检测任务转换器 - 水平边界框"""

    def convert_feature(
        self,
        feature: dict,
        img_width: int,
        img_height: int
    ) -> Optional[str]:
        """
        转换检测任务的feature

        Args:
            feature: GeoJSON feature对象
            img_width: 图像宽度
            img_height: 图像高度

        Returns:
            Optional[str]: YOLO检测格式的标注行 (class_id x_center y_center width height)
        """
        props = feature.get('properties', {})
        class_id = props.get('class_id')
        geometry = feature.get('geometry', {})

        if class_id is None:
            if self.verbose_logging:
                logger.debug("Feature missing class_id")
            return None

        if geometry.get('type') != 'Polygon':
            # 尝试从其他几何类型提取边界框
            if self.verbose_logging:
                logger.debug(f"Attempting to extract bbox from {geometry.get('type')}")
            return self._try_extract_bbox(feature, img_width, img_height, class_id)

        coordinates = geometry.get('coordinates', [])
        if not coordinates:
            if self.verbose_logging:
                logger.debug("Empty coordinates")
            return None

        # 提取外接矩形（AABB）
        polygon_coords = coordinates[0]
        x_center, y_center, width, height = self._calculate_aabb(
            polygon_coords, img_width, img_height
        )

        # 转换class_id
        yolo_class_id = self.get_yolo_class_id(class_id)

        # YOLO检测格式: class_id x_center y_center width height
        return f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

    def _calculate_aabb(
        self,
        coordinates: List[List[float]],
        img_width: int,
        img_height: int
    ) -> Tuple[float, float, float, float]:
        """
        计算轴对齐边界框

        Args:
            coordinates: 多边形坐标列表
            img_width: 图像宽度
            img_height: 图像高度

        Returns:
            Tuple[float, float, float, float]: (x_center, y_center, width, height) 归一化后的值
        """
        # 提取所有x和y坐标
        x_coords = [coord[0] for coord in coordinates]
        y_coords = [coord[1] for coord in coordinates]

        # 计算边界
        min_x = min(x_coords)
        max_x = max(x_coords)
        min_y = min(y_coords)
        max_y = max(y_coords)

        # 归一化并计算中心点和尺寸
        x_center = ((min_x + max_x) / 2) / img_width
        y_center = ((min_y + max_y) / 2) / img_height
        width = (max_x - min_x) / img_width
        height = (max_y - min_y) / img_height

        return x_center, y_center, width, height

    def _try_extract_bbox(
        self,
        feature: dict,
        img_width: int,
        img_height: int,
        class_id: int
    ) -> Optional[str]:
        """
        尝试从非Polygon几何类型提取边界框

        Args:
            feature: GeoJSON feature对象
            img_width: 图像宽度
            img_height: 图像高度
            class_id: 类别ID

        Returns:
            Optional[str]: YOLO检测格式的标注行
        """
        geometry = feature.get('geometry', {})
        geom_type = geometry.get('type')

        # 对于Point，创建一个小的边界框
        if geom_type == 'Point':
            coords = geometry.get('coordinates', [])
            if coords:
                x, y = coords[0], coords[1]
                # 创建1%图像大小的边界框
                box_size = 0.01
                x_norm = x / img_width
                y_norm = y / img_height
                yolo_class_id = self.get_yolo_class_id(class_id)
                return f"{yolo_class_id} {x_norm:.6f} {y_norm:.6f} {box_size:.6f} {box_size:.6f}"

        # 对于LineString，使用其端点计算边界框
        elif geom_type == 'LineString':
            coords = geometry.get('coordinates', [])
            if len(coords) >= 2:
                x_center, y_center, width, height = self._calculate_aabb(
                    coords, img_width, img_height
                )
                yolo_class_id = self.get_yolo_class_id(class_id)
                return f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

        if self.verbose_logging:
            logger.warning(f"Cannot extract bbox from geometry type: {geom_type}")
        return None


class OBBConverter(BaseTaskConverter):
    """旋转框检测转换器"""

    def convert_feature(
        self,
        feature: dict,
        img_width: int,
        img_height: int
    ) -> Optional[str]:
        """
        转换旋转框检测任务的feature

        Args:
            feature: GeoJSON feature对象
            img_width: 图像宽度
            img_height: 图像高度

        Returns:
            Optional[str]: YOLO OBB格式的标注行 (class_id x1 y1 x2 y2 x3 y3 x4 y4)
        """
        props = feature.get('properties', {})
        class_id = props.get('class_id')
        geometry = feature.get('geometry', {})

        if class_id is None:
            if self.verbose_logging:
                logger.debug("Feature missing class_id")
            return None

        if geometry.get('type') != 'Polygon':
            if self.verbose_logging:
                logger.debug(f"OBB requires Polygon geometry, got {geometry.get('type')}")
            return None

        coordinates = geometry.get('coordinates', [])
        if not coordinates:
            if self.verbose_logging:
                logger.debug("Empty coordinates")
            return None

        polygon_coords = coordinates[0]

        # 验证是否为4个角点
        if not self._validate_four_corners(polygon_coords):
            if self.verbose_logging:
                logger.warning(
                    f"OBB requires exactly 4 corner points, got {len(polygon_coords)}"
                )
            return None

        # 归一化四个角点
        normalized_coords = self.normalize_coordinates(
            polygon_coords, img_width, img_height
        )

        # 转换class_id
        yolo_class_id = self.get_yolo_class_id(class_id)

        # YOLO OBB格式: class_id x1 y1 x2 y2 x3 y3 x4 y4
        line = f"{yolo_class_id} " + " ".join(
            f"{x:.6f} {y:.6f}" for x, y in normalized_coords
        )
        return line

    def _validate_four_corners(self, coordinates: List[List[float]]) -> bool:
        """
        验证是否为4个角点

        Args:
            coordinates: 坐标列表

        Returns:
            bool: 是否为4个角点
        """
        return len(coordinates) == 4
