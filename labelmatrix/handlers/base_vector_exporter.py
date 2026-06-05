# -*- coding: utf-8 -*-
"""
Base Vector Exporter - Shared logic for GeoJSON and Shapefile export
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)


class BaseVectorExporter(ABC):
    """Base class for vector data exporters with geographic coordinates"""

    def __init__(
        self,
        geotransform: tuple,
        crs: str,
        img_shape: tuple
    ):
        """
        Args:
            geotransform: GDAL仿射变换参数 (ul_x, x_res, x_rot, ul_y, y_rot, y_res)
            crs: 坐标系字符串
            img_shape: 影像尺寸 (height, width)
        """
        self.geotransform = geotransform
        self.crs = crs
        self.img_shape = img_shape

    def pixel_to_geo(self, x: float, y: float) -> tuple:
        """
        像素坐标转地理坐标

        Args:
            x: 列坐标（像素）
            y: 行坐标（像素）

        Returns:
            (geo_x, geo_y): 地理坐标
        """
        ul_x, x_res, x_rot, ul_y, y_rot, y_res = self.geotransform

        geo_x = ul_x + x * x_res + y * x_rot
        geo_y = ul_y + x * y_rot + y * y_res

        return (float(geo_x), float(geo_y))

    def geo_to_pixel(self, geo_x: float, geo_y: float) -> tuple:
        """
        地理坐标转像素坐标

        Args:
            geo_x: 地理X坐标
            geo_y: 地理Y坐标

        Returns:
            (x, y): 像素坐标
        """
        ul_x, x_res, x_rot, ul_y, y_rot, y_res = self.geotransform

        # 解方程
        denom = x_res * y_res - x_rot * y_rot
        if abs(denom) < 1e-10:
            logger.warning("Degenerate geotransform, using approximate conversion")
            x = (geo_x - ul_x) / x_res if x_res != 0 else 0
            y = (geo_y - ul_y) / y_res if y_res != 0 else 0
        else:
            x = (y_res * (geo_x - ul_x) - x_rot * (geo_y - ul_y)) / denom
            y = (-y_rot * (geo_x - ul_x) + x_res * (geo_y - ul_y)) / denom

        return (x, y)

    @abstractmethod
    def export(
        self,
        merged_result,
        output_path: str,
        class_names: Optional[List[str]] = None,
        naming_config: str = "default"
    ):
        """Export merged results to vector format"""
        pass

    def _generate_filename(self, base_name: str, naming_config: str) -> str:
        """
        生成文件名

        Args:
            base_name: 基础名称
            naming_config: 命名策略

        Returns:
            文件名（不含扩展名）
        """
        from datetime import datetime

        if naming_config == "timestamp":
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return f"{base_name}_{timestamp}"
        elif naming_config == "custom":
            return base_name
        else:  # default
            return base_name
