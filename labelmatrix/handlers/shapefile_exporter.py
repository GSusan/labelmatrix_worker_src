# -*- coding: utf-8 -*-
"""
Shapefile导出器（支持地理坐标）
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon

from .base_vector_exporter import BaseVectorExporter
from .rs_data_structures import MergedResult

logger = logging.getLogger(__name__)


class ShapefileExporter(BaseVectorExporter):
    """Shapefile导出器（支持地理坐标）"""

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
        super().__init__(geotransform, crs, img_shape)

    def export(
        self,
        merged_result: MergedResult,
        output_path: str,
        class_names: Optional[List[str]] = None,
        naming_config: str = "default"
    ) -> Path:
        """
        导出Shapefile文件

        Args:
            merged_result: 合并后的结果
            output_path: 输出路径
            class_names: 类别名称列表
            naming_config: 命名策略 (default/timestamp/custom)

        Returns:
            实际保存的文件路径（.shp文件路径）
        """
        try:
            import fiona
            from fiona import crs as fiona_crs
        except ImportError:
            raise ImportError("Fiona is required for Shapefile export. Install with: pip install fiona")

        output_path = Path(output_path)

        # 确保输出目录存在（但不创建子文件夹）
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 直接使用传入的路径，不创建额外的子文件夹
        final_path = output_path

        # 创建Fiona schema
        schema = self._create_schema()

        # 解析CRS for Fiona
        fiona_crs_dict = self._parse_crs_for_fiona()

        # 收集所有features
        features = []

        # 根据任务类型导出
        if merged_result.merged_boxes is not None:
            # 导出检测框
            features.extend(self._export_boxes_to_shapefile(
                merged_result.merged_boxes,
                merged_result.class_ids,
                merged_result.confidences,
                class_names
            ))

        # 导出多边形（优先级：矢量多边形 > instance_id_map > merged_masks）
        if merged_result.merged_polygons is not None and merged_result.class_ids is not None:
            features.extend(self._export_polygons_to_shapefile(
                merged_result.merged_polygons,
                merged_result.class_ids,
                merged_result.confidences,
                class_names
            ))
        elif merged_result.instance_id_map is not None and merged_result.class_ids is not None:
            # 从 instance_id_map 导出
            features.extend(self._export_masks_from_instance_map(
                merged_result.instance_id_map,
                merged_result.class_ids,
                merged_result.confidences,
                class_names
            ))
        elif merged_result.merged_masks is not None:
            features.extend(self._export_masks_to_shapefile(
                merged_result.merged_masks,
                merged_result.class_ids,
                merged_result.confidences,
                class_names
            ))

        # 写入Shapefile
        # Check if features list is empty
        if not features:
            logger.warning("No features to export, creating empty Shapefile")

        logger.info(f"Writing {len(features)} features to Shapefile: {final_path}")

        with fiona.open(
            str(final_path),
            'w',
            driver='ESRI Shapefile',
            crs=fiona_crs_dict,
            schema=schema,
            encoding='utf-8'
        ) as dst:
            for feature in features:
                dst.write(feature)

        logger.info(f"Exported Shapefile to {final_path}.shp with {len(features)} features")

        return Path(f"{final_path}.shp")

    def _create_schema(self) -> Dict:
        """
        创建Fiona schema（定义字段类型）

        Shapefile字段名限制：最多10个字符
        Returns:
            Fiona schema字典
        """
        return {
            'geometry': 'Polygon',  # 或 'MultiPolygon' 如果需要
            'properties': {
                'fid': 'int',           # Feature ID
                'cls_id': 'int',        # Class ID (shortened for 10-char limit)
                'cls_nm': 'str:254',    # Class name (shortened, limit to 254 characters)
                'confid': 'float',      # Confidence (shortened)
                'area_px': 'float'      # Area in pixels (shortened)
            }
        }

    def _parse_crs_for_fiona(self) -> Dict:
        """
        解析CRS字符串为Fiona格式

        Returns:
            Fiona CRS字典
        """
        # 尝试解析EPSG代码
        if self.crs.startswith('EPSG:'):
            epsg_code = int(self.crs.split(':')[1])
            return {"init": f"EPSG:{epsg_code}"}
        else:
            # 尝试使用WKT
            try:
                from fiona import crs as fiona_crs
                return fiona_crs.from_string(self.crs)
            except Exception:
                logger.warning(f"Could not parse CRS '{self.crs}', using default")
                return {"init": "EPSG:4326"}  # 默认WGS84

    def _validate_polygon(self, polygon: np.ndarray) -> bool:
        """
        Validate polygon for Shapefile export

        Args:
            polygon: Polygon coordinates array

        Returns:
            True if polygon is valid, False otherwise
        """
        if polygon is None or len(polygon) < 3:
            return False
        # Check if points are collinear (degenerate polygon)
        if len(polygon) == 3:
            # For triangles, check if points are collinear
            x1, y1 = polygon[0]
            x2, y2 = polygon[1]
            x3, y3 = polygon[2]
            area = abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)) / 2
            return area > 1e-10
        return True

    def _validate_contour(self, contour: np.ndarray) -> bool:
        """
        Validate contour for Shapefile export

        Args:
            contour: [N, 2] contour points array

        Returns:
            True if valid, False otherwise
        """
        if contour is None or len(contour) < 3:
            return False

        # For triangles, check if points are collinear
        if len(contour) == 3:
            x1, y1 = contour[0]
            x2, y2 = contour[1]
            x3, y3 = contour[2]
            area = abs((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1)) / 2
            return area > 1e-10

        return True

    def _export_boxes_to_shapefile(
        self,
        boxes: np.ndarray,
        class_ids: np.ndarray,
        confidences: np.ndarray,
        class_names: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        导出检测框为Shapefile Features

        Args:
            boxes: [N, 6] (x1, y1, x2, y2, conf, cls)
            class_ids: [N] 类别ID
            confidences: [N] 置信度
            class_names: 类别名称列表

        Returns:
            Feature列表（Fiona格式）
        """
        features = []

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i, :4]

            # 转换为地理坐标
            geo_coords = [
                self.pixel_to_geo(x1, y1),
                self.pixel_to_geo(x2, y1),
                self.pixel_to_geo(x2, y2),
                self.pixel_to_geo(x1, y2),
                self.pixel_to_geo(x1, y1)
            ]

            class_id = int(class_ids[i]) if i < len(class_ids) else 0
            class_name = class_names[class_id] if class_names and class_id < len(class_names) else f"class_{class_id}"
            confidence = float(confidences[i]) if i < len(confidences) else 0.0

            feature = {
                'type': 'Feature',
                'geometry': {
                    'type': 'Polygon',
                    'coordinates': [geo_coords]
                },
                'properties': {
                    'fid': i + 1,
                    'cls_id': class_id,
                    'cls_nm': class_name,
                    'confid': confidence,
                    'area_px': float((x2 - x1) * (y2 - y1))
                }
            }

            features.append(feature)

        return features

    def _export_polygons_to_shapefile(
        self,
        polygons: List[np.ndarray],
        class_ids: np.ndarray,
        confidences: np.ndarray,
        class_names: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        导出矢量多边形为Shapefile Features

        Args:
            polygons: 多边形列表，每个是 [N, 2] 的顶点坐标数组（像素坐标）
            class_ids: [N] 类别ID
            confidences: [N] 置信度
            class_names: 类别名称列表

        Returns:
            Feature列表（Fiona格式）
        """
        features = []

        for i, polygon in enumerate(polygons):
            if not self._validate_polygon(polygon):
                continue

            # 转换为地理坐标
            geo_coords = []
            for point in polygon:
                x, y = float(point[0]), float(point[1])
                geo_x, geo_y = self.pixel_to_geo(x, y)
                geo_coords.append([geo_x, geo_y])

            # 闭合多边形
            if len(geo_coords) > 2:
                geo_coords.append(geo_coords[0])

            class_id = int(class_ids[i]) if i < len(class_ids) else 0
            class_name = class_names[class_id] if class_names and class_id < len(class_names) else f"class_{class_id}"
            confidence = float(confidences[i]) if i < len(confidences) else 0.0

            # 计算面积
            try:
                poly = ShapelyPolygon(polygon)
                area = float(poly.area)
            except Exception:
                area = 0.0

            feature = {
                'type': 'Feature',
                'geometry': {
                    'type': 'Polygon',
                    'coordinates': [geo_coords]
                },
                'properties': {
                    'fid': i + 1,
                    'cls_id': class_id,
                    'cls_nm': class_name,
                    'confid': confidence,
                    'area_px': area
                }
            }

            features.append(feature)

        return features

    def _export_masks_to_shapefile(
        self,
        masks: np.ndarray,
        class_ids: np.ndarray,
        confidences: np.ndarray,
        class_names: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        导出掩膜为Shapefile Features

        Args:
            masks: [N, H, W] 掩膜数组
            class_ids: [N] 类别ID
            confidences: [N] 置信度
            class_names: 类别名称列表

        Returns:
            Feature列表（Fiona格式）
        """
        from skimage import measure

        features = []

        for i in range(len(masks)):
            mask = masks[i]

            # 找到轮廓
            contours = measure.find_contours(mask.astype(np.uint8), 0.5)

            if not contours:
                continue

            # 选择最长的轮廓
            longest_contour = max(contours, key=len)

            # Validate contour
            if not self._validate_contour(longest_contour):
                continue

            # 转换为地理坐标
            geo_coords = []
            for point in longest_contour:
                y, x = point
                geo_x, geo_y = self.pixel_to_geo(x, y)
                geo_coords.append([geo_x, geo_y])

            # 闭合多边形
            if len(geo_coords) > 2:
                geo_coords.append(geo_coords[0])

            class_id = int(class_ids[i]) if i < len(class_ids) else 0
            class_name = class_names[class_id] if class_names and class_id < len(class_names) else f"class_{class_id}"
            confidence = float(confidences[i]) if i < len(confidences) else 0.0

            feature = {
                'type': 'Feature',
                'geometry': {
                    'type': 'Polygon',
                    'coordinates': [geo_coords]
                },
                'properties': {
                    'fid': i + 1,
                    'cls_id': class_id,
                    'cls_nm': class_name,
                    'confid': confidence,
                    'area_px': float(np.sum(mask))
                }
            }

            features.append(feature)

        return features

    def _export_masks_from_instance_map(
        self,
        instance_id_map: np.ndarray,
        class_ids: np.ndarray,
        confidences: np.ndarray,
        class_names: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        从实例ID映射图导出掩膜为Shapefile Features

        Args:
            instance_id_map: [H, W] 实例ID映射图
            class_ids: [N] 每个实例的类别ID
            confidences: [N] 每个实例的置信度
            class_names: 类别名称列表

        Returns:
            Feature列表（Fiona格式）
        """
        from skimage import measure

        unique_ids = np.unique(instance_id_map)
        unique_ids = unique_ids[unique_ids > 0]  # 排除背景(0)

        features = []

        for instance_id in unique_ids:
            idx = int(instance_id) - 1  # 转换为数组索引

            # 创建二值掩膜
            mask = (instance_id_map == instance_id).astype(np.uint8)

            # 找到轮廓
            contours = measure.find_contours(mask, 0.5)

            if not contours:
                continue

            # 选择最长的轮廓
            longest_contour = max(contours, key=len)

            # Validate contour
            if not self._validate_contour(longest_contour):
                continue

            # 简化轮廓（减少点数）
            if len(longest_contour) > 100:
                from skimage.measure import approximate_polygon
                longest_contour = approximate_polygon(longest_contour, tolerance=1.0)

            # 转换为地理坐标
            geo_coords = []
            for point in longest_contour:
                y, x = point
                geo_x, geo_y = self.pixel_to_geo(x, y)
                geo_coords.append([geo_x, geo_y])

            # 闭合多边形
            if len(geo_coords) > 2:
                geo_coords.append(geo_coords[0])

            # 获取类别和置信度
            if idx < len(class_ids):
                class_id = int(class_ids[idx])
            else:
                class_id = 0

            if idx < len(confidences):
                confidence = float(confidences[idx])
            else:
                confidence = 0.0

            class_name = class_names[class_id] if class_names and class_id < len(class_names) else f"class_{class_id}"

            feature = {
                'type': 'Feature',
                'geometry': {
                    'type': 'Polygon',
                    'coordinates': [geo_coords]
                },
                'properties': {
                    'fid': int(instance_id),
                    'cls_id': class_id,
                    'cls_nm': class_name,
                    'confid': confidence,
                    'area_px': float(np.sum(mask))
                }
            }

            features.append(feature)

        return features
