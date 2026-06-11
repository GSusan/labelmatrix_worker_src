# -*- coding: utf-8 -*-
"""
Shapefile导出器（使用GDAL/OGR，避免编码问题）
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np

try:
    from osgeo import gdal, ogr, osr
    GDAL_AVAILABLE = True
except ImportError:
    GDAL_AVAILABLE = False

from .base_vector_exporter import BaseVectorExporter
from .rs_data_structures import MergedResult

logger = logging.getLogger(__name__)


class ShapefileExporter(BaseVectorExporter):
    """Shapefile导出器（使用GDAL/OGR，支持地理坐标）"""

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
        if not GDAL_AVAILABLE:
            raise ImportError("GDAL is required for Shapefile export. Install with: pip install gdal")

        super().__init__(geotransform, crs, img_shape)

        # 关键改进：预先创建SRS对象，避免编码问题
        self._srs = self._create_srs_object()

        # 设置GDAL编码选项（支持中文）
        gdal.SetConfigOption('SHAPE_ENCODING', 'UTF-8')
        gdal.SetConfigOption('GDAL_FILENAME_IS_UTF8', 'YES')

    def _create_srs_object(self) -> osr.SpatialReference:
        """
        从CRS字符串创建SRS对象
        关键方法：处理所有编码问题，后续直接使用对象
        """
        srs = osr.SpatialReference()

        try:
            # 情况1：EPSG代码
            if self.crs.startswith('EPSG:'):
                epsg = int(self.crs.split(':')[1])
                srs.ImportFromEPSG(epsg)
                logger.info(f"Created SRS from EPSG:{epsg}")
                return srs

            # 情况2：Proj4字符串
            if self.crs.startswith('+proj='):
                srs.ImportFromProj4(self.crs)
                logger.info("Created SRS from Proj4")
                return srs

            # 情况3：WKT字符串（可能有编码问题）
            if 'PROJCS' in self.crs or 'GEOGCS' in self.crs:
                # 使用SetFromUserInput，GDAL会自动处理编码
                # 这是最宽容的方法，能处理各种编码的WKT
                result = srs.SetFromUserInput(self.crs)
                if result == 0:  # 成功
                    logger.info("Created SRS from WKT using SetFromUserInput")
                    return srs

                # 回退到直接导入
                srs.ImportFromWkt(self.crs)
                logger.info("Created SRS from WKT using ImportFromWkt")
                return srs

            # 默认使用WGS84
            logger.warning(f"Unrecognized CRS format: {self.crs[:50]}, using WGS84")
            srs.ImportFromEPSG(4326)
            return srs

        except Exception as e:
            logger.error(f"Failed to create SRS: {e}, using WGS84")
            srs.ImportFromEPSG(4326)
            return srs

    def export(
        self,
        merged_result: MergedResult,
        output_path: str,
        class_names: Optional[List[str]] = None,
        naming_config: str = "default"
    ) -> Path:
        """
        导出Shapefile文件（使用GDAL/OGR）

        Args:
            merged_result: 合并后的结果
            output_path: 输出路径
            class_names: 类别名称列表
            naming_config: 命名策略 (default/timestamp/custom)

        Returns:
            实际保存的文件路径（.shp文件路径）
        """
        output_path = Path(output_path)

        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 创建Shapefile驱动
        driver = ogr.GetDriverByName('ESRI Shapefile')
        if driver is None:
            raise RuntimeError("ESRI Shapefile driver not available")

        # 删除已存在的文件
        if output_path.exists():
            driver.DeleteDataSource(str(output_path))

        # 创建数据源
        datasource = driver.CreateDataSource(str(output_path))
        if datasource is None:
            raise RuntimeError(f"Failed to create {output_path}")

        try:
            # 创建图层 - 关键：直接使用SRS对象，无编码问题
            layer = datasource.CreateLayer(
                str(output_path.stem),  # 图层名
                srs=self._srs,          # 直接传递SRS对象
                geom_type=ogr.wkbPolygon
            )

            if layer is None:
                raise RuntimeError("Failed to create layer")

            # 创建属性字段
            self._create_fields(layer)

            # 统计要素数量
            feature_count = 0

            # 根据任务类型导出
            if merged_result.merged_boxes is not None:
                # 导出检测框
                count = self._export_boxes_to_layer(
                    layer,
                    merged_result.merged_boxes,
                    merged_result.class_ids,
                    merged_result.confidences,
                    class_names
                )
                feature_count += count

            # 导出多边形（优先级：矢量多边形 > instance_id_map > merged_masks）
            if merged_result.merged_polygons is not None and merged_result.class_ids is not None:
                count = self._export_polygons_to_layer(
                    layer,
                    merged_result.merged_polygons,
                    merged_result.class_ids,
                    merged_result.confidences,
                    class_names
                )
                feature_count += count
            elif merged_result.instance_id_map is not None and merged_result.class_ids is not None:
                # 从 instance_id_map 导出
                count = self._export_masks_from_instance_map(
                    layer,
                    merged_result.instance_id_map,
                    merged_result.class_ids,
                    merged_result.confidences,
                    class_names
                )
                feature_count += count
            elif merged_result.merged_masks is not None:
                count = self._export_masks_to_layer(
                    layer,
                    merged_result.merged_masks,
                    merged_result.class_ids,
                    merged_result.confidences,
                    class_names
                )
                feature_count += count

            # 同步到磁盘
            datasource.SyncToDisk()

            logger.info(f"Exported {feature_count} features to {output_path}")

        finally:
            datasource = None  # 关闭数据源

        return output_path

    def _create_fields(self, layer):
        """
        创建属性字段

        Shapefile字段名限制：最多10个字符
        """
        # 字段定义：使用字典格式更灵活
        fields = [
            {'name': 'fid', 'type': ogr.OFTInteger, 'width': 10, 'precision': 0},
            {'name': 'cls_id', 'type': ogr.OFTInteger, 'width': 10, 'precision': 0},
            {'name': 'cls_nm', 'type': ogr.OFTString, 'width': 254},
            {'name': 'confid', 'type': ogr.OFTReal, 'width': 10, 'precision': 4},
            {'name': 'area_px', 'type': ogr.OFTReal, 'width': 20, 'precision': 2}
        ]

        for field_def in fields:
            field = ogr.FieldDefn(field_def['name'], field_def['type'])
            field.SetWidth(field_def['width'])
            if 'precision' in field_def and field_def['precision'] > 0:
                field.SetPrecision(field_def['precision'])
            layer.CreateField(field)

    def _create_polygon_geometry(self, polygon: np.ndarray) -> ogr.Geometry:
        """
        从多边形顶点创建OGR多边形几何对象

        Args:
            polygon: [N, 2] 多边形顶点数组（像素坐标）

        Returns:
            OGR多边形几何对象
        """
        # 创建环
        ring = ogr.Geometry(ogr.wkbLinearRing)

        # 添加点（转换为地理坐标）
        for point in polygon:
            x, y = float(point[0]), float(point[1])
            geo_x, geo_y = self.pixel_to_geo(x, y)
            ring.AddPoint(geo_x, geo_y)

        # 闭合环
        ring.AddPoint(ring.GetX(0), ring.GetY(0))

        # 创建多边形
        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(ring)

        return poly

    def _validate_polygon(self, polygon: np.ndarray) -> bool:
        """
        验证多边形有效性

        Args:
            polygon: [N, 2] 多边形顶点数组

        Returns:
            True if valid, False otherwise
        """
        if polygon is None or len(polygon) < 3:
            return False

        # 检查点是否共线（退化多边形）
        if len(polygon) == 3:
            x1, y1 = polygon[0]
            x2, y2 = polygon[1]
            x3, y3 = polygon[2]
            area = abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)) / 2
            return area > 1e-10

        return True

    def _export_boxes_to_layer(
        self,
        layer,
        boxes: np.ndarray,
        class_ids: np.ndarray,
        confidences: np.ndarray,
        class_names: Optional[List[str]] = None
    ) -> int:
        """
        导出检测框到图层

        Args:
            layer: OGR图层对象
            boxes: [N, 6] (x1, y1, x2, y2, conf, cls)
            class_ids: [N] 类别ID
            confidences: [N] 置信度
            class_names: 类别名称列表

        Returns:
            导出的要素数量
        """
        count = 0
        layer_defn = layer.GetLayerDefn()

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i, :4]

            # 创建矩形多边形
            box_polygon = np.array([
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2]
            ])

            # 创建几何
            geom = self._create_polygon_geometry(box_polygon)

            # 创建要素
            feature = ogr.Feature(layer_defn)
            feature.SetGeometry(geom)

            # 设置属性
            class_id = int(class_ids[i]) if i < len(class_ids) else 0
            class_name = (class_names[class_id] if class_names and
                         class_id < len(class_names) else f"class_{class_id}")
            confidence = float(confidences[i]) if i < len(confidences) else 0.0

            feature.SetField('fid', count + 1)
            feature.SetField('cls_id', class_id)
            feature.SetField('cls_nm', class_name)
            feature.SetField('confid', confidence)
            feature.SetField('area_px', float((x2 - x1) * (y2 - y1)))

            # 创建要素
            layer.CreateFeature(feature)
            feature = None  # 释放
            count += 1

        return count

    def _export_polygons_to_layer(
        self,
        layer,
        polygons: List[np.ndarray],
        class_ids: np.ndarray,
        confidences: np.ndarray,
        class_names: Optional[List[str]] = None
    ) -> int:
        """
        导出矢量多边形到图层

        Args:
            layer: OGR图层对象
            polygons: 多边形列表，每个是 [N, 2] 的顶点坐标数组（像素坐标）
            class_ids: [N] 类别ID
            confidences: [N] 置信度
            class_names: 类别名称列表

        Returns:
            导出的要素数量
        """
        count = 0
        layer_defn = layer.GetLayerDefn()

        for i, polygon in enumerate(polygons):
            if not self._validate_polygon(polygon):
                continue

            # 创建几何
            geom = self._create_polygon_geometry(polygon)

            # 创建要素
            feature = ogr.Feature(layer_defn)
            feature.SetGeometry(geom)

            # 设置属性
            class_id = int(class_ids[i]) if i < len(class_ids) else 0
            class_name = (class_names[class_id] if class_names and
                         class_id < len(class_names) else f"class_{class_id}")
            confidence = float(confidences[i]) if i < len(confidences) else 0.0

            # 计算面积
            try:
                from shapely.geometry import Polygon as ShapelyPolygon
                area = float(ShapelyPolygon(polygon).area)
            except Exception:
                area = 0.0

            feature.SetField('fid', i + 1)
            feature.SetField('cls_id', class_id)
            feature.SetField('cls_nm', class_name)
            feature.SetField('confid', confidence)
            feature.SetField('area_px', area)

            # 创建要素
            layer.CreateFeature(feature)
            feature = None  # 释放
            count += 1

        return count

    def _export_masks_to_layer(
        self,
        layer,
        masks: np.ndarray,
        class_ids: np.ndarray,
        confidences: np.ndarray,
        class_names: Optional[List[str]] = None
    ) -> int:
        """
        导出掩膜到图层

        Args:
            layer: OGR图层对象
            masks: [N, H, W] 掩膜数组
            class_ids: [N] 类别ID
            confidences: [N] 置信度
            class_names: 类别名称列表

        Returns:
            导出的要素数量
        """
        from skimage import measure

        count = 0
        layer_defn = layer.GetLayerDefn()

        for i in range(len(masks)):
            mask = masks[i]

            # 找到轮廓
            contours = measure.find_contours(mask.astype(np.uint8), 0.5)

            if not contours:
                continue

            # 选择最长的轮廓
            longest_contour = max(contours, key=len)

            # 验证轮廓
            if len(longest_contour) < 3:
                continue

            # 转换为多边形坐标数组
            polygon = longest_contour[:, ::-1]  # [y, x] -> [x, y]

            # 创建几何
            geom = self._create_polygon_geometry(polygon)

            # 创建要素
            feature = ogr.Feature(layer_defn)
            feature.SetGeometry(geom)

            # 设置属性
            class_id = int(class_ids[i]) if i < len(class_ids) else 0
            class_name = (class_names[class_id] if class_names and
                         class_id < len(class_names) else f"class_{class_id}")
            confidence = float(confidences[i]) if i < len(confidences) else 0.0

            feature.SetField('fid', i + 1)
            feature.SetField('cls_id', class_id)
            feature.SetField('cls_nm', class_name)
            feature.SetField('confid', confidence)
            feature.SetField('area_px', float(np.sum(mask)))

            # 创建要素
            layer.CreateFeature(feature)
            feature = None  # 释放
            count += 1

        return count

    def _export_masks_from_instance_map(
        self,
        layer,
        instance_id_map: np.ndarray,
        class_ids: np.ndarray,
        confidences: np.ndarray,
        class_names: Optional[List[str]] = None
    ) -> int:
        """
        从实例ID映射图导出掩膜到图层

        Args:
            layer: OGR图层对象
            instance_id_map: [H, W] 实例ID映射图
            class_ids: [N] 每个实例的类别ID
            confidences: [N] 每个实例的置信度
            class_names: 类别名称列表

        Returns:
            导出的要素数量
        """
        from skimage import measure

        unique_ids = np.unique(instance_id_map)
        unique_ids = unique_ids[unique_ids > 0]  # 排除背景(0)

        count = 0
        layer_defn = layer.GetLayerDefn()

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

            # 验证轮廓
            if len(longest_contour) < 3:
                continue

            # 简化轮廓（减少点数）
            if len(longest_contour) > 100:
                from skimage.measure import approximate_polygon
                longest_contour = approximate_polygon(longest_contour, tolerance=1.0)

            # 转换为多边形坐标数组
            polygon = longest_contour[:, ::-1]  # [y, x] -> [x, y]

            # 创建几何
            geom = self._create_polygon_geometry(polygon)

            # 创建要素
            feature = ogr.Feature(layer_defn)
            feature.SetGeometry(geom)

            # 获取类别和置信度
            if idx < len(class_ids):
                class_id = int(class_ids[idx])
            else:
                class_id = 0

            if idx < len(confidences):
                confidence = float(confidences[idx])
            else:
                confidence = 0.0

            class_name = (class_names[class_id] if class_names and
                         class_id < len(class_names) else f"class_{class_id}")

            feature.SetField('fid', int(instance_id))
            feature.SetField('cls_id', class_id)
            feature.SetField('cls_nm', class_name)
            feature.SetField('confid', confidence)
            feature.SetField('area_px', float(np.sum(mask)))

            # 创建要素
            layer.CreateFeature(feature)
            feature = None  # 释放
            count += 1

        return count
