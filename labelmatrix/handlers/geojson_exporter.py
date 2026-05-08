# -*- coding: utf-8 -*-
"""
GeoJSON导出器（支持地理坐标 + 并行处理）
"""

import json
import logging
import multiprocessing
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np
from shapely.geometry import Polygon, mapping, shape

from ..handlers.rs_data_structures import MergedResult

logger = logging.getLogger(__name__)


class GeoJSONExporter:
    """GeoJSON导出器（支持地理坐标）"""

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

    def export(
        self,
        merged_result: MergedResult,
        output_path: str,
        class_names: Optional[List[str]] = None,
        naming_config: str = "default"
    ) -> Path:
        """
        导出GeoJSON文件

        Args:
            merged_result: 合并后的结果
            output_path: 输出路径
            class_names: 类别名称列表
            naming_config: 命名策略 (default/timestamp/custom)

        Returns:
            实际保存的文件路径
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 生成文件名
        filename = self._generate_filename(output_path.stem, naming_config)
        final_path = output_path.parent / f"{filename}.geojson"

        features = []

        # 根据任务类型导出
        if merged_result.merged_boxes is not None:
            # 导出检测框
            features.extend(self._export_boxes(
                merged_result.merged_boxes,
                merged_result.class_ids,
                merged_result.confidences,
                class_names
            ))

        # 导出掩膜（优先级：矢量多边形 > instance_id_map > merged_masks）
        logger.info(f"[GeoJSON导出] merged_polygons={merged_result.merged_polygons is not None}, "
                    f"class_ids={merged_result.class_ids is not None}, "
                    f"instance_id_map={merged_result.instance_id_map is not None}")

        if merged_result.merged_polygons is not None and merged_result.class_ids is not None:
            # 【快速路径】直接使用合并后的矢量多边形
            logger.info(f"[GeoJSON导出] 使用矢量多边形路径，多边形数量={len(merged_result.merged_polygons)}")
            features.extend(self._export_polygons(
                merged_result.merged_polygons,
                merged_result.class_ids,
                merged_result.confidences,
                class_names,
                merged_result.debug_polygons_info  # 【调试】传递调试信息
            ))
        elif merged_result.instance_id_map is not None and merged_result.class_ids is not None:
            # 从 instance_id_map 导出掩膜（使用并行处理）
            features.extend(self._export_masks_from_instance_map(
                merged_result.instance_id_map,
                merged_result.class_ids,
                merged_result.confidences,
                class_names,
                use_parallel=True,      # 启用并行
                parallel_threshold=50  # 50个实例以上使用并行
            ))
        elif merged_result.merged_masks is not None:
            # 导出掩膜数组（非并行，作为备选）
            features.extend(self._export_masks(
                merged_result.merged_masks,
                merged_result.class_ids,
                merged_result.confidences,
                class_names
            ))

        # 创建GeoJSON
        geojson = {
            "type": "FeatureCollection",
            "crs": {
                "type": "name",
                "properties": {"name": self.crs}
            },
            "features": features
        }

        # 保存文件
        with open(final_path, 'w', encoding='utf-8') as f:
            json.dump(geojson, f, ensure_ascii=False, indent=2)

        logger.info(f"Exported GeoJSON to {final_path} with {len(features)} features")

        return final_path

    def _export_boxes(
        self,
        boxes: np.ndarray,
        class_ids: np.ndarray,
        confidences: np.ndarray,
        class_names: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        导出检测框为GeoJSON Features

        Args:
            boxes: [N, 6] (x1, y1, x2, y2, conf, cls)
            class_ids: [N] 类别ID
            confidences: [N] 置信度
            class_names: 类别名称列表

        Returns:
            Feature列表
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
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [geo_coords]
                },
                "properties": {
                    "id": i + 1,
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "bbox_pixel": [float(x1), float(y1), float(x2), float(y2)]
                }
            }

            features.append(feature)

        return features

    def _export_masks(
        self,
        masks: np.ndarray,
        class_ids: np.ndarray,
        confidences: np.ndarray,
        class_names: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        导出掩膜为GeoJSON Features

        Args:
            masks: [N, H, W] 掩膜数组
            class_ids: [N] 类别ID
            confidences: [N] 置信度
            class_names: 类别名称列表

        Returns:
            Feature列表
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
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [geo_coords]
                },
                "properties": {
                    "id": i + 1,
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "area_pixels": float(np.sum(mask))
                }
            }

            features.append(feature)

        return features

    def _export_polygons(
        self,
        polygons: List[np.ndarray],
        class_ids: np.ndarray,
        confidences: np.ndarray,
        class_names: Optional[List[str]] = None,
        debug_info: List[dict] = None  # 【新增】调试信息
    ) -> List[Dict]:
        """
        导出矢量多边形为GeoJSON Features（快速路径）

        直接使用已合并的矢量多边形，无需重新提取轮廓。

        Args:
            polygons: 多边形列表，每个是 [N, 2] 的顶点坐标数组（像素坐标）
            class_ids: [N] 类别ID
            confidences: [N] 置信度
            class_names: 类别名称列表
            debug_info: 调试信息列表（可选）

        Returns:
            Feature列表
        """
        features = []

        for i, polygon in enumerate(polygons):
            if polygon is None or len(polygon) < 3:
                continue

            # 转换为地理坐标
            geo_coords = []
            for point in polygon:
                x, y = float(point[0]), float(point[1])  # 确保是 Python float
                geo_x, geo_y = self.pixel_to_geo(x, y)
                geo_coords.append([geo_x, geo_y])

            # 闭合多边形
            if len(geo_coords) > 2:
                geo_coords.append(geo_coords[0])

            class_id = int(class_ids[i]) if i < len(class_ids) else 0
            class_name = class_names[class_id] if class_names and class_id < len(class_names) else f"class_{class_id}"
            confidence = float(confidences[i]) if i < len(confidences) else 0.0

            # 计算面积（使用Shapely）
            from shapely.geometry import Polygon
            try:
                poly = Polygon(polygon)
                area = float(poly.area)
            except Exception:
                area = 0.0

            # 【调试】获取调试信息
            debug_props = {}
            if debug_info and i < len(debug_info):
                info = debug_info[i]
                if 'source_ids' in info:
                    # 确保所有ID都是字符串
                    source_ids_str = [str(sid) for sid in info['source_ids']]
                    debug_props['source_ids'] = ','.join(source_ids_str)
                if 'source_count' in info:
                    debug_props['source_count'] = int(info['source_count'])
                if 'was_merged' in info:
                    debug_props['was_merged'] = bool(info['was_merged'])
                if 'debug_area' in info and info['debug_area'] is not None:
                    debug_props['debug_area'] = float(info['debug_area'])

            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [geo_coords]
                },
                "properties": {
                    "id": i + 1,
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "area_pixels": area
                }
            }

            # 【调试】添加调试字段到properties
            feature["properties"].update(debug_props)

            features.append(feature)

        return features

    def _export_masks_from_instance_map(
        self,
        instance_id_map: np.ndarray,
        class_ids: np.ndarray,
        confidences: np.ndarray,
        class_names: Optional[List[str]] = None,
        use_parallel: bool = True,
        parallel_threshold: int = 100
    ) -> List[Dict]:
        """
        从实例ID映射图导出掩膜为GeoJSON Features（支持并行处理）

        Args:
            instance_id_map: [H, W] 实例ID映射图
            class_ids: [N] 每个实例的类别ID
            confidences: [N] 每个实例的置信度
            class_names: 类别名称列表
            use_parallel: 是否使用并行处理
            parallel_threshold: 启用并行的最小实例数

        Returns:
            Feature列表
        """
        unique_ids = np.unique(instance_id_map)
        unique_ids = unique_ids[unique_ids > 0]  # 排除背景(0)
        n_instances = len(unique_ids)

        logger.info(f"导出 {n_instances} 个实例到GeoJSON")

        # 决定是否使用并行处理
        should_parallel = (
            use_parallel and
            n_instances >= parallel_threshold and
            os.cpu_count() > 1
        )

        if should_parallel:
            return self._export_parallel(unique_ids, instance_id_map, class_ids, confidences, class_names)
        else:
            return self._export_sequential(unique_ids, instance_id_map, class_ids, confidences, class_names)

    def _export_sequential(
        self,
        unique_ids: np.ndarray,
        instance_id_map: np.ndarray,
        class_ids: np.ndarray,
        confidences: np.ndarray,
        class_names: Optional[List[str]] = None
    ) -> List[Dict]:
        """顺序导出（原始逻辑）"""
        from skimage import measure

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
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [geo_coords]
                },
                "properties": {
                    "id": int(instance_id),
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "area_pixels": float(np.sum(mask))
                }
            }

            features.append(feature)

        return features

    def _export_parallel(
        self,
        unique_ids: np.ndarray,
        instance_id_map: np.ndarray,
        class_ids: np.ndarray,
        confidences: np.ndarray,
        class_names: Optional[List[str]] = None
    ) -> List[Dict]:
        """并行导出"""
        import time
        start_time = time.time()

        # 【动态设置】根据 CPU 核心数确定进程数
        cpu_count = os.cpu_count() or 1
        # 使用 CPU 核心数的 75%，最少 2 个，最多 16 个
        n_workers = max(2, min(int(cpu_count * 0.75), 16))

        # 确保不会创建过多进程
        if n_workers > len(unique_ids) // 10:
            # 如果实例数不多，减少进程数
            n_workers = max(1, len(unique_ids) // 50)

        chunk_size = max(10, len(unique_ids) // n_workers)

        # 分块
        chunks = []
        for i in range(0, len(unique_ids), chunk_size):
            chunk = unique_ids[i:i+chunk_size]
            chunk_args = (
                chunk,
                instance_id_map,
                self.geotransform,
                class_ids,
                confidences
            )
            chunks.append(chunk_args)

        logger.info(f"使用 {n_workers} 个进程并行导出 {len(unique_ids)} 个实例 (块大小: {chunk_size})")

        # 并行处理
        try:
            with multiprocessing.Pool(processes=n_workers) as pool:
                features_chunks = pool.map(_export_instances_parallel, chunks)

            # 展平结果
            features = []
            for chunk_features in features_chunks:
                features.extend(chunk_features)

            # 添加类别名称
            if class_names:
                for feature in features:
                    class_id = feature["properties"]["class_id"]
                    if class_id < len(class_names):
                        feature["properties"]["class_name"] = class_names[class_id]
                    else:
                        feature["properties"]["class_name"] = f"class_{class_id}"

        except Exception as e:
            logger.warning(f"并行导出失败，降级到顺序处理: {e}")
            features = self._export_sequential(unique_ids, instance_id_map, class_ids, confidences, class_names)

        elapsed = time.time() - start_time
        logger.info(f"并行导出完成，耗时: {elapsed:.2f}s")

        return features

    def _generate_filename(self, base_name: str, naming_config: str) -> str:
        """
        生成文件名

        Args:
            base_name: 基础名称
            naming_config: 命名策略

        Returns:
            文件名（不含扩展名）
        """
        if naming_config == "timestamp":
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return f"{base_name}_{timestamp}"
        elif naming_config == "custom":
            # 可以从外部传入自定义名称
            return base_name
        else:  # default
            return base_name

    def export_instance_map(
        self,
        merged_result: MergedResult,
        output_path: str
    ) -> Path:
        """
        导出实例ID映射图为图像

        Args:
            merged_result: 合并后的结果
            output_path: 输出路径

        Returns:
            实际保存的文件路径
        """
        if merged_result.instance_id_map is None:
            logger.warning("No instance ID map to export")
            return None

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        from PIL import Image
        import matplotlib.cm as cm

        # 归一化到0-255
        id_map = merged_result.instance_id_map
        max_id = np.max(id_map)

        if max_id > 0:
            # 使用颜色映射
            colored = cm.tab20(id_map / max(max_id, 1))
            colored = (colored[:, :, :3] * 255).astype(np.uint8)
        else:
            colored = np.zeros((id_map.shape[0], id_map.shape[1], 3), dtype=np.uint8)

        Image.fromarray(colored).save(output_path)

        logger.info(f"Exported instance ID map to {output_path}")

        return output_path


# ============================================================================
# 并行处理辅助函数
# ============================================================================

def _process_single_instance(args):
    """
    处理单个实例的轮廓提取（worker函数）

    这个函数必须是模块级函数，以便被 multiprocessing 序列化。

    Args:
        args: (instance_id, instance_id_map, geotransform, idx, class_id, confidence)

    Returns:
        Feature 字典或 None
    """
    instance_id, instance_id_map, geotransform, idx, class_id, confidence = args

    try:
        # 创建二值掩膜
        mask = (instance_id_map == instance_id).astype(np.uint8)

        # 检查掩膜是否有效
        if not np.any(mask):
            return None

        from skimage import measure

        # 找到轮廓
        contours = measure.find_contours(mask, 0.5)

        if not contours:
            return None

        # 选择最长的轮廓
        longest_contour = max(contours, key=len)

        # 简化轮廓（减少点数）
        if len(longest_contour) > 100:
            from skimage.measure import approximate_polygon
            longest_contour = approximate_polygon(longest_contour, tolerance=1.0)

        # 转换为地理坐标
        ul_x, x_res, x_rot, ul_y, y_rot, y_res = geotransform

        geo_coords = []
        for point in longest_contour:
            y, x = point
            geo_x = ul_x + x * x_res + y * x_rot
            geo_y = ul_y + x * y_rot + y * y_res
            geo_coords.append([geo_x, geo_y])

        # 闭合多边形
        if len(geo_coords) > 2:
            geo_coords.append(geo_coords[0])

        # 构建Feature
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [geo_coords]
            },
            "properties": {
                "id": int(instance_id),
                "class_id": class_id,
                "confidence": confidence,
                "area_pixels": float(np.sum(mask))
            }
        }

        return feature

    except Exception as e:
        logger.warning(f"处理实例 {instance_id} 时出错: {e}")
        return None


def _export_instances_parallel(args):
    """
    并行处理一批实例（worker函数）

    Args:
        args: (instance_ids, instance_id_map, geotransform, class_ids, confidences)

    Returns:
        Feature列表
    """
    instance_ids, instance_id_map, geotransform, class_ids, confidences = args

    features = []
    for instance_id in instance_ids:
        idx = int(instance_id) - 1

        # 获取类别和置信度
        if idx < len(class_ids):
            class_id = int(class_ids[idx])
        else:
            class_id = 0

        if idx < len(confidences):
            confidence = float(confidences[idx])
        else:
            confidence = 0.0

        # 处理单个实例
        feature = _process_single_instance(
            (instance_id, instance_id_map, geotransform, idx, class_id, confidence)
        )

        if feature is not None:
            features.append(feature)

    return features
