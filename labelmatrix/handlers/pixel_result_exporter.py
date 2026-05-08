# -*- coding: utf-8 -*-
"""
像素坐标结果导出器
用于输出分块预测的像素坐标结果（第一步识别结果）
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

from ..handlers.rs_data_structures import TileResult

logger = logging.getLogger(__name__)


class PixelResultExporter:
    """像素坐标结果导出器

    将分块预测的原始结果导出为像素坐标格式，
    用于调试和验证第一步识别结果。
    """

    def __init__(self, img_shape: tuple):
        """
        Args:
            img_shape: 原始影像尺寸 (height, width)
        """
        self.img_shape = img_shape

    def export_tile_results(
        self,
        tile_results: List[TileResult],
        output_path: str,
        class_names: List[str] = None
    ) -> Path:
        """
        导出所有分块的预测结果（像素坐标）

        Args:
            tile_results: 分块结果列表
            output_path: 输出文件路径
            class_names: 类别名称列表

        Returns:
            实际保存的文件路径
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        results = {
            "image_shape": {"height": self.img_shape[0], "width": self.img_shape[1]},
            "num_tiles": len(tile_results),
            "tiles": []
        }

        for tile_result in tile_results:
            tile_info = self._format_tile_result(tile_result, class_names)
            results["tiles"].append(tile_info)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info(f"Exported pixel coordinate results to {output_path}")

        return output_path

    def _format_tile_result(
        self,
        tile_result: TileResult,
        class_names: List[str] = None
    ) -> Dict[str, Any]:
        """格式化单个分块结果"""
        tile_info = {
            "tile_id": tile_result.tile_id,
            "offset": {"row": int(tile_result.offset[0]), "col": int(tile_result.offset[1])},
            "shape": {"height": int(tile_result.shape[0]), "width": int(tile_result.shape[1])},
            "num_instances": tile_result.num_instances,
            "predictions": []
        }

        # 添加检测框（像素坐标）
        if tile_result.boxes is not None and len(tile_result.boxes) > 0:
            for i, box in enumerate(tile_result.boxes):
                x1, y1, x2, y2 = box[:4]
                conf = float(box[4]) if len(box) > 4 else 0.0
                cls = int(box[5]) if len(box) > 5 else 0

                class_name = class_names[cls] if class_names and cls < len(class_names) else f"class_{cls}"

                prediction = {
                    "type": "bbox",
                    "id": i,
                    "bbox_pixel": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": conf,
                    "class_id": cls,
                    "class_name": class_name
                }
                tile_info["predictions"].append(prediction)

        # 添加掩膜信息（像素坐标）
        if tile_result.masks is not None:
            if tile_result.masks.ndim == 3:  # [N, H, W]
                for i in range(tile_result.masks.shape[0]):
                    mask = tile_result.masks[i]

                    # 获取掩膜的边界框（像素坐标）
                    rows = np.any(mask, axis=1)
                    cols = np.any(mask, axis=0)
                    if np.any(rows) and np.any(cols):
                        rmin, rmax = np.where(rows)[0][[0, -1]]
                        cmin, cmax = np.where(cols)[0][[0, -1]]

                        # 转换为全局像素坐标
                        global_rmin = int(rmin + tile_result.offset[0])
                        global_cmin = int(cmin + tile_result.offset[1])
                        global_rmax = int(rmax + tile_result.offset[0])
                        global_cmax = int(cmax + tile_result.offset[1])

                        # 获取置信度和类别
                        conf = 0.5
                        cls = 0
                        if tile_result.boxes is not None and i < len(tile_result.boxes):
                            conf = float(tile_result.boxes[i, 4])
                            cls = int(tile_result.boxes[i, 5])

                        class_name = class_names[cls] if class_names and cls < len(class_names) else f"class_{cls}"

                        prediction = {
                            "type": "mask",
                            "id": i,
                            "bbox_pixel": [global_cmin, global_rmin, global_cmax, global_rmax],
                            "tile_offset": [int(tile_result.offset[0]), int(tile_result.offset[1])],
                            "tile_bbox": [int(cmin), int(rmin), int(cmax), int(rmax)],
                            "area_pixels": int(np.sum(mask)),
                            "confidence": conf,
                            "class_id": cls,
                            "class_name": class_name
                        }
                        tile_info["predictions"].append(prediction)

        # 添加多边形信息（像素坐标）
        logger.info(f"[像素导出] tile_id={tile_result.tile_id}, polygons={tile_result.polygons is not None}, "
                    f"num_polygons={len(tile_result.polygons) if tile_result.polygons is not None else 0}")

        if tile_result.polygons is not None and len(tile_result.polygons) > 0:
            for i, polygon in enumerate(tile_result.polygons):
                if polygon is None or len(polygon) < 3:
                    continue

                # 获取置信度和类别
                conf = 0.5
                cls = 0
                if tile_result.boxes is not None and i < len(tile_result.boxes):
                    conf = float(tile_result.boxes[i, 4])
                    cls = int(tile_result.boxes[i, 5])

                class_name = class_names[cls] if class_names and cls < len(class_names) else f"class_{cls}"

                # 转换多边形坐标为列表
                polygon_coords = [[float(p[0]), float(p[1])] for p in polygon]

                prediction = {
                    "type": "polygon",
                    "id": i,
                    "polygon_pixel": polygon_coords,
                    "tile_id": tile_result.tile_id,
                    "tile_offset": [int(tile_result.offset[0]), int(tile_result.offset[1])],
                    "confidence": conf,
                    "class_id": cls,
                    "class_name": class_name
                }
                tile_info["predictions"].append(prediction)

        return tile_info

    def export_polygons_before_merge(
        self,
        tile_results: List[TileResult],
        output_path: str,
        class_names: List[str] = None
    ) -> Path:
        """
        导出合并前的多边形为 GeoJSON（用于调试）

        Args:
            tile_results: 分块结果列表
            output_path: 输出文件路径
            class_names: 类别名称列表

        Returns:
            实际保存的文件路径
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        features = []
        polygon_id = 0

        for tile_result in tile_results:
            # 【修复】优先使用 polygons_with_info，它包含正确的 confidence 和 cls
            if tile_result.polygons_with_info is not None and len(tile_result.polygons_with_info) > 0:
                for poly_info in tile_result.polygons_with_info:
                    polygon = poly_info['polygon']
                    if polygon is None or len(polygon) < 3:
                        continue

                    # 从 polygons_with_info 获取置信度和类别（正确的）
                    conf = float(poly_info['conf'])
                    cls = int(poly_info['cls'])

                    class_name = class_names[cls] if class_names and cls < len(class_names) else f"class_{cls}"

                    # 转换多边形坐标为 GeoJSON 格式
                    coords = [[float(p[0]), float(p[1])] for p in polygon]
                    # 闭合环
                    if len(coords) > 0 and coords[0] != coords[-1]:
                        coords.append(coords[0])

                    feature = {
                        "type": "Feature",
                        "properties": {
                            "id": polygon_id,
                            "tile_id": tile_result.tile_id,
                            "tile_offset": [int(tile_result.offset[0]), int(tile_result.offset[1])],
                            "confidence": conf,
                            "class_id": cls,
                            "class_name": class_name
                        },
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [coords]
                        }
                    }
                    features.append(feature)
                    polygon_id += 1
            # 降级：使用 polygons 字段（向后兼容）
            elif tile_result.polygons is not None and len(tile_result.polygons) > 0:
                for i, polygon in enumerate(tile_result.polygons):
                    if polygon is None or len(polygon) < 3:
                        continue

                    # 获取置信度和类别
                    conf = 0.5
                    cls = 0
                    if tile_result.boxes is not None and i < len(tile_result.boxes):
                        conf = float(tile_result.boxes[i, 4])
                        cls = int(tile_result.boxes[i, 5])

                    class_name = class_names[cls] if class_names and cls < len(class_names) else f"class_{cls}"

                    # 转换多边形坐标为 GeoJSON 格式
                    coords = [[float(p[0]), float(p[1])] for p in polygon]
                    # 闭合环
                    if len(coords) > 0 and coords[0] != coords[-1]:
                        coords.append(coords[0])

                    feature = {
                        "type": "Feature",
                        "properties": {
                            "id": polygon_id,
                            "tile_id": tile_result.tile_id,
                            "tile_offset": [int(tile_result.offset[0]), int(tile_result.offset[1])],
                            "confidence": conf,
                            "class_id": cls,
                            "class_name": class_name
                        },
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [coords]
                        }
                    }
                    features.append(feature)
                    polygon_id += 1

        geojson = {
            "type": "FeatureCollection",
            "crs": {
                "type": "name",
                "properties": {"name": "EPSG:4326"}  # 像素坐标，假设为 WGS84
            },
            "features": features
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(geojson, f, ensure_ascii=False, indent=2)

        logger.info(f"Exported {polygon_id} polygons before merge to {output_path}")
        return output_path

    def export_combined_pixel_results(
        self,
        tile_results: List[TileResult],
        output_path: str,
        class_names: List[str] = None
    ) -> Path:
        """
        导出合并后的像素坐标结果（未做NMS的所有检测结果）

        Args:
            tile_results: 分块结果列表
            output_path: 输出文件路径
            class_names: 类别名称列表

        Returns:
            实际保存的文件路径
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        all_predictions = []

        for tile_result in tile_results:
            if tile_result.boxes is not None and len(tile_result.boxes) > 0:
                for i, box in enumerate(tile_result.boxes):
                    x1, y1, x2, y2 = box[:4]
                    conf = float(box[4]) if len(box) > 4 else 0.0
                    cls = int(box[5]) if len(box) > 5 else 0

                    class_name = class_names[cls] if class_names and cls < len(class_names) else f"class_{cls}"

                    prediction = {
                        "id": len(all_predictions),
                        "bbox_pixel": [float(x1), float(y1), float(x2), float(y2)],
                        "confidence": conf,
                        "class_id": cls,
                        "class_name": class_name,
                        "tile_id": tile_result.tile_id
                    }
                    all_predictions.append(prediction)

        results = {
            "image_shape": {"height": self.img_shape[0], "width": self.img_shape[1]},
            "total_predictions": len(all_predictions),
            "predictions": all_predictions
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info(f"Exported combined pixel results to {output_path} ({len(all_predictions)} predictions)")

        return output_path
