# -*- coding: utf-8 -*-
"""
结果格式转换器
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional


class ResultConverter:
    """推理结果格式转换器"""

    def __init__(self, crs: str = 'EPSG:4326'):
        """
        Args:
            crs: 坐标参考系
        """
        self.crs = crs

    def yolo_results_to_geojson(
        self,
        yolo_results: List[Any],
        output_path: str,
        class_names: Optional[List[str]] = None,
        img_width: int = 0,
        img_height: int = 0
    ):
        """
        将YOLO检测结果转换为GeoJSON

        Args:
            yolo_results: YOLO推理结果列表
            output_path: 输出GeoJSON路径
            class_names: 类别名称列表
            img_width: 原始影像宽度
            img_height: 原始影像高度
        """
        features = []

        for result in yolo_results:
            if not hasattr(result, 'boxes') or result.boxes is None:
                continue

            boxes = result.boxes
            xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes, 'xyxy') else []
            conf = boxes.conf.cpu().numpy() if hasattr(boxes, 'conf') else []
            cls = boxes.cls.cpu().numpy().astype(int) if hasattr(boxes, 'cls') else []

            for i in range(len(xyxy)):
                # 获取边界框坐标
                x_min, y_min, x_max, y_max = xyxy[i]

                # 创建简单的Polygon（像素坐标）
                # 实际使用中需要转换到地理坐标
                polygon = self._create_polygon(x_min, y_min, x_max, y_max)

                class_id = int(cls[i]) if i < len(cls) else 0
                class_name = class_names[class_id] if class_names and class_id < len(class_names) else f'class_{class_id}'
                confidence = float(conf[i]) if i < len(conf) else 0.0

                feature = {
                    'type': 'Feature',
                    'geometry': polygon,
                    'properties': {
                        'class_id': class_id,
                        'class_name': class_name,
                        'confidence': confidence,
                        'bbox': [float(x_min), float(y_min), float(x_max), float(y_max)]
                    }
                }

                features.append(feature)

        # 创建GeoJSON
        geojson = {
            'type': 'FeatureCollection',
            'crs': {
                'type': 'name',
                'properties': {'name': self.crs}
            },
            'features': features
        }

        # 保存
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(geojson, f, ensure_ascii=False, indent=2)

        return output_path

    def _create_polygon(self, x_min: float, y_min: float, x_max: float, y_max: float) -> Dict:
        """创建GeoJSON Polygon"""
        return {
            'type': 'Polygon',
            'coordinates': [[
                [x_min, y_min],
                [x_max, y_min],
                [x_max, y_max],
                [x_min, y_max],
                [x_min, y_min]
            ]]
        }

    def mask_to_geojson(
        self,
        mask: np.ndarray,
        output_path: str,
        class_names: Optional[List[str]] = None,
        min_area: float = 100
    ):
        """
        将分割掩膜转换为GeoJSON

        Args:
            mask: 分割掩膜(H, W)
            output_path: 输出路径
            class_names: 类别名称列表
            min_area: 最小面积（像素平方）
        """
        from skimage import measure

        features = []

        # 对每个类别进行处理
        unique_classes = np.unique(mask)
        unique_classes = unique_classes[unique_classes > 0]  # 跳过背景(0)

        for class_id in unique_classes:
            # 创建二值掩膜
            class_mask = (mask == class_id).astype(np.uint8)

            # 找到轮廓
            contours = measure.find_contours(class_mask, 0.5)

            for contour in contours:
                # 转换为多边形
                if len(contour) < 3:
                    continue

                # 简化为多边形坐标
                coords = contour[:, [1, 0]].tolist()  # 转换为 (x, y) 格式

                # 计算面积（简化）
                if len(coords) < 3:
                    continue

                from shapely.geometry import Polygon
                polygon = Polygon(coords)

                # 过滤小面积
                if polygon.is_empty or polygon.area < min_area:
                    continue

                class_name = class_names[class_id] if class_names and class_id < len(class_names) else f'class_{class_id}'

                feature = {
                    'type': 'Feature',
                    'geometry': polygon.__geo_interface__,
                    'properties': {
                        'class_id': int(class_id),
                        'class_name': class_name,
                        'area': float(polygon.area)
                    }
                }

                features.append(feature)

        # 创建GeoJSON
        geojson = {
            'type': 'FeatureCollection',
            'crs': {
                'type': 'name',
                'properties': {'name': self.crs}
            },
            'features': features
        }

        # 保存
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(geojson, f, ensure_ascii=False, indent=2)

        return output_path

    def save_mask_as_image(
        self,
        mask: np.ndarray,
        output_path: str
    ):
        """
        保存掩膜为图像

        Args:
            mask: 掩膜数组
            output_path: 输出路径
        """
        from PIL import Image

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 转换为uint8
        if mask.dtype != np.uint8:
            mask = (mask * 255).astype(np.uint8)

        # 保存
        Image.fromarray(mask).save(output_path)

        return output_path

    def convert_yolo_results_to_simple_format(
        self,
        yolo_results: List[Any],
        output_dir: str
    ) -> List[Path]:
        """
        将YOLO结果转换为简单格式保存

        Args:
            yolo_results: YOLO推理结果
            output_dir: 输出目录

        Returns:
            保存的文件路径列表
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_files = []

        # 保存检测结果为JSON
        results_data = []

        for i, result in enumerate(yolo_results):
            if not hasattr(result, 'boxes') or result.boxes is None:
                continue

            boxes = result.boxes

            item = {
                'image_id': i,
                'boxes': []
            }

            if hasattr(boxes, 'xyxy'):
                xyxy = boxes.xyxy.cpu().numpy()
                conf = boxes.conf.cpu().numpy() if hasattr(boxes, 'conf') else []
                cls = boxes.cls.cpu().numpy().astype(int) if hasattr(boxes, 'cls') else []

                for j in range(len(xyxy)):
                    box_data = {
                        'bbox': xyxy[j].tolist(),
                        'confidence': float(conf[j]) if j < len(conf) else 0.0,
                        'class_id': int(cls[j]) if j < len(cls) else 0
                    }
                    item['boxes'].append(box_data)

            results_data.append(item)

        # 保存JSON
        json_file = output_dir / 'results.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2)

        saved_files.append(json_file)

        return saved_files
