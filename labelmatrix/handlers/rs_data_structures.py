# -*- coding: utf-8 -*-
"""
遥感预测数据结构
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Any
import numpy as np


@dataclass
class TileWithGeoRef:
    """带地理参考信息的影像分块"""
    tile_id: int                          # 分块ID
    tile_array: np.ndarray                # 分块影像数组 (H, W, C)
    offset: Tuple[int, int]               # 在全图中的偏移 (row_offset, col_offset)
    shape: Tuple[int, int]                # 分块尺寸 (height, width)
    geotransform: tuple                   # 该分块的仿射变换参数
    crs: str                              # 坐标系
    grid_row: int = 0                     # 网格行位置
    grid_col: int = 0                     # 网格列位置


@dataclass
class TileResult:
    """单分块预测结果"""
    tile_id: int                          # 分块ID
    offset: Tuple[int, int]               # 在全图中的偏移
    shape: Tuple[int, int]                # 分块尺寸 (height, width)
    geotransform: tuple                   # 地理变换参数
    # 原始预测结果
    boxes: Optional[np.ndarray] = None    # [N, 6] (x1,y1,x2,y2,conf,cls) - 全局坐标
    masks: Optional[np.ndarray] = None    # [N, H_tile, W_tile] 或 [H_tile, W_tile]
    polygons: Optional[List[np.ndarray]] = None  # [N] 每个实例的轮廓点数组 [[x1,y1], [x2,y2], ...] - 像素坐标
    polygons_with_info: Optional[List[dict]] = None  # [{'polygon': points, 'conf': x, 'cls': y}, ...]
    keypoints: Optional[np.ndarray] = None  # 关键点
    # 元数据
    class_names: Optional[List[str]] = None
    num_instances: int = 0                # 实例数量


@dataclass
class MergedResult:
    """合并后的结果"""
    # 合并后的结果
    merged_boxes: Optional[np.ndarray] = None    # [N, 6] (x1,y1,x2,y2,conf,cls) - 全局坐标
    merged_masks: Optional[np.ndarray] = None    # [N, H, W] 全局掩膜
    merged_polygons: Optional[List[np.ndarray]] = None  # [N] 每个实例的合并后轮廓点数组
    instance_id_map: Optional[np.ndarray] = None # [H, W] 实例ID映射图 (segment专用，兼容保留)
    class_ids: Optional[np.ndarray] = None       # [N] 每个实例的类别ID
    confidences: Optional[np.ndarray] = None     # [N] 每个实例的置信度
    # 【调试】调试信息
    debug_polygons_info: Optional[List[dict]] = None  # [N] 每个实例的调试信息
    # 地理参考
    geotransform: Optional[tuple] = None
    crs: Optional[str] = None
    img_shape: Optional[Tuple[int, int]] = None  # (height, width)
    # 统计信息
    num_tiles: int = 0
    total_instances: int = 0


@dataclass
class RSPredictConfig:
    """遥感预测配置"""
    # 模型配置
    model_path: str = ""
    task_type: str = "detect"  # detect/segment/obb/classify/pose

    # 分块配置
    tile_size: int = 1024
    overlap: float = 0.1

    # 推理配置
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    device: str = "auto"
    batch_size: int = 1
    imgsz: int = 640

    # 合并配置
    merge_iou_threshold: float = 0.5
    segment_high_threshold: float = 0.7
    segment_low_threshold: float = 0.5
    cross_tile_bbox_iou_threshold: float = 0.2  # 跨分块合并的bbox IoU阈值

    # 输出配置
    output_dir: str = "./output"
    save_tiles: bool = True
    file_naming: str = "default"  # default/timestamp/custom

    # 可视化配置
    visualize: bool = True
    show_conf: bool = True
    show_labels: bool = True
    line_width: int = 2

    def __post_init__(self):
        """配置验证"""
        valid_tasks = ["detect", "segment", "obb", "classify", "pose"]
        if self.task_type not in valid_tasks:
            raise ValueError(f"Invalid task_type: {self.task_type}. Must be one of {valid_tasks}")

        if self.overlap < 0 or self.overlap >= 1:
            raise ValueError(f"Invalid overlap: {self.overlap}. Must be in [0, 1)")

        if self.tile_size <= 0:
            raise ValueError(f"Invalid tile_size: {self.tile_size}. Must be > 0")

        valid_naming = ["default", "timestamp", "custom"]
        if self.file_naming not in valid_naming:
            raise ValueError(f"Invalid file_naming: {self.file_naming}. Must be one of {valid_naming}")
