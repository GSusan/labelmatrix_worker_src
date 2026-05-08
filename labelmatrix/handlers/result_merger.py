# -*- coding: utf-8 -*-
"""
分块预测结果合并器
"""

import logging
import multiprocessing
import os
import itertools
from typing import List, Optional, Dict, Set, Tuple
import numpy as np
from scipy import ndimage
from collections import defaultdict

# 尝试导入 numba 进行加速（可选）
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # 创建虚拟装饰器
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def prange(x):
        return range(x)

from ..handlers.rs_data_structures import TileResult, MergedResult
from ..exceptions.remote_sensing_errors import MergeError

logger = logging.getLogger(__name__)


# Numba 加速的 bbox 计算函数
@njit(cache=True)
def _compute_single_mask_bbox_numba(mask: np.ndarray) -> tuple:
    """
    使用 Numba 加速计算单个掩膜的边界框

    Args:
        mask: 掩膜数组 (H, W)

    Returns:
        (cmin, rmin, cmax, rmax) 局部边界框，如果掩膜为空返回 (-1, -1, -1, -1)
    """
    h, w = mask.shape
    rmin, rmax, cmin, cmax = -1, -1, -1, -1

    # 找行范围
    for i in range(h):
        for j in range(w):
            if mask[i, j] > 0:
                rmin = i
                break
        if rmin >= 0:
            break

    # 如果没有找到任何像素
    if rmin < 0:
        return (-1, -1, -1, -1)

    # 找行结束
    for i in range(h - 1, -1, -1):
        for j in range(w):
            if mask[i, j] > 0:
                rmax = i
                break
        if rmax >= 0:
            break

    # 找列范围
    for j in range(w):
        for i in range(h):
            if mask[i, j] > 0:
                cmin = j
                break
        if cmin >= 0:
            break

    # 找列结束
    for j in range(w - 1, -1, -1):
        for i in range(h):
            if mask[i, j] > 0:
                cmax = j
                break
        if cmax >= 0:
            break

    return (cmin, rmin, cmax, rmax)


# Numba 加速的像素重叠检查函数
@njit(cache=True)
def _has_pixel_overlap_numba(
    mask1: np.ndarray,
    offset1_y: int, offset1_x: int,
    shape1_y: int, shape1_x: int,
    mask2: np.ndarray,
    offset2_y: int, offset2_x: int,
    shape2_y: int, shape2_x: int
) -> bool:
    """
    使用 Numba 加速检查两个掩膜是否有像素重叠

    Args:
        mask1: 掩膜1数组
        offset1_y, offset1_x: 掩膜1的偏移
        shape1_y, shape1_x: 掩膜1的尺寸
        mask2: 掩膜2数组
        offset2_y, offset2_x: 掩膜2的偏移
        shape2_y, shape2_x: 掩膜2的尺寸

    Returns:
        是否有像素重叠
    """
    # 计算重叠区域
    y1_min, y1_max = offset1_y, offset1_y + shape1_y
    x1_min, x1_max = offset1_x, offset1_x + shape1_x
    y2_min, y2_max = offset2_y, offset2_y + shape2_y
    x2_min, x2_max = offset2_x, offset2_x + shape2_x

    overlap_y_min = max(y1_min, y2_min)
    overlap_y_max = min(y1_max, y2_max)
    overlap_x_min = max(x1_min, x2_min)
    overlap_x_max = min(x1_max, x2_max)

    if overlap_x_max <= overlap_x_min or overlap_y_max <= overlap_y_min:
        return False

    # 计算局部坐标
    local_y1 = overlap_y_min - y1_min
    local_x1 = overlap_x_min - x1_min
    local_y2 = overlap_y_min - y2_min
    local_x2 = overlap_x_min - x2_min

    overlap_w = overlap_x_max - overlap_x_min
    overlap_h = overlap_y_max - overlap_y_min

    # 检查是否有像素重叠
    for i in range(overlap_h):
        for j in range(overlap_w):
            val1 = mask1[local_y1 + i, local_x1 + j]
            val2 = mask2[local_y2 + i, local_x2 + j]
            if val1 > 0 and val2 > 0:
                return True

    return False


# Numba 加速的 IoU 计算函数
@njit(cache=True)
def _calculate_mask_iou_numba(mask1: np.ndarray, mask2: np.ndarray,
                               offset1_y: int, offset1_x: int,
                               offset2_y: int, offset2_x: int) -> float:
    """
    使用 Numba 加速计算两个掩膜的 IoU

    Args:
        mask1: 掩膜1数组
        mask2: 掩膜2数组
        offset1_y, offset1_x: 掩膜1的全局偏移
        offset2_y, offset2_x: 掩膜2的全局偏移

    Returns:
        IoU值
    """
    h1, w1 = mask1.shape
    h2, w2 = mask2.shape

    # 计算重叠区域在全局坐标中的范围
    y1_min, y1_max = offset1_y, offset1_y + h1
    x1_min, x1_max = offset1_x, offset1_x + w1
    y2_min, y2_max = offset2_y, offset2_y + h2
    x2_min, x2_max = offset2_x, offset2_x + w2

    # 重叠区域
    overlap_y_min = max(y1_min, y2_min)
    overlap_y_max = min(y1_max, y2_max)
    overlap_x_min = max(x1_min, x2_min)
    overlap_x_max = min(x1_max, x2_max)

    if overlap_x_max <= overlap_x_min or overlap_y_max <= overlap_y_min:
        return 0.0

    # 计算在各自掩膜中的局部坐标
    local_y1 = overlap_y_min - y1_min
    local_x1 = overlap_x_min - x1_min
    local_y2 = overlap_y_min - y2_min
    local_x2 = overlap_x_min - x2_min

    overlap_h = overlap_y_max - overlap_y_min
    overlap_w = overlap_x_max - overlap_x_min

    # 计算IoU
    overlap = 0
    union = 0

    for i in range(overlap_h):
        for j in range(overlap_w):
            val1 = mask1[local_y1 + i, local_x1 + j]
            val2 = mask2[local_y2 + i, local_x2 + j]
            if val1 > 0 and val2 > 0:
                overlap += 1
            if val1 > 0 or val2 > 0:
                union += 1

    if union == 0:
        return 0.0

    return float(overlap) / float(union)


# 批量计算 IoU（向量化版本）
@njit(cache=True, parallel=True)
def _calculate_mask_iou_batch_numba(
    masks1: np.ndarray,
    masks2: np.ndarray,
    offsets1: np.ndarray,
    offsets2: np.ndarray
) -> np.ndarray:
    """
    批量计算多对掩膜的 IoU（并行加速）

    Args:
        masks1: 掩膜组1 [N, H, W]
        masks2: 掩膜组2 [M, H, W]
        offsets1: 偏移1 [N, 2]
        offsets2: 偏移2 [M, 2]

    Returns:
        IoU矩阵 [N, M]
    """
    n = masks1.shape[0]
    m = masks2.shape[0]
    iou_matrix = np.zeros((n, m), dtype=np.float32)

    for i in prange(n):
        for j in range(m):
            mask1 = masks1[i]
            mask2 = masks2[j]
            offset1 = offsets1[i]
            offset2 = offsets2[j]

            h1, w1 = mask1.shape
            h2, w2 = mask2.shape

            y1_min, y1_max = offset1[0], offset1[0] + h1
            x1_min, x1_max = offset1[1], offset1[1] + w1
            y2_min, y2_max = offset2[0], offset2[0] + h2
            x2_min, x2_max = offset2[1], offset2[1] + w2

            overlap_y_min = max(y1_min, y2_min)
            overlap_y_max = min(y1_max, y2_max)
            overlap_x_min = max(x1_min, x2_min)
            overlap_x_max = min(x1_max, x2_max)

            if overlap_x_max <= overlap_x_min or overlap_y_max <= overlap_y_min:
                iou_matrix[i, j] = 0.0
                continue

            local_y1 = overlap_y_min - y1_min
            local_x1 = overlap_x_min - x1_min
            local_y2 = overlap_y_min - y2_min
            local_x2 = overlap_x_min - x2_min

            overlap_h = overlap_y_max - overlap_y_min
            overlap_w = overlap_x_max - overlap_x_min

            overlap = 0
            union = 0

            for yi in range(overlap_h):
                for xj in range(overlap_w):
                    val1 = mask1[local_y1 + yi, local_x1 + xj]
                    val2 = mask2[local_y2 + yi, local_x2 + xj]
                    if val1 > 0 and val2 > 0:
                        overlap += 1
                    if val1 > 0 or val2 > 0:
                        union += 1

            if union == 0:
                iou_matrix[i, j] = 0.0
            else:
                iou_matrix[i, j] = float(overlap) / float(union)

    return iou_matrix


class SpatialGridIndex:
    """空间网格索引，用于加速掩膜重叠检测"""

    def __init__(self, cell_size: int = 512):
        """
        Args:
            cell_size: 网格单元大小，建议设置为分块大小
        """
        self.cell_size = cell_size
        self.grid: Dict[tuple, List[int]] = defaultdict(list)

    def insert(self, idx: int, bbox: tuple):
        """
        插入边界框到索引中

        Args:
            idx: 掩膜索引
            bbox: (x1, y1, x2, y2) 边界框
        """
        x1, y1, x2, y2 = bbox
        # 计算覆盖的网格单元
        cell_x1 = int(x1 // self.cell_size)
        cell_y1 = int(y1 // self.cell_size)
        cell_x2 = int(x2 // self.cell_size)
        cell_y2 = int(y2 // self.cell_size)

        # 添加到所有覆盖的网格单元
        for cy in range(cell_y1, cell_y2 + 1):
            for cx in range(cell_x1, cell_x2 + 1):
                self.grid[(cx, cy)].append(idx)

    def query_potential_overlaps(self) -> Set[tuple]:
        """
        查询所有可能重叠的掩膜对

        Returns:
            可能重叠的掩膜对集合 {(i, j), ...} (保证 i < j)
        """
        pairs = set()
        processed = set()

        for cell, mask_indices in self.grid.items():
            if len(mask_indices) < 2:
                continue

            # 对同一网格内的掩膜进行配对
            for i in range(len(mask_indices)):
                for j in range(i + 1, len(mask_indices)):
                    idx1, idx2 = mask_indices[i], mask_indices[j]
                    # 确保有序对，避免重复
                    pair = (min(idx1, idx2), max(idx1, idx2))
                    if pair not in processed:
                        pairs.add(pair)
                        processed.add(pair)

        return pairs


class ResultMerger:
    """分块预测结果合并器"""

    def __init__(
        self,
        task_type: str,
        iou_threshold: float = 0.5,
        segment_high_threshold: float = 0.7,
        segment_low_threshold: float = 0.5,
        cross_tile_bbox_iou_threshold: float = 0.3,
        tile_processor = None
    ):
        """
        Args:
            task_type: 任务类型 (detect/segment/obb/classify/pose)
            iou_threshold: IoU阈值用于合并
            segment_high_threshold: segment模式高IoU阈值
            segment_low_threshold: segment模式低IoU阈值
            cross_tile_bbox_iou_threshold: 跨分块合并的bbox IoU阈值
            tile_processor: TileProcessor实例，用于计算分块重叠区域
        """
        self.task_type = task_type
        self.iou_threshold = iou_threshold
        self.high_threshold = segment_high_threshold
        self.low_threshold = segment_low_threshold
        self.cross_tile_bbox_iou_threshold = cross_tile_bbox_iou_threshold
        self.tile_processor = tile_processor

    def merge(
        self,
        tile_results: List[TileResult],
        img_shape: tuple,
        geotransform: tuple = None,
        crs: str = None,
        output_dir = None
    ) -> MergedResult:
        """
        合并分块预测结果

        Args:
            tile_results: 分块结果列表
            img_shape: 原始影像尺寸 (H, W)
            geotransform: 仿射变换参数
            crs: 坐标系

        Returns:
            MergedResult: 合并后的结果
        """
        if not tile_results:
            return MergedResult(
                img_shape=img_shape,
                geotransform=geotransform,
                crs=crs,
                num_tiles=0,
                total_instances=0
            )

        logger.info(f"Merging {len(tile_results)} tile results for task: {self.task_type}")

        try:
            if self.task_type in ["detect", "obb"]:
                return self._merge_detection(tile_results, img_shape, geotransform, crs)
            elif self.task_type == "segment":
                return self._merge_segmentation(tile_results, img_shape, geotransform, crs, output_dir)
            elif self.task_type == "classify":
                return self._merge_classification(tile_results, img_shape, geotransform, crs)
            else:
                logger.warning(f"Unsupported task type for merging: {self.task_type}, returning empty result")
                return MergedResult(img_shape=img_shape, geotransform=geotransform, crs=crs)

        except Exception as e:
            raise MergeError(f"Failed to merge results: {str(e)}") from e

    def _merge_detection(
        self,
        tile_results: List[TileResult],
        img_shape: tuple,
        geotransform: tuple = None,
        crs: str = None
    ) -> MergedResult:
        """
        合并检测/旋转框结果（使用全局NMS）

        Args:
            tile_results: 分块结果列表
            img_shape: 原始影像尺寸
            geotransform: 仿射变换参数
            crs: 坐标系

        Returns:
            MergedResult: 合并后的结果
        """
        all_boxes = []

        # 收集所有检测框
        for result in tile_results:
            if result.boxes is not None and len(result.boxes) > 0:
                all_boxes.append(result.boxes)

        if not all_boxes:
            return MergedResult(
                img_shape=img_shape,
                geotransform=geotransform,
                crs=crs,
                num_tiles=len(tile_results),
                total_instances=0
            )

        # 合并所有检测框
        merged_boxes = np.vstack(all_boxes)

        # 执行全局NMS
        final_boxes = self._global_nms(merged_boxes, self.iou_threshold)

        # 提取类别和置信度
        if final_boxes.shape[0] > 0 and final_boxes.shape[1] >= 6:
            class_ids = final_boxes[:, 5].astype(int)
            confidences = final_boxes[:, 4]
        else:
            class_ids = np.array([])
            confidences = np.array([])

        return MergedResult(
            merged_boxes=final_boxes,
            class_ids=class_ids,
            confidences=confidences,
            img_shape=img_shape,
            geotransform=geotransform,
            crs=crs,
            num_tiles=len(tile_results),
            total_instances=len(final_boxes)
        )

    def _global_nms(
        self,
        boxes: np.ndarray,
        iou_threshold: float
    ) -> np.ndarray:
        """
        执行全局非极大值抑制

        Args:
            boxes: [N, 6] (x1, y1, x2, y2, conf, cls)
            iou_threshold: IoU阈值

        Returns:
            保留的检测框
        """
        if len(boxes) == 0:
            return boxes

        # 按置信度降序排序
        conf_order = np.argsort(-boxes[:, 4])
        boxes = boxes[conf_order]

        keep = []
        while len(boxes) > 0:
            # 保留置信度最高的框
            keep.append(0)
            if len(boxes) == 1:
                break

            # 计算IoU
            ious = self._calculate_box_iou(boxes[0:1, :4], boxes[1:, :4])

            # 只保留与当前框IoU小于阈值的框，且不同类别
            same_class = boxes[1:, 5] == boxes[0, 5]
            mask = (ious < iou_threshold) | (~same_class)
            boxes = boxes[1:][mask]

        return boxes[keep]

    def _calculate_box_iou(
        self,
        boxes1: np.ndarray,
        boxes2: np.ndarray
    ) -> np.ndarray:
        """
        计算两组框之间的IoU

        Args:
            boxes1: [N, 4]
            boxes2: [M, 4]

        Returns:
            [N, M] IoU矩阵
        """
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

        lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
        rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]

        wh = (rb - lt).clip(min=0)  # [N, M, 2]
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

        union = area1[:, None] + area2 - inter

        iou = inter / (union + 1e-7)
        return iou

    def _merge_segmentation(
        self,
        tile_results: List[TileResult],
        img_shape: tuple,
        geotransform: tuple = None,
        crs: str = None,
        output_dir = None
    ) -> MergedResult:
        """
        合并分割结果（基于分块重叠区域的合并策略）

        只有跨越分块边界的掩膜才需要合并，同分块内的掩膜保持独立。
        优先使用矢量多边形合并，如果不可用则降级到掩膜合并。

        Args:
            tile_results: 分块结果列表
            img_shape: 原始影像尺寸
            geotransform: 仿射变换参数
            crs: 坐标系

        Returns:
            MergedResult: 合并后的结果
        """
        height, width = img_shape[:2]

        # 【新增】尝试使用矢量多边形合并（快速路径）
        if any(result.polygons is not None for result in tile_results):
            logger.info("使用矢量多边形合并策略")
            return self._merge_polygons_vector(tile_results, img_shape, geotransform, crs, output_dir)

        # 检查是否有 tile_processor 可用
        if self.tile_processor is None:
            # 降级到旧策略
            return self._merge_segmentation_fallback(tile_results, img_shape, geotransform, crs, output_dir)

        # 收集所有掩膜信息（使用正确的网格位置）
        all_masks = []
        for tile_idx, result in enumerate(tile_results):
            # 从 result 中获取网格位置（如果有的话）
            grid_row = getattr(result, 'grid_row', tile_idx // self.tile_processor.grid_cols if self.tile_processor.grid_cols > 0 else 0)
            grid_col = getattr(result, 'grid_col', tile_idx % self.tile_processor.grid_cols if self.tile_processor.grid_cols > 0 else 0)

            if result.masks is not None:
                if result.masks.ndim == 3:  # [N, H, W]
                    for i in range(result.masks.shape[0]):
                        conf = result.boxes[i, 4] if result.boxes is not None and i < len(result.boxes) else 0.5
                        cls = result.boxes[i, 5] if result.boxes is not None and i < len(result.boxes) else 0
                        all_masks.append({
                            'mask': result.masks[i],
                            'offset': result.offset,
                            'shape': result.shape,
                            'conf': conf,
                            'cls': cls,
                            'tile_id': tile_idx,
                            'tile_row': grid_row,
                            'tile_col': grid_col,
                            'tile_offset': result.offset,
                            'tile_shape': result.shape
                        })
                elif result.masks.ndim == 2:  # [H, W]
                    all_masks.append({
                        'mask': result.masks,
                        'offset': result.offset,
                        'shape': result.shape,
                        'conf': 0.5,
                        'cls': 0,
                        'tile_id': tile_idx,
                        'tile_row': grid_row,
                        'tile_col': grid_col,
                        'tile_offset': result.offset,
                        'tile_shape': result.shape
                    })

        if not all_masks:
            return MergedResult(
                img_shape=img_shape,
                geotransform=geotransform,
                crs=crs,
                num_tiles=len(tile_results),
                total_instances=0
            )

        # 执行合并（使用基于重叠区域的策略）
        merged_info = self._merge_masks_by_overlap_region(all_masks, img_shape)

        # 【性能优化】移除 merged_masks 构建，只保留 instance_id_map
        # GeoJSON导出器已支持从 instance_id_map 导出，不需要 merged_masks
        merged_masks = None

        # 构建实例ID映射图
        instance_id_map = self._build_sparse_instance_map(merged_info, img_shape)

        confidences_list = [info['conf'] for info in merged_info]
        class_ids_list = [info['cls'] for info in merged_info]

        return MergedResult(
            merged_masks=merged_masks,
            instance_id_map=instance_id_map,
            class_ids=np.array(class_ids_list),
            confidences=np.array(confidences_list),
            img_shape=img_shape,
            geotransform=geotransform,
            crs=crs,
            num_tiles=len(tile_results),
            total_instances=len(merged_info)
        )

    def _merge_segmentation_fallback(
        self,
        tile_results: List[TileResult],
        img_shape: tuple,
        geotransform: tuple = None,
        crs: str = None,
        output_dir = None
    ) -> MergedResult:
        """
        合并分割结果（降级策略，当没有 tile_processor 时使用）

        Args:
            tile_results: 分块结果列表
            img_shape: 原始影像尺寸
            geotransform: 仿射变换参数
            crs: 坐标系

        Returns:
            MergedResult: 合并后的结果
        """
        height, width = img_shape[:2]

        # 收集所有掩膜信息
        all_masks = []
        for tile_idx, result in enumerate(tile_results):
            if result.masks is not None:
                if result.masks.ndim == 3:
                    for i in range(result.masks.shape[0]):
                        conf = result.boxes[i, 4] if result.boxes is not None and i < len(result.boxes) else 0.5
                        cls = result.boxes[i, 5] if result.boxes is not None and i < len(result.boxes) else 0
                        all_masks.append({
                            'mask': result.masks[i],
                            'offset': result.offset,
                            'shape': result.shape,
                            'conf': conf,
                            'cls': cls,
                            'tile_id': tile_idx,
                            'tile_row': tile_idx // 100,
                            'tile_col': tile_idx % 100
                        })
                elif result.masks.ndim == 2:
                    all_masks.append({
                        'mask': result.masks,
                        'offset': result.offset,
                        'shape': result.shape,
                        'conf': 0.5,
                        'cls': 0,
                        'tile_id': tile_idx,
                        'tile_row': tile_idx // 100,
                        'tile_col': tile_idx % 100
                    })

        if not all_masks:
            return MergedResult(
                img_shape=img_shape,
                geotransform=geotransform,
                crs=crs,
                num_tiles=len(tile_results),
                total_instances=0
            )

        # 使用原有的优化版本
        merged_info = self._merge_masks_with_iou_optimized(all_masks, img_shape)

        # 【性能优化】移除 merged_masks 构建，只保留 instance_id_map
        merged_masks = None

        # 构建实例ID映射图
        instance_id_map = self._build_sparse_instance_map(merged_info, img_shape)

        confidences_list = [info['conf'] for info in merged_info]
        class_ids_list = [info['cls'] for info in merged_info]

        return MergedResult(
            merged_masks=merged_masks,
            instance_id_map=instance_id_map,
            class_ids=np.array(class_ids_list),
            confidences=np.array(confidences_list),
            img_shape=img_shape,
            geotransform=geotransform,
            crs=crs,
            num_tiles=len(tile_results),
            total_instances=len(merged_info)
        )

    def _build_sparse_instance_map(self, merged_info: List[dict], img_shape: tuple) -> np.ndarray:
        """
        构建稀疏实例ID映射图（内存优化，适配局部掩膜）

        使用稀疏存储：只分配实际需要的边界框区域

        Args:
            merged_info: 合并后的掩膜信息（可能包含 mask_local 或 mask_full）
            img_shape: 影像尺寸

        Returns:
            稀疏实例ID映射图
        """
        height, width = img_shape[:2]

        # 如果实例数量很少或图像很大，使用稀疏表示
        num_instances = len(merged_info)
        if num_instances > 100 and height * width > 10000000:  # 大于1000万像素
            # 使用更紧凑的表示
            return self._build_compact_instance_map(merged_info, img_shape)

        # 对于小规模数据，创建完整的映射图
        instance_id_map = np.zeros((height, width), dtype=np.int32)
        for idx, info in enumerate(merged_info, start=1):
            # 【优化】优先使用 mask_local，如果没有则使用 mask_full
            if 'mask_local' in info and info['mask_local'] is not None:
                mask_local = info['mask_local']
                offset = info['offset']
                shape = info['shape']
                # 将局部掩膜放置到全局映射图中
                instance_id_map[offset[0]:offset[0]+shape[0], offset[1]:offset[1]+shape[1]][mask_local > 0] = idx
            elif 'mask_full' in info and info['mask_full'] is not None:
                mask_full = info['mask_full']
                instance_id_map[mask_full > 0] = idx

        return instance_id_map

    def _merge_polygons_vector(
        self,
        tile_results: List[TileResult],
        img_shape: tuple,
        geotransform: tuple = None,
        crs: str = None,
        output_dir = None
    ) -> MergedResult:
        """
        使用矢量多边形合并策略（快速路径）

        直接合并YOLO输出的矢量多边形，避免栅格化再重矢量化的开销。

        Args:
            tile_results: 分块结果列表
            img_shape: 原始影像尺寸
            geotransform: 仿射变换参数
            crs: 坐标系

        Returns:
            MergedResult: 合并后的结果
        """
        import time
        start_time = time.time()

        # 收集所有多边形
        all_polygons = []

        for tile_idx, result in enumerate(tile_results):
            # 优先使用 polygons_with_info（包含conf和cls）
            if result.polygons_with_info is not None and len(result.polygons_with_info) > 0:
                for poly_info in result.polygons_with_info:
                    polygon = poly_info['polygon']
                    if polygon is None or len(polygon) < 3:
                        continue

                    # 计算边界框
                    x_coords = polygon[:, 0]
                    y_coords = polygon[:, 1]
                    bbox = (float(np.min(x_coords)), float(np.min(y_coords)),
                           float(np.max(x_coords)), float(np.max(y_coords)))

                    all_polygons.append({
                        'polygon': polygon,
                        'conf': poly_info['conf'],
                        'cls': poly_info['cls'],
                        'tile_id': tile_idx,
                        'bbox': bbox
                    })
            # 降级：使用 polygons 字段
            elif result.polygons is not None and len(result.polygons) > 0:
                # 获取类别和置信度
                if result.boxes is not None:
                    confidences = result.boxes[:, 4]
                    class_ids = result.boxes[:, 5]
                else:
                    confidences = [0.5] * len(result.polygons)
                    class_ids = [0] * len(result.polygons)

                for i, polygon in enumerate(result.polygons):
                    if polygon is None or len(polygon) < 3:
                        continue

                    # 【调试】检查接收到的多边形数据
                    if tile_idx == 0 and i < 3:
                        logger.info(f"[DEBUG] 收集多边形: 分块{tile_idx}, 多边形{i}, shape={polygon.shape}, "
                                   f"x范围=[{polygon[:,0].min():.1f}, {polygon[:,0].max():.1f}], "
                                   f"y范围=[{polygon[:,1].min():.1f}, {polygon[:,1].max():.1f}]")

                    # 计算边界框
                    x_coords = polygon[:, 0]
                    y_coords = polygon[:, 1]
                    bbox = (float(np.min(x_coords)), float(np.min(y_coords)),
                           float(np.max(x_coords)), float(np.max(y_coords)))

                    all_polygons.append({
                        'polygon': polygon,
                        'conf': float(confidences[i]) if i < len(confidences) else 0.5,
                        'cls': int(class_ids[i]) if i < len(class_ids) else 0,
                        'tile_id': tile_idx,
                        'bbox': bbox,
                        # 【调试】添加唯一ID，用于追踪
                        'source_id': f"{tile_idx}_{i}"  # 格式: "分块ID_多边形序号"
                    })

        if not all_polygons:
            return MergedResult(
                img_shape=img_shape,
                geotransform=geotransform,
                crs=crs,
                num_tiles=len(tile_results),
                total_instances=0
            )

        logger.info(f"收集到 {len(all_polygons)} 个多边形，开始矢量合并...")

        # 执行矢量合并
        merged_polygons = self._merge_polygons_by_iou(all_polygons, output_dir)

        # 简化多边形（减少顶点数，平滑边界）
        final_polygons = []
        for poly_info in merged_polygons:
            # 统一使用0.5容差
            tolerance = 0.1
            simplified = _simplify_polygon(poly_info['polygon'], tolerance=tolerance)

            # 计算简化后的面积
            from shapely.geometry import Polygon
            try:
                final_area = float(Polygon(simplified).area)
            except Exception:
                final_area = 0.0

            final_polygons.append({
                'polygon': simplified,
                'conf': poly_info['conf'],
                'cls': poly_info['cls'],
                # 【调试】保留原始ID和合并信息
                'source_ids': poly_info.get('source_ids', [str(poly_info.get('source_id', 'unknown'))]),
                'source_count': poly_info.get('source_count', 1),
                'was_merged': poly_info.get('was_merged', False),
                'debug_area': final_area
            })

        class_ids = np.array([p['cls'] for p in final_polygons])
        confidences = np.array([p['conf'] for p in final_polygons])
        polygons_list = [p['polygon'] for p in final_polygons]

        # 【调试】保留调试信息
        debug_info_list = []
        for p in final_polygons:
            debug_info_list.append({
                'source_ids': p.get('source_ids', []),
                'source_count': p.get('source_count', 1),
                'was_merged': p.get('was_merged', False),
                'debug_area': p.get('debug_area', None)
            })

        elapsed = time.time() - start_time
        logger.info(f"矢量合并完成: {len(final_polygons)} 个实例，耗时: {elapsed:.2f}s")

        return MergedResult(
            merged_masks=None,  # 不需要栅格掩膜
            merged_polygons=polygons_list,  # 保存合并后的矢量多边形
            debug_polygons_info=debug_info_list,  # 【调试】调试信息
            instance_id_map=None,  # 不需要实例ID映射
            class_ids=class_ids,
            confidences=confidences,
            img_shape=img_shape,
            geotransform=geotransform,
            crs=crs,
            num_tiles=len(tile_results),
            total_instances=len(final_polygons)
        )

    def _merge_polygons_by_iou(self, polygons: List[dict], output_dir = None) -> List[dict]:
        """
        基于IoU阈值合并重叠多边形

        合并条件：
        1. 只合并来自不同分块的多边形
        2. 高IoU (> 0.7)：直接合并
        3. 中等IoU (0.5 - 0.7) + 重叠比例 > 0.2：合并
        4. 低IoU (<= 0.5) + 覆盖比例 >= 0.3：包含关系合并

        Args:
            polygons: 多边形信息列表，每项包含 {polygon, conf, cls, bbox, tile_id}
            output_dir: 保留参数，兼容性使用（已不再使用）

        Returns:
            合并后的多边形信息列表
        """
        if not polygons:
            return []

        from shapely.geometry import Polygon
        from shapely.strtree import STRtree

        # 按类别分组（不同类别不应合并）
        by_class = {}
        for poly_info in polygons:
            cls = poly_info['cls']
            if cls not in by_class:
                by_class[cls] = []
            by_class[cls].append(poly_info)

        merged_results = []

        for cls, class_polygons in by_class.items():
            if len(class_polygons) == 1:
                # 单个多边形，添加默认标记
                result = class_polygons[0].copy()
                result['was_merged'] = False
                if 'source_id' not in result:
                    result['source_id'] = f"{result['tile_id']}_0"
                result['source_ids'] = [result['source_id']]
                result['source_count'] = 1
                merged_results.append(result)
                continue

            n = len(class_polygons)

            # 构建 STRtree 空间索引
            shapely_polys = []
            for poly_info in class_polygons:
                points = poly_info['polygon']
                shapely_poly = Polygon(points)
                if not shapely_poly.is_valid:
                    shapely_poly = shapely_poly.buffer(0)
                shapely_polys.append(shapely_poly)

            tree = STRtree(shapely_polys)

            # 使用并查集合并重叠多边形
            parent = list(range(n))

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def union(x, y):
                px, py = find(x), find(y)
                if px != py:
                    parent[px] = py

            # 使用空间索引找出候选重叠对
            checked_pairs = set()
            overlap_pairs_count = 0
            cross_tile_pairs_count = 0

            # 统计各种合并情况
            high_iou_count = 0          # 高IoU合并
            medium_iou_count = 0        # 中等IoU合并
            containment_count = 0       # 包含关系合并
            same_tile_count = 0         # 同分块跳过

            # 预先计算每个多边形的面积
            poly_areas = []
            for idx, poly_info in enumerate(class_polygons):
                poly = Polygon(poly_info['polygon'])
                if not poly.is_valid:
                    poly = poly.buffer(0)
                poly_areas.append(poly.area)

            for i in range(n):
                # 不指定 predicate，获取所有 MBR 相交的对（包括包含关系）
                # intersects 谓词可能遗漏完全包含的情况
                potential_matches = tree.query(shapely_polys[i])

                for j in potential_matches:
                    if i >= j:  # 避免重复检查和自检
                        continue

                    pair_key = (i, j)
                    if pair_key in checked_pairs:
                        continue
                    checked_pairs.add(pair_key)

                    overlap_pairs_count += 1

                    # 只合并来自不同分块的多边形
                    tile_i = class_polygons[i]['tile_id']
                    tile_j = class_polygons[j]['tile_id']

                    # 同分块处理：只检查包含关系，不进行其他合并
                    if tile_i == tile_j:
                        # 计算包含关系
                        poly1 = shapely_polys[i]
                        poly2 = shapely_polys[j]
                        inter = poly1.intersection(poly2).area
                        min_area = min(poly1.area, poly2.area)
                        coverage_ratio = inter / min_area if min_area > 0 else 0

                        # 同分块内只合并包含关系
                        if coverage_ratio >= 0.3:
                            union(i, j)
                            containment_count += 1
                        same_tile_count += 1
                        continue

                    cross_tile_pairs_count += 1

                    # 计算精确的多边形IoU
                    poly_iou = _calculate_polygon_iou(
                        class_polygons[i]['polygon'],
                        class_polygons[j]['polygon']
                    )

                    # 合并策略
                    merged = False

                    # 1. 高IoU (> 0.7)：直接合并
                    if poly_iou > self.high_threshold:
                        union(i, j)
                        merged = True
                        high_iou_count += 1
                    # 2. 中等IoU (0.5 - 0.7)：检查重叠比例
                    elif poly_iou > self.low_threshold:
                        poly1 = shapely_polys[i]
                        poly2 = shapely_polys[j]
                        inter = poly1.intersection(poly2).area
                        min_area = min(poly1.area, poly2.area)
                        # 重叠面积占较小多边形的比例
                        overlap_ratio = inter / min_area if min_area > 0 else 0

                        if overlap_ratio > 0.2:
                            union(i, j)
                            merged = True
                            medium_iou_count += 1

                    # 3. 低IoU但包含关系检查（小多边形大部分在大多边形内）
                    if not merged and poly_iou <= 0.5:
                        poly1 = shapely_polys[i]
                        poly2 = shapely_polys[j]
                        inter = poly1.intersection(poly2).area
                        min_area = min(poly1.area, poly2.area)
                        # 覆盖比例：交集面积占较小多边形面积的比例
                        coverage_ratio = inter / min_area if min_area > 0 else 0

                        if coverage_ratio >= 0.3:
                            union(i, j)
                            containment_count += 1

            # 收集每个组的成员
            groups = {}
            for i in range(n):
                root = find(i)
                if root not in groups:
                    groups[root] = []
                groups[root].append(i)

            # 为每个组合并多边形
            for root, group in groups.items():
                if len(group) == 1:
                    # 单个成员，添加标记
                    result = class_polygons[group[0]].copy()
                    result['was_merged'] = False
                    merged_results.append(result)
                else:
                    # 检查是否真正跨分块（有多个不同tile_id）
                    is_cross_tile = len(set(class_polygons[idx]['tile_id'] for idx in group)) > 1

                    # 合并组内所有多边形
                    max_conf = 0
                    for idx in group:
                        max_conf = max(max_conf, class_polygons[idx]['conf'])

                    try:
                        from shapely.geometry import Polygon
                        from shapely.ops import unary_union

                        # 使用 Shapely 合并多边形（更稳定，能处理复杂拓扑）
                        shapely_polys_to_merge = []
                        source_ids_in_group = []
                        for idx in group:
                            points = class_polygons[idx]['polygon']
                            poly = Polygon(points)
                            # 修复无效多边形
                            if not poly.is_valid:
                                poly = poly.buffer(0)
                            if not poly.is_empty:
                                shapely_polys_to_merge.append(poly)
                                source_ids_in_group.append(class_polygons[idx].get('source_id', idx))

                        if not shapely_polys_to_merge:
                            # 所有多边形都无效，保留原始
                            for idx in group:
                                result = class_polygons[idx].copy()
                                result['was_merged'] = False
                                merged_results.append(result)
                            continue

                        # 使用 unary_union 合并所有多边形
                        merged = unary_union(shapely_polys_to_merge)

                        # 检查结果类型
                        geom_type = merged.geom_type

                        if geom_type == 'MultiPolygon':

                            # 策略：将每个子多边形作为一个独立的结果
                            # 需要找出每个子多边形实际包含哪些原始多边形
                            for geom_idx, geom in enumerate(merged.geoms):
                                if geom.geom_type == 'Polygon' and geom.area > 0:
                                    coords = np.array(geom.exterior.coords)
                                    if len(coords) > 1 and np.allclose(coords[0], coords[-1]):
                                        coords = coords[:-1]
                                    if len(coords) >= 3:
                                        from shapely.geometry import Polygon
                                        final_area = float(Polygon(coords).area)

                                        # 找出这个子多边形实际包含的原始多边形
                                        actual_source_ids = []
                                        for idx in group:
                                            orig_poly = Polygon(class_polygons[idx]['polygon'])
                                            # 检查原始多边形是否与当前子多边形有显著重叠
                                            inter = geom.intersection(orig_poly)
                                            if inter.area > orig_poly.area * 0.5:  # 至少50%重叠
                                                sid = class_polygons[idx].get('source_id', str(idx))
                                                actual_source_ids.append(sid)

                                        # 如果没找到任何原始多边形，使用group的source_id
                                        if not actual_source_ids:
                                            actual_source_ids = [class_polygons[idx].get('source_id', str(idx)) for idx in group]

                                        merged_results.append({
                                            'polygon': coords,
                                            'conf': max_conf,
                                            'cls': cls,
                                            'was_merged': True,
                                            'source_ids': actual_source_ids,  # 使用实际包含的源ID
                                            'source_count': len(actual_source_ids),
                                            'debug_area': final_area
                                        })
                        elif geom_type == 'Polygon':
                            # 检查是否有内环（空间上不连续的标志）
                            num_interior_rings = len(merged.interiors)

                            if num_interior_rings > 0:
                                # 有内环（孔洞），说明包含关系
                                # 只输出外边界，忽略内环（内环表示被包含的区域，不需要单独输出）

                                exterior_coords = np.array(merged.exterior.coords)
                                if len(exterior_coords) > 1 and np.allclose(exterior_coords[0], exterior_coords[-1]):
                                    exterior_coords = exterior_coords[:-1]

                                if len(exterior_coords) >= 3:
                                    from shapely.geometry import Polygon
                                    exterior_area = float(Polygon(exterior_coords).area)

                                    # 找出与外边界对应的原始多边形
                                    actual_source_ids = []
                                    for idx in group:
                                        orig_poly = Polygon(class_polygons[idx]['polygon'])
                                        inter = merged.intersection(orig_poly)
                                        if inter.area > orig_poly.area * 0.1:  # 至少10%重叠
                                            actual_source_ids.append(class_polygons[idx].get('source_id', idx))

                                    merged_results.append({
                                        'polygon': exterior_coords,
                                        'conf': max_conf,
                                        'cls': cls,
                                        'was_merged': True,
                                        'source_ids': actual_source_ids,
                                        'source_count': len(actual_source_ids),
                                        'debug_area': exterior_area
                                    })
                            else:
                                # 没有内环，正常处理
                                coords = np.array(merged.exterior.coords)
                                if len(coords) > 1 and np.allclose(coords[0], coords[-1]):
                                    coords = coords[:-1]

                                if len(coords) >= 3:
                                    from shapely.geometry import Polygon
                                    final_area = float(Polygon(coords).area)

                                    merged_results.append({
                                        'polygon': coords,
                                        'conf': max_conf,
                                        'cls': cls,
                                        'was_merged': True,
                                        'source_ids': [class_polygons[idx].get('source_id', idx) for idx in group],
                                        'source_count': len(group),
                                        'debug_area': final_area
                                    })
                                else:
                                    # 结果点太少，保留原始
                                    for idx in group:
                                        result = class_polygons[idx].copy()
                                        result['was_merged'] = False
                                        result['source_ids'] = [result.get('source_id', f"{result['tile_id']}_{idx}")]
                                        result['source_count'] = 1
                                        merged_results.append(result)
                        elif geom_type == 'GeometryCollection':
                            # GeometryCollection，尝试提取有效的 Polygon
                            has_valid_polygon = False
                            for geom in merged.geoms:
                                if geom.geom_type == 'Polygon' and geom.area > 0:
                                    coords = np.array(geom.exterior.coords)
                                    if len(coords) > 1 and np.allclose(coords[0], coords[-1]):
                                        coords = coords[:-1]
                                    if len(coords) >= 3:
                                        # 计算简化后的面积
                                        from shapely.geometry import Polygon
                                        final_area = float(Polygon(coords).area)

                                        merged_results.append({
                                            'polygon': coords,
                                            'conf': max_conf,
                                            'cls': cls,
                                            'was_merged': True,
                                            'source_ids': [class_polygons[idx].get('source_id', idx) for idx in group],
                                            'source_count': len(group),
                                            'debug_area': final_area
                                        })
                                        has_valid_polygon = True
                                        break  # 只取第一个有效的
                            if not has_valid_polygon:
                                # 没有有效的多边形，保留原始
                                for idx in group:
                                    result = class_polygons[idx].copy()
                                    result['was_merged'] = False
                                    merged_results.append(result)
                                logger.warning("GeometryCollection 中没有有效的多边形，保留原始")
                        else:
                            # 其他类型，保留原始
                            for idx in group:
                                result = class_polygons[idx].copy()
                                result['was_merged'] = False
                                merged_results.append(result)
                            logger.warning(f"Shapely Union 返回意外类型: {geom_type}")

                    except ImportError:
                        # Shapely 不可用，保留原始多边形
                        for idx in group:
                            result = class_polygons[idx].copy()
                            result['was_merged'] = False
                            merged_results.append(result)
                        logger.warning("Shapely 不可用，跳过多边形合并，保留原始结果")
                    except Exception as e:
                        # Union 失败，保留原始多边形
                        for idx in group:
                            result = class_polygons[idx].copy()
                            result['was_merged'] = False
                            merged_results.append(result)
                        logger.warning(f"Shapely Union 失败: {e}，保留原始多边形")

        logger.info(f"多边形合并: 输入 {len(polygons)} 个，输出 {len(merged_results)} 个")
        return merged_results

    def _find_spatially_connected_groups(self, poly_infos: List[dict], shapely_polys: List) -> list:
        """
        使用连通分量分析找出空间上真正相连的多边形组

        Args:
            poly_infos: 多边形信息列表
            shapely_polys: 对应的 Shapely Polygon 对象列表

        Returns:
            空间连通组的列表，每组包含 (索引列表, Shapely Polygon列表)
        """
        n = len(shapely_polys)
        if n <= 1:
            return [([list(range(n))], shapely_polys)]

        # 构建连通图：两个多边形相交或距离很近则为连通
        visited = [False] * n
        groups = []

        for i in range(n):
            if visited[i]:
                continue

            # BFS 找出所有与 i 连通的多边形
            group_indices = []
            queue = [i]
            visited[i] = True

            while queue:
                curr = queue.pop(0)
                group_indices.append(curr)

                # 检查所有未访问的多边形
                for j in range(n):
                    if visited[j]:
                        continue

                    # 判断是否连通：相交或距离很近
                    poly_i = shapely_polys[curr]
                    poly_j = shapely_polys[j]

                    # 方法1: 直接相交
                    if poly_i.intersects(poly_j):
                        # 检查相交面积是否显著（不是点或线的接触）
                        inter = poly_i.intersection(poly_j)
                        if inter.area > 1:  # 至少1像素的相交面积
                            visited[j] = True
                            queue.append(j)
                            continue

                    # 方法2: 距离很近（相接）
                    distance = poly_i.distance(poly_j)
                    if distance < 5:  # 距离小于5像素
                        visited[j] = True
                        queue.append(j)
                        continue

                    # 方法3: bbox 相交且有重叠
                    bbox_i = poly_infos[curr]['bbox']
                    bbox_j = poly_infos[j]['bbox']
                    x_left = max(bbox_i[0], bbox_j[0])
                    y_top = max(bbox_i[1], bbox_j[1])
                    x_right = min(bbox_i[2], bbox_j[2])
                    y_bottom = min(bbox_i[3], bbox_j[3])

                    if x_right > x_left and y_bottom > y_top:
                        # bbox 有重叠
                        inter_area = (x_right - x_left) * (y_bottom - y_top)
                        if inter_area > 10:  # 至少10像素的重叠
                            visited[j] = True
                            queue.append(j)
                            continue

            # 将该组加入结果
            group_polys = [shapely_polys[idx] for idx in group_indices]
            groups.append((group_indices, group_polys))

        return groups

    def _build_compact_instance_map(self, merged_info: List[dict], img_shape: tuple) -> np.ndarray:
        """
        构建紧凑实例ID映射图（【性能优化】直接使用int32，跳过类型转换）

        Args:
            merged_info: 合并后的掩膜信息（可能包含 mask_local 或 mask_full）
            img_shape: 影像尺寸

        Returns:
            实例ID映射图（int32类型）
        """
        height, width = img_shape[:2]

        # 【性能优化】直接使用 int32，跳过 dtype 选择和类型转换
        # 这避免了后续的 .astype(np.int32) 复制操作
        instance_id_map = np.zeros((height, width), dtype=np.int32)

        for idx, info in enumerate(merged_info, start=1):
            # 优先使用 mask_local，如果没有则使用 mask_full
            if 'mask_local' in info and info['mask_local'] is not None:
                mask_local = info['mask_local']
                offset = info['offset']
                shape = info['shape']
                # 将局部掩膜放置到全局映射图中
                instance_id_map[offset[0]:offset[0]+shape[0], offset[1]:offset[1]+shape[1]][mask_local > 0] = idx
            elif 'mask_full' in info and info['mask_full'] is not None:
                mask_full = info['mask_full']
                instance_id_map[mask_full > 0] = idx

        return instance_id_map

    def _merge_masks_by_overlap_region(
        self,
        all_masks: List[dict],
        img_shape: tuple
    ) -> List[dict]:
        """
        基于分块重叠区域的合并策略（优化版本：提前分离重叠/非重叠掩膜）

        核心优化：
        1. 找出所有与重叠区域相交的掩膜（跨越边界的掩膜）
        2. 不与任何重叠区域相交的掩膜直接保存，不参与合并
        3. 只对跨越边界的掩膜进行像素重叠判断和合并

        Args:
            all_masks: 掩膜信息列表（需包含 tile_row, tile_col, tile_offset, tile_shape）
            img_shape: 影像尺寸

        Returns:
            合并后的掩膜信息列表
        """
        if not all_masks:
            return []

        # 快速路径：如果掩膜数量很少，直接返回
        if len(all_masks) <= 1:
            return self._build_single_mask_results(all_masks, img_shape)

        # 【优化】预计算所有掩膜的bbox
        bbox_cache = self._precompute_all_bboxes(all_masks)

        # 【关键优化】找出所有与重叠区域相交的掩膜索引
        overlapping_mask_indices = self._find_overlapping_mask_indices(all_masks, bbox_cache)

        logger.debug(f"Found {len(overlapping_mask_indices)} overlapping masks out of {len(all_masks)} total")

        # 分离非重叠掩膜和重叠掩膜
        non_overlapping_results = []
        overlapping_masks = []
        overlapping_index_map = {}  # 原始索引 -> 重叠掩膜列表中的新索引

        for idx, mask_info in enumerate(all_masks):
            if idx in overlapping_mask_indices:
                # 跨越边界的掩膜，需要合并处理
                overlapping_index_map[idx] = len(overlapping_masks)
                overlapping_masks.append(mask_info)
            else:
                # 非重叠掩膜，直接保存为结果（不需要合并）
                non_overlapping_results.append({
                    'mask_local': mask_info['mask'],
                    'offset': mask_info['offset'],
                    'shape': mask_info['shape'],
                    'conf': mask_info['conf'],
                    'cls': mask_info['cls'],
                    'mask_full': None
                })

        # 只对重叠掩膜进行合并
        if overlapping_masks:
            # 为重叠掩膜构建bbox缓存子集
            overlapping_bbox_cache = {}
            for orig_idx in overlapping_mask_indices:
                new_idx = overlapping_index_map[orig_idx]
                overlapping_bbox_cache[new_idx] = bbox_cache[orig_idx]

            merged_overlapping = self._merge_overlapping_masks_only(overlapping_masks, img_shape, overlapping_bbox_cache)
            # 合并非重叠结果和合并后的重叠结果
            return non_overlapping_results + merged_overlapping
        else:
            # 没有需要合并的掩膜，直接返回非重叠结果
            return non_overlapping_results

    def _find_overlapping_mask_indices(self, all_masks: List[dict], bbox_cache: dict) -> Set[int]:
        """
        找出所有与分块重叠区域相交的掩膜索引

        这些掩膜跨越了分块边界，需要参与合并。

        Args:
            all_masks: 掩膜信息列表
            bbox_cache: bbox缓存

        Returns:
            与重叠区域相交的掩膜索引集合
        """
        overlapping_indices = set()

        # 按分块分组
        tiles_map = {}
        for idx, mask_info in enumerate(all_masks):
            tile_key = (mask_info['tile_row'], mask_info['tile_col'])
            if tile_key not in tiles_map:
                tiles_map[tile_key] = []
            tiles_map[tile_key].append(idx)

        # 8方向偏移
        directions = [(-1, -1), (-1, 0), (-1, 1),
                      (0, -1),           (0, 1),
                      (1, -1),  (1, 0), (1, 1)]

        processed_tile_pairs = set()

        for (row, col), mask_indices in tiles_map.items():
            for dr, dc in directions:
                neighbor_key = (row + dr, col + dc)
                if neighbor_key not in tiles_map:
                    continue

                # 避免重复处理同一对分块
                pair_key = tuple(sorted([ (row, col), neighbor_key ]))
                if pair_key in processed_tile_pairs:
                    continue
                processed_tile_pairs.add(pair_key)

                # 获取当前分块和邻居分块的信息
                current_tile_info = all_masks[mask_indices[0]]
                neighbor_tile_info = all_masks[tiles_map[neighbor_key][0]]

                # 计算两个分块之间的重叠区域
                overlap_region = self.tile_processor.get_overlap_region_between_tiles(
                    current_tile_info['tile_offset'],
                    current_tile_info['tile_shape'],
                    neighbor_tile_info['tile_offset'],
                    neighbor_tile_info['tile_shape']
                )

                if overlap_region is None:
                    continue

                # 检查两个分块中的所有掩膜是否与重叠区域相交
                for tile_key in [(row, col), neighbor_key]:
                    for idx in tiles_map[tile_key]:
                        if idx in overlapping_indices:
                            continue  # 已经标记过

                        mask_bbox = bbox_cache.get(idx)
                        if mask_bbox and self._bbox_intersects_overlap_region(mask_bbox, overlap_region):
                            overlapping_indices.add(idx)

        return overlapping_indices

    def _merge_overlapping_masks_only(self, overlapping_masks: List[dict], img_shape: tuple, bbox_cache: dict = None) -> List[dict]:
        """
        只对跨越边界的掩膜进行合并（并行优化版本 + 性能分析）

        核心优化：
        - 只比较来自相邻分块的掩膜对
        - 使用 Numba 加速的像素重叠检查和 IoU 计算
        - 批量收集候选对，统一计算 mask IoU
        - 根据数据规模自动选择单进程或多进程并行
        - 性能分析：记录各阶段耗时

        合并原则（不改变算法逻辑）：
        - 只合并重叠区域的掩膜
        - 不同类别的掩膜不合并
        - mask_iou > 0.7 → 直接合并
        - 0.5 < mask_iou ≤ 0.7 → 合并
        - mask_iou ≤ 0.5 → 不合并

        Args:
            overlapping_masks: 跨越边界的掩膜列表
            img_shape: 影像尺寸
            bbox_cache: bbox缓存 {idx: bbox}（可选，用于性能优化）

        Returns:
            合并后的掩膜信息列表
        """
        import time
        if not overlapping_masks:
            return []

        # 性能计时开始
        total_start = time.time()

        # 初始化并行调度器和 IoU 计算器
        scheduler = ParallelScheduler(parallel_threshold=100)
        calculator = IoUCalculator(use_numba=NUMBA_AVAILABLE)

        # 使用并查集管理合并关系
        parent = list(range(len(overlapping_masks)))
        rank = [0] * len(overlapping_masks)

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                if rank[px] < rank[py]:
                    px, py = py, px
                parent[py] = px
                if rank[px] == rank[py]:
                    rank[px] += 1

        # 【优化】预计算掩膜数据，避免重复访问字典
        mask_data = [m['mask'] for m in overlapping_masks]
        mask_offsets = [m['offset'] for m in overlapping_masks]
        mask_shapes = [m['shape'] for m in overlapping_masks]

        # 按分块分组，只比较相邻分块的掩膜
        tiles_map = {}
        for idx, mask_info in enumerate(overlapping_masks):
            tile_key = (mask_info['tile_row'], mask_info['tile_col'])
            if tile_key not in tiles_map:
                tiles_map[tile_key] = []
            tiles_map[tile_key].append(idx)

        # 8方向偏移
        directions = [(-1, -1), (-1, 0), (-1, 1),
                      (0, -1),           (0, 1),
                      (1, -1),  (1, 0), (1, 1)]

        processed_pairs = set()

        # 【第一阶段】收集所有需要计算的候选掩膜对
        stage1_start = time.time()
        candidate_pairs = []

        for (row, col), mask_indices in tiles_map.items():
            for dr, dc in directions:
                neighbor_key = (row + dr, col + dc)
                if neighbor_key not in tiles_map:
                    continue

                neighbor_indices = tiles_map[neighbor_key]

                # 避免重复处理同一对分块
                pair_key = tuple(sorted([ (row, col), neighbor_key ]))
                if pair_key in processed_pairs:
                    continue
                processed_pairs.add(pair_key)

                # 收集这两个相邻分块中的所有掩膜对
                for idx in mask_indices:
                    for neighbor_idx in neighbor_indices:
                        if idx == neighbor_idx:
                            continue

                        ordered_pair = tuple(sorted([idx, neighbor_idx]))
                        if ordered_pair in processed_pairs:
                            continue
                        processed_pairs.add(ordered_pair)

                        # 【快速筛选】bbox重叠检查
                        bbox1 = bbox_cache.get(idx) if bbox_cache else None
                        bbox2 = bbox_cache.get(neighbor_idx) if bbox_cache else None
                        if not self._masks_overlap(overlapping_masks[idx], overlapping_masks[neighbor_idx], bbox1, bbox2):
                            continue

                        # 【快速筛选】类别检查
                        if overlapping_masks[idx]['cls'] != overlapping_masks[neighbor_idx]['cls']:
                            continue

                        # 【快速筛选】像素重叠检查
                        offset1 = mask_offsets[idx]
                        offset2 = mask_offsets[neighbor_idx]
                        shape1 = mask_shapes[idx]
                        shape2 = mask_shapes[neighbor_idx]

                        if NUMBA_AVAILABLE:
                            has_overlap = _has_pixel_overlap_numba(
                                mask_data[idx], offset1[0], offset1[1], shape1[0], shape1[1],
                                mask_data[neighbor_idx], offset2[0], offset2[1], shape2[0], shape2[1]
                            )
                        else:
                            has_overlap = self._has_pixel_overlap_vectorized(
                                overlapping_masks[idx], overlapping_masks[neighbor_idx]
                            )

                        if has_overlap:
                            candidate_pairs.append((idx, neighbor_idx))

        stage1_time = time.time() - stage1_start
        logger.info(f"候选收集: {len(candidate_pairs)} 对掩膜 (耗时: {stage1_time:.2f}s)")

        # 【第二阶段】根据数据规模选择计算方式
        stage2_start = time.time()
        iou_cache = {}

        if not candidate_pairs:
            pass  # 没有候选对，直接跳过
        elif scheduler.should_parallelize(len(candidate_pairs)):
            # 多进程并行计算
            try:
                logger.info(f"使用多进程并行计算 {len(candidate_pairs)} 对掩膜的 IoU (进程数: {scheduler.max_workers})")

                # 分块
                chunks = scheduler.split_into_chunks(candidate_pairs)
                chunk_args = [
                    (chunk, mask_data, mask_offsets)
                    for chunk in chunks
                ]

                # 并行计算
                iou_results = scheduler.execute_parallel(_calculate_iou_chunk, chunk_args)

                # 存入缓存
                for (idx, neighbor_idx), iou in zip(candidate_pairs, iou_results):
                    iou_cache[(idx, neighbor_idx)] = iou

            except Exception as e:
                logger.warning(f"并行计算失败，降级到单进程: {e}")
                # 降级到单进程
                iou_results = calculator.calculate_batch(candidate_pairs, mask_data, mask_offsets)
                for (idx, neighbor_idx), iou in zip(candidate_pairs, iou_results):
                    iou_cache[(idx, neighbor_idx)] = iou
        else:
            # 单进程批量计算
            logger.debug(f"使用单进程批量计算 {len(candidate_pairs)} 对掩膜的 IoU")
            iou_results = calculator.calculate_batch(candidate_pairs, mask_data, mask_offsets)
            for (idx, neighbor_idx), iou in zip(candidate_pairs, iou_results):
                iou_cache[(idx, neighbor_idx)] = iou

        stage2_time = time.time() - stage2_start
        logger.info(f"IoU计算: {len(iou_cache)} 对掩膜 (耗时: {stage2_time:.2f}s)")

        # 【第三阶段】根据 IoU 结果执行合并
        stage3_start = time.time()
        for (idx, neighbor_idx), mask_iou in iou_cache.items():
            # 使用 mask IoU 决定是否合并
            if mask_iou > self.high_threshold:
                union(idx, neighbor_idx)
            elif mask_iou > self.low_threshold:
                union(idx, neighbor_idx)
            # mask_iou <= low_threshold (0.5): 不合并，保持独立

        stage3_time = time.time() - stage3_start
        # 统计合并次数：计算最终组数，减去原始掩膜数，再减去单成员组
        final_groups = {}
        for i in range(len(overlapping_masks)):
            root = find(i)
            if root not in final_groups:
                final_groups[root] = []
            final_groups[root].append(i)
        merged_count = sum(len(members) - 1 for members in final_groups.values() if len(members) > 1)
        logger.info(f"合并执行: {merged_count} 次合并 (耗时: {stage3_time:.2f}s)")

        # 为每个组合并掩膜
        result = self._build_merged_results(final_groups, overlapping_masks, img_shape)

        total_time = time.time() - total_start
        logger.info(f"合并总耗时: {total_time:.2f}s (阶段1: {stage1_time:.2f}s, 阶段2: {stage2_time:.2f}s, 阶段3: {stage3_time:.2f}s)")

        return result

    def _find_cross_tile_candidates(self, all_masks: List[dict], bbox_cache: dict = None) -> Set[tuple]:
        """
        找出跨越分块边界的候选掩膜对

        判断标准：掩膜与相邻分块的重叠区域相交

        Args:
            all_masks: 掩膜信息列表
            bbox_cache: 预计算的bbox缓存 {idx: (x_min, y_min, x_max, y_max)}

        Returns:
            候选掩膜对集合 {(i, j), ...}
        """
        candidates = set()

        # 如果没有提供缓存，使用旧方法
        if bbox_cache is None:
            bbox_cache = {}

        # 按分块分组
        tiles_map = {}
        for idx, mask_info in enumerate(all_masks):
            tile_key = (mask_info['tile_row'], mask_info['tile_col'])
            if tile_key not in tiles_map:
                tiles_map[tile_key] = []
            tiles_map[tile_key].append(idx)

        # 8方向偏移
        directions = [(-1, -1), (-1, 0), (-1, 1),
                      (0, -1),           (0, 1),
                      (1, -1),  (1, 0), (1, 1)]

        for (row, col), mask_indices in tiles_map.items():
            for dr, dc in directions:
                neighbor_key = (row + dr, col + dc)
                if neighbor_key not in tiles_map:
                    continue

                neighbor_indices = tiles_map[neighbor_key]

                # 获取当前分块和邻居分块的信息
                if mask_indices and neighbor_indices:
                    # 使用第一个掩膜的分块信息（同一分块内所有掩膜的分块信息相同）
                    current_tile_info = all_masks[mask_indices[0]]
                    neighbor_tile_info = all_masks[neighbor_indices[0]]

                    # 计算两个分块之间的重叠区域
                    overlap_region = self.tile_processor.get_overlap_region_between_tiles(
                        current_tile_info['tile_offset'],
                        current_tile_info['tile_shape'],
                        neighbor_tile_info['tile_offset'],
                        neighbor_tile_info['tile_shape']
                    )

                    if overlap_region is None:
                        continue

                    # 检查当前分块内的每个掩膜是否与重叠区域相交
                    for idx in mask_indices:
                        # 【优化】从缓存获取bbox
                        if idx in bbox_cache:
                            mask_bbox = bbox_cache[idx]
                        else:
                            mask_bbox = self._get_mask_bbox_global(all_masks[idx])

                        if self._bbox_intersects_overlap_region(mask_bbox, overlap_region):
                            # 与重叠区域相交，检查邻居分块中的掩膜是否也相交
                            for neighbor_idx in neighbor_indices:
                                # 【优化】从缓存获取bbox
                                if neighbor_idx in bbox_cache:
                                    neighbor_bbox = bbox_cache[neighbor_idx]
                                else:
                                    neighbor_bbox = self._get_mask_bbox_global(all_masks[neighbor_idx])

                                if self._bbox_intersects_overlap_region(neighbor_bbox, overlap_region):
                                    # 【优化】使用缓存的bbox计算IoU
                                    bbox_iou = self._calculate_bbox_iou_from_cache(
                                        mask_bbox, neighbor_bbox
                                    )
                                    if bbox_iou >= self.cross_tile_bbox_iou_threshold:
                                        # 两个掩膜彼此有足够的重叠，是合并候选
                                        pair = (min(idx, neighbor_idx), max(idx, neighbor_idx))
                                        candidates.add(pair)

        return candidates

    def _calculate_bbox_iou_from_cache(
        self,
        bbox1: Tuple[int, int, int, int],
        bbox2: Tuple[int, int, int, int]
    ) -> float:
        """
        从缓存的bbox计算IoU

        Args:
            bbox1: (x_min, y_min, x_max, y_max)
            bbox2: (x_min, y_min, x_max, y_max)

        Returns:
            bbox IoU值
        """
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2

        # 计算交集
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0

        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

        # 计算并集
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area

        if union_area <= 0:
            return 0.0

        return inter_area / union_area

    def _get_mask_bbox_global(self, mask_info: dict) -> Tuple[int, int, int, int]:
        """
        获取掩膜的全局边界框

        从掩膜数组计算边界框，然后加上偏移量转换为全局坐标

        Args:
            mask_info: 掩膜信息

        Returns:
            (x_min, y_min, x_max, y_max) 全局边界框
        """
        mask = mask_info['mask']
        offset = mask_info['offset']  # (row_offset, col_offset)

        # 计算掩膜的局部边界框
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        if not np.any(rows) or not np.any(cols):
            # 空掩膜，返回偏移位置
            return (offset[1], offset[0], offset[1], offset[0])

        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        # 转换为全局坐标
        # offset 是 (row, col)，bbox 是 (x, y) = (col, row)
        x_min = offset[1] + cmin
        y_min = offset[0] + rmin
        x_max = offset[1] + cmax + 1
        y_max = offset[0] + rmax + 1

        return (x_min, y_min, x_max, y_max)

    def _precompute_all_bboxes(self, all_masks: List[dict]) -> dict:
        """
        预计算所有掩膜的全局bbox并缓存（向量化版本，超高效）

        使用 np.argmax 替代 np.where，避免创建完整的索引数组

        Args:
            all_masks: 掩膜信息列表

        Returns:
            {idx: (x_min, y_min, x_max, y_max)} bbox缓存字典
        """
        bbox_cache = {}

        for idx, mask_info in enumerate(all_masks):
            mask = mask_info['mask']
            offset = mask_info['offset']

            # 【向量化优化】使用 np.any 计算行列是否有像素
            rows = np.any(mask, axis=1)  # [H] bool
            cols = np.any(mask, axis=0)  # [W] bool

            if not np.any(rows) or not np.any(cols):
                # 空掩膜
                bbox_cache[idx] = (offset[1], offset[0], offset[1], offset[0])
            else:
                # 【进一步优化】使用 np.argmax 找第一个True的位置，比 np.where 更快
                rmin = np.argmax(rows)  # 第一个有像素的行
                rmax = len(rows) - np.argmax(rows[::-1]) - 1  # 最后一个有像素的行
                cmin = np.argmax(cols)  # 第一个有像素的列
                cmax = len(cols) - np.argmax(cols[::-1]) - 1  # 最后一个有像素的列

                # 转换为全局坐标
                x_min = offset[1] + cmin
                y_min = offset[0] + rmin
                x_max = offset[1] + cmax + 1
                y_max = offset[0] + rmax + 1
                bbox_cache[idx] = (x_min, y_min, x_max, y_max)

        return bbox_cache

    def _get_mask_bbox_from_cache(self, idx: int, bbox_cache: dict) -> Tuple[int, int, int, int]:
        """
        从缓存获取掩膜bbox

        Args:
            idx: 掩膜索引
            bbox_cache: bbox缓存字典

        Returns:
            (x_min, y_min, x_max, y_max)
        """
        return bbox_cache.get(idx, (0, 0, 0, 0))

    def _calculate_bbox_iou_between_masks(
        self,
        mask_info1: dict,
        mask_info2: dict
    ) -> float:
        """
        计算两个掩膜全局bbox之间的IoU

        Args:
            mask_info1: 掩膜信息1
            mask_info2: 掩膜信息2

        Returns:
            bbox IoU值
        """
        bbox1 = self._get_mask_bbox_global(mask_info1)
        bbox2 = self._get_mask_bbox_global(mask_info2)

        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2

        # 计算交集
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0

        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

        # 计算并集
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area

        if union_area <= 0:
            return 0.0

        return inter_area / union_area

    def _bbox_intersects_overlap_region(
        self,
        bbox: Tuple[int, int, int, int],
        overlap_region: Tuple[int, int, int, int]
    ) -> bool:
        """
        检查边界框是否与重叠区域相交

        Args:
            bbox: (x_min, y_min, x_max, y_max)
            overlap_region: (x_min, y_min, x_max, y_max)

        Returns:
            是否相交
        """
        x1_min, y1_min, x1_max, y1_max = bbox
        x2_min, y2_min, x2_max, y2_max = overlap_region

        # 检查是否相交
        return not (x1_max <= x2_min or x2_max <= x1_min or
                    y1_max <= y2_min or y2_max <= y1_min)

    def _has_pixel_overlap_fast(
        self,
        mask_info1: dict,
        mask_info2: dict
    ) -> bool:
        """
        快速检查两个掩膜是否有像素重叠（使用Numba加速）

        Args:
            mask_info1: 掩膜信息1
            mask_info2: 掩膜信息2

        Returns:
            是否有像素重叠
        """
        mask1 = mask_info1['mask']
        offset1 = mask_info1['offset']
        shape1 = mask_info1['shape']

        mask2 = mask_info2['mask']
        offset2 = mask_info2['offset']
        shape2 = mask_info2['shape']

        return _has_pixel_overlap_numba(
            mask1, offset1[0], offset1[1], shape1[0], shape1[1],
            mask2, offset2[0], offset2[1], shape2[0], shape2[1]
        )

    def _has_pixel_overlap_vectorized(
        self,
        mask_info1: dict,
        mask_info2: dict
    ) -> bool:
        """
        快速检查两个掩膜是否有像素重叠（向量化版本，不依赖Numba）

        使用 NumPy 数组操作替代嵌套循环，利用广播机制加速

        Args:
            mask_info1: 掩膜信息1
            mask_info2: 掩膜信息2

        Returns:
            是否有像素重叠
        """
        mask1 = mask_info1['mask']
        offset1 = mask_info1['offset']
        shape1 = mask_info1['shape']

        mask2 = mask_info2['mask']
        offset2 = mask_info2['offset']
        shape2 = mask_info2['shape']

        # 计算重叠区域
        y1_min, y1_max = offset1[0], offset1[0] + shape1[0]
        x1_min, x1_max = offset1[1], offset1[1] + shape1[1]
        y2_min, y2_max = offset2[0], offset2[0] + shape2[0]
        x2_min, x2_max = offset2[1], offset2[1] + shape2[1]

        overlap_y_min = max(y1_min, y2_min)
        overlap_y_max = min(y1_max, y2_max)
        overlap_x_min = max(x1_min, x2_min)
        overlap_x_max = min(x1_max, x2_max)

        if overlap_x_max <= overlap_x_min or overlap_y_max <= overlap_y_min:
            return False

        # 计算局部坐标并提取重叠区域
        local_y1 = overlap_y_min - y1_min
        local_x1 = overlap_x_min - x1_min
        local_y2 = overlap_y_min - y2_min
        local_x2 = overlap_x_min - x2_min

        overlap_w = overlap_x_max - overlap_x_min
        overlap_h = overlap_y_max - overlap_y_min

        # 【向量化优化】使用 NumPy 逻辑与运算检查重叠
        overlap_mask1 = mask1[local_y1:local_y1+overlap_h, local_x1:local_x1+overlap_w]
        overlap_mask2 = mask2[local_y2:local_y2+overlap_h, local_x2:local_x2+overlap_w]

        # 检查是否有两个掩膜都非零的位置
        return np.any((overlap_mask1 > 0) & (overlap_mask2 > 0))

    def _merge_masks_with_iou(
        self,
        all_masks: List[dict],
        img_shape: tuple
    ) -> List[dict]:
        """
        使用双阈值IoU策略合并掩膜（优化版本 - 空间索引 + 内存优化）

        Args:
            all_masks: 掩膜信息列表
            img_shape: 影像尺寸

        Returns:
            合并后的掩膜信息列表
        """
        height, width = img_shape[:2]

        if not all_masks:
            return []

        # 快速路径：如果掩膜数量很少，直接返回
        if len(all_masks) <= 1:
            return self._build_single_mask_results(all_masks, img_shape)

        # 使用优化的并查集（路径压缩）
        parent = list(range(len(all_masks)))
        rank = [0] * len(all_masks)

        def find(x):
            # 带路径压缩的查找
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            # 按秩合并
            px, py = find(x), find(y)
            if px != py:
                if rank[px] < rank[py]:
                    px, py = py, px
                parent[py] = px
                if rank[px] == rank[py]:
                    rank[px] += 1

        # 优化：使用空间网格索引快速找出可能重叠的掩膜对
        potential_pairs = self._find_potential_overlap_pairs_optimized(all_masks)

        logger.debug(f"Found {len(potential_pairs)} potential overlap pairs out of {len(all_masks) * (len(all_masks) - 1) // 2} possible pairs")

        # 只对可能重叠的掩膜对计算IoU
        # 分级策略：先使用box IoU快速判断，必要时使用mask IoU精确计算
        for i, j in potential_pairs:
            mask_info_i = all_masks[i]
            mask_info_j = all_masks[j]

            # 第一步：快速检查边界框是否重叠（空间预筛选）
            if not self._masks_overlap(mask_info_i, mask_info_j):
                continue

            # 第二步：检查是否有真实的像素重叠（关键！）
            if not self._has_pixel_overlap(mask_info_i, mask_info_j):
                continue  # 边界框重叠但像素不重叠，保持原始掩膜

            # 第三步：有真实像素重叠，使用分级策略决定是否合并
            box_iou = self._calculate_mask_bbox_iou(mask_info_i, mask_info_j)

            if box_iou > self.high_threshold:
                # 高置信度重叠，直接合并
                union(i, j)
            elif box_iou > self.low_threshold:
                # 中等置信度，使用mask IoU精确计算
                mask_iou = self._calculate_mask_iou_fast(mask_info_i, mask_info_j)
                if mask_iou > self.low_threshold:
                    union(i, j)
            # box_iou <= low_threshold: 不合并，保持各自独立的掩膜
                    union(i, j)
            # box_iou <= low_threshold: 不合并，跳过

        # 收集每个最终ID的所有成员
        final_groups = {}
        for i in range(len(all_masks)):
            root = find(i)
            if root not in final_groups:
                final_groups[root] = []
            final_groups[root].append(i)

        # 为每个组合并掩膜（使用延迟分配策略）
        return self._build_merged_results(final_groups, all_masks, img_shape)

    def _build_single_mask_results(self, all_masks: List[dict], img_shape: tuple) -> List[dict]:
        """为单个掩膜构建结果（优化：使用局部掩膜）"""
        result = []
        for mask_info in all_masks:
            # 【优化】保持局部掩膜，不创建全局掩膜
            result.append({
                'mask_local': mask_info['mask'],  # 保存局部掩膜
                'offset': mask_info['offset'],
                'shape': mask_info['shape'],
                'conf': mask_info['conf'],
                'cls': mask_info['cls'],
                'mask_full': None  # 延迟构建
            })
        return result

    def _build_merged_results(self, final_groups: dict, all_masks: List[dict], img_shape: tuple) -> List[dict]:
        """
        构建合并后的结果（优化：使用局部掩膜 + 延迟构建全局掩膜）

        Args:
            final_groups: 分组信息 {root: [members]}
            all_masks: 原始掩膜列表
            img_shape: 影像尺寸

        Returns:
            合并结果列表
        """
        height, width = img_shape[:2]
        result = []

        for root, members in final_groups.items():
            if len(members) == 1:
                # 【优化】单个成员，保持局部掩膜，不创建全局掩膜
                mask_info = all_masks[members[0]]
                result.append({
                    'mask_local': mask_info['mask'],  # 保存局部掩膜
                    'offset': mask_info['offset'],
                    'shape': mask_info['shape'],
                    'conf': mask_info['conf'],
                    'cls': mask_info['cls'],
                    'mask_full': None  # 延迟构建
                })
            else:
                # 多个成员，计算边界框只分配需要的区域
                merged_result = self._merge_multiple_masks(members, all_masks, img_shape)
                result.append(merged_result)

        return result

    def _merge_multiple_masks(self, members: List[int], all_masks: List[dict], img_shape: tuple) -> dict:
        """
        合并多个掩膜（使用逻辑或运算，真正融合重叠区域）

        Args:
            members: 成员索引列表
            all_masks: 原始掩膜列表
            img_shape: 影像尺寸

        Returns:
            合并结果
        """
        height, width = img_shape[:2]

        # 计算所有成员的联合边界框
        bboxes = []
        total_conf = 0
        cls_counts = {}

        for member in members:
            mask_info = all_masks[member]
            offset = mask_info['offset']
            shape = mask_info['shape']
            bboxes.append((offset[1], offset[0], offset[1] + shape[1], offset[0] + shape[0]))
            total_conf += mask_info['conf']
            cls = mask_info['cls']
            cls_counts[cls] = cls_counts.get(cls, 0) + 1

        # 联合边界框
        x_min = min(bbox[0] for bbox in bboxes)
        y_min = min(bbox[1] for bbox in bboxes)
        x_max = max(bbox[2] for bbox in bboxes)
        y_max = max(bbox[3] for bbox in bboxes)

        bbox_h = y_max - y_min
        bbox_w = x_max - x_min

        # 使用逻辑或运算合并掩膜（任何掩膜有值的位置都保留）
        merged_mask = np.zeros((bbox_h, bbox_w), dtype=np.uint8)

        for member in members:
            mask_info = all_masks[member]
            mask = mask_info['mask']
            offset = mask_info['offset']
            shape = mask_info['shape']

            # 计算在边界框中的相对位置
            rel_y = offset[0] - y_min
            rel_x = offset[1] - x_min
            h, w = shape

            # 使用逻辑或运算：merged_mask = merged_mask OR mask
            # 这样重叠区域会被保留，不会被截断
            mask_region = mask.astype(np.uint8)
            merged_mask[rel_y:rel_y+h, rel_x:rel_x+w] = \
                np.logical_or(
                    merged_mask[rel_y:rel_y+h, rel_x:rel_x+w],
                    mask_region
                ).astype(np.uint8)

        # 选择最常见的类别和平均置信度
        final_cls = max(cls_counts.items(), key=lambda x: x[1])[0]
        final_conf = total_conf / len(members)

        # 【优化】不创建全局掩膜，只在边界框区域分配内存
        # 返回局部掩膜 + 边界框信息，按需构建全局掩膜
        return {
            'mask_local': merged_mask,  # 边界框区域内的合并掩膜
            'bbox': (x_min, y_min, x_max, y_max),  # 边界框
            'offset': (y_min, x_min),  # 边界框偏移
            'shape': (bbox_h, bbox_w),  # 边界框尺寸
            'conf': final_conf,
            'cls': final_cls,
            'mask_full': None  # 延迟构建
        }

    def _merge_masks_with_iou_optimized(
        self,
        all_masks: List[dict],
        img_shape: tuple
    ) -> List[dict]:
        """
        使用双阈值IoU策略合并掩膜（超级优化版本 - 邻接过滤 + 掩膜优化）

        优化策略：
        1. 分块邻接过滤：只合并相邻分块的掩膜
        2. 掩膜轮廓表示：使用轮廓点而非完整掩膜
        3. 更激进的IoU阈值：减少精确计算

        Args:
            all_masks: 掩膜信息列表（需包含 tile_id, tile_row, tile_col）
            img_shape: 影像尺寸

        Returns:
            合并后的掩膜信息列表
        """
        height, width = img_shape[:2]

        if not all_masks:
            return []

        # 快速路径：如果掩膜数量很少，直接返回
        if len(all_masks) <= 1:
            return self._build_single_mask_results(all_masks, img_shape)

        # 使用优化的并查集（路径压缩 + 按秩合并）
        parent = list(range(len(all_masks)))
        rank = [0] * len(all_masks)

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                if rank[px] < rank[py]:
                    px, py = py, px
                parent[py] = px
                if rank[px] == rank[py]:
                    rank[px] += 1

        # 超级优化：分块邻接过滤 + 空间索引
        potential_pairs = self._find_adjacent_tile_pairs(all_masks)

        logger.debug(f"Found {len(potential_pairs)} potential overlap pairs (adjacent tiles only)")

        # 只对可能重叠的掩膜对计算IoU
        for i, j in potential_pairs:
            mask_info_i = all_masks[i]
            mask_info_j = all_masks[j]

            # 第一步：快速检查边界框是否重叠（空间预筛选）
            if not self._masks_overlap(mask_info_i, mask_info_j):
                continue  # 边界框不重叠，跳过

            # 第二步：检查是否有真实的像素重叠（关键！）
            if not self._has_pixel_overlap(mask_info_i, mask_info_j):
                continue  # 边界框重叠但像素不重叠，跳过（保持原始掩膜）

            # 第三步：有真实像素重叠，使用分级策略决定是否合并
            box_iou = self._calculate_mask_bbox_iou(mask_info_i, mask_info_j)

            if box_iou > self.high_threshold:
                # 高置信度重叠，直接合并
                union(i, j)
            elif box_iou > self.low_threshold:
                # 中等置信度，使用mask IoU精确计算
                mask_iou = self._calculate_mask_iou_fast(mask_info_i, mask_info_j)
                if mask_iou > self.low_threshold:
                    union(i, j)
            # box_iou <= low_threshold: 不合并，保持各自独立的掩膜

        # 收集每个最终ID的所有成员
        final_groups = {}
        for i in range(len(all_masks)):
            root = find(i)
            if root not in final_groups:
                final_groups[root] = []
            final_groups[root].append(i)

        # 为每个组合并掩膜
        return self._build_merged_results(final_groups, all_masks, img_shape)

    def _find_adjacent_tile_pairs(self, all_masks: List[dict]) -> Set[tuple]:
        """
        找出相邻分块中可能重叠的掩膜对（分块邻接优化）

        只比较相邻分块（上下左右）中的掩膜，大幅减少比较次数

        Args:
            all_masks: 掩膜信息列表（需包含 tile_row, tile_col）

        Returns:
            可能重叠的掩膜对集合 {(i, j), ...}
        """
        if not all_masks:
            return set()

        # 按分块分组
        tiles_map = {}  # (row, col) -> [mask_indices]
        for idx, mask_info in enumerate(all_masks):
            tile_row = mask_info.get('tile_row', 0)
            tile_col = mask_info.get('tile_col', 0)
            key = (tile_row, tile_col)
            if key not in tiles_map:
                tiles_map[key] = []
            tiles_map[key].append(idx)

        pairs = set()
        processed_tiles = set()

        # 8方向邻接：上下左右 + 对角线
        directions = [(-1, -1), (-1, 0), (-1, 1),
                      (0, -1),          (0, 1),
                      (1, -1),  (1, 0), (1, 1)]

        for (row, col), mask_indices in tiles_map.items():
            # 注意：不检查当前分块内的掩膜对，因为同分块内的掩膜不应该合并
            # 只有跨越分块边界的掩膜才需要合并

            # 检查相邻分块
            for dr, dc in directions:
                neighbor_key = (row + dr, col + dc)
                if neighbor_key in tiles_map:
                    # 避免重复检查
                    check_key = tuple(sorted([ (row, col), neighbor_key ]))
                    if check_key in processed_tiles:
                        continue
                    processed_tiles.add(check_key)

                    # 比较两个相邻分块中的所有掩膜对
                    for idx1 in mask_indices:
                        for idx2 in tiles_map[neighbor_key]:
                            # 快速边界框检查
                            if self._masks_overlap(all_masks[idx1], all_masks[idx2]):
                                pairs.add((min(idx1, idx2), max(idx1, idx2)))

        return pairs

    def _find_potential_overlap_pairs(self, all_masks: List[dict]) -> List[tuple]:
        """
        基于分块位置快速找出可能重叠的掩膜对（旧版本 - O(N²)复杂度）

        Args:
            all_masks: 掩膜信息列表

        Returns:
            可能重叠的掩膜对列表 [(i, j), ...]
        """
        pairs = []

        # 计算每个掩膜的边界框（全局坐标）
        bboxes = []
        for mask_info in all_masks:
            offset = mask_info['offset']
            shape = mask_info['shape']
            h, w = shape
            # 边界框 (x1, y1, x2, y2)
            bbox = (offset[1], offset[0], offset[1] + w, offset[0] + h)
            bboxes.append(bbox)

        # 只检查边界框重叠的对 - O(N²)复杂度
        for i in range(len(all_masks)):
            for j in range(i + 1, len(all_masks)):
                if self._bbox_overlap(bboxes[i], bboxes[j]):
                    pairs.append((i, j))

        return pairs

    def _find_potential_overlap_pairs_optimized(self, all_masks: List[dict]) -> Set[tuple]:
        """
        基于空间网格索引快速找出可能重叠的掩膜对（优化版本）

        使用空间网格索引将O(N²)复杂度降低到接近O(N)

        Args:
            all_masks: 掩膜信息列表

        Returns:
            可能重叠的掩膜对集合 {(i, j), ...} (保证 i < j)
        """
        # 估算网格大小：使用平均掩膜大小
        if not all_masks:
            return set()

        # 计算平均掩膜尺寸作为网格单元大小
        total_area = 0
        for mask_info in all_masks:
            shape = mask_info['shape']
            total_area += shape[0] * shape[1]

        avg_area = total_area // len(all_masks) if all_masks else 512 * 512
        cell_size = int(np.sqrt(avg_area))

        # 创建空间索引
        spatial_index = SpatialGridIndex(cell_size=cell_size)

        # 插入所有掩膜的边界框
        for idx, mask_info in enumerate(all_masks):
            offset = mask_info['offset']
            shape = mask_info['shape']
            h, w = shape
            # 边界框 (x1, y1, x2, y2)
            bbox = (offset[1], offset[0], offset[1] + w, offset[0] + h)
            spatial_index.insert(idx, bbox)

        # 查询可能重叠的对
        return spatial_index.query_potential_overlaps()

    def _masks_overlap(self, mask_info1: dict, mask_info2: dict, bbox_cache1: tuple = None, bbox_cache2: tuple = None) -> bool:
        """
        检查两个掩膜的边界框是否重叠（修复：使用实际掩膜bbox而非分块bbox）

        Args:
            mask_info1: 掩膜信息1
            mask_info2: 掩膜信息2
            bbox_cache1: 掩膜1的bbox缓存（可选，避免重复计算）
            bbox_cache2: 掩膜2的bbox缓存（可选，避免重复计算）

        Returns:
            是否重叠
        """
        # 【修复】计算掩膜的实际bbox，而不是分块的bbox
        if bbox_cache1 is not None:
            x1_min, y1_min, x1_max, y1_max = bbox_cache1
        else:
            bbox1 = self._get_mask_bbox_global(mask_info1)
            x1_min, y1_min, x1_max, y1_max = bbox1

        if bbox_cache2 is not None:
            x2_min, y2_min, x2_max, y2_max = bbox_cache2
        else:
            bbox2 = self._get_mask_bbox_global(mask_info2)
            x2_min, y2_min, x2_max, y2_max = bbox2

        # 检查是否重叠（使用严格不等号，只有真正重叠才返回True）
        return not (x1_max <= x2_min or x2_max <= x1_min or
                    y1_max <= y2_min or y2_max <= y1_min)

    def _calculate_mask_bbox_iou(self, mask_info1: dict, mask_info2: dict) -> float:
        """
        计算两个掩膜边界框的IoU（快速版本）

        Args:
            mask_info1: 掩膜信息1
            mask_info2: 掩膜信息2

        Returns:
            边界框IoU值
        """
        offset1 = mask_info1['offset']
        shape1 = mask_info1['shape']
        offset2 = mask_info2['offset']
        shape2 = mask_info2['shape']

        # 计算两个边界框
        x1_min, y1_min = offset1[1], offset1[0]
        x1_max, y1_max = x1_min + shape1[1], y1_min + shape1[0]
        x2_min, y2_min = offset2[1], offset2[0]
        x2_max, y2_max = x2_min + shape2[1], y2_min + shape2[0]

        # 计算交集
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0

        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

        # 计算并集
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area

        if union_area <= 0:
            return 0.0

        return inter_area / union_area

    def _has_pixel_overlap(self, mask_info1: dict, mask_info2: dict) -> bool:
        """
        快速检查两个掩膜是否有真实的像素重叠

        只检查重叠区域内是否两个掩膜都有非零像素，不计算精确IoU
        这是区分"边界框重叠"和"真实掩膜重叠"的关键

        Args:
            mask_info1: 掩膜信息1
            mask_info2: 掩膜信息2

        Returns:
            是否有像素重叠
        """
        mask1 = mask_info1['mask']
        offset1 = mask_info1['offset']
        shape1 = mask_info1['shape']

        mask2 = mask_info2['mask']
        offset2 = mask_info2['offset']
        shape2 = mask_info2['shape']

        # 计算重叠区域在全局坐标中的范围
        x1_min, y1_min = offset1[1], offset1[0]
        x1_max, y1_max = x1_min + shape1[1], y1_min + shape1[0]
        x2_min, y2_min = offset2[1], offset2[0]
        x2_max, y2_max = x2_min + shape2[1], y2_min + shape2[0]

        # 重叠区域
        overlap_x_min = max(x1_min, x2_min)
        overlap_y_min = max(y1_min, y2_min)
        overlap_x_max = min(x1_max, x2_max)
        overlap_y_max = min(y1_max, y2_max)

        if overlap_x_max <= overlap_x_min or overlap_y_max <= overlap_y_min:
            return False

        # 计算在各自掩膜中的局部坐标
        local_x1 = int(overlap_x_min - x1_min)
        local_y1 = int(overlap_y_min - y1_min)
        local_x2 = int(overlap_x_min - x2_min)
        local_y2 = int(overlap_y_min - y2_min)

        overlap_w = int(overlap_x_max - overlap_x_min)
        overlap_h = int(overlap_y_max - overlap_y_min)

        # 提取重叠区域的掩膜
        overlap_mask1 = mask1[local_y1:local_y1+overlap_h, local_x1:local_x1+overlap_w]
        overlap_mask2 = mask2[local_y2:local_y2+overlap_h, local_x2:local_x2+overlap_w]

        # 检查是否有像素重叠（两个掩膜都非零的位置）
        # 使用 any() 快速检查，避免完整计算
        has_overlap = np.any((overlap_mask1 > 0) & (overlap_mask2 > 0))

        return has_overlap

    def _calculate_mask_iou_fast(self, mask_info1: dict, mask_info2: dict) -> float:
        """
        快速计算两个掩膜的IoU（只计算重叠区域）

        Args:
            mask_info1: 掩膜信息1
            mask_info2: 掩膜信息2

        Returns:
            IoU值
        """
        mask1 = mask_info1['mask']
        offset1 = mask_info1['offset']
        shape1 = mask_info1['shape']

        mask2 = mask_info2['mask']
        offset2 = mask_info2['offset']
        shape2 = mask_info2['shape']

        # 计算重叠区域在全局坐标中的范围
        x1_min, y1_min = offset1[1], offset1[0]
        x1_max, y1_max = x1_min + shape1[1], y1_min + shape1[0]
        x2_min, y2_min = offset2[1], offset2[0]
        x2_max, y2_max = x2_min + shape2[1], y2_min + shape2[0]

        # 重叠区域
        overlap_x_min = max(x1_min, x2_min)
        overlap_y_min = max(y1_min, y2_min)
        overlap_x_max = min(x1_max, x2_max)
        overlap_y_max = min(y1_max, y2_max)

        if overlap_x_max <= overlap_x_min or overlap_y_max <= overlap_y_min:
            return 0.0

        # 计算在各自掩膜中的局部坐标
        local_x1 = overlap_x_min - x1_min
        local_y1 = overlap_y_min - y1_min
        local_x2 = overlap_x_max - x1_min
        local_y2 = overlap_y_max - y1_min

        local2_x1 = overlap_x_min - x2_min
        local2_y1 = overlap_y_min - y2_min
        local2_x2 = overlap_x_max - x2_min
        local2_y2 = overlap_y_max - y2_min

        # 提取重叠区域的掩膜
        overlap_mask1 = mask1[local_y1:local_y2, local_x1:local_x2]
        overlap_mask2 = mask2[local2_y1:local2_y2, local2_x1:local2_x2]

        # 计算IoU
        overlap = (overlap_mask1 > 0) & (overlap_mask2 > 0)
        union = (overlap_mask1 > 0) | (overlap_mask2 > 0)

        if np.sum(union) == 0:
            return 0.0

        return float(np.sum(overlap) / np.sum(union))

    def _calculate_mask_iou_precise(self, mask_info1: dict, mask_info2: dict) -> float:
        """
        精确计算两个掩膜的IoU（使用连通性检查）

        Args:
            mask_info1: 掩膜信息1
            mask_info2: 掩膜信息2

        Returns:
            IoU值
        """
        # 构建临时的全局掩膜用于连通性检查
        mask1 = mask_info1['mask']
        offset1 = mask_info1['offset']
        shape1 = mask_info1['shape']

        mask2 = mask_info2['mask']
        offset2 = mask_info2['offset']
        shape2 = mask_info2['shape']

        # 计算需要的区域大小
        all_x = [offset1[1], offset1[1] + shape1[1], offset2[1], offset2[1] + shape2[1]]
        all_y = [offset1[0], offset1[0] + shape1[0], offset2[0], offset2[0] + shape2[0]]
        region_x_min, region_x_max = min(all_x), max(all_x)
        region_y_min, region_y_max = min(all_y), max(all_y)

        region_w = region_x_max - region_x_min
        region_h = region_y_max - region_y_min

        # 创建区域掩膜
        region_mask1 = np.zeros((region_h, region_w), dtype=np.uint8)
        region_mask2 = np.zeros((region_h, region_w), dtype=np.uint8)

        # 放置第一个掩膜
        local_x1 = offset1[1] - region_x_min
        local_y1 = offset1[0] - region_y_min
        region_mask1[local_y1:local_y1+shape1[0], local_x1:local_x1+shape1[1]] = mask1

        # 放置第二个掩膜
        local_x2 = offset2[1] - region_x_min
        local_y2 = offset2[0] - region_y_min
        region_mask2[local_y2:local_y2+shape2[0], local_x2:local_x2+shape2[1]] = mask2

        # 检查连通性
        return self._check_contour_connectivity_with_iou(region_mask1, region_mask2)

    def _check_contour_connectivity_with_iou(
        self,
        mask1: np.ndarray,
        mask2: np.ndarray
    ) -> float:
        """
        检查两个掩膜的连通性并返回IoU

        Args:
            mask1: 掩膜1
            mask2: 掩膜2

        Returns:
            IoU值
        """
        overlap = (mask1 > 0).astype(np.uint8) & (mask2 > 0).astype(np.uint8)

        if np.sum(overlap) == 0:
            return 0.0

        union = (mask1 > 0).astype(np.uint8) | (mask2 > 0).astype(np.uint8)

        # 检查连通性
        labeled, num_features = ndimage.label(overlap)

        if num_features > 0:
            sizes = ndimage.sum(overlap, labeled, range(1, num_features + 1))
            if np.max(sizes) > 10:  # 至少10个像素的连通区域
                return float(np.sum(overlap) / (np.sum(union) + 1e-7))

        return float(np.sum(overlap) / (np.sum(union) + 1e-7))

    def _get_mask_bbox(self, mask: np.ndarray) -> tuple:
        """获取掩膜的边界框 (x1, y1, x2, y2)"""
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not np.any(rows) or not np.any(cols):
            return (0, 0, 0, 0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return (cmin, rmin, cmax, rmax)

    def _find_overlapping_pairs(
        self,
        temp_masks: dict
    ) -> List[tuple]:
        """
        找出所有可能重叠的掩膜对

        Args:
            temp_masks: {temp_id: mask_info}

        Returns:
            可能重叠的掩膜对列表 [(i, j), ...]
        """
        pairs = []
        ids = list(temp_masks.keys())

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id_i, id_j = ids[i], ids[j]

                bbox_i = temp_masks[id_i]['bbox']
                bbox_j = temp_masks[id_j]['bbox']

                # 检查边界框是否重叠
                if self._bbox_overlap(bbox_i, bbox_j):
                    pairs.append((id_i, id_j))

        return pairs

    def _bbox_overlap(self, bbox1: tuple, bbox2: tuple) -> bool:
        """检查两个边界框是否重叠"""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2

        return not (x1_max < x2_min or x2_max < x1_min or
                    y1_max < y2_min or y2_max < y1_min)

    def _calculate_mask_iou_in_overlap(
        self,
        mask1: np.ndarray,
        mask2: np.ndarray
    ) -> float:
        """
        计算两个掩膜在重叠区域内的IoU

        Args:
            mask1: 掩膜1
            mask2: 掩膜2

        Returns:
            IoU值
        """
        # 找到重叠区域
        overlap = mask1.astype(np.uint8) & mask2.astype(np.uint8)

        if np.sum(overlap) == 0:
            return 0.0

        # 计算并集
        union = mask1.astype(np.uint8) | mask2.astype(np.uint8)

        # 计算IoU（在重叠区域范围内）
        iou = np.sum(overlap) / (np.sum(union) + 1e-7)

        return float(iou)

    def _check_contour_connectivity(
        self,
        mask1: np.ndarray,
        mask2: np.ndarray
    ) -> bool:
        """
        检查两个掩膜在重叠区域是否连通

        Args:
            mask1: 掩膜1
            mask2: 掩膜2

        Returns:
            是否连通
        """
        # 找到重叠区域
        overlap = (mask1 > 0).astype(np.uint8) & (mask2 > 0).astype(np.uint8)

        if np.sum(overlap) == 0:
            return False

        # 检查连通性
        labeled, num_features = ndimage.label(overlap)

        # 如果有较大连通区域，认为是连通的
        if num_features > 0:
            sizes = ndimage.sum(overlap, labeled, range(1, num_features + 1))
            if np.max(sizes) > 10:  # 至少10个像素的连通区域
                return True

        return False

    def _merge_classification(
        self,
        tile_results: List[TileResult],
        img_shape: tuple,
        geotransform: tuple = None,
        crs: str = None
    ) -> MergedResult:
        """
        合并分类结果（多数投票）

        Args:
            tile_results: 分块结果列表
            img_shape: 原始影像尺寸
            geotransform: 仿射变换参数
            crs: 坐标系

        Returns:
            MergedResult: 合并后的结果
        """
        # 收集所有分类结果
        class_votes = []

        for result in tile_results:
            if result.boxes is not None and len(result.boxes) > 0:
                # 分类结果通常在boxes中，格式为 [conf, cls]
                if result.boxes.shape[1] >= 2:
                    cls = int(result.boxes[0, 1])  # 假设类别在第二列
                    conf = result.boxes[0, 0] if result.boxes.shape[1] > 0 else 0.5
                    class_votes.append((cls, conf))

        if not class_votes:
            return MergedResult(
                img_shape=img_shape,
                geotransform=geotransform,
                crs=crs,
                num_tiles=len(tile_results),
                total_instances=0
            )

        # 使用加权投票
        class_scores = {}
        for cls, conf in class_votes:
            class_scores[cls] = class_scores.get(cls, 0) + conf

        # 选择最高分的类别
        final_class = max(class_scores.items(), key=lambda x: x[1])[0]
        final_conf = class_scores[final_class] / len(class_votes)

        return MergedResult(
            class_ids=np.array([final_class]),
            confidences=np.array([final_conf]),
            img_shape=img_shape,
            geotransform=geotransform,
            crs=crs,
            num_tiles=len(tile_results),
            total_instances=1
        )


# ============================================================================
# 并行处理辅助类
# ============================================================================

class IoUCalculator:
    """统一的掩膜 IoU 计算接口"""

    def __init__(self, use_numba=True):
        """
        Args:
            use_numba: 是否使用 Numba 加速
        """
        self.use_numba = use_numba and NUMBA_AVAILABLE
        self._compiled = False

    def _ensure_compiled(self):
        """确保 Numba 函数已编译"""
        if self.use_numba and not self._compiled:
            # 预热编译
            dummy_mask = np.zeros((10, 10), dtype=np.uint8)
            _calculate_mask_iou_numba(
                dummy_mask, dummy_mask, 0, 0, 0, 0
            )
            self._compiled = True

    def calculate_batch(self, pairs, mask_data, mask_offsets):
        """
        批量计算多对掩膜的 IoU

        Args:
            pairs: 掩膜对列表 [(idx1, idx2), ...]
            mask_data: 掩膜数据列表
            mask_offsets: 掩膜偏移列表

        Returns:
            IoU 值列表
        """
        self._ensure_compiled()

        if not pairs:
            return []

        # 【性能优化】对于大量数据，使用批量向量化版本
        if self.use_numba and len(pairs) >= 50:
            return self._calculate_batch_vectorized(pairs, mask_data, mask_offsets)

        # 小量数据：逐个计算
        results = []
        for idx1, idx2 in pairs:
            iou = _calculate_mask_iou_numba(
                mask_data[idx1], mask_data[idx2],
                mask_offsets[idx1][0], mask_offsets[idx1][1],
                mask_offsets[idx2][0], mask_offsets[idx2][1]
            )
            results.append(iou)
        return results

    def _calculate_batch_vectorized(self, pairs, mask_data, mask_offsets):
        """
        批量向量化计算（针对大规模数据优化）

        将所有掩膜组织成 numpy 数组，使用批量 Numba 函数计算

        Args:
            pairs: 掩膜对列表
            mask_data: 掩膜数据列表
            mask_offsets: 掩膜偏移列表

        Returns:
            IoU 值列表
        """
        try:
            # 提取索引
            idx1_list = [p[0] for p in pairs]
            idx2_list = [p[1] for p in pairs]

            # 找到最大尺寸，用于填充
            max_h = max(mask_data[i].shape[0] for i in idx1_list + idx2_list)
            max_w = max(mask_data[i].shape[1] for i in idx1_list + idx2_list)

            # 构建批量数组
            n_pairs = len(pairs)
            masks1 = np.zeros((n_pairs, max_h, max_w), dtype=np.uint8)
            masks2 = np.zeros((n_pairs, max_h, max_w), dtype=np.uint8)
            offsets1 = np.zeros((n_pairs, 2), dtype=np.int32)
            offsets2 = np.zeros((n_pairs, 2), dtype=np.int32)

            for i, (idx1, idx2) in enumerate(pairs):
                h1, w1 = mask_data[idx1].shape
                h2, w2 = mask_data[idx2].shape
                masks1[i, :h1, :w1] = mask_data[idx1]
                masks2[i, :h2, :w2] = mask_data[idx2]
                offsets1[i] = mask_offsets[idx1]
                offsets2[i] = mask_offsets[idx2]

            # 使用批量 Numba 函数计算
            iou_matrix = _calculate_mask_iou_batch_numba(
                masks1, masks2, offsets1, offsets2
            )

            # 提取对角线元素（因为我们计算的是配对）
            return [iou_matrix[i, i] for i in range(n_pairs)]

        except Exception as e:
            logger.warning(f"批量计算失败，降级到逐个计算: {e}")
            # 降级到逐个计算
            results = []
            for idx1, idx2 in pairs:
                iou = _calculate_mask_iou_numba(
                    mask_data[idx1], mask_data[idx2],
                    mask_offsets[idx1][0], mask_offsets[idx1][1],
                    mask_offsets[idx2][0], mask_offsets[idx2][1]
                )
                results.append(iou)
            return results

    def _calculate_fallback(self, mask1, mask2, offset1, offset2):
        """降级计算版本（不使用 Numba）"""
        y1_min, y1_max = offset1[0], offset1[0] + mask1.shape[0]
        x1_min, x1_max = offset1[1], offset1[1] + mask1.shape[1]
        y2_min, y2_max = offset2[0], offset2[0] + mask2.shape[0]
        x2_min, x2_max = offset2[1], offset2[1] + mask2.shape[1]

        overlap_x_min = max(x1_min, x2_min)
        overlap_y_min = max(y1_min, y2_min)
        overlap_x_max = min(x1_max, x2_max)
        overlap_y_max = min(y1_max, y2_max)

        if overlap_x_max <= overlap_x_min or overlap_y_max <= overlap_y_min:
            return 0.0

        local_y1 = overlap_y_min - y1_min
        local_x1 = overlap_x_min - x1_min
        local_y2 = overlap_y_min - y2_min
        local_x2 = overlap_x_min - x2_min

        overlap_w = overlap_x_max - overlap_x_min
        overlap_h = overlap_y_max - overlap_y_min

        overlap_mask1 = mask1[local_y1:local_y1+overlap_h, local_x1:local_x1+overlap_w]
        overlap_mask2 = mask2[local_y2:local_y2+overlap_h, local_x2:local_x2+overlap_w]

        overlap = (overlap_mask1 > 0) & (overlap_mask2 > 0)
        union = (overlap_mask1 > 0) | (overlap_mask2 > 0)

        if np.sum(union) == 0:
            return 0.0
        return float(np.sum(overlap) / np.sum(union))


class ParallelScheduler:
    """并行任务调度器"""

    def __init__(self, parallel_threshold=100, max_workers=None):
        """
        Args:
            parallel_threshold: 启用并行化的最小任务数
            max_workers: 最大进程数（None 表示自动计算）
        """
        self.parallel_threshold = parallel_threshold
        self.cpu_count = os.cpu_count() or 1

        if max_workers is None:
            # 【动态设置】使用 CPU 核心数的 75%，最少 2 个，最多 16 个
            self.max_workers = max(2, min(int(self.cpu_count * 0.75), 16))
        else:
            self.max_workers = max_workers

    def should_parallelize(self, n_pairs):
        """判断是否值得并行化"""
        return n_pairs >= self.parallel_threshold

    def split_into_chunks(self, items, n_chunks=None):
        """
        将数据分成多个块

        Args:
            items: 要分割的数据列表
            n_chunks: 块数量（None 表示使用 max_workers）

        Returns:
            数据块列表
        """
        if n_chunks is None:
            n_chunks = self.max_workers

        chunk_size = (len(items) + n_chunks - 1) // n_chunks
        return [items[i:i+chunk_size] for i in range(0, len(items), chunk_size)]

    def execute_parallel(self, func, data_chunks):
        """
        并行执行任务

        Args:
            func: 执行函数
            data_chunks: 数据块列表

        Returns:
            合并后的结果列表
        """
        import traceback
        import sys

        try:
            logger.debug(f"开始并行执行，chunks 数量: {len(data_chunks)}")
            for i, chunk in enumerate(data_chunks):
                logger.debug(f"Chunk {i} 大小: {len(chunk) if hasattr(chunk, '__len__') else 'N/A'}")

            with multiprocessing.Pool(processes=self.max_workers) as pool:
                results = pool.map(func, data_chunks)

            logger.debug(f"并行执行完成，results 数量: {len(results)}")
            for i, r in enumerate(results):
                logger.debug(f"Result {i} 类型: {type(r)}, 长度: {len(r) if isinstance(r, list) else 'N/A'}")

            # 展平结果列表
            flattened = []
            for i, r in enumerate(results):
                if isinstance(r, list):
                    flattened.extend(r)
                else:
                    logger.warning(f"Chunk {i} 返回了意外的类型: {type(r)}, 期望 list. 值: {r}")
                    flattened.append(r)
            return flattened
        except Exception as e:
            logger.warning(f"并行执行失败: {e}")
            logger.warning(traceback.format_exc())
            raise


# ============================================================================
# 全局 worker 函数（可被 pickle 序列化）
# ============================================================================

def _calculate_iou_chunk(args):
    """
    计算一个 chunk 的 IoU 值（worker 函数）

    这个函数必须是模块级函数，以便被 multiprocessing 序列化。

    Args:
        args: (pairs_chunk, mask_data, mask_offsets)
            - pairs_chunk: 掩膜对列表
            - mask_data: 掩膜数据列表
            - mask_offsets: 掩膜偏移列表

    Returns:
        IoU 值列表
    """
    pairs_chunk, mask_data, mask_offsets = args

    results = []
    for idx1, idx2 in pairs_chunk:
        if NUMBA_AVAILABLE:
            iou = _calculate_mask_iou_numba(
                mask_data[idx1], mask_data[idx2],
                mask_offsets[idx1][0], mask_offsets[idx1][1],
                mask_offsets[idx2][0], mask_offsets[idx2][1]
            )
        else:
            # 降级版本
            y1_min, y1_max = mask_offsets[idx1][0], mask_offsets[idx1][0] + mask_data[idx1].shape[0]
            x1_min, x1_max = mask_offsets[idx1][1], mask_offsets[idx1][1] + mask_data[idx1].shape[1]
            y2_min, y2_max = mask_offsets[idx2][0], mask_offsets[idx2][0] + mask_data[idx2].shape[0]
            x2_min, x2_max = mask_offsets[idx2][1], mask_offsets[idx2][1] + mask_data[idx2].shape[1]

            overlap_x_min = max(x1_min, x2_min)
            overlap_y_min = max(y1_min, y2_min)
            overlap_x_max = min(x1_max, x2_max)
            overlap_y_max = min(y1_max, y2_max)

            if overlap_x_max <= overlap_x_min or overlap_y_max <= overlap_y_min:
                iou = 0.0
            else:
                local_y1 = overlap_y_min - y1_min
                local_x1 = overlap_x_min - x1_min
                local_y2 = overlap_y_min - y2_min
                local_x2 = overlap_x_min - x2_min

                overlap_w = overlap_x_max - overlap_x_min
                overlap_h = overlap_y_max - overlap_y_min

                overlap_mask1 = mask_data[idx1][local_y1:local_y1+overlap_h, local_x1:local_x1+overlap_w]
                overlap_mask2 = mask_data[idx2][local_y2:local_y2+overlap_h, local_x2:local_x2+overlap_w]

                overlap = (overlap_mask1 > 0) & (overlap_mask2 > 0)
                union = (overlap_mask1 > 0) | (overlap_mask2 > 0)

                if np.sum(union) == 0:
                    iou = 0.0
                else:
                    iou = float(np.sum(overlap) / np.sum(union))

        results.append(iou)

    return results


# ============================================================================
# 矢量多边形合并函数
# ============================================================================

def _calculate_polygon_iou(poly1_points: np.ndarray, poly2_points: np.ndarray) -> float:
    """
    计算两个多边形的IoU（基于Shapely）

    Args:
        poly1_points: [N, 2] 多边形1的顶点坐标
        poly2_points: [M, 2] 多边形2的顶点坐标

    Returns:
        IoU值
    """
    try:
        from shapely.geometry import Polygon

        if len(poly1_points) < 3 or len(poly2_points) < 3:
            return 0.0

        poly1 = Polygon(poly1_points)
        poly2 = Polygon(poly2_points)

        if not poly1.is_valid or not poly2.is_valid:
            poly1 = poly1.buffer(0)
            poly2 = poly2.buffer(0)

        if not poly1.is_valid or not poly2.is_valid:
            return 0.0

        inter_area = poly1.intersection(poly2).area
        union_area = poly1.union(poly2).area

        if union_area == 0:
            return 0.0

        return inter_area / union_area

    except Exception as e:
        logger.warning(f"计算多边形IoU时出错: {e}")
        return 0.0


def _merge_two_polygons(poly1_points: np.ndarray, poly2_points: np.ndarray) -> np.ndarray:
    """
    合并两个多边形（使用Shapely的union操作）

    Args:
        poly1_points: [N, 2] 多边形1的顶点坐标
        poly2_points: [M, 2] 多边形2的顶点坐标

    Returns:
        合并后的多边形顶点坐标 [K, 2]
    """
    try:
        from shapely.geometry import Polygon

        if len(poly1_points) < 3:
            return poly2_points
        if len(poly2_points) < 3:
            return poly1_points

        poly1 = Polygon(poly1_points)
        poly2 = Polygon(poly2_points)

        if not poly1.is_valid:
            poly1 = poly1.buffer(0)
        if not poly2.is_valid:
            poly2 = poly2.buffer(0)

        merged = poly1.union(poly2)

        if merged.is_empty:
            # union失败，返回面积较大的
            return poly1_points if poly1.area >= poly2.area else poly2_points

        # 如果结果是MultiPolygon，返回最大的那个
        if merged.geom_type == 'MultiPolygon':
            largest = max(merged.geoms, key=lambda g: g.area)
            merged = largest

        # 提取外轮廓坐标
        if merged.geom_type == 'Polygon':
            coords = np.array(merged.exterior.coords)
            # 移除最后一个重复点（闭合点）
            if len(coords) > 1 and np.allclose(coords[0], coords[-1]):
                coords = coords[:-1]
            return coords
        else:
            # 降级：返回面积较大的原始多边形
            return poly1_points if poly1.area >= poly2.area else poly2_points

    except Exception as e:
        logger.warning(f"合并多边形时出错: {e}，返回第一个多边形")
        return poly1_points


def _simplify_polygon(points: np.ndarray, tolerance: float = 0.1) -> np.ndarray:
    """
    简化多边形（减少顶点数量）

    Args:
        points: [N, 2] 多边形顶点
        tolerance: 简化容差

    Returns:
        简化后的顶点
    """
    try:
        from shapely.geometry import Polygon

        if len(points) < 4:
            return points

        poly = Polygon(points)
        if not poly.is_valid:
            poly = poly.buffer(0)

        simplified = poly.simplify(tolerance, preserve_topology=True)

        if simplified.is_empty or not simplified.is_valid:
            return points

        coords = np.array(simplified.exterior.coords)
        if len(coords) > 1 and np.allclose(coords[0], coords[-1]):
            coords = coords[:-1]

        return coords if len(coords) >= 3 else points

    except Exception:
        return points
