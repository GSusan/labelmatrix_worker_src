# -*- coding: utf-8 -*-
"""
大影像分块处理器
"""

from typing import Generator, Tuple, Union, Dict, Set, List
import numpy as np
from pathlib import Path


class TileProcessor:
    """大影像分块处理器"""

    def __init__(self, tile_size: int = 512, overlap: float = 0.1):
        """
        Args:
            tile_size: 切片大小（像素）
            overlap: 切片重叠比例（0-1）
        """
        self.tile_size = tile_size
        self.overlap = overlap
        self.overlap_pixels = int(tile_size * overlap)

        # 存储分块网格信息
        self.grid_rows = 0
        self.grid_cols = 0
        self.tile_adjacency: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}

    def _calculate_grid_info(self, height: int, width: int) -> Tuple[int, int]:
        """
        计算分块网格信息

        Args:
            height: 影像高度
            width: 影像宽度

        Returns:
            (grid_rows, grid_cols): 网格行列数
        """
        step = self.tile_size - self.overlap_pixels
        grid_rows = (height + step - 1) // step  # 向上取整
        grid_cols = (width + step - 1) // step
        return grid_rows, grid_cols

    def _build_adjacency_map(self, grid_rows: int, grid_cols: int) -> None:
        """
        构建分块邻接关系图（8方向）

        Args:
            grid_rows: 网格行数
            grid_cols: 网格列数
        """
        self.tile_adjacency = {}

        # 8方向偏移
        directions = [(-1, -1), (-1, 0), (-1, 1),
                      (0, -1),           (0, 1),
                      (1, -1),  (1, 0), (1, 1)]

        for row in range(grid_rows):
            for col in range(grid_cols):
                neighbors = []
                for dr, dc in directions:
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < grid_rows and 0 <= nc < grid_cols:
                        neighbors.append((nr, nc))
                self.tile_adjacency[(row, col)] = neighbors

    def get_overlap_region_between_tiles(
        self,
        tile1_offset: Tuple[int, int],
        tile1_shape: Tuple[int, int],
        tile2_offset: Tuple[int, int],
        tile2_shape: Tuple[int, int]
    ) -> Tuple[int, int, int, int]:
        """
        计算两个分块之间的重叠区域（全局坐标）

        Args:
            tile1_offset: 分块1偏移 (row, col)
            tile1_shape: 分块1尺寸 (height, width)
            tile2_offset: 分块2偏移 (row, col)
            tile2_shape: 分块2尺寸 (height, width)

        Returns:
            (x_min, y_min, x_max, y_max) 重叠区域的边界框，如果不重叠返回 None
        """
        r1, c1 = tile1_offset
        h1, w1 = tile1_shape
        r2, c2 = tile2_offset
        h2, w2 = tile2_shape

        # 计算重叠区域
        overlap_r_min = max(r1, r2)
        overlap_c_min = max(c1, c2)
        overlap_r_max = min(r1 + h1, r2 + h2)
        overlap_c_max = min(c1 + w1, c2 + w2)

        if overlap_r_max <= overlap_r_min or overlap_c_max <= overlap_c_min:
            return None

        return (overlap_c_min, overlap_r_min, overlap_c_max, overlap_r_max)

    def get_tile_grid_position(self, tile_id: int) -> Tuple[int, int]:
        """
        根据tile_id获取网格位置

        Args:
            tile_id: 分块ID

        Returns:
            (row, col): 网格位置
        """
        if self.grid_cols == 0:
            return (0, 0)
        return (tile_id // self.grid_cols, tile_id % self.grid_cols)

    def generate_tiles(
        self,
        image_array: np.ndarray
    ) -> Generator[Tuple[np.ndarray, Tuple[int, int]], None, None]:
        """
        生成影像切片

        Args:
            image_array: 输入影像数组 (H, W, C) 或 (H, W)

        Yields:
            (tile_array, (row_offset, col_offset)): 切片数组及其偏移
        """
        if image_array.ndim == 2:
            image_array = image_array[:, :, np.newaxis]

        height, width = image_array.shape[:2]

        # 计算步长
        step = self.tile_size - self.overlap_pixels

        # 遍历影像
        for row_start in range(0, height, step):
            for col_start in range(0, width, step):
                # 计算实际窗口大小（处理边界）
                row_end = min(row_start + self.tile_size, height)
                col_end = min(col_start + self.tile_size, width)

                # 读取切片
                tile = image_array[row_start:row_end, col_start:col_end].copy()

                yield tile, (row_start, col_start)

    def merge_tiles(
        self,
        tiles: List[Tuple[np.ndarray, Tuple[int, int]]],
        output_shape: Tuple[int, int, int]
    ) -> np.ndarray:
        """
        合并切片回完整影像

        Args:
            tiles: 切片列表，每项为(tile_array, (row_offset, col_offset))
            output_shape: 输出形状(H, W, C)

        Returns:
            合并后的完整影像
        """
        height, width, channels = output_shape
        result = np.zeros(output_shape, dtype=np.float32)
        weights = np.zeros((height, width, 1), dtype=np.float32)

        for tile, (row_off, col_off) in tiles:
            tile_h, tile_w = tile.shape[:2]

            # 创建边缘渐变权重
            weight_mask = self._create_weight_mask(tile_h, tile_w, channels)

            # 计算实际边界
            row_end = min(row_off + tile_h, height)
            col_end = min(col_off + tile_w, width)
            actual_h = row_end - row_off
            actual_w = col_end - col_off

            # 扩展tile维度如果需要
            if tile.ndim == 2:
                tile = tile[:, :, np.newaxis]

            # 累加结果和权重
            result[row_off:row_end, col_off:col_end] += \
                tile[:actual_h, :actual_w] * weight_mask[:actual_h, :actual_w]
            weights[row_off:row_end, col_off:col_end] += \
                weight_mask[:actual_h, :actual_w]

        # 归一化
        result = result / (weights + 1e-8)

        return result

    def _create_weight_mask(self, height: int, width: int, channels: int) -> np.ndarray:
        """创建边缘渐变权重掩膜"""
        mask = np.ones((height, width, 1), dtype=np.float32)

        # 边缘渐变区域大小
        border = max(1, self.overlap_pixels // 2)

        if border > 0:
            # 创建渐变
            for i in range(border):
                alpha = (i + 1) / (border + 1)

                # 上边界
                mask[i, :, :] *= alpha
                # 下边界
                mask[height - 1 - i, :, :] *= alpha
                # 左边界
                mask[:, i, :] *= alpha
                # 右边界
                mask[:, width - 1 - i, :] *= alpha

        # 扩展到所有通道
        return np.repeat(mask, channels, axis=2)

    def generate_tiles_with_georef(
        self,
        image_array: np.ndarray,
        geotransform: tuple,
        crs: str
    ) -> Generator['TileWithGeoRef', None, None]:
        """
        生成带地理参考信息的影像分块

        Args:
            image_array: 输入影像数组 (H, W, C) 或 (H, W)
            geotransform: GDAL仿射变换参数 (ul_x, x_res, x_rot, ul_y, y_rot, y_res)
            crs: 坐标系字符串

        Yields:
            TileWithGeoRef: 带地理参考的分块信息（包含grid_row, grid_col）
        """
        from ..handlers.rs_data_structures import TileWithGeoRef

        if image_array.ndim == 2:
            image_array = image_array[:, :, np.newaxis]

        height, width = image_array.shape[:2]

        # 计算网格信息
        self.grid_rows, self.grid_cols = self._calculate_grid_info(height, width)
        self._build_adjacency_map(self.grid_rows, self.grid_cols)

        # 计算步长
        step = self.tile_size - self.overlap_pixels

        tile_id = 0
        grid_row = 0
        # 遍历影像
        for row_start in range(0, height, step):
            grid_col = 0
            for col_start in range(0, width, step):
                # 计算实际窗口大小（处理边界）
                row_end = min(row_start + self.tile_size, height)
                col_end = min(col_start + self.tile_size, width)

                # 读取切片
                tile = image_array[row_start:row_end, col_start:col_end].copy()

                # 计算该分块的仿射变换参数
                tile_geotransform = self._get_tile_geotransform(
                    geotransform, row_start, col_start
                )

                yield TileWithGeoRef(
                    tile_id=tile_id,
                    tile_array=tile,
                    offset=(row_start, col_start),
                    shape=(row_end - row_start, col_end - col_start),
                    geotransform=tile_geotransform,
                    crs=crs,
                    grid_row=grid_row,
                    grid_col=grid_col
                )

                tile_id += 1
                grid_col += 1
            grid_row += 1

    def _get_tile_geotransform(
        self,
        global_geotransform: tuple,
        row_offset: int,
        col_offset: int
    ) -> tuple:
        """
        计算分块的仿射变换参数

        Args:
            global_geotransform: 全局仿射变换参数 (ul_x, x_res, x_rot, ul_y, y_rot, y_res)
            row_offset: 行偏移
            col_offset: 列偏移

        Returns:
            分块的仿射变换参数
        """
        ul_x, x_res, x_rot, ul_y, y_rot, y_res = global_geotransform

        # 计算分块左上角地理坐标
        tile_ul_x = ul_x + col_offset * x_res
        tile_ul_y = ul_y + row_offset * y_res

        return (tile_ul_x, x_res, x_rot, tile_ul_y, y_rot, y_res)


class ImageTileGenerator:
    """简单的影像切片生成器（不依赖GDAL）"""

    @staticmethod
    def generate_tiles(
        image: np.ndarray,
        tile_size: int = 512,
        overlap: int = 0
    ) -> Generator[Tuple[np.ndarray, int, int], None, None]:
        """
        生成影像切片

        Args:
            image: 输入影像 (H, W, C)
            tile_size: 切片大小
            overlap: 切片重叠像素数

        Yields:
            (tile, row, col): 切片及其位置
        """
        h, w = image.shape[:2]
        step = tile_size - overlap

        for row in range(0, h, step):
            for col in range(0, w, step):
                # 计算实际边界
                row_end = min(row + tile_size, h)
                col_end = min(col + tile_size, w)

                # 提取切片
                tile = image[row:row_end, col:col_end].copy()

                # 如果切片小于tile_size，进行填充
                if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                    padded_tile = np.zeros((tile_size, tile_size, image.shape[2]), dtype=image.dtype)
                    padded_tile[:tile.shape[0], :tile.shape[1]] = tile
                    tile = padded_tile

                yield tile, row, col

    @staticmethod
    def merge_tiles(
        tiles: List[Tuple[np.ndarray, int, int]],
        image_shape: Tuple[int, int, int]
    ) -> np.ndarray:
        """
        合并切片

        Args:
            tiles: (tile, row, col) 列表
            image_shape: 输出影像形状 (H, W, C)

        Returns:
            合并后的影像
        """
        h, w, c = image_shape
        result = np.zeros(image_shape, dtype=np.float32)
        count = np.zeros((h, w, 1), dtype=np.float32)

        for tile, row, col in tiles:
            tile_h, tile_w = tile.shape[:2]

            # 计算实际边界
            row_end = min(row + tile_h, h)
            col_end = min(col + tile_w, w)
            actual_h = row_end - row
            actual_w = col_end - col

            # 累加
            result[row:row_end, col:col_end] += tile[:actual_h, :actual_w]
            count[row:row_end, col:col_end] += 1

        # 平均
        result = result / (count + 1e-8)

        return result.astype(np.uint8)
