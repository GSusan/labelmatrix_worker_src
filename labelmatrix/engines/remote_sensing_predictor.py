# -*- coding: utf-8 -*-
"""
遥感影像大幅面预测器
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from tqdm import tqdm

try:
    from osgeo import gdal
    from osgeo import osr
    GDAL_AVAILABLE = True
except ImportError:
    GDAL_AVAILABLE = False

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

import numpy as np

from .base_engine import BaseEngine, PredictResult, TrainResult
from ..handlers.rs_data_structures import TileResult, TileWithGeoRef, MergedResult
from ..handlers.tile_processor import TileProcessor
from ..handlers.result_merger import ResultMerger
from ..handlers.geojson_exporter import GeoJSONExporter
from ..handlers.shapefile_exporter import ShapefileExporter
from ..handlers.pixel_result_exporter import PixelResultExporter
from ..handlers.visualization_exporter import LargeImageVisualizer
from ..exceptions.remote_sensing_errors import (
    ImageReadError,
    InvalidGeoReferenceError,
    ModelLoadError,
    RemoteSensingError
)

logger = logging.getLogger(__name__)


class RemoteSensingPredictor(BaseEngine):
    """遥感影像大幅面预测器

    专门用于大幅面遥感影像的分块预测，支持：
    - 自动分块处理大幅面影像
    - 地理参考信息保持
    - NMS去重合并
    - GeoJSON/Shapefile导出
    - 像素坐标结果导出（第一步识别结果）
    - 可视化成果导出（叠加检测结果）
    """

    SUPPORTED_TASKS = ['detect', 'segment', 'obb', 'classify', 'pose']

    def __init__(self, config: Dict[str, Any]):
        """
        初始化遥感预测器

        Args:
            config: 配置字典，包含:
                - task_type: 任务类型
                - model_path: 模型路径
                - predict: 预测配置，包含 source, save_dir, tile_size, tile_overlap 等
                - output_dir: 输出目录
        """
        super().__init__(config)

        # 检查依赖
        if not GDAL_AVAILABLE:
            raise ImportError("GDAL is required for remote sensing prediction. Install with: pip install gdal")
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("Ultralytics is required. Install with: pip install ultralytics")

        # 获取预测配置
        self.predict_config = config.get('predict', {})

        # 分块配置
        self.tile_size = self.predict_config.get('tile_size', 512)
        self.overlap = self.predict_config.get('tile_overlap', 0.1)

        # 推理配置
        self.conf_threshold = self.predict_config.get('conf_thres', 0.25)
        self.iou_threshold = self.predict_config.get('iou_thres', 0.45)
        self.imgsz = config.get('hyperparameters', {}).get('imgsz', 640)

        # 合并配置
        self.merge_iou_threshold = self.predict_config.get('merge_iou_threshold', 0.5)
        self.segment_high_threshold = self.predict_config.get('segment_high_threshold', 0.7)
        self.segment_low_threshold = self.predict_config.get('segment_low_threshold', 0.5)

        # 输出配置
        self.save_tiles = self.predict_config.get('save_tiles', False)
        self.save_pixel_results = self.predict_config.get('save_pixel_results', True)
        self.save_visualization = self.predict_config.get('save_visualization', True)
        self.file_naming = self.predict_config.get('file_naming', 'default')

        # 可视化配置
        self.show_conf = self.predict_config.get('show_conf', True)
        self.show_labels = self.predict_config.get('show_labels', True)
        self.line_width = self.predict_config.get('line_width', 2)
        self.vis_conf_threshold = self.predict_config.get('vis_conf_threshold', 0.3)

        # 初始化组件
        self.model = None
        self.device = self._get_device()
        self.tile_processor = TileProcessor(
            tile_size=self.tile_size,
            overlap=self.overlap
        )
        self.result_merger = ResultMerger(
            task_type=self.task_type,
            iou_threshold=self.merge_iou_threshold,
            segment_high_threshold=self.segment_high_threshold,
            segment_low_threshold=self.segment_low_threshold,
            cross_tile_bbox_iou_threshold=self.predict_config.get('cross_tile_bbox_iou_threshold', 0.2),
            tile_processor=self.tile_processor  # 传递 tile_processor 用于计算重叠区域
        )

        # 存储地理参考信息
        self._geotransform = None
        self._crs = None
        self._img_shape = None
        self._original_image = None  # 保存原始影像用于可视化

    def predict(self) -> PredictResult:
        """
        执行遥感影像预测（从配置中获取图像路径）

        Returns:
            PredictResult: 预测结果
        """
        try:
            # 从配置获取图像路径
            image_path = self.predict_config.get('source')
            if not image_path:
                error_msg = "Image path not specified in config['predict']['source']"
                logger.error(error_msg)
                return PredictResult(
                    success=False,
                    result_files=[],
                    error=error_msg
                )

            logger.info(f"Starting remote sensing prediction: {image_path}")
            self._update_status('running', progress=0)

            # 获取模型路径
            model_path = self.config.get('model_path')
            if not model_path:
                raise ModelLoadError("Model path not specified in config", model_path=None)

            # 加载模型
            logger.info(f"Loading model: {model_path}")
            self.model = YOLO(model_path)

            # 读取影像
            image_array, geotransform, crs = self._load_image_with_gdal(image_path)

            # 保存原始影像用于可视化
            self._original_image = image_array

            # 存储地理参考信息
            self._geotransform = geotransform
            self._crs = crs
            self._img_shape = image_array.shape[:2]

            # 分块预测
            tile_results = self._predict_tiles(image_array, geotransform, crs)

            # 导出第一步像素坐标结果（原始分块结果）
            pixel_result_path = None
            if self.save_pixel_results:
                pixel_result_path = self._export_pixel_results(tile_results, image_path)
                logger.info(f"Step 1 pixel results exported to: {pixel_result_path}")

            # 合并结果
            merged_result = self.result_merger.merge(
                tile_results,
                self._img_shape,
                geotransform,
                crs,
                output_dir=self.output_dir
            )

            # 导出GeoJSON或Shapefile（最终大幅面矢量结果）
            vector_path = self._export_results(merged_result, image_path)

            # 收集所有结果文件
            result_files = [str(vector_path)]

            # 添加像素坐标结果文件
            if pixel_result_path:
                result_files.append(str(pixel_result_path))

            # 导出可视化成果
            vis_path = None
            if self.save_visualization and self._original_image is not None:
                try:
                    vis_path = self._export_visualization(
                        self._original_image, merged_result, image_path
                    )
                    result_files.append(str(vis_path))
                    logger.info(f"Visualization exported to: {vis_path}")
                except Exception as e:
                    logger.warning(f"Failed to export visualization: {e}")

            # 保存实例ID映射图（如果有）
            if merged_result.instance_id_map is not None:
                instance_map_path = self.output_dir / f"{Path(image_path).stem}_instance_map.png"
                self._save_instance_map(merged_result, instance_map_path)
                result_files.append(str(instance_map_path))

            # 保存分块可视化（如果配置要求）
            if self.save_tiles:
                tiles_dir = self.output_dir / "tiles"
                if tiles_dir.exists():
                    result_files.append(str(tiles_dir))

            self._update_status('completed', progress=100)
            logger.info("Remote sensing prediction completed successfully")

            # 获取保存格式用于metadata
            save_format = self.predict_config.get('save_format', 'geojson').lower()

            return PredictResult(
                success=True,
                result_files=result_files,
                metadata={
                    "num_tiles": merged_result.num_tiles,
                    "total_instances": merged_result.total_instances,
                    "img_shape": merged_result.img_shape,
                    "crs": merged_result.crs,
                    "pixel_results_path": str(pixel_result_path) if pixel_result_path else None,
                    "vector_path": str(vector_path),
                    "vector_format": save_format,
                    "visualization_path": str(vis_path) if vis_path else None
                }
            )

        except Exception as e:
            error_msg = f"Remote sensing prediction failed: {str(e)}"
            logger.exception(error_msg)
            self._update_status('failed', error=error_msg)
            return PredictResult(
                success=False,
                result_files=[],
                error=error_msg
            )

    def _load_image_with_gdal(
        self,
        image_path: str
    ) -> tuple:
        """
        使用GDAL读取遥感影像

        Args:
            image_path: 影像路径

        Returns:
            (image_array, geotransform, crs)
        """
        dataset = gdal.Open(str(image_path))
        if dataset is None:
            raise ImageReadError(f"Failed to open image with GDAL", image_path=image_path)

        try:
            # 获取影像尺寸
            width = dataset.RasterXSize
            height = dataset.RasterYSize

            # 获取地理参考信息
            geotransform = dataset.GetGeoTransform()
            if geotransform is None or geotransform == (0, 1, 0, 0, 0, 1):
                logger.warning("Image has no valid geotransform, using default")
                geotransform = (0, 1, 0, height, 0, -1)  # 默认：左上角为原点

            # 获取坐标系
            crs = dataset.GetProjection()
            if not crs:
                logger.warning("Image has no CRS, using EPSG:4326")
                crs = "EPSG:4326"
            else:
                # 尝试提取EPSG代码
                srs = osr.SpatialReference(wkt=crs)
                epsg = srs.GetAuthorityCode(None)
                if epsg:
                    crs = f"EPSG:{epsg}"

            # 读取影像数据
            if dataset.RasterCount == 1:
                image_array = dataset.ReadAsArray()
                # 确保是3通道
                if image_array.ndim == 2:
                    image_array = np.stack([image_array] * 3, axis=-1)
            elif dataset.RasterCount >= 3:
                # 读取前3个波段
                bands = []
                for i in range(1, 4):
                    band = dataset.GetRasterBand(i)
                    bands.append(band.ReadAsArray())
                image_array = np.stack(bands, axis=-1)
            else:
                raise ImageReadError(f"Unsupported band count: {dataset.RasterCount}", image_path=image_path)

            # 确保数据类型正确
            if image_array.dtype != np.uint8:
                # 归一化到0-255
                image_array = self._normalize_image(image_array)

            logger.info(f"Loaded image: {image_array.shape}, CRS: {crs}")

            return image_array, geotransform, crs

        finally:
            dataset = None  # 关闭数据集

    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        归一化影像到0-255（针对float32等数据类型优化）

        Args:
            image: 输入影像

        Returns:
            归一化后的影像
        """
        # 记录原始数据类型和范围信息
        original_dtype = image.dtype
        data_range = (image.min(), image.max())

        logger.info(f"Normalizing image from {original_dtype}, range: {data_range}")

        # 根据数据类型选择不同的归一化策略
        if original_dtype == np.float32 or original_dtype == np.float64:
            # 对于浮点数据，使用更精确的归一化
            return self._normalize_float_image(image)
        else:
            # 对于整数类型，使用百分比拉伸
            return self._normalize_integer_image(image)

    def _normalize_float_image(self, image: np.ndarray) -> np.ndarray:
        """
        专门处理float32/float64影像的归一化

        Args:
            image: 浮点型影像数据

        Returns:
            归一化到uint8的影像
        """
        # 检查是否为NaN或Inf
        if np.any(np.isnan(image)) or np.any(np.isinf(image)):
            logger.warning("Image contains NaN or Inf values, replacing with 0")
            image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

        # 获取实际数据范围
        img_min = np.min(image)
        img_max = np.max(image)

        # 如果数据范围很小（接近常数），避免除零
        if img_max - img_min < 1e-6:
            logger.warning(f"Image has very small range ({img_max - img_min:.6f}), using threshold normalization")
            # 使用阈值方法
            normalized = np.zeros_like(image, dtype=np.uint8)
            # 将大于平均值+1标准差的设为255，其余为0
            mean_val = np.mean(image)
            std_val = np.std(image)
            mask = image > (mean_val + std_val)
            normalized[mask] = 255
            return normalized

        # 方法1：最小-最大归一化（保留全部动态范围）
        min_max_normalized = (image - img_min) / (img_max - img_min)

        # 方法2：百分比归一化（去除极端值）
        p2 = np.percentile(image, 2)
        p98 = np.percentile(image, 98)
        percentile_normalized = (image - p2) / (p98 - p2 + 1e-7)

        # 根据数据分布选择更好的方法
        # 如果数据分布比较均匀，使用最小-最大
        # 如果有极端值，使用百分比
        range_ratio = (img_max - img_min) / (p98 - p2 + 1e-7)

        if range_ratio > 2.0:  # 说明有极端值，使用百分比归一化
            logger.info(f"Using percentile normalization (range ratio: {range_ratio:.2f})")
            normalized = percentile_normalized
        else:  # 数据分布相对均匀，使用最小-最大归一化
            logger.info(f"Using min-max normalization (range ratio: {range_ratio:.2f})")
            normalized = min_max_normalized

        # 转换到uint8
        normalized = np.clip(normalized * 255, 0, 255).astype(np.uint8)

        return normalized

    def _normalize_integer_image(self, image: np.ndarray) -> np.ndarray:
        """
        处理整数类型影像的归一化

        Args:
            image: 整数型影像数据

        Returns:
            归一化到uint8的影像
        """
        # 对于整数类型，使用百分比拉伸
        img_min = np.percentile(image, 2)
        img_max = np.percentile(image, 98)

        normalized = (image - img_min) / (img_max - img_min + 1e-7)
        normalized = np.clip(normalized * 255, 0, 255).astype(np.uint8)

        return normalized

    def _predict_tiles(
        self,
        image_array: np.ndarray,
        geotransform: tuple,
        crs: str
    ) -> List[TileResult]:
        """
        逐块预测

        Args:
            image_array: 影像数组
            geotransform: 仿射变换参数
            crs: 坐标系

        Returns:
            分块结果列表
        """
        tile_results = []

        # 生成带地理参考的分块
        tiles_generator = self.tile_processor.generate_tiles_with_georef(
            image_array, geotransform, crs
        )

        # 收集所有分块以便显示进度
        all_tiles = list(tiles_generator)
        total_tiles = len(all_tiles)

        logger.info(f"Processing {total_tiles} tiles...")

        for tile_info in tqdm(all_tiles, desc="Predicting tiles"):
            try:
                tile_result = self._predict_single_tile(tile_info)
                if tile_result is not None:
                    tile_results.append(tile_result)

                    # 保存分块可视化（如果配置要求）
                    if self.save_tiles:
                        self._save_tile_visualization(tile_info, tile_result)

            except Exception as e:
                import traceback
                logger.warning(f"Failed to predict tile {tile_info.tile_id}: {str(e)}")
                logger.warning(f"Traceback:\n{traceback.format_exc()}")
                continue

        logger.info(f"Successfully processed {len(tile_results)}/{total_tiles} tiles")

        return tile_results

    def _predict_single_tile(self, tile_info: TileWithGeoRef) -> Optional[TileResult]:
        """
        预测单个分块

        采用与 UltralyticsEngine 完全一致的预测逻辑：
        1. 将分块保存为临时文件
        2. 使用相同的参数调用 model.predict()
        3. 读取结果后删除临时文件

        Args:
            tile_info: 分块信息

        Returns:
            分块预测结果
        """
        import tempfile
        import os
        from PIL import Image

        # 创建临时目录
        temp_dir = tempfile.mkdtemp(prefix='labelmatrix_tile_')
        temp_img_path = os.path.join(temp_dir, f'tile_{tile_info.tile_id}.jpg')

        try:
            # 1. 将分块保存为临时图像文件
            # 确保是 uint8 格式
            tile_array = tile_info.tile_array
            if tile_array.dtype != np.uint8:
                tile_array = tile_array.astype(np.uint8)

            # 使用 PIL 保存为 JPEG
            img = Image.fromarray(tile_array)
            img.save(temp_img_path, quality=95)

            # 2. 使用与 UltralyticsEngine 完全一致的参数进行预测
            # 参考 UltralyticsEngine._build_predict_args()
            hyperparams = self.config.get('hyperparameters', {})
            predict_args = {
                'source': temp_img_path,  # 使用文件路径，而非 numpy 数组
                'conf': self.conf_threshold,
                'iou': self.iou_threshold,
                'imgsz': hyperparams.get('imgsz', 640),
                'device': self.device,
                'save': False,  # 不自动保存，我们手动处理结果
                'verbose': False
            }

            # 添加可选参数（与 UltralyticsEngine 保持一致）
            if self.predict_config.get('augment'):
                predict_args['augment'] = True
            if self.predict_config.get('half'):
                predict_args['half'] = True

            results = self.model.predict(**predict_args)

            if not results or len(results) == 0:
                return TileResult(
                    tile_id=tile_info.tile_id,
                    offset=tile_info.offset,
                    shape=tile_info.shape,
                    geotransform=tile_info.geotransform,
                    num_instances=0
                )

            result = results[0]

            # 提取结果并转换为全局坐标
            boxes_global = None
            masks_global = None
            polygons_global = None
            polygons_with_info = None
            class_names = result.names if hasattr(result, 'names') else None

            # 获取偏移（提前计算，供后续使用）
            row_offset, col_offset = tile_info.offset

            # 处理检测框
            if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
                boxes_local = result.boxes.xyxy.cpu().numpy()  # [N, 4]
                conf = result.boxes.conf.cpu().numpy()  # [N]
                cls = result.boxes.cls.cpu().numpy()  # [N]

                if len(boxes_local) > 0:
                    # 转换到全局坐标
                    boxes_global = np.zeros((len(boxes_local), 6), dtype=np.float32)
                    boxes_global[:, 0] = boxes_local[:, 0] + col_offset  # x1
                    boxes_global[:, 1] = boxes_local[:, 1] + row_offset  # y1
                    boxes_global[:, 2] = boxes_local[:, 2] + col_offset  # x2
                    boxes_global[:, 3] = boxes_local[:, 3] + row_offset  # y2
                    boxes_global[:, 4] = conf
                    boxes_global[:, 5] = cls

            # 处理掩膜和矢量多边形
            if hasattr(result, 'masks') and result.masks is not None:
                masks_data = result.masks.data
                if masks_data is not None:
                    masks_local = masks_data.cpu().numpy()  # [N, H, W] 或 [H, W]

                    # 【新增】从掩膜提取矢量多边形（使用连通域分析，确保断开区域分离）
                    if masks_local is not None:
                        # 获取 boxes 用于传递置信度和类别
                        boxes_for_poly = boxes_global if boxes_global is not None else None

                        polygons_with_info = self._extract_polygons_from_masks(
                            masks_local,
                            col_offset,
                            row_offset,
                            boxes_for_poly
                        )
                        # polygons_with_info 是 [{'polygon': points, 'conf': x, 'cls': y}, ...]
                        polygons_global = [p['polygon'] for p in polygons_with_info]

                    # 如果掩膜尺寸与分块尺寸不一致，需要调整
                    tile_h, tile_w = tile_info.shape
                    if masks_local.ndim == 3:
                        # [N, H_pred, W_pred] -> [N, H_tile, W_tile]
                        masks_global = []
                        for i in range(masks_local.shape[0]):
                            mask = masks_local[i]
                            if mask.shape[0] != tile_h or mask.shape[1] != tile_w:
                                import cv2
                                mask_resized = cv2.resize(mask.astype(np.uint8), (tile_w, tile_h), interpolation=cv2.INTER_NEAREST)
                                masks_global.append(mask_resized)
                            else:
                                masks_global.append(mask)
                        masks_global = np.array(masks_global)
                    else:
                        # [H_pred, W_pred] -> [H_tile, W_tile]
                        if masks_local.shape[0] != tile_h or masks_local.shape[1] != tile_w:
                            import cv2
                            masks_global = cv2.resize(masks_local.astype(np.uint8), (tile_w, tile_h), interpolation=cv2.INTER_NEAREST)
                        else:
                            masks_global = masks_local

            num_instances = len(boxes_global) if boxes_global is not None else len(polygons_global) if polygons_global is not None else 0

            return TileResult(
                tile_id=tile_info.tile_id,
                offset=tile_info.offset,
                shape=tile_info.shape,
                geotransform=tile_info.geotransform,
                boxes=boxes_global,
                masks=masks_global,
                polygons=polygons_global,  # 矢量多边形列表
                polygons_with_info=polygons_with_info if 'polygons_with_info' in dir() else None,  # 带信息的多边形
                class_names=class_names,
                num_instances=num_instances
            )

        finally:
            # 清理临时文件
            try:
                if os.path.exists(temp_img_path):
                    os.remove(temp_img_path)
                # 如果目录为空，也删除
                if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                    os.rmdir(temp_dir)
                # 如果目录不为空（可能有其他临时文件），递归删除
                elif os.path.exists(temp_dir):
                    import shutil
                    shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp files: {e}")

    def _debug_save_polygons(
        self,
        tile_info: TileWithGeoRef,
        polygons: list,
        tile_array: np.ndarray
    ):
        """
        【调试】保存多边形可视化，检查YOLO输出是否正常

        Args:
            tile_info: 分块信息
            polygons: 多边形列表
            tile_array: 分块图像
        """
        import cv2
        from pathlib import Path

        debug_dir = Path(self.output_dir) / "debug_polygons"
        debug_dir.mkdir(parents=True, exist_ok=True)

        # 复制图像用于绘制
        if tile_array.ndim == 3:
            img = tile_array.copy()
        else:
            img = cv2.cvtColor(tile_array, cv2.COLOR_GRAY2BGR)

        # 绘制每个多边形
        for i, poly in enumerate(polygons):
            # 转换为整数坐标
            points = poly.astype(np.int32)

            # 用不同颜色绘制每个多边形
            color = tuple(np.random.randint(0, 255, 3).tolist())
            cv2.polylines(img, [points], True, color, 2)

            # 标注序号
            center = points.mean(axis=0).astype(np.int32)
            cv2.putText(img, str(i), tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 保存
        output_path = debug_dir / f"tile_{tile_info.tile_id}_polygons.png"
        cv2.imwrite(str(output_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        logger.info(f"[DEBUG] 多边形可视化已保存: {output_path}")

    def _debug_save_yolo_masks(
        self,
        result,
        tile_array: np.ndarray,
        tile_id: int
    ):
        """
        【调试】保存 YOLO 原始掩膜，检查掩膜是否正常

        Args:
            result: YOLO 预测结果
            tile_array: 分块图像
            tile_id: 分块ID
        """
        import cv2
        from pathlib import Path

        debug_dir = Path(self.output_dir) / "debug_polygons"
        debug_dir.mkdir(parents=True, exist_ok=True)

        # 获取掩膜数据
        if hasattr(result, 'masks') and result.masks is not None:
            masks_data = result.masks.data
            if masks_data is not None:
                masks = masks_data.cpu().numpy()

                # 创建可视化图像
                h, w = masks.shape[1] if masks.ndim == 3 else masks.shape[0], \
                       masks.shape[2] if masks.ndim == 3 else masks.shape[1]

                # 为每个掩膜创建彩色图像
                for i, mask in enumerate(masks):
                    if masks.ndim == 3:
                        mask = masks[i]

                    # 归一化到 0-255
                    mask_vis = (mask * 255).astype(np.uint8)

                    # 应用颜色映射
                    mask_colored = cv2.applyColorMap(mask_vis, cv2.COLORMAP_JET)

                    # 保存
                    output_path = debug_dir / f"tile_{tile_id}_mask_{i}.png"
                    cv2.imwrite(str(output_path), mask_colored)

                # 同时保存所有掩膜叠加的结果
                overlay = np.zeros((h, w, 3), dtype=np.uint8)
                for i, mask in enumerate(masks):
                    if masks.ndim == 3:
                        mask = masks[i]
                    # 为每个掩膜分配不同颜色
                    color = tuple(np.random.randint(50, 255, 3).tolist())
                    mask_bool = mask > 0.5
                    overlay[mask_bool] = color

                output_path = debug_dir / f"tile_{tile_id}_masks_overlay.png"
                cv2.imwrite(str(output_path), overlay)

                logger.info(f"[DEBUG] YOLO掩膜已保存到: {debug_dir}")

    def _extract_polygons_from_masks(
        self,
        masks: np.ndarray,
        col_offset: int,
        row_offset: int,
        boxes = None
    ) -> list:
        """
        从掩膜中提取多边形（使用连通域分析，确保断开区域分离）

        Args:
            masks: [N, H, W] 掩膜数组
            col_offset: 列偏移
            row_offset: 行偏移
            boxes: YOLO boxes 对象或 [N, 6] numpy 数组 (可选)

        Returns:
            多边形信息列表，每项包含 {'polygon': points, 'conf': x, 'cls': y}
        """
        import cv2

        polygons = []

        # 处理 boxes 参数 - 可能是 YOLO 对象或 numpy 数组
        boxes_array = None
        if boxes is not None:
            if hasattr(boxes, 'cpu'):
                # YOLO boxes 对象，转换为 numpy 数组
                boxes_array = boxes.xyxy.cpu().numpy()
                conf_array = boxes.conf.cpu().numpy()
                cls_array = boxes.cls.cpu().numpy()
                # 组合成 [N, 6] 格式
                boxes_array = np.column_stack([boxes_array, conf_array, cls_array])
            elif isinstance(boxes, np.ndarray):
                boxes_array = boxes
            else:
                logger.warning(f"未知的 boxes 类型: {type(boxes)}")

        if masks.ndim == 2:
            masks = masks[np.newaxis, ...]

        for mask_idx, mask in enumerate(masks):
            # 获取当前掩膜对应的置信度和类别
            if boxes_array is not None and mask_idx < len(boxes_array):
                conf = float(boxes_array[mask_idx, 4])
                cls = int(boxes_array[mask_idx, 5])
            else:
                conf = 0.5
                cls = 0

            # 确保掩膜是二值的
            mask_binary = (mask > 0.5).astype(np.uint8)

            # 使用连通域分析找到所有独立区域
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                mask_binary, connectivity=8
            )

            # 跳过背景 (label 0)
            for label in range(1, num_labels):
                # 提取当前连通域的掩膜
                single_mask = (labels == label).astype(np.uint8)

                # 检查面积，过滤太小的噪点
                area = stats[label, cv2.CC_STAT_AREA]
                if area < 10:  # 面积小于10像素的视为噪点
                    continue

                # 提取轮廓
                contours, _ = cv2.findContours(
                    single_mask,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_TC89_KCOS
                )

                if not contours:
                    continue

                # 选择最大的轮廓
                largest_contour = max(contours, key=cv2.contourArea)

                # 简化轮廓（减少点数）
                epsilon = 0.001 * cv2.arcLength(largest_contour, True)
                simplified = cv2.approxPolyDP(largest_contour, epsilon, True)

                # 转换格式：[N, 1, 2] -> [N, 2]
                points = simplified.squeeze().astype(np.float32)

                if len(points) < 3:
                    continue

                # 转换为全局坐标
                points[:, 0] += col_offset
                points[:, 1] += row_offset

                polygons.append({
                    'polygon': points,
                    'conf': conf,
                    'cls': cls
                })

        return polygons

    def _fix_and_simplify_polygon(self, points: np.ndarray) -> np.ndarray:
        """
        修复和简化多边形，去除自相交和重复点

        Args:
            points: [N, 2] 多边形顶点

        Returns:
            修复后的多边形顶点
        """
        if len(points) < 3:
            return points

        try:
            from shapely.geometry import Polygon

            # 创建 Shapely 多边形
            poly = Polygon(points)

            # 如果多边形无效，使用 buffer(0) 修复
            if not poly.is_valid:
                poly = poly.buffer(0)

            # 简化多边形（Douglas-Peucker 算法）
            # tolerance=1.0 表示保留1像素精度的细节
            simplified = poly.simplify(tolerance=1.0, preserve_topology=True)

            if simplified.is_empty or not simplified.is_valid:
                return points

            # 提取外边界坐标
            coords = np.array(simplified.exterior.coords)
            # 移除闭合点（最后一个点与第一个点重复）
            if len(coords) > 1 and np.allclose(coords[0], coords[-1]):
                coords = coords[:-1]

            return coords if len(coords) >= 3 else points

        except Exception as e:
            logger.warning(f"多边形修复失败: {e}")
            return points

    def _save_tile_visualization(
        self,
        tile_info: TileWithGeoRef,
        tile_result: TileResult
    ):
        """
        保存分块可视化结果

        Args:
            tile_info: 分块信息
            tile_result: 分块预测结果
        """
        tiles_dir = self.output_dir / "tiles"
        tiles_dir.mkdir(parents=True, exist_ok=True)

        # 重新运行预测以获取可视化图像
        # 使用与 UltralyticsEngine 完全一致的预测逻辑
        import tempfile
        import os
        from PIL import Image

        temp_dir = tempfile.mkdtemp(prefix='labelmatrix_tile_vis_')
        temp_img_path = os.path.join(temp_dir, f'tile_{tile_info.tile_id}.jpg')

        try:
            # 将分块保存为临时文件
            tile_array = tile_info.tile_array
            if tile_array.dtype != np.uint8:
                tile_array = tile_array.astype(np.uint8)
            img = Image.fromarray(tile_array)
            img.save(temp_img_path, quality=95)

            # 使用与 UltralyticsEngine 一致的参数
            hyperparams = self.config.get('hyperparameters', {})
            predict_args = {
                'source': temp_img_path,
                'conf': self.conf_threshold,
                'iou': self.iou_threshold,
                'imgsz': hyperparams.get('imgsz', 640),
                'device': self.device,
                'save': False,
                'verbose': False
            }

            if self.predict_config.get('augment'):
                predict_args['augment'] = True
            if self.predict_config.get('half'):
                predict_args['half'] = True

            results = self.model.predict(**predict_args)
        finally:
            # 清理临时文件
            try:
                if os.path.exists(temp_img_path):
                    os.remove(temp_img_path)
                if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                    os.rmdir(temp_dir)
                elif os.path.exists(temp_dir):
                    import shutil
                    shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass

        if results and len(results) > 0:
            # 绘制结果
            plotted = results[0].plot(
                line_width=self.line_width,
                conf=self.show_conf,
                labels=self.show_labels
            )

            # 保存
            from PIL import Image
            tile_img_path = tiles_dir / f"tile_{tile_info.tile_id:04d}.jpg"
            Image.fromarray(plotted).save(tile_img_path)

    def _export_pixel_results(
        self,
        tile_results: List[TileResult],
        image_path: str
    ) -> Path:
        """
        导出第一步像素坐标结果（原始分块预测结果）

        Args:
            tile_results: 分块结果列表
            image_path: 原始影像路径

        Returns:
            像素坐标结果文件路径
        """
        # 获取保存目录
        save_dir = self.predict_config.get('save_dir')
        if save_dir:
            output_dir = Path(save_dir)
        else:
            output_dir = self.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # 创建像素坐标导出器
        pixel_exporter = PixelResultExporter(img_shape=self._img_shape)

        # 导出分块原始结果
        pixel_result_path = output_dir / f"{Path(image_path).stem}_pixel_results.json"

        # 从模型获取类别名称
        class_names = None
        if self.model is not None and hasattr(self.model, 'names'):
            class_names = self.model.names

        return pixel_exporter.export_tile_results(
            tile_results=tile_results,
            output_path=str(pixel_result_path),
            class_names=class_names
        )

    def _export_visualization(
        self,
        image_array: np.ndarray,
        merged_result: MergedResult,
        image_path: str
    ) -> Path:
        """
        导出可视化成果（叠加检测结果）

        Args:
            image_array: 原始影像数组
            merged_result: 合并后的结果
            image_path: 原始影像路径

        Returns:
            可视化文件路径
        """
        # 获取保存目录
        save_dir = self.predict_config.get('save_dir')
        if save_dir:
            output_dir = Path(save_dir)
        else:
            output_dir = self.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # 从模型获取类别名称
        class_names = None
        if self.model is not None and hasattr(self.model, 'names'):
            class_names = self.model.names

        # 创建大幅面可视化器
        visualizer = LargeImageVisualizer(
            task_type=self.task_type,
            class_names=class_names,
            line_width=self.line_width,
            show_labels=self.show_labels,
            show_conf=self.show_conf
        )

        # 确定输出格式
        vis_format = self.predict_config.get('vis_format', 'jpg')
        if vis_format == 'tif' or vis_format == 'tiff':
            output_path = output_dir / f"{Path(image_path).stem}_visualized.tif"
        else:
            output_path = output_dir / f"{Path(image_path).stem}_visualized.{vis_format}"

        # 导出可视化
        return visualizer.export_large_visualization(
            image_array=image_array,
            merged_result=merged_result,
            output_path=str(output_path),
            conf_threshold=self.vis_conf_threshold
        )

    def _export_results(
        self,
        merged_result: MergedResult,
        image_path: str
    ) -> Path:
        """
        导出预测结果为GeoJSON或Shapefile（最终大幅面矢量结果）

        Args:
            merged_result: 合并后的结果
            image_path: 原始影像路径

        Returns:
            输出文件路径
        """
        # 获取保存目录
        save_dir = self.predict_config.get('save_dir')
        if save_dir:
            output_dir = Path(save_dir)
        else:
            output_dir = self.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # 从模型获取类别名称
        class_names = None
        if self.model is not None and hasattr(self.model, 'names'):
            class_names = self.model.names

        # 获取保存格式
        save_format = self.predict_config.get('save_format', 'geojson').lower()

        # 根据格式选择导出器
        if save_format == 'shapefile':
            exporter = ShapefileExporter(
                geotransform=self._geotransform,
                crs=self._crs,
                img_shape=self._img_shape
            )
            output_path = output_dir / f"{Path(image_path).stem}.shp"
        elif save_format == 'both':
            # Export both formats
            geojson_exporter = GeoJSONExporter(
                geotransform=self._geotransform,
                crs=self._crs,
                img_shape=self._img_shape
            )
            geojson_path = output_dir / f"{Path(image_path).stem}.geojson"
            geojson_result = geojson_exporter.export(
                merged_result=merged_result,
                output_path=str(geojson_path),
                class_names=class_names,
                naming_config=self.file_naming
            )

            shapefile_exporter = ShapefileExporter(
                geotransform=self._geotransform,
                crs=self._crs,
                img_shape=self._img_shape
            )
            shp_path = output_dir / f"{Path(image_path).stem}.shp"
            shapefile_result = shapefile_exporter.export(
                merged_result=merged_result,
                output_path=str(shp_path),
                class_names=class_names,
                naming_config=self.file_naming
            )

            # Return GeoJSON path for compatibility (already exported above)
            return geojson_result
        else:  # default to geojson
            exporter = GeoJSONExporter(
                geotransform=self._geotransform,
                crs=self._crs,
                img_shape=self._img_shape
            )
            output_path = output_dir / f"{Path(image_path).stem}.geojson"

        return exporter.export(
            merged_result=merged_result,
            output_path=str(output_path),
            class_names=class_names,
            naming_config=self.file_naming
        )

    def _save_instance_map(
        self,
        merged_result: MergedResult,
        output_path: Path
    ):
        """
        保存实例ID映射图

        Args:
            merged_result: 合并后的结果
            output_path: 输出路径
        """
        exporter = GeoJSONExporter(
            geotransform=self._geotransform,
            crs=self._crs,
            img_shape=self._img_shape
        )

        exporter.export_instance_map(merged_result, str(output_path))

    def train(self) -> TrainResult:
        """
        RemoteSensingPredictor does not support training.

        Raises:
            NotImplementedError: Training is not supported
        """
        raise NotImplementedError("RemoteSensingPredictor does not support training. Use UltralyticsEngine or LabelMatrixTrainer for training.")

    def resume(self) -> TrainResult:
        """
        RemoteSensingPredictor does not support resume.

        Raises:
            NotImplementedError: Resume is not supported
        """
        raise NotImplementedError("RemoteSensingPredictor does not support resume. Use UltralyticsEngine or LabelMatrixTrainer for training/resume.")
