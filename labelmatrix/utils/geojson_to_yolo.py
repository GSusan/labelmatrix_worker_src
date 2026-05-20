# -*- coding: utf-8 -*-
"""
GeoJSON 到 YOLO 格式转换器
"""

import json
import logging
import random
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import yaml
from PIL import Image
import cv2
import numpy as np

from labelmatrix.exceptions.dataset_errors import (
    DatasetConversionError,
    GeoJSONFormatError,
    MissingCategoriesError
)

logger = logging.getLogger(__name__)


class GeoJSONToYOLOConverter:
    """GeoJSON格式到YOLO格式的转换器"""

    def __init__(
        self,
        source_dataset_path: str,
        train_val_split: float = 0.8,
        random_seed: int = 42,
        verbose_logging: bool = False
    ):
        """
        初始化转换器

        Args:
            source_dataset_path: 源数据集根目录路径
            train_val_split: 训练集比例，默认0.8（80%训练，20%验证）
            random_seed: 随机种子，用于保证划分可复现，默认42
            verbose_logging: 是否使用详细日志模式，默认False
        """
        # 使用绝对路径，避免相对路径被解析为当前工作目录
        self.source_path = Path(source_dataset_path).resolve().absolute()
        self.source_config = self.source_path / 'data.yaml'

        # 训练/验证集划分配置（需要在_load_categories之前设置）
        self.train_val_split = train_val_split
        self.random_seed = random_seed
        self.verbose_logging = verbose_logging

        # 保存原始配置信息（需要在_load_categories之前初始化）
        self._source_config_data: Dict[str, Any] = {}

        # 读取源配置获取类别信息
        self.categories = self._load_categories()

        # 读取task类型
        self.task = self._load_task()

        # 创建对应任务的转换器
        self.task_converter = self._create_task_converter()

        # 输出目录 - 在源数据集目录内部创建yolo_format
        # 例如: D:\DLProjects\Dataset\sat_farmland_test_20260415\yolo_format\
        self.output_path = self.source_path / 'yolo_format'

        # 文件划分结果
        self._train_files: List[str] = []
        self._val_files: List[str] = []

    def _load_categories(self) -> Dict[int, str]:
        """
        从源data.yaml加载类别信息和配置

        Returns:
            Dict[int, str]: 类别ID到类别名称的映射（用于YOLO格式，从0开始）
        """
        if not self.source_config.exists():
            raise MissingCategoriesError(f"Source data.yaml not found: {self.source_config}")

        with open(self.source_config, 'r', encoding='utf-8') as f:
            self._source_config_data = yaml.safe_load(f)

        if 'names' not in self._source_config_data:
            raise MissingCategoriesError("No 'names' field found in data.yaml")

        names = self._source_config_data['names']
        categories = {}

        if isinstance(names, dict):
            # {0: 'building', 1: 'road', ...} 或 {1: 'building', 2: 'road', ...}
            # 检查是否是从1开始的索引
            keys = [int(k) for k in names.keys()]
            min_key = min(keys) if keys else 0

            if min_key == 1:
                # 从1开始，需要转换为从0开始
                for k, v in names.items():
                    original_id = int(k)
                    yolo_id = original_id - 1
                    categories[yolo_id] = v
            else:
                # 已经从0开始，直接使用
                categories = {int(k): v for k, v in names.items()}

        elif isinstance(names, list):
            # ['building', 'road', ...]
            # GeoJSON中class_id从1开始，对应列表索引
            # 例如：class_id=1 → names[0], class_id=2 → names[1]
            # 转换为YOLO格式：{0: names[0], 1: names[1], ...}
            categories = {i: name for i, name in enumerate(names)}
        else:
            raise MissingCategoriesError(f"Unsupported 'names' format: {type(names)}")

        logger.info(f"Loaded {len(categories)} categories: {categories}")
        return categories

    def _load_task(self) -> str:
        """
        从data.yaml加载task类型

        Returns:
            str: task类型 (segment/detect/obb)

        Raises:
            UnsupportedTaskError: 当task类型不支持时
        """
        from labelmatrix.exceptions.dataset_errors import UnsupportedTaskError

        task = self._source_config_data.get('task', 'segment')

        # 验证task类型
        valid_tasks = {'segment', 'detect', 'obb'}
        if task not in valid_tasks:
            raise UnsupportedTaskError(
                f"Unsupported task type: '{task}'. "
                f"Valid tasks are: {', '.join(valid_tasks)}"
            )

        logger.info(f"Task type: {task}")
        return task

    def _create_task_converter(self) -> 'BaseTaskConverter':
        """
        根据task类型创建对应的转换器

        Returns:
            BaseTaskConverter: 任务转换器实例
        """
        from labelmatrix.utils.task_converters import (
            SegmentConverter,
            DetectConverter,
            OBBConverter
        )

        converter_map = {
            'segment': SegmentConverter,
            'detect': DetectConverter,
            'obb': OBBConverter
        }

        converter_class = converter_map.get(self.task, SegmentConverter)
        return converter_class(
            self.categories,
            self._source_config_data,
            verbose_logging=self.verbose_logging
        )

    def convert(self) -> str:
        """
        执行转换

        Returns:
            str: 新数据集的data.yaml路径
        """
        logger.info(f"Starting conversion: {self.source_path} -> {self.output_path}")
        logger.info(f"Train/val split: {self.train_val_split:.1%} / {1-self.train_val_split:.1%}, seed={self.random_seed}")

        # 创建输出目录结构
        self._create_output_structure()

        # 划分训练集和验证集文件列表
        self._split_files()

        # 复制图像文件
        train_count, val_count = self._copy_images()

        # 转换标注文件
        train_labels, val_labels = self._convert_labels()

        # 生成新的data.yaml
        new_config_path = self._generate_data_yaml()

        logger.info(f"Conversion completed: {train_count} train images, {val_count} val images")
        logger.info(f"Labels: {train_labels} train, {val_labels} val")
        logger.info(f"New dataset: {new_config_path}")

        # 返回绝对路径，确保训练时能正确找到数据集
        return str(new_config_path.absolute())

    def _create_output_structure(self) -> None:
        """创建输出目录结构"""
        directories = [
            self.output_path / 'images' / 'train2017',
            self.output_path / 'images' / 'val2017',
            self.output_path / 'labels' / 'train2017',
            self.output_path / 'labels' / 'val2017',
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

        logger.debug(f"Created directory structure: {self.output_path}")

    def _split_files(self) -> None:
        """
        划分训练集和验证集文件列表

        使用随机种子保证可复现性
        """
        source_geojsons = self.source_path / 'geojsons'
        if not source_geojsons.exists():
            raise DatasetConversionError(f"Source geojsons directory not found: {source_geojsons}")

        # 获取所有geojson文件名（不含扩展名）
        all_files = [f.stem for f in source_geojsons.glob('*.geojson')]

        if not all_files:
            raise DatasetConversionError("No GeoJSON files found in source directory")

        # 设置随机种子并打乱
        random.seed(self.random_seed)
        random.shuffle(all_files)

        # 按比例划分
        split_idx = int(len(all_files) * self.train_val_split)
        self._train_files = all_files[:split_idx]
        self._val_files = all_files[split_idx:]

        logger.info(f"Dataset split: {len(self._train_files)} train, {len(self._val_files)} val")

    def _copy_images(self) -> Tuple[int, int]:
        """
        复制图像文件到输出目录，将.tif格式转换为.jpg

        Returns:
            Tuple[int, int]: (训练集图像数量, 验证集图像数量)
        """
        source_images = self.source_path / 'images'
        if not source_images.exists():
            logger.warning(f"Source images directory not found: {source_images}")
            return 0, 0

        train_count = 0
        val_count = 0
        # 支持的图像格式
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

        # 获取所有图像文件
        image_files = [
            f for f in source_images.rglob('*')
            if f.suffix.lower() in image_extensions
        ]

        # 创建文件名到文件对象的映射
        image_map = {}
        for img_file in image_files:
            # 使用不含扩展名的文件名作为key
            name_without_ext = img_file.stem
            image_map[name_without_ext] = img_file

        # 复制/转换训练集图像
        for file_stem in self._train_files:
            if file_stem in image_map:
                src_file = image_map[file_stem]
                dest_dir = self.output_path / 'images' / 'train2017'

                # 如果是.tif格式，转换为.jpg
                if src_file.suffix.lower() in {'.tif', '.tiff'}:
                    dest_file = dest_dir / (src_file.stem + '.jpg')
                    if self._convert_tif_to_jpg(src_file, dest_file):
                        train_count += 1
                else:
                    # 其他格式直接复制
                    dest_file = dest_dir / src_file.name
                    shutil.copy2(src_file, dest_file)
                    train_count += 1

        # 复制/转换验证集图像
        for file_stem in self._val_files:
            if file_stem in image_map:
                src_file = image_map[file_stem]
                dest_dir = self.output_path / 'images' / 'val2017'

                # 如果是.tif格式，转换为.jpg
                if src_file.suffix.lower() in {'.tif', '.tiff'}:
                    dest_file = dest_dir / (src_file.stem + '.jpg')
                    if self._convert_tif_to_jpg(src_file, dest_file):
                        val_count += 1
                else:
                    # 其他格式直接复制
                    dest_file = dest_dir / src_file.name
                    shutil.copy2(src_file, dest_file)
                    val_count += 1

        logger.debug(f"Copied/converted {train_count} train images, {val_count} val images")
        return train_count, val_count

    def _convert_tif_to_jpg(self, src_file: Path, dest_file: Path) -> bool:
        """
        将TIF文件转换为JPG格式，支持GeoTIFF

        Args:
            src_file: 源TIF文件路径
            dest_file: 目标JPG文件路径

        Returns:
            bool: 转换是否成功
        """
        try:
            # 首先尝试使用rasterio读取（支持GeoTIFF）
            try:
                import rasterio
                with rasterio.open(src_file) as src:
                    # 读取图像数据
                    img_data = src.read()
                    # 转换维度顺序: (C, H, W) -> (H, W, C)
                    if img_data.ndim == 3:
                        img_data = np.moveaxis(img_data, 0, -1)

                    # 处理数据类型和范围
                    if img_data.dtype != np.uint8:
                        # 归一化到0-255
                        img_data = self._normalize_to_uint8(img_data)

                    # 如果是单波段或大于3波段，取前3个波段
                    if img_data.shape[-1] >= 3:
                        img_rgb = img_data[:, :, :3]
                    elif img_data.shape[-1] == 2:
                        # 2波段情况，复制第二个通道
                        img_rgb = np.stack([img_data[:, :, 0], img_data[:, :, 1], img_data[:, :, 1]], axis=-1)
                    elif img_data.shape[-1] == 1:
                        # 单波段，复制到RGB
                        img_rgb = np.stack([img_data[:, :, 0]] * 3, axis=-1)
                    else:
                        img_rgb = img_data

                    # 保存为JPG
                    cv2.imwrite(str(dest_file), img_rgb)
                    return True
            except ImportError:
                pass
            except Exception as e:
                logger.debug(f"rasterio conversion failed: {e}, trying PIL")

            # 如果rasterio失败，尝试使用PIL
            try:
                img = Image.open(src_file)
                img.convert('RGB').save(dest_file, 'JPEG')
                return True
            except Exception as e:
                logger.debug(f"PIL conversion failed: {e}")

            # 如果都失败，尝试使用opencv
            try:
                img = cv2.imread(str(src_file))
                if img is not None:
                    cv2.imwrite(str(dest_file), img)
                    return True
            except Exception as e:
                logger.debug(f"OpenCV conversion failed: {e}")

            logger.warning(f"Failed to convert {src_file.name}: all methods failed")
            return False

        except Exception as e:
            logger.warning(f"Unexpected error converting {src_file.name}: {e}")
            return False

    def _normalize_to_uint8(self, img_data: np.ndarray) -> np.ndarray:
        """
        将图像数据归一化到uint8范围

        Args:
            img_data: 输入图像数据

        Returns:
            np.ndarray: uint8范围的图像数据
        """
        # 获取数据范围
        min_val = np.nanmin(img_data)
        max_val = np.nanmax(img_data)

        if max_val > min_val:
            # 归一化到0-255
            normalized = ((img_data - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        else:
            # 如果所有值相同，设置为0
            normalized = np.zeros_like(img_data, dtype=np.uint8)

        return normalized

    def _convert_labels(self) -> Tuple[int, int]:
        """
        转换GeoJSON标注到YOLO TXT格式

        Returns:
            Tuple[int, int]: (训练集标注数量, 验证集标注数量)
        """
        source_geojsons = self.source_path / 'geojsons'
        if not source_geojsons.exists():
            raise DatasetConversionError(f"Source geojsons directory not found: {source_geojsons}")

        train_count = 0
        val_count = 0

        # 转换训练集标注
        for file_stem in self._train_files:
            geojson_file = source_geojsons / f'{file_stem}.geojson'
            if geojson_file.exists():
                try:
                    self._convert_single_geojson(geojson_file, split='train')
                    train_count += 1
                except GeoJSONFormatError as e:
                    logger.warning(f"Failed to convert {geojson_file.name}: {e}")
                except Exception as e:
                    logger.error(f"Unexpected error converting {geojson_file.name}: {e}")

        # 转换验证集标注
        for file_stem in self._val_files:
            geojson_file = source_geojsons / f'{file_stem}.geojson'
            if geojson_file.exists():
                try:
                    self._convert_single_geojson(geojson_file, split='val')
                    val_count += 1
                except GeoJSONFormatError as e:
                    logger.warning(f"Failed to convert {geojson_file.name}: {e}")
                except Exception as e:
                    logger.error(f"Unexpected error converting {geojson_file.name}: {e}")

        logger.debug(f"Converted {train_count} train labels, {val_count} val labels")
        return train_count, val_count

    def _convert_single_geojson(self, geojson_file: Path, split: str = 'train') -> None:
        """
        转换单个GeoJSON文件

        Args:
            geojson_file: GeoJSON文件路径
            split: 数据集划分，'train' 或 'val'
        """
        with open(geojson_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if data.get('type') != 'FeatureCollection':
            raise GeoJSONFormatError("Not a FeatureCollection")

        features = data.get('features', [])
        if not features:
            logger.debug(f"No features in {geojson_file.name}")
            return

        # 从第一个feature获取图像信息
        first_feature = features[0]
        props = first_feature.get('properties', {})
        image_size = props.get('image_size', [512, 512])  # 默认512x512
        img_width, img_height = image_size[0], image_size[1]

        # 获取对应的图像文件名
        image_ref = props.get('image_ref', '')
        if not image_ref:
            # 使用geojson文件名
            image_ref = geojson_file.stem + '.tif'

        # 转换图像扩展名
        if image_ref.endswith('.tif'):
            image_ref = image_ref[:-4] + '.jpg'

        # 生成YOLO标注内容（使用task_converter）
        yolo_annotations = []
        skipped_count = 0

        for feature in features:
            try:
                line = self.task_converter.convert_feature(feature, img_width, img_height)
                if line:
                    yolo_annotations.append(line)
                else:
                    skipped_count += 1
            except Exception as e:
                if self.verbose_logging:
                    logger.warning(f"Failed to convert feature in {geojson_file.name}: {e}")
                skipped_count += 1

        if skipped_count > 0 and self.verbose_logging:
            logger.debug(f"Skipped {skipped_count} features in {geojson_file.name}")

        # 根据split选择输出目录
        split_dir = 'train2017' if split == 'train' else 'val2017'
        output_dir = self.output_path / 'labels' / split_dir
        output_file = output_dir / (geojson_file.stem + '.txt')

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(yolo_annotations))

        logger.debug(f"Converted: {geojson_file.name} -> {split_dir}/{output_file.name}")

    def _normalize_polygon(
        self, coordinates: List[List[float]], img_width: int, img_height: int
    ) -> List[tuple]:
        """
        归一化多边形坐标到0-1范围

        Args:
            coordinates: 原始坐标列表 [[x, y, z], ...]
            img_width: 图像宽度
            img_height: 图像高度

        Returns:
            List[tuple]: 归一化后的坐标列表 [(x, y), ...]
        """
        normalized = []
        for coord in coordinates:
            x = coord[0] / img_width
            y = coord[1] / img_height
            normalized.append((x, y))

        return normalized

    def _generate_data_yaml(self) -> Path:
        """
        生成标准的Ultralytics data.yaml

        按照以下格式生成：
        task: <task>
        description: <description>
        names:
          0: <class_name>
          1: <class_name>
          ...
        train: images/train2017
        val: images/val2017

        Returns:
            Path: 生成的data.yaml文件路径
        """
        # 获取task和description
        # task: 使用原始值或默认值
        task = self._source_config_data.get('task')
        if task is None or task == '':
            task = 'segment'  # 默认值

        # description: 始终使用数据集目录名
        description = self.source_path.name

        # 构建类别字典（索引: 类别名称）
        names_dict = {i: name for i, name in self.categories.items()}

        output_file = self.output_path / 'data.yaml'

        # 手动写入文件以控制格式
        with open(output_file, 'w', encoding='utf-8') as f:
            # 第一行：task
            f.write(f"task: {task}\n")
            # 第二行：description（使用数据集目录名）
            f.write(f"description: {description}\n")
            # 第三行：names（多行格式）
            f.write("names:\n")
            for idx, name in names_dict.items():
                f.write(f"  {idx}: {name}\n")
            # train和val
            f.write("train: images/train2017\n")
            f.write("val: images/val2017\n")

        logger.debug(f"Generated data.yaml: {output_file}")
        return output_file
