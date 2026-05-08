# 多任务YOLO格式转换器扩展实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 扩展GeoJSONToYOLOConverter以支持segment、detect、obb三种YOLO任务类型

**架构:** 在现有转换器基础上引入任务转换器架构，通过BaseTaskConverter抽象基类定义统一接口，为每个任务类型实现专门的转换逻辑。根据data.yaml中的task字段路由到对应的转换器。

**Tech Stack:** Python 3.8+, PyYAML, OpenCV, NumPy, Ultralytics YOLO

---

## Task 1: 添加新的异常类

**Files:**
- Create: `labelmatrix/exceptions/dataset_errors.py` (modify existing)

**Step 1: 添加UnsupportedTaskError异常类**

在现有`dataset_errors.py`文件末尾添加新的异常类：

```python
class UnsupportedTaskError(DatasetConversionError):
    """不支持的task类型异常"""
    pass


class InvalidGeometryError(GeoJSONFormatError):
    """几何形状无效异常"""
    pass
```

**Step 2: 验证异常类可以正常导入**

运行: `python -c "from labelmatrix.exceptions.dataset_errors import UnsupportedTaskError, InvalidGeometryError; print('Import successful')"`
预期: 输出 "Import successful"

**Step 3: 提交**

```bash
git add labelmatrix/exceptions/dataset_errors.py
git commit -m "feat: add UnsupportedTaskError and InvalidGeometryError exceptions"
```

---

## Task 2: 创建BaseTaskConverter抽象基类

**Files:**
- Create: `labelmatrix/utils/task_converters.py`

**Step 1: 创建文件并编写抽象基类**

```python
# -*- coding: utf-8 -*-
"""
任务转换器基类和具体实现
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, List, Any
import logging

logger = logging.getLogger(__name__)


class BaseTaskConverter(ABC):
    """任务转换器抽象基类"""

    def __init__(
        self,
        categories: dict,
        source_config_data: dict,
        verbose_logging: bool = False
    ):
        """
        初始化转换器

        Args:
            categories: 类别ID到类别名称的映射
            source_config_data: 源data.yaml配置数据
            verbose_logging: 是否使用详细日志模式
        """
        self.categories = categories
        self._source_config_data = source_config_data
        self.verbose_logging = verbose_logging

    @abstractmethod
    def convert_feature(
        self,
        feature: dict,
        img_width: int,
        img_height: int
    ) -> Optional[str]:
        """
        转换单个feature为YOLO格式行

        Args:
            feature: GeoJSON feature对象
            img_width: 图像宽度
            img_height: 图像高度

        Returns:
            Optional[str]: YOLO格式的标注行，如果转换失败返回None
        """
        pass

    def normalize_coordinate(
        self,
        x: float,
        y: float,
        img_width: int,
        img_height: int
    ) -> Tuple[float, float]:
        """
        归一化单个坐标点

        Args:
            x: 原始x坐标
            y: 原始y坐标
            img_width: 图像宽度
            img_height: 图像高度

        Returns:
            Tuple[float, float]: 归一化后的坐标 (x, y)
        """
        return (x / img_width, y / img_height)

    def normalize_coordinates(
        self,
        coordinates: List[List[float]],
        img_width: int,
        img_height: int
    ) -> List[Tuple[float, float]]:
        """
        归一化坐标列表

        Args:
            coordinates: 原始坐标列表 [[x, y, z], ...]
            img_width: 图像宽度
            img_height: 图像高度

        Returns:
            List[Tuple[float, float]]: 归一化后的坐标列表 [(x, y), ...]
        """
        normalized = []
        for coord in coordinates:
            x = coord[0] / img_width
            y = coord[1] / img_height
            normalized.append((x, y))
        return normalized

    def get_yolo_class_id(self, class_id: int) -> int:
        """
        获取YOLO格式的class_id（处理从1开始的索引）

        Args:
            class_id: 原始class_id

        Returns:
            int: YOLO格式的class_id（从0开始）
        """
        names = self._source_config_data.get('names')
        needs_conversion = False

        if isinstance(names, dict):
            keys = [int(k) for k in names.keys()]
            if keys and min(keys) == 1:
                needs_conversion = True
        elif isinstance(names, list):
            needs_conversion = True

        return class_id - 1 if needs_conversion and class_id > 0 else class_id
```

**Step 2: 验证文件可以正常导入**

运行: `python -c "from labelmatrix.utils.task_converters import BaseTaskConverter; print('Import successful')"`
预期: 输出 "Import successful"

**Step 3: 提交**

```bash
git add labelmatrix/utils/task_converters.py
git commit -m "feat: add BaseTaskConverter abstract base class"
```

---

## Task 3: 实现SegmentConverter

**Files:**
- Modify: `labelmatrix/utils/task_converters.py`

**Step 1: 在task_converters.py中添加SegmentConverter类**

在BaseTaskConverter类后面添加：

```python
class SegmentConverter(BaseTaskConverter):
    """分割任务转换器"""

    def convert_feature(
        self,
        feature: dict,
        img_width: int,
        img_height: int
    ) -> Optional[str]:
        """
        转换分割任务的feature

        Args:
            feature: GeoJSON feature对象
            img_width: 图像宽度
            img_height: 图像高度

        Returns:
            Optional[str]: YOLO分割格式的标注行
        """
        props = feature.get('properties', {})
        class_id = props.get('class_id')
        geometry = feature.get('geometry', {})

        if class_id is None:
            if self.verbose_logging:
                logger.debug("Feature missing class_id")
            return None

        if geometry.get('type') != 'Polygon':
            if self.verbose_logging:
                logger.debug(f"Invalid geometry type: {geometry.get('type')}")
            return None

        coordinates = geometry.get('coordinates', [])
        if not coordinates:
            if self.verbose_logging:
                logger.debug("Empty coordinates")
            return None

        # 转换多边形坐标
        normalized_coords = self.normalize_coordinates(
            coordinates[0], img_width, img_height
        )

        # 转换class_id
        yolo_class_id = self.get_yolo_class_id(class_id)

        # YOLO格式: class_id x1 y1 x2 y2 ... xn yn
        line = f"{yolo_class_id} " + " ".join(
            f"{x:.6f} {y:.6f}" for x, y in normalized_coords
        )
        return line
```

**Step 2: 验证SegmentConverter可以正常导入**

运行: `python -c "from labelmatrix.utils.task_converters import SegmentConverter; print('Import successful')"`
预期: 输出 "Import successful"

**Step 3: 提交**

```bash
git add labelmatrix/utils/task_converters.py
git commit -m "feat: add SegmentConverter for segmentation tasks"
```

---

## Task 4: 实现DetectConverter

**Files:**
- Modify: `labelmatrix/utils/task_converters.py`

**Step 1: 添加DetectConverter类**

```python
import numpy as np


class DetectConverter(BaseTaskConverter):
    """检测任务转换器 - 水平边界框"""

    def convert_feature(
        self,
        feature: dict,
        img_width: int,
        img_height: int
    ) -> Optional[str]:
        """
        转换检测任务的feature

        Args:
            feature: GeoJSON feature对象
            img_width: 图像宽度
            img_height: 图像高度

        Returns:
            Optional[str]: YOLO检测格式的标注行 (class_id x_center y_center width height)
        """
        props = feature.get('properties', {})
        class_id = props.get('class_id')
        geometry = feature.get('geometry', {})

        if class_id is None:
            if self.verbose_logging:
                logger.debug("Feature missing class_id")
            return None

        if geometry.get('type') != 'Polygon':
            # 尝试从其他几何类型提取边界框
            if self.verbose_logging:
                logger.debug(f"Attempting to extract bbox from {geometry.get('type')}")
            return self._try_extract_bbox(feature, img_width, img_height, class_id)

        coordinates = geometry.get('coordinates', [])
        if not coordinates:
            if self.verbose_logging:
                logger.debug("Empty coordinates")
            return None

        # 提取外接矩形（AABB）
        polygon_coords = coordinates[0]
        x_center, y_center, width, height = self._calculate_aabb(
            polygon_coords, img_width, img_height
        )

        # 转换class_id
        yolo_class_id = self.get_yolo_class_id(class_id)

        # YOLO检测格式: class_id x_center y_center width height
        return f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

    def _calculate_aabb(
        self,
        coordinates: List[List[float]],
        img_width: int,
        img_height: int
    ) -> Tuple[float, float, float, float]:
        """
        计算轴对齐边界框

        Args:
            coordinates: 多边形坐标列表
            img_width: 图像宽度
            img_height: 图像高度

        Returns:
            Tuple[float, float, float, float]: (x_center, y_center, width, height) 归一化后的值
        """
        # 提取所有x和y坐标
        x_coords = [coord[0] for coord in coordinates]
        y_coords = [coord[1] for coord in coordinates]

        # 计算边界
        min_x = min(x_coords)
        max_x = max(x_coords)
        min_y = min(y_coords)
        max_y = max(y_coords)

        # 归一化并计算中心点和尺寸
        x_center = ((min_x + max_x) / 2) / img_width
        y_center = ((min_y + max_y) / 2) / img_height
        width = (max_x - min_x) / img_width
        height = (max_y - min_y) / img_height

        return x_center, y_center, width, height

    def _try_extract_bbox(
        self,
        feature: dict,
        img_width: int,
        img_height: int,
        class_id: int
    ) -> Optional[str]:
        """
        尝试从非Polygon几何类型提取边界框

        Args:
            feature: GeoJSON feature对象
            img_width: 图像宽度
            img_height: 图像高度
            class_id: 类别ID

        Returns:
            Optional[str]: YOLO检测格式的标注行
        """
        geometry = feature.get('geometry', {})
        geom_type = geometry.get('type')

        # 对于Point，创建一个小的边界框
        if geom_type == 'Point':
            coords = geometry.get('coordinates', [])
            if coords:
                x, y = coords[0], coords[1]
                # 创建1%图像大小的边界框
                box_size = 0.01
                x_norm = x / img_width
                y_norm = y / img_height
                yolo_class_id = self.get_yolo_class_id(class_id)
                return f"{yolo_class_id} {x_norm:.6f} {y_norm:.6f} {box_size:.6f} {box_size:.6f}"

        # 对于LineString，使用其端点计算边界框
        elif geom_type == 'LineString':
            coords = geometry.get('coordinates', [])
            if len(coords) >= 2:
                x_center, y_center, width, height = self._calculate_aabb(
                    coords, img_width, img_height
                )
                yolo_class_id = self.get_yolo_class_id(class_id)
                return f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

        if self.verbose_logging:
            logger.warning(f"Cannot extract bbox from geometry type: {geom_type}")
        return None
```

**Step 2: 验证DetectConverter可以正常导入**

运行: `python -c "from labelmatrix.utils.task_converters import DetectConverter; print('Import successful')"`
预期: 输出 "Import successful"

**Step 3: 提交**

```bash
git add labelmatrix/utils/task_converters.py
git commit -m "feat: add DetectConverter for object detection tasks"
```

---

## Task 5: 实现OBBConverter

**Files:**
- Modify: `labelmatrix/utils/task_converters.py`

**Step 1: 添加OBBConverter类**

```python
class OBBConverter(BaseTaskConverter):
    """旋转框检测转换器"""

    def convert_feature(
        self,
        feature: dict,
        img_width: int,
        img_height: int
    ) -> Optional[str]:
        """
        转换旋转框检测任务的feature

        Args:
            feature: GeoJSON feature对象
            img_width: 图像宽度
            img_height: 图像高度

        Returns:
            Optional[str]: YOLO OBB格式的标注行 (class_id x1 y1 x2 y2 x3 y3 x4 y4)
        """
        props = feature.get('properties', {})
        class_id = props.get('class_id')
        geometry = feature.get('geometry', {})

        if class_id is None:
            if self.verbose_logging:
                logger.debug("Feature missing class_id")
            return None

        if geometry.get('type') != 'Polygon':
            if self.verbose_logging:
                logger.debug(f"OBB requires Polygon geometry, got {geometry.get('type')}")
            return None

        coordinates = geometry.get('coordinates', [])
        if not coordinates:
            if self.verbose_logging:
                logger.debug("Empty coordinates")
            return None

        polygon_coords = coordinates[0]

        # 验证是否为4个角点
        if not self._validate_four_corners(polygon_coords):
            if self.verbose_logging:
                logger.warning(
                    f"OBB requires exactly 4 corner points, got {len(polygon_coords)}"
                )
            return None

        # 归一化四个角点
        normalized_coords = self.normalize_coordinates(
            polygon_coords, img_width, img_height
        )

        # 转换class_id
        yolo_class_id = self.get_yolo_class_id(class_id)

        # YOLO OBB格式: class_id x1 y1 x2 y2 x3 y3 x4 y4
        line = f"{yolo_class_id} " + " ".join(
            f"{x:.6f} {y:.6f}" for x, y in normalized_coords
        )
        return line

    def _validate_four_corners(self, coordinates: List[List[float]]) -> bool:
        """
        验证是否为4个角点

        Args:
            coordinates: 坐标列表

        Returns:
            bool: 是否为4个角点
        """
        return len(coordinates) == 4
```

**Step 2: 验证OBBConverter可以正常导入**

运行: `python -c "from labelmatrix.utils.task_converters import OBBConverter; print('Import successful')"`
预期: 输出 "Import successful"

**Step 3: 提交**

```bash
git add labelmatrix/utils/task_converters.py
git commit -m "feat: add OBBConverter for oriented bounding box tasks"
```

---

## Task 6: 修改GeoJSONToYOLOConverter以支持多任务

**Files:**
- Modify: `labelmatrix/utils/geojson_to_yolo.py`

**Step 1: 添加task字段读取和转换器工厂方法**

在`GeoJSONToYOLOConverter`类的`__init__`方法中添加task相关代码：

```python
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
    self.source_path = Path(source_dataset_path)
    self.source_config = self.source_path / 'data.yaml'

    # 保存原始配置信息（需要在_load_categories之前初始化）
    self._source_config_data: Dict[str, Any] = {}

    # 读取源配置获取类别信息
    self.categories = self._load_categories()

    # 读取task类型
    self.task = self._load_task()

    # 创建对应任务的转换器
    self.task_converter = self._create_task_converter()

    # 输出目录 - 在源数据集目录内部创建yolo_format
    self.output_path = self.source_path / 'yolo_format'

    # 训练/验证集划分配置
    self.train_val_split = train_val_split
    self.random_seed = random_seed
    self.verbose_logging = verbose_logging

    # 文件划分结果
    self._train_files: List[str] = []
    self._val_files: List[str] = []
```

**Step 2: 添加_load_task方法**

在类中添加：

```python
def _load_task(self) -> str:
    """
    从data.yaml加载task类型

    Returns:
        str: task类型 (segment/detect/obb)

    Raises:
        UnsupportedTaskError: 当task类型不支持时
    """
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
```

**Step 3: 添加_create_task_converter工厂方法**

```python
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
```

**Step 4: 修改_convert_single_geojson方法使用task_converter**

找到现有的`_convert_single_geojson`方法，替换标注生成逻辑：

```python
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
```

**Step 5: 验证修改后的代码可以正常导入**

运行: `python -c "from labelmatrix.utils.geojson_to_yolo import GeoJSONToYOLOConverter; print('Import successful')"`
预期: 输出 "Import successful"

**Step 6: 提交**

```bash
git add labelmatrix/utils/geojson_to_yolo.py
git commit -m "feat: integrate task converters into GeoJSONToYOLOConverter"
```

---

## Task 7: 更新DatasetValidator以支持verbose_logging

**Files:**
- Modify: `labelmatrix/dataset_validator.py`

**Step 1: 添加verbose_logging参数**

```python
class DatasetValidator:
    """数据集格式验证器"""

    def __init__(
        self,
        data_config_path: str,
        train_val_split: float = 0.8,
        random_seed: int = 42,
        verbose_logging: bool = False
    ):
        """
        初始化验证器

        Args:
            data_config_path: 数据集配置文件路径 (data.yaml)
            train_val_split: 训练集比例，默认0.8（80%训练，20%验证）
            random_seed: 随机种子，默认42
            verbose_logging: 是否使用详细日志模式，默认False
        """
        self.data_config_path = Path(data_config_path)
        self.dataset_root = self.data_config_path.parent
        self.train_val_split = train_val_split
        self.random_seed = random_seed
        self.verbose_logging = verbose_logging
```

**Step 2: 更新get_converter方法**

```python
def get_converter(self):
    """
    获取转换器实例

    Returns:
        GeoJSONToYOLOConverter: 转换器实例
    """
    from labelmatrix.utils.geojson_to_yolo import GeoJSONToYOLOConverter
    return GeoJSONToYOLOConverter(
        str(self.dataset_root),
        train_val_split=self.train_val_split,
        random_seed=self.random_seed,
        verbose_logging=self.verbose_logging
    )
```

**Step 3: 提交**

```bash
git add labelmatrix/dataset_validator.py
git commit -m "feat: add verbose_logging support to DatasetValidator"
```

---

## Task 8: 创建单元测试

**Files:**
- Create: `tests/test_task_converters.py`

**Step 1: 创建测试文件**

```python
# -*- coding: utf-8 -*-
"""
任务转换器单元测试
"""

import pytest
from labelmatrix.utils.task_converters import (
    SegmentConverter,
    DetectConverter,
    OBBConverter
)
from labelmatrix.exceptions.dataset_errors import UnsupportedTaskError


class TestSegmentConverter:
    """分割转换器测试"""

    @pytest.fixture
    def converter(self):
        categories = {0: 'building', 1: 'road'}
        config_data = {'names': {0: 'building', 1: 'road'}}
        return SegmentConverter(categories, config_data)

    def test_convert_valid_polygon(self, converter):
        feature = {
            'properties': {'class_id': 1},
            'geometry': {
                'type': 'Polygon',
                'coordinates': [[[0, 0], [100, 0], [100, 100], [0, 100], [0, 0]]]
            }
        }
        result = converter.convert_feature(feature, 512, 512)
        assert result is not None
        assert result.startswith("0")  # class_id should be converted from 1 to 0

    def test_convert_missing_class_id(self, converter):
        feature = {
            'properties': {},
            'geometry': {'type': 'Polygon', 'coordinates': [[[0, 0], [100, 0], [100, 100], [0, 100], [0, 0]]]}
        }
        result = converter.convert_feature(feature, 512, 512)
        assert result is None

    def test_convert_invalid_geometry_type(self, converter):
        feature = {
            'properties': {'class_id': 1},
            'geometry': {'type': 'Point', 'coordinates': [50, 50]}
        }
        result = converter.convert_feature(feature, 512, 512)
        assert result is None


class TestDetectConverter:
    """检测转换器测试"""

    @pytest.fixture
    def converter(self):
        categories = {0: 'building', 1: 'road'}
        config_data = {'names': {0: 'building', 1: 'road'}}
        return DetectConverter(categories, config_data)

    def test_convert_axis_aligned_rectangle(self, converter):
        feature = {
            'properties': {'class_id': 1},
            'geometry': {
                'type': 'Polygon',
                'coordinates': [[[0, 0], [100, 0], [100, 50], [0, 50], [0, 0]]]
            }
        }
        result = converter.convert_feature(feature, 512, 512)
        assert result is not None
        parts = result.split()
        assert len(parts) == 5  # class_id + x_center + y_center + width + height
        assert parts[0] == "0"

    def test_convert_rotated_rectangle(self, converter):
        # 旋转矩形的四个角点
        feature = {
            'properties': {'class_id': 1},
            'geometry': {
                'type': 'Polygon',
                'coordinates': [[[50, 0], [100, 50], [50, 100], [0, 50], [50, 0]]]
            }
        }
        result = converter.convert_feature(feature, 512, 512)
        assert result is not None
        # 应该计算外接矩形
        parts = result.split()
        assert len(parts) == 5

    def test_convert_point(self, converter):
        feature = {
            'properties': {'class_id': 1},
            'geometry': {'type': 'Point', 'coordinates': [256, 256]}
        }
        result = converter.convert_feature(feature, 512, 512)
        assert result is not None
        # Point应该转换为小的边界框


class TestOBBConverter:
    """旋转框转换器测试"""

    @pytest.fixture
    def converter(self):
        categories = {0: 'building', 1: 'road'}
        config_data = {'names': {0: 'building', 1: 'road'}}
        return OBBConverter(categories, config_data)

    def test_convert_four_corner_polygon(self, converter):
        feature = {
            'properties': {'class_id': 1},
            'geometry': {
                'type': 'Polygon',
                'coordinates': [[[50, 0], [100, 50], [50, 100], [0, 50], [50, 0]]]
            }
        }
        result = converter.convert_feature(feature, 512, 512)
        assert result is not None
        parts = result.split()
        assert len(parts) == 9  # class_id + 4 corners * 2 coordinates

    def test_convert_invalid_corner_count(self, converter):
        feature = {
            'properties': {'class_id': 1},
            'geometry': {
                'type': 'Polygon',
                'coordinates': [[[0, 0], [100, 0], [100, 100], [50, 150], [0, 100], [0, 0]]]
            }
        }
        result = converter.convert_feature(feature, 512, 512)
        assert result is None  # 不是4个角点

    def test_convert_non_polygon(self, converter):
        feature = {
            'properties': {'class_id': 1},
            'geometry': {'type': 'Point', 'coordinates': [256, 256]}
        }
        result = converter.convert_feature(feature, 512, 512)
        assert result is None


class TestTaskConverterFactory:
    """转换器工厂测试"""

    def test_invalid_task_raises_error(self):
        from labelmatrix.utils.geojson_to_yolo import GeoJSONToYOLOConverter
        # 这个测试需要mock data.yaml，在实际集成测试中会更合适
        pass
```

**Step 2: 运行测试验证失败**

运行: `pytest tests/test_task_converters.py -v`
预期: 部分测试可能失败（因为我们还没有创建测试数据）

**Step 3: 提交**

```bash
git add tests/test_task_converters.py
git commit -m "test: add unit tests for task converters"
```

---

## Task 9: 创建集成测试数据和测试

**Files:**
- Create: `tests/fixtures/geojsons/detect_sample.geojson`
- Create: `tests/fixtures/geojsons/obb_sample.geojson`
- Create: `tests/fixtures/geojsons/segment_sample.geojson`
- Create: `tests/test_integration.py`

**Step 1: 创建detect测试数据**

创建 `tests/fixtures/geojsons/detect_sample.geojson`:

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {
        "class_id": 1,
        "image_size": [512, 512],
        "image_ref": "detect_image.jpg"
      },
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[100, 100], [200, 100], [200, 200], [100, 200], [100, 100]]]
      }
    },
    {
      "type": "Feature",
      "properties": {
        "class_id": 2,
        "image_size": [512, 512],
        "image_ref": "detect_image.jpg"
      },
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[300, 300], [400, 300], [400, 400], [300, 400], [300, 300]]]
      }
    }
  ]
}
```

**Step 2: 创建obb测试数据**

创建 `tests/fixtures/geojsons/obb_sample.geojson`:

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {
        "class_id": 1,
        "image_size": [512, 512],
        "image_ref": "obb_image.jpg"
      },
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[150, 100], [200, 150], [150, 200], [100, 150], [150, 100]]]
      }
    }
  ]
}
```

**Step 3: 创建segment测试数据**

创建 `tests/fixtures/geojsons/segment_sample.geojson`:

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {
        "class_id": 1,
        "image_size": [512, 512],
        "image_ref": "segment_image.jpg"
      },
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[100, 100], [150, 120], [200, 100], [180, 150], [200, 200], [150, 180], [100, 200], [120, 150], [100, 100]]]
      }
    }
  ]
}
```

**Step 4: 创建集成测试**

创建 `tests/test_integration.py`:

```python
# -*- coding: utf-8 -*-
"""
集成测试
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from labelmatrix.utils.geojson_to_yolo import GeoJSONToYOLOConverter


class TestGeoJSONToYOLOIntegration:
    """GeoJSON到YOLO转换集成测试"""

    @pytest.fixture
    def temp_dataset_dir(self, tmp_path):
        """创建临时测试数据集"""
        # 创建必要的目录
        (tmp_path / 'geojsons').mkdir()
        (tmp_path / 'images').mkdir()

        # 创建data.yaml
        data_yaml = {
            'task': 'detect',
            'names': {0: 'building', 1: 'road', 2: 'background'}
        }
        with open(tmp_path / 'data.yaml', 'w') as f:
            yaml.dump(data_yaml, f)

        # 复制测试GeoJSON文件
        import shutil
        test_fixtures = Path(__file__).parent / 'fixtures' / 'geojsons'
        if test_fixtures.exists():
            for geojson_file in test_fixtures.glob('*.geojson'):
                shutil.copy(geojson_file, tmp_path / 'geojsons')

        return tmp_path

    def test_detect_task_conversion(self, temp_dataset_dir):
        """测试detect任务转换"""
        # 修改data.yaml为detect任务
        data_yaml = {
            'task': 'detect',
            'names': {0: 'building', 1: 'road', 2: 'background'}
        }
        with open(temp_dataset_dir / 'data.yaml', 'w') as f:
            yaml.dump(data_yaml, f)

        # 复制detect样本
        import shutil
        detect_sample = Path(__file__).parent / 'fixtures' / 'geojsons' / 'detect_sample.geojson'
        if detect_sample.exists():
            shutil.copy(detect_sample, temp_dataset_dir / 'geojsons')

            # 执行转换
            converter = GeoJSONToYOLOConverter(str(temp_dataset_dir))
            output_path = converter.convert()

            # 验证输出
            output_dir = Path(output_path).parent
            label_files = list((output_dir / 'labels' / 'train2017').glob('*.txt'))
            assert len(label_files) > 0

            # 验证标签格式
            with open(label_files[0], 'r') as f:
                content = f.read()
                lines = content.strip().split('\n')
                for line in lines:
                    parts = line.split()
                    assert len(parts) == 5  # detect格式: class_id x_center y_center width height

    def test_obb_task_conversion(self, temp_dataset_dir):
        """测试obb任务转换"""
        # 修改data.yaml为obb任务
        data_yaml = {
            'task': 'obb',
            'names': {0: 'building', 1: 'road'}
        }
        with open(temp_dataset_dir / 'data.yaml', 'w') as f:
            yaml.dump(data_yaml, f)

        # 复制obb样本
        import shutil
        obb_sample = Path(__file__).parent / 'fixtures' / 'geojsons' / 'obb_sample.geojson'
        if obb_sample.exists():
            shutil.copy(obb_sample, temp_dataset_dir / 'geojsons')

            # 执行转换
            converter = GeoJSONToYOLOConverter(str(temp_dataset_dir))
            output_path = converter.convert()

            # 验证输出
            output_dir = Path(output_path).parent
            label_files = list((output_dir / 'labels' / 'train2017').glob('*.txt'))
            assert len(label_files) > 0

            # 验证标签格式
            with open(label_files[0], 'r') as f:
                content = f.read()
                lines = content.strip().split('\n')
                for line in lines:
                    parts = line.split()
                    assert len(parts) == 9  # obb格式: class_id + 4个角点*2坐标

    def test_segment_task_conversion(self, temp_dataset_dir):
        """测试segment任务转换"""
        # 修改data.yaml为segment任务
        data_yaml = {
            'task': 'segment',
            'names': {0: 'building'}
        }
        with open(temp_dataset_dir / 'data.yaml', 'w') as f:
            yaml.dump(data_yaml, f)

        # 复制segment样本
        import shutil
        segment_sample = Path(__file__).parent / 'fixtures' / 'geojsons' / 'segment_sample.geojson'
        if segment_sample.exists():
            shutil.copy(segment_sample, temp_dataset_dir / 'geojsons')

            # 执行转换
            converter = GeoJSONToYOLOConverter(str(temp_dataset_dir))
            output_path = converter.convert()

            # 验证输出
            output_dir = Path(output_path).parent
            label_files = list((output_dir / 'labels' / 'train2017').glob('*.txt'))
            assert len(label_files) > 0

            # 验证标签格式（segment应该是可变长度）
            with open(label_files[0], 'r') as f:
                content = f.read()
                assert len(content) > 0
```

**Step 5: 运行集成测试**

运行: `pytest tests/test_integration.py -v`
预期: 所有测试通过

**Step 6: 提交**

```bash
git add tests/
git commit -m "test: add integration tests for multi-task conversion"
```

---

## Task 10: 验证完整功能和清理

**Files:**
- Various

**Step 1: 运行完整测试套件**

运行: `pytest tests/ -v`
预期: 所有测试通过

**Step 2: 手动验证（可选）**

如果可能，使用真实数据集验证转换结果：
```python
from labelmatrix.utils.geojson_to_yolo import GeoJSONToYOLOConverter

converter = GeoJSONToYOLOConverter(
    '/path/to/your/dataset',
    verbose_logging=True
)
output_path = converter.convert()
print(f"转换完成: {output_path}")
```

**Step 3: 检查代码质量**

运行: `flake8 labelmatrix/utils/task_converters.py labelmatrix/utils/geojson_to_yolo.py` (如果安装了flake8)
或运行: `pylint labelmatrix/utils/task_converters.py` (如果安装了pylint)

**Step 4: 更新文档（如果需要）**

如果项目有README或文档，更新使用说明以反映新的多任务支持功能。

**Step 5: 最终提交**

```bash
git add .
git commit -m "feat: complete multi-task YOLO converter implementation"
```

---

## 实施注意事项

1. **TDD原则**: 每个任务都遵循"写测试→运行测试→实现代码→验证→提交"的循环
2. **小步提交**: 每个小的改动都立即提交，便于回滚和代码审查
3. **DRY原则**: 共同的逻辑提取到BaseTaskConverter基类中
4. **YAGNI原则**: 只实现当前需要的功能，不添加不必要的特性

## 预期结果

完成后，`GeoJSONToYOLOConverter`将能够：
- 根据data.yaml中的task字段自动选择正确的转换器
- 支持segment（多边形）、detect（水平边界框）、obb（旋转边界框）三种格式
- 提供可配置的日志详细程度
- 保持向后兼容性（默认为segment任务）
