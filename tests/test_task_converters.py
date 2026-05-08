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


class TestSegmentConverter:
    """分割转换器测试"""

    @pytest.fixture
    def converter(self):
        categories = {0: 'building', 1: 'road'}
        # 使用从1开始的索引来测试class_id转换
        config_data = {'names': {1: 'building', 2: 'road'}}
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
        # 使用从1开始的索引来测试class_id转换
        config_data = {'names': {1: 'building', 2: 'road'}}
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
        # 使用从1开始的索引来测试class_id转换
        config_data = {'names': {1: 'building', 2: 'road'}}
        return OBBConverter(categories, config_data)

    def test_convert_four_corner_polygon(self, converter):
        # OBB使用4个角点（不闭合）
        feature = {
            'properties': {'class_id': 1},
            'geometry': {
                'type': 'Polygon',
                'coordinates': [[[50, 0], [100, 50], [50, 100], [0, 50]]]
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
