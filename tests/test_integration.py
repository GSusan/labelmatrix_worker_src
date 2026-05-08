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
            # 创建虚拟图像文件以避免警告
            (temp_dataset_dir / 'images' / 'detect_image.jpg').parent.mkdir(parents=True, exist_ok=True)
            (temp_dataset_dir / 'images' / 'detect_image.jpg').touch()

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
            # 创建虚拟图像文件以避免警告
            (temp_dataset_dir / 'images' / 'obb_image.jpg').parent.mkdir(parents=True, exist_ok=True)
            (temp_dataset_dir / 'images' / 'obb_image.jpg').touch()

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
                # 确保内容不为空
                assert len(content) > 0, "Label file is empty"
                lines = content.strip().split('\n')
                # 过滤空行
                lines = [line for line in lines if line.strip()]
                assert len(lines) > 0, "No valid lines in label file"
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
            # 创建虚拟图像文件以避免警告
            (temp_dataset_dir / 'images' / 'segment_image.jpg').parent.mkdir(parents=True, exist_ok=True)
            (temp_dataset_dir / 'images' / 'segment_image.jpg').touch()

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
