# -*- coding: utf-8 -*-
"""
元数据构建器模块
"""

from typing import Dict, Any, List
from pathlib import Path
from datetime import datetime, timedelta
import json


class MetadataBuilder:
    """元数据构建器"""

    def build(self, config: Dict[str, Any], result) -> Dict[str, Any]:
        """
        构建训练元数据

        Args:
            config: 配置字典
            result: 训练结果对象

        Returns:
            元数据字典
        """
        task_id = config['task_id']
        task_type = config['task_type']
        model_arch = config.get('model_architecture', 'unknown')

        # 从数据集配置中获取类别信息
        class_names = self._extract_class_names(config)

        # 计算训练时长
        start_time = datetime.now()
        end_time = datetime.now()
        duration = end_time - start_time

        metadata = {
            'model_id': task_id,
            'task_type': task_type,
            'model_architecture': model_arch,
            'train_start_time': start_time.isoformat(),
            'train_end_time': end_time.isoformat(),
            'train_duration': self._format_duration(duration),
            'best_metric': result.metrics if hasattr(result, 'metrics') and result.metrics else {},
            'config_file': 'config.yaml',
            'weight_file': 'best_model.pt',
            'dataset_path': config.get('data_config', ''),
            'class_names': class_names
        }

        return metadata

    def save(self, metadata: Dict[str, Any], output_dir: Path):
        """
        保存元数据到文件

        Args:
            metadata: 元数据字典
            output_dir: 输出目录
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        metadata_file = output_dir / 'metadata.json'

        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    def _extract_class_names(self, config: Dict[str, Any]) -> List[str]:
        """
        从配置中提取类别名称

        Args:
            config: 配置字典

        Returns:
            类别名称列表
        """
        # 尝试从数据集配置中读取
        data_config = config.get('data_config', '')
        if data_config:
            class_names = self._read_class_names_from_yaml(data_config)
            if class_names:
                return class_names

        # 尝试从配置中直接获取
        if 'class_names' in config:
            return config['class_names']

        # 默认类别
        return []

    def _read_class_names_from_yaml(self, yaml_path: str) -> List[str]:
        """
        从数据集YAML文件中读取类别名称

        Args:
            yaml_path: YAML文件路径

        Returns:
            类别名称列表
        """
        try:
            import yaml

            yaml_file = Path(yaml_path)
            if not yaml_file.exists():
                return []

            with open(yaml_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            # Ultralytics格式
            if 'names' in data:
                names = data['names']
                if isinstance(names, dict):
                    return [names[i] for i in sorted(names.keys())]
                elif isinstance(names, list):
                    return names

            # 其他格式
            if 'classes' in data:
                return data['classes']

        except Exception:
            pass

        return []

    def _format_duration(self, duration: timedelta) -> str:
        """
        格式化时长

        Args:
            duration: 时长

        Returns:
            格式化后的时长字符串
        """
        total_seconds = int(duration.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60

        if hours > 0:
            return f"{hours}h{minutes}m"
        elif minutes > 0:
            return f"{minutes}m{seconds}s"
        else:
            return f"{seconds}s"
