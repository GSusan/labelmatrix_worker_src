# -*- coding: utf-8 -*-
"""
引擎抽象基类
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import threading
import json
import logging
from datetime import datetime


logger = logging.getLogger(__name__)


@dataclass
class TrainResult:
    """训练结果"""
    success: bool
    best_model_path: Optional[Path]
    last_model_path: Optional[Path]
    metrics: Dict[str, float]
    error: Optional[str] = None
    test_metrics: Optional[Dict[str, float]] = None
    test_evaluated: bool = False
    test_skipped_reason: Optional[str] = None


@dataclass
class PredictResult:
    """推理结果"""
    success: bool
    result_files: List[Path]
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseEngine(ABC):
    """所有引擎的抽象基类"""

    # 支持的任务类型和模式
    SUPPORTED_TASKS: List[str] = []
    SUPPORTED_MODES: List[str] = ['train', 'predict', 'resume']

    def __init__(self, config: Dict[str, Any]):
        """
        初始化引擎

        Args:
            config: 配置字典，包含任务的所有必要参数
        """
        self.config = config
        self.task_id = config['task_id']
        self.task_type = config['task_type']
        self.mode = config.get('mode', 'train')
        self.output_dir = Path(config['output_dir']) / self.task_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 创建子目录
        (self.output_dir / 'logs').mkdir(exist_ok=True)
        (self.output_dir / 'results').mkdir(exist_ok=True)

        # 状态控制
        self._stop_flag = threading.Event()
        self._status = 'initialized'
        self._progress = 0
        self._metrics: Dict[str, float] = {}
        self._error: Optional[str] = None

        # 状态文件路径
        self._state_file_path = self.output_dir / 'state.json'
        self._state_lock = threading.Lock()  # 用于状态文件写入的线程锁

        # 时间戳
        self._start_time: Optional[datetime] = None
        self._end_time: Optional[datetime] = None

    @abstractmethod
    def train(self) -> TrainResult:
        """
        执行训练任务

        Returns:
            TrainResult: 训练结果对象
        """
        pass

    @abstractmethod
    def predict(self) -> PredictResult:
        """
        执行推理任务

        Returns:
            PredictResult: 推理结果对象
        """
        pass

    @abstractmethod
    def resume(self) -> TrainResult:
        """
        继续训练

        Returns:
            TrainResult: 训练结果对象
        """
        pass

    def stop(self) -> bool:
        """
        请求停止当前任务

        Returns:
            bool: 停止信号是否成功发送
        """
        self._stop_flag.set()
        self._status = 'stopping'
        return True

    def get_status(self) -> Dict[str, Any]:
        """
        获取当前状态

        Returns:
            状态字典，包含 status, progress, metrics, error 等
        """
        return {
            'status': self._status,
            'progress': self._progress,
            'metrics': self._metrics.copy(),
            'error': self._error,
            'task_id': self.task_id,
            'task_type': self.task_type
        }

    def _update_status(self, status: str, progress: int = None, metrics: Dict[str, float] = None, error: str = None):
        """更新内部状态"""
        self._status = status
        if progress is not None:
            self._progress = max(0, min(100, progress))
        if metrics:
            self._metrics.update(metrics)
        if error:
            self._error = error

    def _is_stopped(self) -> bool:
        """检查是否收到停止信号"""
        return self._stop_flag.is_set()

    def _get_device(self):
        """获取设备对象"""
        import torch
        device_config = self.config.get('hardware', {}).get('device', 'cpu')
        if device_config == 'cpu':
            return torch.device('cpu')
        return torch.device(device_config)

    def read_state_file(self) -> Dict[str, Any]:
        """
        从状态文件读取状态

        Returns:
            状态字典，如果文件不存在或解析失败则返回空字典
        """
        if not self._state_file_path.exists():
            return {}

        try:
            with open(self._state_file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to read state file: {e}")
            return {}

    def _write_state_atomically(self, state_data: Dict[str, Any]) -> bool:
        """
        原子写入状态文件

        Args:
            state_data: 要写入的状态数据

        Returns:
            是否成功写入
        """
        with self._state_lock:
            try:
                # 先写入临时文件
                temp_file = self._state_file_path.with_suffix('.tmp')
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(state_data, f, indent=2, ensure_ascii=False)

                # 原子重命名
                temp_file.replace(self._state_file_path)
                return True
            except (IOError, OSError) as e:
                logger.warning(f"Failed to write state file: {e}")
                return False

    def _build_state_dict(self) -> Dict[str, Any]:
        """
        构建状态字典

        Returns:
            完整的状态字典
        """
        return {
            'task_id': self.task_id,
            'status': self._status,
            'progress': self._progress,
            'metrics': self._metrics.copy(),
            'error': self._error,
            'task_type': self.task_type,
            'timestamp': datetime.now().isoformat(),
            'start_time': self._start_time.isoformat() if self._start_time else None,
            'end_time': self._end_time.isoformat() if self._end_time else None
        }
