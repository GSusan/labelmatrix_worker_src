# -*- coding: utf-8 -*-
"""
任务状态管理器
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import threading
import json
from pathlib import Path


class StateManager:
    """任务状态管理器"""

    VALID_STATUSES = [
        'initialized',
        'running',
        'completed',
        'failed',
        'stopped',
        'stopping'
    ]

    def __init__(self, task_id: str, output_dir: Path):
        """
        Args:
            task_id: 任务ID
            output_dir: 输出目录
        """
        self.task_id = task_id
        self.output_dir = output_dir
        self._lock = threading.Lock()

        # 状态数据
        self._status = 'initialized'
        self._progress = 0
        self._current_epoch = 0
        self._total_epochs = 0
        self._metrics: Dict[str, Any] = {}
        self._metrics_history: List[Dict[str, Any]] = []
        self._error: Optional[str] = None
        self._start_time: Optional[datetime] = None
        self._end_time: Optional[datetime] = None

        # 状态持久化文件
        self._state_file = output_dir / 'state.json'

    def get_state(self) -> Dict[str, Any]:
        """获取当前完整状态"""
        with self._lock:
            return {
                'task_id': self.task_id,
                'status': self._status,
                'progress': self._progress,
                'current_epoch': self._current_epoch,
                'total_epochs': self._total_epochs,
                'metrics': self._metrics.copy(),
                'error': self._error,
                'start_time': self._start_time.isoformat() if self._start_time else None,
                'end_time': self._end_time.isoformat() if self._end_time else None
            }

    def set_status(self, status: str):
        """设置状态"""
        if status not in self.VALID_STATUSES:
            raise ValueError(f"Invalid status: {status}")

        with self._lock:
            self._status = status
            if status == 'running' and self._start_time is None:
                self._start_time = datetime.now()
            elif status in ['completed', 'failed', 'stopped']:
                self._end_time = datetime.now()

            self._persist()

    def set_progress(self, progress: int):
        """设置进度(0-100)"""
        with self._lock:
            self._progress = max(0, min(100, progress))
            self._persist()

    def set_epoch(self, current: int, total: int):
        """设置当前epoch"""
        with self._lock:
            self._current_epoch = current
            self._total_epochs = total
            progress = int(current / total * 100) if total > 0 else 0
            self._progress = progress
            self._persist()

    def update_metrics(self, metrics: Dict[str, float], epoch: int = None):
        """更新指标"""
        with self._lock:
            self._metrics.update(metrics)

            if epoch is not None:
                self._metrics_history.append({
                    'epoch': epoch,
                    'timestamp': datetime.now().isoformat(),
                    **metrics
                })

            self._persist()

    def set_error(self, error: str):
        """设置错误信息"""
        with self._lock:
            self._error = error
            self._status = 'failed'
            self._end_time = datetime.now()
            self._persist()

    def get_metrics_history(self) -> List[Dict[str, Any]]:
        """获取指标历史"""
        with self._lock:
            return self._metrics_history.copy()

    def _persist(self):
        """持久化状态到文件"""
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)

            state_data = {
                'task_id': self.task_id,
                'status': self._status,
                'progress': self._progress,
                'current_epoch': self._current_epoch,
                'total_epochs': self._total_epochs,
                'metrics': self._metrics,
                'error': self._error,
                'start_time': self._start_time.isoformat() if self._start_time else None,
                'end_time': self._end_time.isoformat() if self._end_time else None,
                'metrics_history': self._metrics_history
            }

            with open(self._state_file, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, ensure_ascii=False, indent=2)
        except Exception:
            # 持久化失败不影响主流程
            pass

    @classmethod
    def load(cls, state_file: Path) -> 'StateManager':
        """从文件加载状态"""
        with open(state_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        manager = cls(data['task_id'], state_file.parent)
        manager._status = data['status']
        manager._progress = data['progress']
        manager._current_epoch = data['current_epoch']
        manager._total_epochs = data['total_epochs']
        manager._metrics = data['metrics']
        manager._error = data['error']
        manager._metrics_history = data.get('metrics_history', [])

        if data['start_time']:
            manager._start_time = datetime.fromisoformat(data['start_time'])
        if data['end_time']:
            manager._end_time = datetime.fromisoformat(data['end_time'])

        return manager
