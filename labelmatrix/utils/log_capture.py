# -*- coding: utf-8 -*-
"""
日志捕获工具

用于捕获训练过程中的日志输出，支持内存缓冲和文件存储。
"""

import logging
import threading
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any


class LogCaptureHandler(logging.Handler):
    """
    日志捕获处理器

    捕获日志记录到内存缓冲区，并可选地写入文件。
    支持线程安全的日志追加和按时间戳查询。
    """

    def __init__(self,
                 max_buffer_size: int = 1000,
                 log_file_path: Optional[Path] = None):
        """
        初始化日志捕获处理器

        Args:
            max_buffer_size: 内存缓冲区最大条目数
            log_file_path: 日志文件路径，如果为None则不写文件
        """
        super().__init__()
        self.max_buffer_size = max_buffer_size
        self.log_file_path = log_file_path
        self._buffer: deque = deque(maxlen=max_buffer_size)
        self._lock = threading.Lock()
        self._file_lock = threading.Lock()

        # 设置日志格式
        self.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))

    def emit(self, record: logging.LogRecord) -> None:
        """
        处理日志记录

        Args:
            record: 日志记录对象
        """
        try:
            # 构建日志条目
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'level': record.levelname,
                'logger': record.name,
                'message': self.format(record)
            }

            # 线程安全地添加到缓冲区
            with self._lock:
                self._buffer.append(log_entry)

            # 可选地写入文件
            if self.log_file_path:
                self._write_to_file(log_entry)

        except Exception:
            # 防止日志处理器本身抛出异常
            self.handleError(record)

    def get_logs(self, since: Optional[str] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        获取日志条目

        Args:
            since: ISO格式的时间戳，只返回此时间之后的日志
            limit: 最大返回条目数

        Returns:
            日志条目列表
        """
        with self._lock:
            logs = list(self._buffer)

        # 按时间戳过滤
        if since:
            try:
                since_dt = datetime.fromisoformat(since)
                logs = [log for log in logs if datetime.fromisoformat(log['timestamp']) > since_dt]
            except ValueError:
                pass  # 无效的时间戳格式，返回所有日志

        # 应用限制
        if limit and limit > 0:
            logs = logs[-limit:]

        return logs

    def get_all_logs(self) -> List[Dict[str, Any]]:
        """
        获取所有日志条目

        Returns:
            所有日志条目列表
        """
        with self._lock:
            return list(self._buffer)

    def clear(self) -> None:
        """清空日志缓冲区"""
        with self._lock:
            self._buffer.clear()

    def get_buffer_size(self) -> int:
        """
        获取当前缓冲区大小

        Returns:
            当前缓冲区中的日志条目数
        """
        return len(self._buffer)

    def _write_to_file(self, log_entry: Dict[str, Any]) -> None:
        """
        将日志条目写入文件

        Args:
            log_entry: 日志条目
        """
        if not self.log_file_path:
            return

        with self._file_lock:
            try:
                # 确保日志文件目录存在
                self.log_file_path.parent.mkdir(parents=True, exist_ok=True)

                # 追加写入日志文件（每行一个JSON对象）
                with open(self.log_file_path, 'a', encoding='utf-8') as f:
                    import json
                    f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')

            except (IOError, OSError) as e:
                # 文件写入失败不应影响日志捕获
                # 这里使用print而不是logging避免递归
                print(f"Warning: Failed to write log to file: {e}")

    def read_log_file(self, since: Optional[str] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        从日志文件读取日志条目

        Args:
            since: ISO格式的时间戳
            limit: 最大返回条目数

        Returns:
            日志条目列表
        """
        if not self.log_file_path or not self.log_file_path.exists():
            return []

        logs = []
        try:
            import json
            with open(self.log_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        log_entry = json.loads(line.strip())
                        logs.append(log_entry)
                    except json.JSONDecodeError:
                        continue
        except (IOError, OSError):
            return []

        # 按时间戳过滤
        if since:
            try:
                since_dt = datetime.fromisoformat(since)
                logs = [log for log in logs if datetime.fromisoformat(log['timestamp']) > since_dt]
            except ValueError:
                pass

        # 应用限制
        if limit and limit > 0:
            logs = logs[-limit:]

        return logs


def setup_log_capture(log_dir: Path,
                      max_buffer_size: int = 1000,
                      logger_names: Optional[List[str]] = None) -> LogCaptureHandler:
    """
    设置日志捕获

    Args:
        log_dir: 日志目录
        max_buffer_size: 内存缓冲区最大大小
        logger_names: 要捕获的logger名称列表，如果为None则捕获所有

    Returns:
        LogCaptureHandler实例
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    handler = LogCaptureHandler(
        max_buffer_size=max_buffer_size,
        log_file_path=log_file_path
    )

    # 添加到指定的logger或根logger
    if logger_names:
        for name in logger_names:
            logger = logging.getLogger(name)
            logger.addHandler(handler)
    else:
        # 添加到ultralytics相关的logger
        for name in ['ultralytics', 'labelmatrix']:
            logger = logging.getLogger(name)
            logger.addHandler(handler)

    return handler
