# -*- coding: utf-8 -*-
"""
日志工具模块
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler


def setup_logger(
    name: str,
    log_file: Path = None,
    level=logging.INFO,
    max_bytes=10 * 1024 * 1024,  # 10MB
    backup_count=5
) -> logging.Logger:
    """
    设置日志记录器

    Args:
        name: 记录器名称
        log_file: 日志文件路径
        level: 日志级别
        max_bytes: 单个日志文件最大大小
        backup_count: 保留的日志文件数量

    Returns:
        配置好的Logger实例
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 清除现有处理器
    logger.handlers.clear()

    # 防止传播到父记录器
    logger.propagate = False

    # 日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件处理器
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    获取已配置的记录器

    Args:
        name: 记录器名称

    Returns:
        Logger实例
    """
    return logging.getLogger(name)
