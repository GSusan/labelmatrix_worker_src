#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LabelMatrix Worker - 主入口程序

使用方式:
    python labelmatrix_worker.py --config config.yaml [--port PORT]
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from labelmatrix.config.parser import ConfigParser
from labelmatrix.engines import create_engine
from labelmatrix.server import WorkerServer
from labelmatrix.utils import setup_logger, GPUMemoryChecker, MetadataBuilder
from labelmatrix.exceptions import (
    LabelMatrixException,
    ConfigFileError,
    ConfigValidationError,
    InsufficientGPUMemoryError
)

# 全局变量用于存储结果
_train_result = None
_predict_result = None


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='LabelMatrix Worker - 深度学习训练与推理引擎',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python labelmatrix_worker.py --config train.yaml
  python labelmatrix_worker.py --config predict.yaml --port 8888
  python labelmatrix_worker.py --config train.yaml --resume  # 继续训练
        """
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='YAML配置文件路径'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=0,
        help='HTTP服务器端口 (0表示自动分配)'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='强制使用resume模式继续训练（覆盖配置文件中的mode设置）'
    )
    parser.add_argument(
        '--resume-from',
        type=str,
        default=None,
        help='指定检查点文件路径用于继续训练（覆盖配置文件中的resume_from设置）'
    )
    parser.add_argument(
        '--load-from',
        type=str,
        default=None,
        help='指定预训练模型路径用于训练（覆盖配置文件中的model设置或预训练权重）'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='启用详细日志输出'
    )
    return parser.parse_args()


def check_gpu_resources(config, logger):
    """
    检查GPU资源

    Args:
        config: 配置字典
        logger: 日志记录器
    """
    hardware = config.get('hardware', {})
    device = hardware.get('device', 'cpu')

    if device.startswith('cuda'):
        checker = GPUMemoryChecker()

        # 检查PyTorch是否可用
        if not checker.is_torch_available():
            logger.error("PyTorch is not available. Cannot use CUDA.")
            print("ERROR: PyTorch with CUDA support is required for GPU training")
            sys.exit(1)

        # 获取GPU ID
        gpu_id = int(device.split(':')[1]) if ':' in device else 0

        # 检查GPU可用性
        if not checker.is_gpu_available(gpu_id):
            logger.error(f"GPU {gpu_id} is not available")
            print(f"ERROR: GPU {gpu_id} is not available")
            sys.exit(1)

        # 获取GPU信息
        gpu_name = checker.get_device_name(gpu_id)
        total_memory = checker.get_total_memory(gpu_id)
        available_memory = checker.get_available_memory(gpu_id)

        logger.info(f"GPU {gpu_id}: {gpu_name}")
        logger.info(f"  Total memory: {total_memory} MB")
        logger.info(f"  Available memory: {available_memory} MB")

        # 检查显存要求
        min_memory = hardware.get('min_memory_mb')
        if min_memory and available_memory < min_memory:
            error_msg = (
                f"Insufficient GPU memory. "
                f"Required: {min_memory}MB, Available: {available_memory}MB"
            )
            logger.error(error_msg)
            print(f"ERROR: {error_msg}")
            sys.exit(1)


def save_metadata(config, result, output_dir: Path, logger):
    """保存训练元数据"""
    try:
        builder = MetadataBuilder()
        metadata = builder.build(config, result)
        builder.save(metadata, output_dir)
        logger.info(f"Metadata saved to {output_dir / 'metadata.json'}")
    except Exception as e:
        logger.warning(f"Failed to save metadata: {e}")


def main():
    """主函数"""
    args = parse_args()

    # 解析配置
    try:
        parser = ConfigParser(args.config)
        config = parser.parse()
    except ConfigFileError as e:
        print(f"CONFIG_ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except ConfigValidationError as e:
        print(f"VALIDATION_ERROR: {e}", file=sys.stderr)
        if e.errors:
            for error in e.errors:
                print(f"  - {error}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to parse config: {e}", file=sys.stderr)
        sys.exit(1)

    # 获取任务信息
    task_id = config['task_id']
    task_type = config['task_type']
    mode = config.get('mode', 'train')

    # 如果命令行指定了--resume，覆盖配置文件中的mode
    if args.resume:
        mode = 'resume'
        logger_msg = "Mode forced to 'resume' by command line argument"
        print(f"INFO: {logger_msg}")

    # 如果命令行指定了--resume-from，覆盖配置文件中的resume_from
    if args.resume_from:
        config['resume_from'] = args.resume_from
        logger_msg = f"Resume checkpoint overridden: {args.resume_from}"
        print(f"INFO: {logger_msg}")

    # 如果命令行指定了--load-from，覆盖配置文件中的设置
    if args.load_from:
        config['load_from'] = args.load_from
        logger_msg = f"Pretrained model overridden: {args.load_from}"
        print(f"INFO: {logger_msg}")

    output_dir = Path(config['output_dir']) / task_id

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    # 设置日志
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logger(
        name='labelmatrix',
        log_file=output_dir / 'logs' / 'worker.log',
        level=log_level
    )

    logger.info("=" * 60)
    logger.info("LabelMatrix Worker Starting")
    logger.info("=" * 60)
    logger.info(f"Task ID: {task_id}")
    logger.info(f"Task Type: {task_type}")
    logger.info(f"Mode: {mode}")
    logger.info(f"Output Directory: {output_dir}")

    # 保存配置副本，添加类别标签信息
    import shutil
    import yaml

    # 读取原始配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)

    # 如果配置中指定了data_config，从中解析类别标签
    data_config_path = config.get('data_config')
    if data_config_path:
        try:
            with open(data_config_path, 'r', encoding='utf-8') as f:
                data_yaml = yaml.safe_load(f)
                # 提取类别信息
                if 'names' in data_yaml:
                    config_data['class_names'] = data_yaml['names']
                    logger.info(f"Added class_names from {data_config_path}")
        except Exception as e:
            logger.warning(f"Failed to read class names from {data_config_path}: {e}")

    # 保存增强后的配置文件
    config_copy_path = output_dir / 'config.yaml'
    with open(config_copy_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    logger.info(f"Config saved to {config_copy_path}")

    # 检查GPU资源
    check_gpu_resources(config, logger)

    # 创建引擎
    try:
        engine = create_engine(config)
        logger.info(f"Engine created: {engine.__class__.__name__}")
    except Exception as e:
        logger.error(f"Failed to create engine: {e}")
        print(f"ENGINE_ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    # 启动HTTP服务器
    server = WorkerServer(engine, port=args.port)
    actual_port = server.start()

    # 打印端口号供前端捕获（重要！）
    print(f"PORT:{actual_port}", file=sys.stdout)
    sys.stdout.flush()

    logger.info(f"HTTP server started on port {actual_port}")
    logger.info("Waiting for status requests...")

    # 执行任务
    global _train_result, _predict_result

    try:
        if mode == 'train':
            logger.info("Starting training...")
            _train_result = engine.train()

            if _train_result.success:
                logger.info("Training completed successfully!")
                logger.info(f"Best model: {_train_result.best_model_path}")
                logger.info(f"Last model: {_train_result.last_model_path}")
                logger.info(f"Metrics: {_train_result.metrics}")
            else:
                logger.error(f"Training failed: {_train_result.error}")

        elif mode == 'predict':
            logger.info("Starting prediction...")
            _predict_result = engine.predict()

            if _predict_result.success:
                logger.info("Prediction completed successfully!")
                logger.info(f"Result files: {_predict_result.result_files}")
            else:
                logger.error(f"Prediction failed: {_predict_result.error}")

        elif mode == 'resume':
            logger.info("Resuming training...")
            _train_result = engine.resume()

            if _train_result.success:
                logger.info("Resumed training completed successfully!")
                logger.info(f"Best model: {_train_result.best_model_path}")
                logger.info(f"Last model: {_train_result.last_model_path}")
                logger.info(f"Metrics: {_train_result.metrics}")
            else:
                logger.error(f"Resume training failed: {_train_result.error}")

        else:
            logger.error(f"Unknown mode: {mode}")
            print(f"ERROR: Unknown mode: {mode}", file=sys.stderr)
            sys.exit(1)

        # 保存元数据（仅训练任务）
        if mode in ['train', 'resume'] and _train_result and _train_result.success:
            save_metadata(config, _train_result, output_dir, logger)

    except KeyboardInterrupt:
        logger.info("Task stopped by user (Ctrl+C)")
        engine.stop()
        print("\nINFO: Task stopped by user")

    except Exception as e:
        logger.exception("Task failed with exception")
        print(f"RUNTIME_ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    finally:
        # 停止HTTP服务器
        server.stop()
        logger.info("HTTP server stopped")

    logger.info("=" * 60)
    logger.info("LabelMatrix Worker Finished")
    logger.info("=" * 60)

    # 返回退出码
    if mode in ['train', 'resume']:
        if _train_result and _train_result.success:
            return 0
        else:
            return 1
    else:
        if _predict_result and _predict_result.success:
            return 0
        else:
            return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
