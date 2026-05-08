# -*- coding: utf-8 -*-
"""
Ultralytics YOLO引擎实现
"""

from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
import logging

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

from .base_engine import BaseEngine, TrainResult, PredictResult
from ..exceptions.engine_errors import ModelLoadError, TrainingError


logger = logging.getLogger(__name__)


class UltralyticsEngine(BaseEngine):
    """基于Ultralytics YOLO的引擎实现"""

    SUPPORTED_TASKS = ['detect', 'segment', 'classify', 'obb']

    def __init__(self, config: Dict[str, Any]):
        """
        初始化Ultralytics引擎

        Args:
            config: 配置字典
        """
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("Ultralytics is not installed. Install with: pip install ultralytics")

        super().__init__(config)
        self.model = None
        self.device = self._get_device()

        # 存储训练结果
        self._training_results = None

    def train(self) -> TrainResult:
        """
        执行YOLO模型训练

        Returns:
            TrainResult: 训练结果
        """
        try:
            self._update_status('running', progress=0)
            logger.info(f"Starting training: {self.task_id}")

            # 初始化模型
            model_arch = self.config['model_architecture']
            logger.info(f"Loading model: {model_arch}")
            self.model = YOLO(model_arch)

            # 构建训练参数
            train_args = self._build_train_args()

            logger.info(f"Training arguments: {train_args}")

            # 设置回调
            self._setup_training_callbacks()

            # 开始训练
            results = self.model.train(**train_args)
            self._training_results = results

            # 训练完成，获取模型路径
            project_dir = self.output_dir / 'train'
            best_model = project_dir / 'weights' / 'best.pt'
            last_model = project_dir / 'weights' / 'last.pt'

            # 提取指标
            metrics = self._extract_final_metrics()

            self._update_status('completed', progress=100, metrics=metrics)
            logger.info("Training completed successfully")

            return TrainResult(
                success=True,
                best_model_path=best_model if best_model.exists() else None,
                last_model_path=last_model if last_model.exists() else None,
                metrics=metrics
            )

        except Exception as e:
            error_msg = f"Training failed: {str(e)}"
            logger.exception(error_msg)
            self._update_status('failed', error=error_msg)
            return TrainResult(
                success=False,
                best_model_path=None,
                last_model_path=None,
                metrics={},
                error=error_msg
            )

    def predict(self) -> PredictResult:
        """
        执行YOLO模型推理

        Returns:
            PredictResult: 推理结果
        """
        try:
            self._update_status('running', progress=0)
            logger.info(f"Starting prediction: {self.task_id}")

            # 加载模型
            model_path = self.config.get('model_path', self.config.get('model_architecture'))
            logger.info(f"Loading model: {model_path}")
            self.model = YOLO(model_path)

            # 构建推理参数
            predict_config = self.config.get('predict', {})
            predict_args = self._build_predict_args(predict_config)

            logger.info(f"Prediction arguments: {predict_args}")

            # 执行推理
            results = self.model.predict(**predict_args)

            # 处理结果
            result_files = self._process_prediction_results(results, predict_config)

            self._update_status('completed', progress=100)
            logger.info("Prediction completed successfully")

            return PredictResult(
                success=True,
                result_files=result_files
            )

        except Exception as e:
            error_msg = f"Prediction failed: {str(e)}"
            logger.exception(error_msg)
            self._update_status('failed', error=error_msg)
            return PredictResult(
                success=False,
                result_files=[],
                error=error_msg
            )

    def resume(self) -> TrainResult:
        """
        继续训练

        Returns:
            TrainResult: 训练结果
        """
        try:
            self._update_status('running', progress=0)
            logger.info(f"Resuming training: {self.task_id}")

            # 获取resume_from路径
            resume_from = self.config.get('resume_from')

            # 如果没有指定resume_from，尝试从输出目录查找last.pt
            if not resume_from:
                last_model = self.output_dir / 'train' / 'weights' / 'last.pt'
                if last_model.exists():
                    resume_from = str(last_model)
                    logger.info(f"Found last checkpoint: {resume_from}")
                else:
                    # 如果没有找到检查点，尝试从头开始训练
                    logger.warning("No checkpoint found, starting new training instead")
                    return self.train()

            logger.info(f"Resuming from checkpoint: {resume_from}")
            self.model = YOLO(resume_from)

            train_args = self._build_train_args(resume=True)

            self._setup_training_callbacks()
            results = self.model.train(**train_args)
            self._training_results = results

            project_dir = self.output_dir / 'train'
            best_model = project_dir / 'weights' / 'best.pt'
            last_model = project_dir / 'weights' / 'last.pt'

            metrics = self._extract_final_metrics()

            self._update_status('completed', progress=100, metrics=metrics)
            logger.info("Resumed training completed successfully")

            return TrainResult(
                success=True,
                best_model_path=best_model if best_model.exists() else None,
                last_model_path=last_model if last_model.exists() else None,
                metrics=metrics
            )

        except Exception as e:
            error_msg = f"Resume training failed: {str(e)}"
            logger.exception(error_msg)
            self._update_status('failed', error=error_msg)
            return TrainResult(
                success=False,
                best_model_path=None,
                last_model_path=None,
                metrics={},
                error=error_msg
            )

    def _build_train_args(self, resume: bool = False) -> Dict[str, Any]:
        """构建训练参数"""
        hyperparams = self.config.get('hyperparameters', {})
        hardware = self.config.get('hardware', {})

        args = {
            'data': self.config['data_config'],
            'epochs': hyperparams.get('epochs', 100),
            'batch': hyperparams.get('batch', 16),
            'imgsz': hyperparams.get('imgsz', 640),
            'device': self.device,
            'workers': hardware.get('workers', 8),
            'project': str(self.output_dir),
            'name': 'train',
            'exist_ok': True,
            'verbose': True,
            'amp': hyperparams.get('amp', True)
        }

        # 优化器配置
        if 'optimizer' in hyperparams:
            args['optimizer'] = hyperparams['optimizer']
        if 'lr0' in hyperparams:
            args['lr0'] = hyperparams['lr0']
        if 'weight_decay' in hyperparams:
            args['weight_decay'] = hyperparams['weight_decay']

        # 恢复训练
        if resume:
            args['resume'] = True

        return args

    def _build_predict_args(self, predict_config: Dict[str, Any]) -> Dict[str, Any]:
        """构建推理参数"""
        hyperparams = self.config.get('hyperparameters', {})

        args = {
            'source': predict_config.get('source'),
            'conf': predict_config.get('conf_thres', 0.25),
            'iou': predict_config.get('iou_thres', 0.45),
            'imgsz': hyperparams.get('imgsz', 640),
            'device': self.device,
            'save': True,
            'project': str(self.output_dir / 'results'),
            'name': 'predict',
            'exist_ok': True
        }

        # 可选参数
        if predict_config.get('augment'):
            args['augment'] = True
        if predict_config.get('half'):
            args['half'] = True

        return args

    def _setup_training_callbacks(self):
        """设置训练回调以更新状态"""
        # Ultralytics使用callbacks系统
        # 这里可以注册自定义回调来更新进度
        # 暂时留空，后续可以实现更详细的进度跟踪
        pass

    def _extract_final_metrics(self) -> Dict[str, float]:
        """提取最终训练指标"""
        metrics = {}

        # 尝试从训练结果中获取指标
        if self._training_results is not None:
            try:
                # Ultralytics results对象包含指标
                # 这里简化处理，实际需要根据具体结果结构提取
                if hasattr(self._training_results, 'results_dict'):
                    metrics = self._training_results.results_dict
            except Exception:
                pass

        # 如果没有获取到指标，返回默认值
        if not metrics:
            metrics = {
                'mAP50': 0.0,
                'mAP50-95': 0.0
            }

        return metrics

    def _process_prediction_results(self, results, predict_config: Dict[str, Any]) -> List[Path]:
        """
        处理推理结果

        Args:
            results: Ultralytics推理结果
            predict_config: 推理配置

        Returns:
            结果文件路径列表
        """
        result_files = []

        # Ultralytics默认保存结果到runs目录
        # 我们需要找到并返回这些文件
        predict_dir = self.output_dir / 'results' / 'predict'

        if predict_dir.exists():
            # 根据任务类型添加结果文件
            if self.task_type == 'detect':
                # 目标检测：labels文件
                labels_files = list(predict_dir.glob('*.txt'))
                result_files.extend(labels_files)

                # 图片结果
                image_files = list(predict_dir.glob('*.jpg'))
                image_files.extend(predict_dir.glob('*.png'))
                result_files.extend(image_files)

            elif self.task_type == 'segment':
                # 分割：mask文件
                mask_files = list(predict_dir.glob('*.png'))
                result_files.extend(mask_files)

            # 添加预测目录本身
            result_files.append(predict_dir)

        return result_files
