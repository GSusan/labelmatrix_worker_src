# -*- coding: utf-8 -*-
"""
LabelMatrix Trainer - 集成Ultralytics YOLO的训练引擎

通过Ultralytics callbacks系统实现实时状态监控和日志捕获。
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .base_engine import BaseEngine, TrainResult, PredictResult
from ..exceptions.engine_errors import TrainingError, ModelLoadError
from ..utils.log_capture import LogCaptureHandler


logger = logging.getLogger(__name__)


class LabelMatrixTrainer(BaseEngine):
    """
    基于Ultralytics YOLO的训练引擎

    通过callbacks系统实现实时状态监控、日志捕获和资源跟踪。

    支持的任务类型: detect, segment, classify, obb
    """

    SUPPORTED_TASKS = ['detect', 'segment', 'classify', 'obb']

    def __init__(self, config: Dict[str, Any]):
        """
        初始化LabelMatrixTrainer

        Args:
            config: 配置字典
        """
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("Ultralytics is not installed. Install with: pip install ultralytics")

        super().__init__(config)

        # 扩展状态属性
        self._current_epoch: int = 0
        self._total_epochs: int = 0
        self._resources: Dict[str, Any] = {}
        self._learning_rate: float = 0.0

        # 日志捕获
        self._log_handler: Optional[LogCaptureHandler] = None
        self._log_file_path = self.output_dir / 'logs' / 'training.log'

        # YOLO模型实例
        self.model: Optional[Any] = None

        # 训练结果存储
        self._training_results: Optional[Any] = None

    def train(self) -> TrainResult:
        """
        执行YOLO模型训练

        Returns:
            TrainResult: 训练结果
        """
        try:
            # 数据集验证与转换
            from labelmatrix.dataset_validator import DatasetValidator
            from labelmatrix.exceptions.dataset_errors import DatasetConversionError

            data_config = self.config['data_config']
            validator = DatasetValidator(data_config)

            if not validator.validate():
                logger.info(f"Dataset format validation failed, starting conversion...")
                try:
                    converter = validator.get_converter()
                    new_data_config = converter.convert()

                    # 动态更新配置，使用新数据集路径
                    self.config['data_config'] = new_data_config
                    logger.info(f"Dataset converted, new config: {new_data_config}")
                except DatasetConversionError as e:
                    error_msg = f"Dataset conversion failed: {str(e)}"
                    logger.error(error_msg)
                    self._update_status('failed', error=error_msg)

                    return TrainResult(
                        success=False,
                        best_model_path=None,
                        last_model_path=None,
                        metrics={},
                        error=error_msg
                    )

            # 设置开始时间
            self._start_time = datetime.now()
            self._update_status('running', progress=0)

            logger.info(f"Starting training: {self.task_id}")

            # 初始化模型
            # 优先使用load_from（预训练模型），否则使用model_architecture
            model_path = self.config.get('load_from', self.config['model_architecture'])
            logger.info(f"Loading model: {model_path}")
            self.model = YOLO(model_path)

            # 设置日志捕获
            self._setup_log_capture()

            # 设置训练回调
            self._setup_training_callbacks()

            # 构建训练参数
            train_args = self._build_train_args()
            logger.info(f"Training arguments: {train_args}")

            # 初始化状态文件
            self._init_state_file()

            # 开始训练
            results = self.model.train(**train_args)
            self._training_results = results

            # 训练完成
            self._end_time = datetime.now()

            # 获取模型路径
            project_dir = self.output_dir / 'train'
            best_model = project_dir / 'weights' / 'best.pt'
            last_model = project_dir / 'weights' / 'last.pt'

            # 提取最终指标
            metrics = self._extract_final_metrics()

            # 自动test评估（三层安全检查）
            test_metrics = None
            test_evaluated = False
            test_skipped_reason = None

            should_run, skip_reason = self._should_run_test_evaluation()
            if should_run and best_model.exists():
                logger.info("Test set available, running automatic test evaluation...")
                test_metrics = self._run_test_evaluation(best_model)
                test_evaluated = test_metrics is not None
                if not test_evaluated:
                    test_skipped_reason = "test evaluation execution failed"
                    logger.warning(f"Test evaluation failed, skipping: {test_skipped_reason}")
            else:
                test_evaluated = False
                test_skipped_reason = skip_reason
                if skip_reason:
                    logger.info(f"Test evaluation skipped: {skip_reason}")

            self._update_status('completed', progress=100, metrics=metrics)
            self._persist_state()

            logger.info("Training completed successfully")
            if test_evaluated:
                logger.info(f"Test evaluation completed successfully: {test_metrics}")
            elif test_skipped_reason:
                logger.info(f"Test evaluation skipped: {test_skipped_reason}")

            return TrainResult(
                success=True,
                best_model_path=best_model if best_model.exists() else None,
                last_model_path=last_model if last_model.exists() else None,
                metrics=metrics,
                test_metrics=test_metrics,
                test_evaluated=test_evaluated,
                test_skipped_reason=test_skipped_reason
            )

        except Exception as e:
            self._end_time = datetime.now()
            error_msg = f"Training failed: {str(e)}"
            logger.exception(error_msg)
            self._update_status('failed', error=error_msg)
            self._persist_state()

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
            self._start_time = datetime.now()
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
                    logger.warning("No checkpoint found, starting new training instead")
                    return self.train()

            logger.info(f"Resuming from checkpoint: {resume_from}")
            self.model = YOLO(resume_from)

            # 设置日志捕获和回调
            self._setup_log_capture()
            self._setup_training_callbacks()

            train_args = self._build_train_args(resume=True)

            # 初始化状态文件
            self._init_state_file()

            results = self.model.train(**train_args)
            self._training_results = results

            self._end_time = datetime.now()

            project_dir = self.output_dir / 'train'
            best_model = project_dir / 'weights' / 'best.pt'
            last_model = project_dir / 'weights' / 'last.pt'

            metrics = self._extract_final_metrics()

            # 自动test评估（三层安全检查）
            test_metrics = None
            test_evaluated = False
            test_skipped_reason = None

            should_run, skip_reason = self._should_run_test_evaluation()
            if should_run and best_model.exists():
                logger.info("Test set available, running automatic test evaluation...")
                test_metrics = self._run_test_evaluation(best_model)
                test_evaluated = test_metrics is not None
                if not test_evaluated:
                    test_skipped_reason = "test evaluation execution failed"
                    logger.warning(f"Test evaluation failed, skipping: {test_skipped_reason}")
            else:
                test_evaluated = False
                test_skipped_reason = skip_reason
                if skip_reason:
                    logger.info(f"Test evaluation skipped: {skip_reason}")

            self._update_status('completed', progress=100, metrics=metrics)
            self._persist_state()

            logger.info("Resumed training completed successfully")
            if test_evaluated:
                logger.info(f"Test evaluation completed successfully: {test_metrics}")
            elif test_skipped_reason:
                logger.info(f"Test evaluation skipped: {test_skipped_reason}")

            return TrainResult(
                success=True,
                best_model_path=best_model if best_model.exists() else None,
                last_model_path=last_model if last_model.exists() else None,
                metrics=metrics,
                test_metrics=test_metrics,
                test_evaluated=test_evaluated,
                test_skipped_reason=test_skipped_reason
            )

        except Exception as e:
            self._end_time = datetime.now()
            error_msg = f"Resume training failed: {str(e)}"
            logger.exception(error_msg)
            self._update_status('failed', error=error_msg)
            self._persist_state()

            return TrainResult(
                success=False,
                best_model_path=None,
                last_model_path=None,
                metrics={},
                error=error_msg
            )

    def get_status(self) -> Dict[str, Any]:
        """
        获取当前训练状态（包含epoch、resources等扩展字段）

        Returns:
            完整的状态字典，包含status、progress、metrics、epoch、resources等
        """
        status = super().get_status()
        # 添加训练特有的字段
        # 注意：Ultralytics使用0-based epoch索引，但用户习惯1-based显示
        # 因此current_epoch需要+1后返回给前端
        display_epoch = self._current_epoch + 1 if self._current_epoch >= 0 else 0
        status.update({
            'current_epoch': display_epoch,
            'total_epochs': self._total_epochs,
            'resources': self._resources.copy() if self._resources else {},
            'start_time': self._start_time.isoformat() if self._start_time else None,
            'end_time': self._end_time.isoformat() if self._end_time else None,
            'lr': self._learning_rate
        })
        # 同时将lr放入resources中，方便前端统一访问
        if self._learning_rate > 0:
            status['resources']['lr'] = self._learning_rate
        return status

    def get_logs(self, since: Optional[str] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        获取日志条目

        Args:
            since: ISO格式的时间戳
            limit: 最大返回条目数

        Returns:
            日志条目列表
        """
        if self._log_handler:
            return self._log_handler.get_logs(since=since, limit=limit)
        return []

    # ==================== 私有方法 ====================

    def _setup_log_capture(self) -> None:
        """设置日志捕获"""
        self._log_handler = LogCaptureHandler(
            max_buffer_size=1000,
            log_file_path=self._log_file_path
        )

        # 添加到ultralytics和labelmatrix的logger
        ultralytics_logger = logging.getLogger('ultralytics')
        ultralytics_logger.addHandler(self._log_handler)
        ultralytics_logger.setLevel(logging.INFO)

        labelmatrix_logger = logging.getLogger('labelmatrix')
        labelmatrix_logger.addHandler(self._log_handler)
        labelmatrix_logger.setLevel(logging.INFO)

        logger.info("Log capture handler installed")

    def _setup_training_callbacks(self) -> None:
        """设置Ultralytics训练回调"""
        if not self.model:
            return

        # 注册各种回调
        self.model.add_callback("on_train_start", self._on_train_start)
        self.model.add_callback("on_train_epoch_end", self._on_train_epoch_end)
        self.model.add_callback("on_train_batch_end", self._on_train_batch_end)
        self.model.add_callback("on_train_end", self._on_train_end)
        self.model.add_callback("on_fit_epoch_end", self._on_fit_epoch_end)
        self.model.add_callback("on_pretrain_routine_end", self._on_pretrain_routine_end)

        logger.info("Training callbacks registered")

    def _init_state_file(self) -> None:
        """初始化状态文件"""
        state_data = self._build_full_state_dict()
        self._write_state_atomically(state_data)

    def _persist_state(self) -> None:
        """持久化当前状态到文件"""
        state_data = self._build_full_state_dict()
        self._write_state_atomically(state_data)

    def _build_full_state_dict(self) -> Dict[str, Any]:
        """构建完整的状态字典"""
        state = self._build_state_dict()
        # 转换为1-based epoch显示（用户习惯）
        display_epoch = self._current_epoch + 1 if self._current_epoch >= 0 else 0
        state.update({
            'current_epoch': display_epoch,
            'total_epochs': self._total_epochs,
            'resources': self._resources,
            'lr': self._learning_rate,
            # 添加test评估相关状态
            'test_evaluated': False,
            'test_skipped_reason': None
        })
        return state

    # ==================== Ultralytics Callbacks ====================

    def _on_pretrain_routine_end(self, trainer) -> None:
        """预训练例程结束回调 - 在此获取总epoch数"""
        try:
            if hasattr(trainer, 'epochs'):
                self._total_epochs = trainer.epochs
            elif hasattr(trainer.args, 'epochs'):
                self._total_epochs = trainer.args.epochs

            logger.debug(f"Total epochs set to: {self._total_epochs}")
            self._persist_state()

        except Exception as e:
            logger.warning(f"Error in on_pretrain_routine_end: {e}")

    def _on_train_start(self, trainer) -> None:
        """训练开始回调"""
        try:
            self._current_epoch = getattr(trainer, 'epoch', 0)
            self._update_status('running', progress=0)

            logger.info(f"Training started, total epochs: {self._total_epochs}")
            self._persist_state()

        except Exception as e:
            logger.warning(f"Error in on_train_start: {e}")

    def _on_train_epoch_end(self, trainer) -> None:
        """每个epoch结束回调"""
        try:
            # 更新epoch
            self._current_epoch = getattr(trainer, 'epoch', self._current_epoch)

            # 计算进度
            if self._total_epochs > 0:
                progress = int((self._current_epoch + 1) / self._total_epochs * 100)
            else:
                progress = 0

            # 提取验证指标
            metrics = {}
            if hasattr(trainer, 'metrics') and trainer.metrics:
                # Ultralytics metrics对象
                if hasattr(trainer.metrics, 'keys'):
                    keys = trainer.metrics.keys() if callable(trainer.metrics.keys) else trainer.metrics.keys
                    for key in keys:
                        try:
                            value = trainer.metrics[key]
                            if hasattr(value, 'item'):  # tensor
                                value = value.item()
                            metrics[key] = float(value)
                        except (AttributeError, TypeError):
                            pass

            # 更新状态
            self._update_status('running', progress=progress, metrics=metrics)

            # 持久化
            self._persist_state()

            logger.debug(f"Epoch {self._current_epoch} completed, progress: {progress}%")

        except Exception as e:
            logger.warning(f"Error in on_train_epoch_end: {e}")

    def _on_train_batch_end(self, trainer) -> None:
        """每个batch结束回调"""
        try:
            # 捕获loss值
            if hasattr(trainer, 'loss') and trainer.loss is not None:
                if hasattr(trainer.loss, 'item'):  # tensor
                    loss_value = float(trainer.loss.item())
                else:
                    loss_value = float(trainer.loss)

                self._metrics['loss'] = loss_value

            # 捕获学习率
            if hasattr(trainer, 'optimizer') and trainer.optimizer:
                for param_group in trainer.optimizer.param_groups:
                    if 'lr' in param_group:
                        self._learning_rate = float(param_group['lr'])
                        break

        except Exception as e:
            logger.debug(f"Error in on_train_batch_end: {e}")

    def _on_fit_epoch_end(self, trainer) -> None:
        """拟合epoch结束回调 - 用于资源监控"""
        try:
            resources = {}

            # GPU内存使用
            if TORCH_AVAILABLE and torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_reserved() / 1024**3  # GB
                resources['gpu_memory_gb'] = round(gpu_memory, 2)

            # 训练速度
            if hasattr(trainer, 'epoch_time') and trainer.epoch_time:
                if hasattr(trainer, 'train_loader') and trainer.train_loader:
                    dataset_size = len(trainer.train_loader.dataset)
                    speed = dataset_size / trainer.epoch_time
                    resources['speed_img_sec'] = round(speed, 2)

            self._resources = resources

            logger.debug(f"Resources: {resources}")

        except Exception as e:
            logger.debug(f"Error in on_fit_epoch_end: {e}")

    def _on_train_end(self, trainer) -> None:
        """训练结束回调"""
        try:
            # 提取最终指标
            metrics = {}
            if hasattr(trainer, 'metrics') and trainer.metrics:
                if hasattr(trainer.metrics, 'keys'):
                    keys = trainer.metrics.keys() if callable(trainer.metrics.keys) else trainer.metrics.keys
                    for key in keys:
                        try:
                            value = trainer.metrics[key]
                            if hasattr(value, 'item'):
                                value = value.item()
                            metrics[key] = float(value)
                        except (AttributeError, TypeError):
                            pass

            self._metrics = metrics

            logger.info(f"Training ended, final metrics: {metrics}")
            self._persist_state()

        except Exception as e:
            logger.warning(f"Error in on_train_end: {e}")

    # ==================== 参数构建 ====================

    def _build_train_args(self, resume: bool = False) -> Dict[str, Any]:
        """构建训练参数"""
        hyperparams = self.config.get('hyperparameters', {})
        hardware = self.config.get('hardware', {})

        args = {
            'data': self.config['data_config'],
            'epochs': hyperparams.get('epochs', 100),
            'batch': hyperparams.get('batch', 16),
            'imgsz': hyperparams.get('imgsz', 640),
            'device': self._get_device(),
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
            'device': self._get_device(),
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

    def _extract_final_metrics(self) -> Dict[str, float]:
        """提取最终训练指标"""
        metrics = {}

        # 尝试从训练结果中获取指标
        if self._training_results is not None:
            try:
                if hasattr(self._training_results, 'results_dict'):
                    metrics = self._training_results.results_dict
            except Exception:
                pass

        # 合合当前metrics
        if self._metrics:
            metrics.update(self._metrics)

        # 如果没有获取到指标，返回默认值
        if not metrics:
            metrics = {
                'mAP50': 0.0,
                'mAP50-95': 0.0
            }

        return metrics

    def _process_prediction_results(self, results, predict_config: Dict[str, Any]) -> List[Path]:
        """处理推理结果"""
        result_files = []

        # Ultralytics默认保存结果到runs目录
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

    # ==================== Test集自动评估 ====================

    def _should_run_test_evaluation(self) -> Tuple[bool, Optional[str]]:
        """
        三层安全检查：判断是否应该运行test集评估

        Returns:
            (should_run, skip_reason): 是否应该运行，如果不应运行则返回原因
        """
        try:
            # 第一层：配置层检查
            data_config_path = self.config.get('data_config', '')
            if not data_config_path:
                return False, "data_config not specified"

            # 读取数据集配置文件
            from pathlib import Path
            import yaml

            data_yaml_path = Path(data_config_path)
            if not data_yaml_path.exists():
                return False, f"data config file not found: {data_config_path}"

            with open(data_yaml_path, 'r', encoding='utf-8') as f:
                data_config = yaml.safe_load(f)

            # 检查是否有test字段
            if 'test' not in data_config or not data_config['test']:
                return False, "test field not configured in data.yaml"

            # 第二层：路径层检查
            test_path = data_config['test']
            dataset_root = Path(data_config.get('path', data_yaml_path.parent))
            full_test_path = dataset_root / test_path

            if not full_test_path.exists():
                return False, f"test path does not exist: {full_test_path}"

            # 第三层：内容层检查
            # 检查是否有图像文件
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
            image_files = list(full_test_path.glob('*.jpg')) + \
                          list(full_test_path.glob('*.jpeg')) + \
                          list(full_test_path.glob('*.png')) + \
                          list(full_test_path.glob('*.bmp')) + \
                          list(full_test_path.glob('*.webp'))

            if not image_files:
                return False, f"no image files found in test path: {full_test_path}"

            # 检查是否有对应的标注文件
            labels_path = full_test_path.parent / f"{full_test_path.name}_labels"
            labels_exist = False
            if labels_path.exists():
                label_files = list(labels_path.glob('*.txt'))
                if label_files:
                    labels_exist = True
            else:
                # 检查同目录下是否有txt标注文件
                label_files = list(full_test_path.glob('*.txt'))
                if label_files:
                    labels_exist = True

            if not labels_exist:
                return False, f"no label files found for test set"

            # 所有检查通过
            return True, None

        except Exception as e:
            logger.warning(f"Error during test evaluation check: {e}")
            return False, f"error checking test availability: {str(e)}"

    def _run_test_evaluation(self, model_path: Path) -> Optional[Dict[str, float]]:
        """
        执行test集评估

        Args:
            model_path: 训练好的模型路径

        Returns:
            test评估指标，如果评估失败则返回None
        """
        try:
            if not self.model:
                logger.warning("Model not loaded, cannot run test evaluation")
                return None

            logger.info(f"Starting test evaluation with model: {model_path}")

            # 执行test集验证
            test_results = self.model.val(
                split='test',
                data=self.config['data_config'],
                batch=self.config.get('hyperparameters', {}).get('batch', 16),
                project=str(self.output_dir),
                name='test_evaluation',
                save_json=False,
                plots=True,
                verbose=True
            )

            # 提取test指标
            test_metrics = {}
            if hasattr(test_results, 'box'):
                test_metrics['mAP50'] = float(test_results.box.map50)
                test_metrics['mAP50-95'] = float(test_results.box.map)
                if hasattr(test_results.box, 'mp'):
                    test_metrics['precision'] = float(test_results.box.mp)
                if hasattr(test_results.box, 'mr'):
                    test_metrics['recall'] = float(test_results.box.mr)

            logger.info(f"Test evaluation completed: {test_metrics}")
            return test_metrics

        except Exception as e:
            logger.error(f"Test evaluation failed: {e}")
            return None
