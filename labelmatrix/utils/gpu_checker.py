# -*- coding: utf-8 -*-
"""
GPU检查工具模块
"""

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class GPUMemoryChecker:
    """GPU显存检查工具"""

    @staticmethod
    def is_torch_available() -> bool:
        """
        检查PyTorch是否可用

        Returns:
            PyTorch是否可用
        """
        return TORCH_AVAILABLE

    @staticmethod
    def is_gpu_available(gpu_id: int = 0) -> bool:
        """
        检查GPU是否可用

        Args:
            gpu_id: GPU设备ID

        Returns:
            GPU是否可用
        """
        if not TORCH_AVAILABLE:
            return False

        if not torch.cuda.is_available():
            return False

        try:
            device_count = torch.cuda.device_count()
            if gpu_id >= device_count:
                return False

            # 尝试分配少量内存以验证GPU正常
            test_tensor = torch.zeros(1).cuda(gpu_id)
            del test_tensor
            torch.cuda.empty_cache()
            return True
        except Exception:
            return False

    @staticmethod
    def get_available_memory(gpu_id: int = 0) -> int:
        """
        获取GPU可用显存（MB）

        Args:
            gpu_id: GPU设备ID

        Returns:
            可用显存大小（MB）
        """
        if not TORCH_AVAILABLE:
            return 0

        if not torch.cuda.is_available():
            return 0

        try:
            device_count = torch.cuda.device_count()
            if gpu_id >= device_count:
                return 0

            props = torch.cuda.get_device_properties(gpu_id)
            total = props.total_memory

            # 获取已用显存
            reserved = torch.cuda.memory_reserved(gpu_id)

            available = total - reserved
            return int(available / 1024 / 1024)
        except Exception:
            return 0

    @staticmethod
    def get_total_memory(gpu_id: int = 0) -> int:
        """
        获取GPU总显存（MB）

        Args:
            gpu_id: GPU设备ID

        Returns:
            总显存大小（MB）
        """
        if not TORCH_AVAILABLE:
            return 0

        if not torch.cuda.is_available():
            return 0

        try:
            device_count = torch.cuda.device_count()
            if gpu_id >= device_count:
                return 0

            props = torch.cuda.get_device_properties(gpu_id)
            return int(props.total_memory / 1024 / 1024)
        except Exception:
            return 0

    @staticmethod
    def get_device_name(gpu_id: int = 0) -> str:
        """
        获取GPU设备名称

        Args:
            gpu_id: GPU设备ID

        Returns:
            GPU设备名称
        """
        if not TORCH_AVAILABLE:
            return "PyTorch not available"

        if not torch.cuda.is_available():
            return "CUDA not available"

        try:
            device_count = torch.cuda.device_count()
            if gpu_id >= device_count:
                return f"Invalid GPU ID: {gpu_id}"

            return torch.cuda.get_device_name(gpu_id)
        except Exception:
            return "Unknown"

    @staticmethod
    def list_available_gpus():
        """
        列出所有可用的GPU

        Returns:
            GPU信息列表
        """
        gpus = []

        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return gpus

        device_count = torch.cuda.device_count()

        for i in range(device_count):
            try:
                props = torch.cuda.get_device_properties(i)
                gpus.append({
                    'id': i,
                    'name': props.name,
                    'total_memory_mb': int(props.total_memory / 1024 / 1024),
                    'available_memory_mb': GPUMemoryChecker.get_available_memory(i)
                })
            except Exception:
                continue

        return gpus
