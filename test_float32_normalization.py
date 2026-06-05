#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试改进的float32归一化方法
"""

import numpy as np
from pathlib import Path


class ImageNormalizationTester:
    """图像归一化测试器"""

    def __init__(self):
        """初始化测试器"""
        self.output_dir = Path("test_normalization_results")
        self.output_dir.mkdir(exist_ok=True)

    def test_normalize_float_image(self):
        """测试改进的float32归一化方法"""
        print("=" * 60)
        print("测试改进的Float32归一化方法")
        print("=" * 60)

        # 模拟遥感局部的normalize_float_image方法
        def normalize_float_image(image):
            """改进的float32归一化"""
            # 检查NaN和Inf
            if np.any(np.isnan(image)) or np.any(np.isinf(image)):
                print("  警告: 图像包含NaN或Inf值")
                image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

            # 获取实际数据范围
            img_min = np.min(image)
            img_max = np.max(image)

            print(f"  原始数据范围: [{img_min:.4f}, {img_max:.4f}]")

            # 如果数据范围很小
            if img_max - img_min < 1e-6:
                print("  警告: 数据范围很小，使用阈值归一化")
                normalized = np.zeros_like(image, dtype=np.uint8)
                mean_val = np.mean(image)
                std_val = np.std(image)
                mask = image > (mean_val + std_val)
                normalized[mask] = 255
                return normalized

            # 两种归一化方法
            p2 = np.percentile(image, 2)
            p98 = np.percentile(image, 98)
            range_ratio = (img_max - img_min) / (p98 - p2 + 1e-7)

            print(f"  2%分位数: {p2:.4f}, 98%分位数: {p98:.4f}")
            print(f"  范围比率: {range_ratio:.2f}")

            # 选择合适的归一化方法
            if range_ratio > 2.0:
                print("  使用百分比归一化（有极端值）")
                normalized = (image - p2) / (p98 - p2 + 1e-7)
            else:
                print("  使用最小-最大归一化（分布均匀）")
                normalized = (image - img_min) / (img_max - img_min)

            return np.clip(normalized * 255, 0, 255).astype(np.uint8)

        def normalize_old_method(image):
            """旧的归一化方法（仅使用百分比）"""
            img_min = np.percentile(image, 2)
            img_max = np.percentile(image, 98)
            normalized = (image - img_min) / (img_max - img_min + 1e-7)
            return np.clip(normalized * 255, 0, 255).astype(np.uint8)

        # 测试用例
        test_cases = [
            {
                "name": "正常分布的float32数据",
                "data": self._generate_normal_float_data((100, 100, 3), mean=0.5, std=0.2)
            },
            {
                "name": "有极端值的float32数据",
                "data": self._generate_outlier_float_data((100, 100, 3))
            },
            {
                "name": "低对比度float32数据",
                "data": self._generate_low_contrast_data((100, 100, 3))
            },
            {
                "name": "模拟遥感影像数据",
                "data": self._generate_remote_sensing_like_data((100, 100, 3))
            }
        ]

        for i, test_case in enumerate(test_cases):
            print(f"\n测试用例 {i+1}: {test_case['name']}")
            print("-" * 40)

            test_data = test_case['data']

            # 测试新方法
            print("新方法结果:")
            result_new = normalize_float_image(test_data)

            # 测试旧方法
            print("旧方法结果:")
            result_old = normalize_old_method(test_data)

            # 比较结果
            self._compare_results(test_case['name'], test_data, result_new, result_old)

        print("\n" + "=" * 60)
        print("测试完成")
        print("=" * 60)

    def _generate_normal_float_data(self, shape, mean=0.5, std=0.2):
        """生成正态分布的float32数据"""
        np.random.seed(42)
        data = np.random.normal(mean, std, shape).astype(np.float32)
        return np.clip(data, 0, 1)  # 限制在0-1范围

    def _generate_outlier_float_data(self, shape):
        """生成包含极端值的float32数据"""
        np.random.seed(42)
        # 基础数据
        data = np.random.uniform(0.3, 0.7, shape).astype(np.float32)

        # 添加一些极端值
        outlier_mask = np.random.random(shape[:2]) < 0.01  # 1%的极端值
        for c in range(shape[2]):
            data[outlier_mask, c] = np.random.uniform(0.0, 0.1, size=np.sum(outlier_mask))
            data[outlier_mask, c] = np.random.uniform(0.9, 1.0, size=np.sum(outlier_mask))

        return data

    def _generate_low_contrast_data(self, shape):
        """生成低对比度数据"""
        np.random.seed(42)
        # 数据范围很小，比如0.45-0.55
        data = np.random.uniform(0.45, 0.55, shape).astype(np.float32)
        return data

    def _generate_remote_sensing_like_data(self, shape):
        """生成模拟遥感影像数据"""
        np.random.seed(42)
        # 模拟多波段遥感数据，每个波段有不同的分布
        data = np.zeros(shape, dtype=np.float32)

        # 红外波段（较高值）
        data[:, :, 0] = np.random.uniform(0.6, 0.9, shape[:2])

        # 红光波段（中等值）
        data[:, :, 1] = np.random.uniform(0.3, 0.6, shape[:2])

        # 绿光波段（较低值）
        data[:, :, 2] = np.random.uniform(0.1, 0.4, shape[:2])

        return data

    def _compare_results(self, test_name, original_data, result_new, result_old):
        """比较新旧方法的结果"""
        # 计算统计信息
        new_mean = np.mean(result_new)
        new_std = np.std(result_new)
        new_min = np.min(result_new)
        new_max = np.max(result_new)

        old_mean = np.mean(result_old)
        old_std = np.std(result_old)
        old_min = np.min(result_old)
        old_max = np.max(result_old)

        print(f"  新方法统计: 均值={new_mean:.1f}, 标准差={new_std:.1f}, 范围=[{new_min}, {new_max}]")
        print(f"  旧方法统计: 均值={old_mean:.1f}, 标准差={old_std:.1f}, 范围=[{old_min}, {old_max}]")

        # 判断哪个方法更好
        new_dynamic_range = new_max - new_min
        old_dynamic_range = old_max - old_min

        if new_dynamic_range > old_dynamic_range:
            print(f"  ✓ 新方法提供更大的动态范围 ({new_dynamic_range} vs {old_dynamic_range})")
        elif new_dynamic_range < old_dynamic_range:
            print(f"  ⚠ 旧方法提供更大的动态范围 ({old_dynamic_range} vs {new_dynamic_range})")
        else:
            print(f"  = 两种方法动态范围相同 ({new_dynamic_range})")


def main():
    """主函数"""
    print("Float32归一化方法改进测试")
    print()

    tester = ImageNormalizationTester()
    tester.test_normalize_float_image()

    print("\n关键改进:")
    print("1. 区分float32和整数数据类型")
    print("2. 对float32使用智能的归一化策略选择")
    print("3. 处理NaN和Inf值")
    print("4. 根据数据分布选择最小-最大或百分比归一化")
    print("5. 提供详细的日志信息用于调试")


if __name__ == "__main__":
    main()
