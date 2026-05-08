# -*- coding: utf-8 -*-
"""
预测结果可视化导出器
用于在大幅面遥感影像上叠加检测结果，生成可视化成果
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from ..handlers.rs_data_structures import MergedResult

logger = logging.getLogger(__name__)


# 定义颜色方案（BGR格式用于OpenCV，RGB格式用于PIL）
CLASS_COLORS = {
    0: (0, 255, 0),      # 绿色
    1: (255, 0, 0),      # 蓝色
    2: (0, 0, 255),      # 红色
    3: (255, 255, 0),    # 青色
    4: (255, 0, 255),    # 品红
    5: (0, 255, 255),    # 黄色
    6: (128, 0, 128),    # 紫色
    7: (255, 165, 0),    # 橙色
    8: (255, 192, 203),  # 粉色
    9: (0, 128, 128),    # 深青
}


class VisualizationExporter:
    """预测结果可视化导出器

    在大幅面遥感影像上叠加检测结果，生成可视化成果图
    """

    def __init__(
        self,
        task_type: str,
        class_names: Optional[List[str]] = None,
        line_width: int = 2,
        show_labels: bool = True,
        show_conf: bool = True,
        font_size: int = 12
    ):
        """
        Args:
            task_type: 任务类型 (detect/segment/obb)
            class_names: 类别名称列表
            line_width: 线宽
            show_labels: 是否显示标签
            show_conf: 是否显示置信度
            font_size: 字体大小
        """
        self.task_type = task_type
        self.class_names = class_names or {}
        self.line_width = line_width
        self.show_labels = show_labels
        self.show_conf = show_conf
        self.font_size = font_size

    def export_visualization(
        self,
        image_array: np.ndarray,
        merged_result: MergedResult,
        output_path: str,
        conf_threshold: float = 0.0
    ) -> Path:
        """
        导出可视化结果

        Args:
            image_array: 原始影像数组 (H, W, 3)
            merged_result: 合并后的预测结果
            output_path: 输出文件路径
            conf_threshold: 置信度阈值，低于此值的结果不显示

        Returns:
            实际保存的文件路径
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 复制图像数组
        vis_image = image_array.copy()

        # 根据任务类型绘制结果
        if self.task_type == "detect" or self.task_type == "obb":
            vis_image = self._draw_detection_boxes(vis_image, merged_result, conf_threshold)
        elif self.task_type == "segment":
            vis_image = self._draw_segmentation_masks(vis_image, merged_result, conf_threshold)

        # 保存图像
        if output_path.suffix.lower() in ['.tif', '.tiff']:
            # 对于GeoTIFF，使用OpenCV保存
            if CV2_AVAILABLE:
                cv2.imwrite(str(output_path), cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
            else:
                # 降级为PNG
                png_path = output_path.with_suffix('.png')
                Image.fromarray(vis_image).save(png_path)
                output_path = png_path
        else:
            # 使用PIL保存
            Image.fromarray(vis_image).save(output_path)

        logger.info(f"Exported visualization to {output_path}")

        return output_path

    def _draw_detection_boxes(
        self,
        image: np.ndarray,
        merged_result: MergedResult,
        conf_threshold: float
    ) -> np.ndarray:
        """
        绘制检测框

        Args:
            image: 图像数组 (H, W, 3) RGB
            merged_result: 合并后的结果
            conf_threshold: 置信度阈值

        Returns:
            绘制后的图像
        """
        if merged_result.merged_boxes is None:
            return image

        # 使用PIL绘制
        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image)

        # 尝试加载字体
        try:
            font = ImageFont.truetype("arial.ttf", self.font_size)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", self.font_size)
            except:
                font = ImageFont.load_default()

        boxes = merged_result.merged_boxes
        confidences = merged_result.confidences
        class_ids = merged_result.class_ids

        for i in range(len(boxes)):
            conf = float(confidences[i]) if i < len(confidences) else 0.5
            if conf < conf_threshold:
                continue

            x1, y1, x2, y2 = boxes[i, :4].astype(int)
            class_id = int(class_ids[i]) if i < len(class_ids) else 0

            # 获取颜色
            color = self._get_class_color(class_id)

            # 绘制边界框
            draw.rectangle([x1, y1, x2, y2], outline=color, width=self.line_width)

            # 绘制标签
            if self.show_labels or self.show_conf:
                label = self._get_label(class_id, conf)
                self._draw_label(draw, label, (x1, y1), color, font)

        return np.array(pil_image)

    def _draw_segmentation_masks(
        self,
        image: np.ndarray,
        merged_result: MergedResult,
        conf_threshold: float
    ) -> np.ndarray:
        """
        绘制分割掩膜

        Args:
            image: 图像数组 (H, W, 3) RGB
            merged_result: 合并后的结果
            conf_threshold: 置信度阈值

        Returns:
            绘制后的图像
        """
        if merged_result.merged_masks is None:
            return image

        # 使用OpenCV绘制掩膜（更好的性能）
        if CV2_AVAILABLE:
            return self._draw_masks_opencv(image, merged_result, conf_threshold)
        else:
            return self._draw_masks_pil(image, merged_result, conf_threshold)

    def _draw_masks_opencv(
        self,
        image: np.ndarray,
        merged_result: MergedResult,
        conf_threshold: float
    ) -> np.ndarray:
        """使用OpenCV绘制掩膜"""
        # 转换为BGR
        vis_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR).copy()

        masks = merged_result.merged_masks
        confidences = merged_result.confidences
        class_ids = merged_result.class_ids

        for i in range(len(masks)):
            conf = float(confidences[i]) if i < len(confidences) else 0.5
            if conf < conf_threshold:
                continue

            mask = masks[i]
            class_id = int(class_ids[i]) if i < len(class_ids) else 0

            # 获取颜色
            color = self._get_class_color(class_id, bgr=True)

            # 创建彩色掩膜
            color_mask = np.zeros_like(vis_image)
            color_mask[mask > 0] = color

            # 半透明叠加
            alpha = 0.4
            vis_image = cv2.addWeighted(vis_image, 1, color_mask, alpha, 0)

            # 绘制轮廓
            contours, _ = cv2.findContours(
                mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(vis_image, contours, -1, color, self.line_width)

            # 绘制标签（在掩膜中心）
            if self.show_labels or self.show_conf:
                # 计算掩膜中心
                M = cv2.moments(mask.astype(np.uint8))
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    label = self._get_label(class_id, conf)

                    # 绘制标签背景
                    (text_w, text_h), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                    )
                    cv2.rectangle(vis_image,
                                (cx - text_w // 2 - 2, cy - text_h - 2),
                                (cx + text_w // 2 + 2, cy + 2),
                                color, -1)
                    # 绘制标签文字
                    cv2.putText(vis_image, label,
                              (cx - text_w // 2, cy),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)

    def _draw_masks_pil(
        self,
        image: np.ndarray,
        merged_result: MergedResult,
        conf_threshold: float
    ) -> np.ndarray:
        """使用PIL绘制掩膜"""
        pil_image = Image.fromarray(image)

        masks = merged_result.merged_masks
        confidences = merged_result.confidences
        class_ids = merged_result.class_ids

        # 创建覆盖层
        overlay = Image.new('RGBA', pil_image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        for i in range(len(masks)):
            conf = float(confidences[i]) if i < len(confidences) else 0.5
            if conf < conf_threshold:
                continue

            mask = masks[i]
            class_id = int(class_ids[i]) if i < len(class_ids) else 0

            # 获取颜色
            color_rgb = self._get_class_color(class_id)
            color_rgba = (*color_rgb, 100)  # 半透明

            # 绘制掩膜像素
            rows, cols = np.where(mask > 0)
            for r, c in zip(rows, cols):
                draw.point((c, r), fill=color_rgba)

        # 合并图像
        pil_image = pil_image.convert('RGBA')
        pil_image = Image.alpha_composite(pil_image, overlay)
        pil_image = pil_image.convert('RGB')

        return np.array(pil_image)

    def _get_class_color(self, class_id: int, bgr: bool = False) -> tuple:
        """
        获取类别对应的颜色

        Args:
            class_id: 类别ID
            bgr: 是否返回BGR格式

        Returns:
            颜色元组
        """
        if class_id in CLASS_COLORS:
            color = CLASS_COLORS[class_id]
        else:
            # 根据类别ID生成颜色
            np.random.seed(class_id)
            color = tuple(np.random.randint(0, 255, 3).tolist())

        if bgr:
            return (color[2], color[1], color[0])
        return color

    def _get_label(self, class_id: int, conf: float) -> str:
        """
        生成标签文本

        Args:
            class_id: 类别ID
            conf: 置信度

        Returns:
            标签文本
        """
        parts = []

        # 类别名称
        if self.show_labels:
            if class_id in self.class_names:
                parts.append(self.class_names[class_id])
            else:
                parts.append(f"class_{class_id}")

        # 置信度
        if self.show_conf:
            parts.append(f"{conf:.2f}")

        return " ".join(parts)

    def _draw_label(
        self,
        draw: ImageDraw.ImageDraw,
        label: str,
        position: tuple,
        color: tuple,
        font
    ):
        """
        绘制标签

        Args:
            draw: PIL Draw对象
            label: 标签文本
            position: 位置 (x, y)
            color: 颜色
            font: 字体
        """
        # 获取文本大小
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        x, y = position

        # 绘制背景
        draw.rectangle(
            [x, y - text_h - 4, x + text_w + 4, y],
            fill=color
        )

        # 绘制文字
        draw.text((x + 2, y - text_h - 2), label, fill=(255, 255, 255), font=font)


class LargeImageVisualizer:
    """大幅面影像可视化处理器

    处理超大影像的可视化，支持分块绘制和拼接
    """

    def __init__(
        self,
        task_type: str,
        class_names: Optional[List[str]] = None,
        line_width: int = 3,
        show_labels: bool = True,
        show_conf: bool = True
    ):
        """
        Args:
            task_type: 任务类型
            class_names: 类别名称列表
            line_width: 线宽
            show_labels: 是否显示标签
            show_conf: 是否显示置信度
        """
        self.exporter = VisualizationExporter(
            task_type=task_type,
            class_names=class_names,
            line_width=line_width,
            show_labels=show_labels,
            show_conf=show_conf
        )

    def export_large_visualization(
        self,
        image_array: np.ndarray,
        merged_result: MergedResult,
        output_path: str,
        conf_threshold: float = 0.0,
        max_size: int = 20000
    ) -> Path:
        """
        导出大幅面可视化结果

        Args:
            image_array: 原始影像数组 (H, W, 3)
            merged_result: 合并后的预测结果
            output_path: 输出文件路径
            conf_threshold: 置信度阈值
            max_size: 最大允许的影像尺寸（超过此尺寸将分块处理）

        Returns:
            实际保存的文件路径
        """
        height, width = image_array.shape[:2]

        # 如果影像尺寸在允许范围内，直接处理
        if height <= max_size and width <= max_size:
            return self.exporter.export_visualization(
                image_array, merged_result, output_path, conf_threshold
            )

        # 影像过大，使用分块处理
        logger.info(f"Large image detected ({height}x{width}), using tiled visualization")

        return self._export_tiled_visualization(
            image_array, merged_result, output_path, conf_threshold, max_size
        )

    def _export_tiled_visualization(
        self,
        image_array: np.ndarray,
        merged_result: MergedResult,
        output_path: str,
        conf_threshold: float,
        max_size: int
    ) -> Path:
        """分块可视化处理"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        height, width = image_array.shape[:2]

        # 计算分块
        tile_size = max_size // 2
        overlap = 100

        # 计算行列数
        rows = (height + tile_size - 1) // tile_size
        cols = (width + tile_size - 1) // tile_size

        logger.info(f"Processing in {rows}x{cols} tiles")

        # 创建输出图像
        vis_image = image_array.copy()

        # 分块绘制
        for row in range(rows):
            for col in range(cols):
                y1 = row * tile_size
                x1 = col * tile_size
                y2 = min(y1 + tile_size, height)
                x2 = min(x1 + tile_size, width)

                # 提取分块图像
                tile_img = image_array[y1:y2, x1:x2]

                # 筛选在此分块内的检测结果
                tile_result = self._filter_results_by_tile(
                    merged_result, y1, x1, y2, x2, conf_threshold
                )

                # 绘制分块结果
                if tile_result.total_instances > 0:
                    tile_vis = self.exporter.export_visualization(
                        tile_img, tile_result, str(output_path) + ".temp", conf_threshold
                    )

                    # 将绘制结果放回原图
                    vis_image[y1:y2, x1:x2] = np.array(Image.open(str(output_path) + ".temp"))

                    # 删除临时文件
                    Path(str(output_path) + ".temp").unlink()

        # 保存最终结果
        Image.fromarray(vis_image).save(output_path)

        logger.info(f"Exported tiled visualization to {output_path}")

        return output_path

    def _filter_results_by_tile(
        self,
        merged_result: MergedResult,
        y1: int, x1: int, y2: int, x2: int,
        conf_threshold: float
    ) -> MergedResult:
        """筛选在指定分块内的检测结果"""
        # 创建新的结果对象
        filtered_boxes = []
        filtered_confidences = []
        filtered_class_ids = []

        if merged_result.merged_boxes is not None:
            for i, box in enumerate(merged_result.merged_boxes):
                bx1, by1, bx2, by2 = box[:4]

                # 检查边界框是否与分块相交
                if bx2 < x1 or bx1 > x2 or by2 < y1 or by1 > y2:
                    continue

                # 检查置信度
                conf = float(merged_result.confidences[i]) if i < len(merged_result.confidences) else 0.5
                if conf < conf_threshold:
                    continue

                filtered_boxes.append(box)
                filtered_confidences.append(conf)
                filtered_class_ids.append(int(merged_result.class_ids[i]) if i < len(merged_result.class_ids) else 0)

        return MergedResult(
            merged_boxes=np.array(filtered_boxes) if filtered_boxes else None,
            class_ids=np.array(filtered_class_ids) if filtered_class_ids else None,
            confidences=np.array(filtered_confidences) if filtered_confidences else None,
            img_shape=(y2 - y1, x2 - x1),
            total_instances=len(filtered_boxes)
        )
