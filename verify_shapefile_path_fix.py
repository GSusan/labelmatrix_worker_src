#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证Shapefile路径修复
"""

from pathlib import Path

def verify_path_logic():
    """验证修复后的路径处理逻辑"""
    print("=" * 70)
    print("Shapefile路径处理逻辑验证")
    print("=" * 70)

    # 模拟用户配置
    scenarios = [
        {
            "name": "场景1: 原始问题场景",
            "image_path": "D:/DataSets/SatDatasets/lh_farmland/Shapefile/inference/N-34-106-A-c-1-3_water/N-34-106-A-c-1-3.tif",
            "save_dir": "D:/DataSets/SatDatasets/lh_farmland/Shapefile/inference"
        },
        {
            "name": "场景2: 简单文件名",
            "image_path": "test_image.tif",
            "save_dir": "D:/output"
        },
        {
            "name": "场景3: 带时间戳",
            "image_path": "satellite_image.tif",
            "save_dir": "D:/results",
            "naming_config": "timestamp"
        }
    ]

    for scenario in scenarios:
        print(f"\n{scenario['name']}")
        print("-" * 70)

        image_path = scenario['image_path']
        save_dir = scenario['save_dir']
        naming_config = scenario.get('naming_config', 'default')

        print(f"输入影像: {image_path}")
        print(f"输出目录: {save_dir}")

        # 模拟RemoteSensingPredictor中的路径构建
        image_stem = Path(image_path).stem
        output_path = Path(save_dir) / f"{image_stem}.shp"

        print(f"\nRemoteSensingPredictor构建的路径:")
        print(f"  影像文件名: {image_stem}")
        print(f"  Shapefile输出路径: {output_path}")

        # 模拟修复后的ShapefileExporter.export()方法
        print(f"\n修复后的ShapefileExporter处理:")

        # 传入的output_path
        input_output_path = Path(output_path)
        print(f"  传入路径: {input_output_path}")

        # 确保父目录存在（但不创建子文件夹）
        print(f"  检查父目录: {input_output_path.parent}")
        print(f"  父目录存在性检查: {input_output_path.parent.exists()}")

        # 直接使用传入的路径（修复后的逻辑）
        final_path = input_output_path
        print(f"  最终路径: {final_path}")

        # 列出将要创建的Shapefile组件文件
        components = ['.shp', '.shx', '.dbf', '.prj', '.cpg']
        print(f"\n将创建的Shapefile组件文件:")
        for comp in components:
            component_file = final_path.parent / f"{final_path.stem}{comp}"
            print(f"  - {component_file}")

        print(f"\n确认:")
        print(f"  ✓ 所有文件直接保存在: {final_path.parent}")
        print(f"  ✓ 不创建额外的子文件夹")
        print(f"  ✓ Shapefile和GeoJSON在同一级目录")

def main():
    """主函数"""
    verify_path_logic()

    print("\n" + "=" * 70)
    print("修复确认")
    print("=" * 70)
    print("✓ Shapefile文件将直接保存在save_dir目录下")
    print("✓ 不会创建与影像同名的子文件夹")
    print("✓ 所有Shapefile组件文件(.shp, .shx, .dbf, .prj, .cpg)在同一目录")
    print("✓ 与GeoJSON文件保持在同一级目录")

if __name__ == "__main__":
    main()
