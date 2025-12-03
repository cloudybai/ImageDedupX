#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图片相似度检测脚本
基于similarities库的CLIP模型实现图片相似度计算和筛选

Requirements:
pip install similarities torch torchvision pillow numpy
"""

import os
import sys
import argparse
from typing import List, Tuple, Dict
import numpy as np
from PIL import Image
import torch

try:
    from similarities import ClipSimilarity
except ImportError:
    print("请安装similarities库：pip install similarities")
    sys.exit(1)

# ==================== 配置区域 ====================
# 在这里预设常用的图片路径，避免每次运行时手动输入

# 预设路径配置
# PRESET_PATHS = {
#     # 预设的图片目录路径
#     "image_directories": {
#         "default": "./images",  # 默认图片目录
#         "photos": "./photos",  # 照片目录
#         "downloads": "./downloads",  # 下载目录
#         "dataset": "./dataset/images",  # 数据集目录
#         "test": "./test_images",  # 测试图片目录
#     },

PRESET_PATHS = {
    # 预设的图片目录路径
    "image_directories": {
        "default": "./examples/data/shanqiimage",  # 默认图片目录
        "photos": ".examples/data/shanqiimage",  # 照片目录
        "downloads": "./data",  # 下载目录
        "dataset": "./examples/data/shanqiimage",  # 数据集目录
        "test": "./examples/data/shanqiimage",  # 测试图片目录
    },

    # 预设的目标图片路径
    "target_images": {
        "sample1": "./examples/data/shanqiimage/1-1.jpg",
        "sample2": "/examples/data/shanqiimage/1-2.jpg.png",
        "test_img": "./examples/data/shanqiimage/1-3.jpg",
        "reference": "./examples/data/shanqiimage/1-4.jpg",
    },

    # 预设的输出文件路径
    "output_files": {
        "default": "./similarity_results.txt",
        "detailed": "./detailed_results.txt",
        "batch": "./batch_results.txt",
    }
}

# 默认配置
DEFAULT_CONFIG = {
    "model_name": "openai/clip-vit-base-patch32",  # 默认模型
    "threshold": 0.9998,  # 默认相似度阈值
    "max_results": None,  # 默认最大结果数
    "batch_mode": False,  # 默认非批量模式
}


# ==================== 配置区域结束 ====================


class ImageSimilarityDetector:
    """图片相似度检测器"""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """
        初始化检测器

        Args:
            model_name: CLIP模型名称，默认使用openai/clip-vit-base-patch32
                       也可以使用中文模型：OFA-Sys/chinese-clip-vit-base-patch16
        """
        print(f"正在加载模型: {model_name}")
        try:
            self.model = ClipSimilarity(model_name_or_path=model_name)
            print("模型加载成功！")
        except Exception as e:
            print(f"模型加载失败: {e}")
            sys.exit(1)

    def load_image(self, image_path: str) -> Image.Image:
        """
        加载图片

        Args:
            image_path: 图片路径

        Returns:
            PIL Image对象
        """
        try:
            image = Image.open(image_path)
            # 转换为RGB模式（处理RGBA等格式）
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return image
        except Exception as e:
            print(f"加载图片失败 {image_path}: {e}")
            return None

    def get_supported_formats(self) -> List[str]:
        """获取支持的图片格式"""
        return ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']

    def find_images_in_directory(self, directory: str) -> List[str]:
        """
        在目录中查找所有支持的图片文件

        Args:
            directory: 目录路径

        Returns:
            图片文件路径列表
        """
        image_paths = []
        supported_formats = self.get_supported_formats()

        for root, dirs, files in os.walk(directory):
            for file in files:
                if any(file.lower().endswith(fmt) for fmt in supported_formats):
                    image_paths.append(os.path.join(root, file))

        return sorted(image_paths)

    def calculate_similarity(self, image1_path: str, image2_path: str) -> float:
        """
        计算两张图片的相似度

        Args:
            image1_path: 第一张图片路径
            image2_path: 第二张图片路径

        Returns:
            相似度分数 (0-1之间，1表示完全相似)
        """
        try:
            similarity_score = self.model.similarity(image1_path, image2_path)
            return float(similarity_score)
        except Exception as e:
            print(f"计算相似度失败 {image1_path} vs {image2_path}: {e}")
            return 0.0

    def find_similar_images(
            self,
            target_image: str,
            candidate_images: List[str],
            threshold: float = 0.9998,
            max_results: int = None
    ) -> List[Tuple[str, float]]:
        """
        找出与目标图片相似的图片

        Args:
            target_image: 目标图片路径
            candidate_images: 候选图片路径列表
            threshold: 相似度阈值 (0-1之间)
            max_results: 最大返回结果数量，None表示不限制

        Returns:
            相似图片列表，每个元素为(图片路径, 相似度分数)的元组
        """
        print(f"正在与目标图片比较: {target_image}")
        print(f"相似度阈值: {threshold}")
        print(f"候选图片数量: {len(candidate_images)}")

        similar_images = []

        for i, candidate_image in enumerate(candidate_images):
            # 跳过目标图片本身
            if os.path.abspath(candidate_image) == os.path.abspath(target_image):
                continue

            # 显示进度
            if (i + 1) % 10 == 0 or i == len(candidate_images) - 1:
                print(f"进度: {i + 1}/{len(candidate_images)}")

            similarity_score = self.calculate_similarity(target_image, candidate_image)

            if similarity_score >= threshold:
                similar_images.append((candidate_image, similarity_score))
                print(f"  找到相似图片: {candidate_image} (相似度: {similarity_score:.4f})")

        # 按相似度降序排序
        similar_images.sort(key=lambda x: x[1], reverse=True)

        # 限制返回结果数量
        if max_results is not None:
            similar_images = similar_images[:max_results]

        return similar_images

    def batch_find_similar_images(
            self,
            images_directory: str,
            threshold: float = 0.9998,
            output_file: str = None
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        批量查找相似图片（找出目录中所有相互相似的图片对）

        Args:
            images_directory: 图片目录
            threshold: 相似度阈值
            output_file: 输出文件路径（可选）

        Returns:
            字典，键为图片路径，值为相似图片列表
        """
        image_paths = self.find_images_in_directory(images_directory)
        print(f"找到 {len(image_paths)} 张图片")

        results = {}
        total_comparisons = len(image_paths) * (len(image_paths) - 1) // 2
        current_comparison = 0

        for i, image1 in enumerate(image_paths):
            similar_to_image1 = []

            for j, image2 in enumerate(image_paths[i + 1:], i + 1):
                current_comparison += 1

                if current_comparison % 50 == 0:
                    print(f"批量比较进度: {current_comparison}/{total_comparisons}")

                similarity_score = self.calculate_similarity(image1, image2)

                if similarity_score >= threshold:
                    similar_to_image1.append((image2, similarity_score))

                    # 同时记录反向关系
                    if image2 not in results:
                        results[image2] = []
                    results[image2].append((image1, similarity_score))

            if similar_to_image1:
                results[image1] = similar_to_image1

        # 保存结果到文件
        if output_file:
            self.save_results_to_file(results, output_file)

        return results

    def save_results_to_file(self, results: Dict[str, List[Tuple[str, float]]], output_file: str):
        """
        将结果保存到文件

        Args:
            results: 相似度检测结果
            output_file: 输出文件路径
        """
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("图片相似度检测结果\n")
                f.write("=" * 50 + "\n\n")

                for target_image, similar_images in results.items():
                    f.write(f"目标图片: {target_image}\n")
                    f.write(f"相似图片数量: {len(similar_images)}\n")

                    for similar_image, score in similar_images:
                        f.write(f"  - {similar_image} (相似度: {score:.4f})\n")
                    f.write("\n")

            print(f"结果已保存到: {output_file}")

        except Exception as e:
            print(f"保存结果失败: {e}")


def resolve_preset_path(path_input: str, path_type: str) -> str:
    """
    解析预设路径

    Args:
        path_input: 用户输入的路径（可能是预设名称或实际路径）
        path_type: 路径类型 ('target_images', 'image_directories', 'output_files')

    Returns:
        解析后的实际路径
    """
    if not path_input:
        return None

    # 如果输入的是预设名称，则返回对应的预设路径
    if path_type in PRESET_PATHS and path_input in PRESET_PATHS[path_type]:
        resolved_path = PRESET_PATHS[path_type][path_input]
        print(f"使用预设路径 '{path_input}': {resolved_path}")
        return resolved_path

    # 否则返回原始输入（假设是实际路径）
    return path_input


def list_preset_paths():
    """显示所有预设路径"""
    print("可用的预设路径:")
    print("=" * 50)

    print("\n📁 图片目录 (--directory/-d):")
    for name, path in PRESET_PATHS["image_directories"].items():
        exists = "✓" if os.path.exists(path) else "✗"
        print(f"  {name:12} -> {path} {exists}")

    print("\n🎯 目标图片 (--target/-t):")
    for name, path in PRESET_PATHS["target_images"].items():
        exists = "✓" if os.path.exists(path) else "✗"
        print(f"  {name:12} -> {path} {exists}")

    print("\n📄 输出文件 (--output/-o):")
    for name, path in PRESET_PATHS["output_files"].items():
        print(f"  {name:12} -> {path}")

    print(f"\n默认配置:")
    print(f"  模型: {DEFAULT_CONFIG['model_name']}")
    print(f"  阈值: {DEFAULT_CONFIG['threshold']}")
    print(f"  批量模式: {DEFAULT_CONFIG['batch_mode']}")


def run_with_preset_config():
    """使用预设配置运行"""
    print("使用预设配置运行...")
    print(f"目标图片: {PRESET_PATHS['target_images']['sample1']}")
    print(f"图片目录: {PRESET_PATHS['image_directories']['default']}")
    print(f"相似度阈值: {DEFAULT_CONFIG['threshold']}")

    # 检查预设路径是否存在
    target_path = PRESET_PATHS['target_images']['sample1']
    directory_path = PRESET_PATHS['image_directories']['default']

    if not os.path.exists(target_path):
        print(f"警告: 预设目标图片不存在: {target_path}")
        return False

    if not os.path.exists(directory_path):
        print(f"警告: 预设图片目录不存在: {directory_path}")
        return False

    # 使用预设配置运行检测
    detector = ImageSimilarityDetector(model_name=DEFAULT_CONFIG['model_name'])
    candidate_images = detector.find_images_in_directory(directory_path)

    similar_images = detector.find_similar_images(
        target_image=target_path,
        candidate_images=candidate_images,
        threshold=DEFAULT_CONFIG['threshold'],
        max_results=DEFAULT_CONFIG['max_results']
    )

    # 打印结果
    print(f"\n检测完成！")
    print(f"找到 {len(similar_images)} 张相似图片:")

    for image_path, score in similar_images:
        print(f"  {image_path} (相似度: {score:.4f})")

    return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="图片相似度检测工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用预设路径示例:
  %(prog)s --target sample1 --directory default
  %(prog)s -t sample1 -d photos --output detailed
  %(prog)s --batch -d dataset --output batch

查看预设路径:
  %(prog)s --list-presets

使用默认预设配置:
  %(prog)s --run-preset
        """
    )

    parser.add_argument("--target", "-t", type=str,default='/Users/cloudbai/PycharmProjects/imagesim/similarities/examples/data/shanqiimage/1-1.jpg',
                       help="目标图片路径或预设名称 (如: sample1, test_img)")
    parser.add_argument("--directory", "-d", type=str,default='/Users/cloudbai/PycharmProjects/imagesim/similarities/examples/data/shanqiimage',
                        help="图片目录路径或预设名称 (如: default, photos, dataset)")
    parser.add_argument("--threshold", "-th", type=float, default=DEFAULT_CONFIG['threshold'],
                        help=f"相似度阈值 (0-1之间，默认{DEFAULT_CONFIG['threshold']})")
    # parser.add_argument("--max-results", "-m", type=int, default=DEFAULT_CONFIG['max_results'],
    #                     help="最大返回结果数量")
    parser.add_argument("--max-results", "-m", type=int, default=5,
                        help="最大返回结果数量")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="输出文件路径或预设名称 (如: default, detailed, batch)")
    parser.add_argument("--model", type=str, default=DEFAULT_CONFIG['model_name'],
                        help="CLIP模型名称")
    parser.add_argument("--batch", "-b", action="store_true",
                        help="批量模式：找出目录中所有相互相似的图片对")
    parser.add_argument("--list-presets", action="store_true",
                        help="显示所有预设路径")
    parser.add_argument("--run-preset", action="store_true",
                        help="使用默认预设配置运行")

    args = parser.parse_args()

    # 显示预设路径
    if args.list_presets:
        list_preset_paths()
        return

    # 使用预设配置运行
    if args.run_preset:
        success = run_with_preset_config()
        if not success:
            print("\n请修改脚本顶部的PRESET_PATHS配置，设置正确的路径")
        return

    # 解析预设路径
    target_image = resolve_preset_path(args.target, "target_images")
    images_directory = resolve_preset_path(args.directory, "image_directories")
    output_file = resolve_preset_path(args.output, "output_files")

    # 验证参数
    if not args.batch and not target_image:
        print("错误：请指定目标图片路径 (--target) 或使用批量模式 (--batch)")
        print("提示：使用 --list-presets 查看可用的预设路径")
        return

    if not images_directory:
        print("错误：请指定图片目录路径 (--directory)")
        print("提示：使用 --list-presets 查看可用的预设路径")
        return

    if not os.path.exists(images_directory):
        print(f"错误：目录不存在 {images_directory}")
        return

    if target_image and not os.path.exists(target_image):
        print(f"错误：目标图片不存在 {target_image}")
        return

    # 初始化检测器
    detector = ImageSimilarityDetector(model_name=args.model)

    try:
        if args.batch:
            # 批量模式
            print("开始批量相似度检测...")
            results = detector.batch_find_similar_images(
                images_directory=images_directory,
                threshold=args.threshold,
                output_file=output_file
            )

            # 打印结果摘要
            total_similar_pairs = sum(len(similar_images) for similar_images in results.values()) // 2
            print(f"\n检测完成！")
            print(f"找到 {total_similar_pairs} 对相似图片")

        else:
            # 单目标模式
            candidate_images = detector.find_images_in_directory(images_directory)

            similar_images = detector.find_similar_images(
                target_image=target_image,
                candidate_images=candidate_images,
                threshold=args.threshold,
                max_results=args.max_results
            )

            # 打印结果
            print(f"\n检测完成！")
            print(f"找到 {len(similar_images)} 张相似图片:")

            for image_path, score in similar_images:
                print(f"  {image_path} (相似度: {score:.4f})")

            # 保存结果到文件
            if output_file:
                results = {target_image: similar_images}
                detector.save_results_to_file(results, output_file)

    except KeyboardInterrupt:
        print("\n检测被用户中断")
    except Exception as e:
        print(f"检测过程中出现错误: {e}")


if __name__ == "__main__":
    main()

# 使用示例：
"""
# 1. 查看所有预设路径
python image_similarity_detector.py --list-presets

# 2. 使用默认预设配置快速运行
python image_similarity_detector.py --run-preset

# 3. 使用预设路径名称（推荐）
python image_similarity_detector.py --target sample1 --directory default
python image_similarity_detector.py -t sample1 -d photos --output detailed

# 4. 混合使用预设名称和实际路径
python image_similarity_detector.py --target sample1 --directory /path/to/actual/dir

# 5. 批量模式使用预设路径
python image_similarity_detector.py --batch --directory dataset --output batch

# 6. 传统方式（仍然支持）
python image_similarity_detector.py --target /path/to/target.jpg --directory /path/to/images --threshold 0.8

# 修改预设配置：
# 1. 在脚本顶部的PRESET_PATHS中添加您的常用路径
# 2. 在DEFAULT_CONFIG中修改默认参数
"""