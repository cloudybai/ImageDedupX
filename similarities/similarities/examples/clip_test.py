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
    print("请安装similarities相关安装包和库：pip install similarities")
    sys.exit(1)


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
            threshold: float = 0.8,
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
            threshold: float = 0.8,
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


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="图片相似度检测工具")
    parser.add_argument("--target", "-t", type=str, help="目标图片路径")
    parser.add_argument("--directory", "-d", type=str, help="图片目录路径")
    parser.add_argument("--threshold", "-th", type=float, default=0.8,
                        help="相似度阈值 (0-1之间，默认0.8)")
    parser.add_argument("--max-results", "-m", type=int, default=None,
                        help="最大返回结果数量")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="输出文件路径")
    parser.add_argument("--model", type=str, default="openai/clip-vit-base-patch32",
                        help="CLIP模型名称")
    parser.add_argument("--batch", "-b", action="store_true",
                        help="批量模式：找出目录中所有相互相似的图片对")

    args = parser.parse_args()

    # 验证参数
    if not args.batch and not args.target:
        print("错误：请指定目标图片路径 (--target) 或使用批量模式 (--batch)")
        return

    if not args.directory:
        print("错误：请指定图片目录路径 (--directory)")
        return

    if not os.path.exists(args.directory):
        print(f"错误：目录不存在 {args.directory}")
        return

    if args.target and not os.path.exists(args.target):
        print(f"错误：目标图片不存在 {args.target}")
        return

    # 初始化检测器
    detector = ImageSimilarityDetector(model_name=args.model)

    try:
        if args.batch:
            # 批量模式
            print("开始批量相似度检测...")
            results = detector.batch_find_similar_images(
                images_directory=args.directory,
                threshold=args.threshold,
                output_file=args.output
            )

            # 打印结果摘要
            total_similar_pairs = sum(len(similar_images) for similar_images in results.values()) // 2
            print(f"\n检测完成！")
            print(f"找到 {total_similar_pairs} 对相似图片")

        else:
            # 单目标模式
            candidate_images = detector.find_images_in_directory(args.directory)

            similar_images = detector.find_similar_images(
                target_image=args.target,
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
            if args.output:
                results = {args.target: similar_images}
                detector.save_results_to_file(results, args.output)

    except KeyboardInterrupt:
        print("\n检测被用户中断")
    except Exception as e:
        print(f"检测过程中出现错误: {e}")


if __name__ == "__main__":
    main()

# 使用示例：
"""
# 基本用法：找出与指定图片相似的图片
python image_similarity_detector.py --target /path/to/target.jpg --directory /path/to/images --threshold 0.8

python clip_test.py --target /Users/cloudbai/PycharmProjects/imagesim/similarities/examples/data/shanqiimage/1-1.jpg --directory /Users/cloudbai/PycharmProjects/imagesim/similarities/examples/data/shanqiimage --threshold 0.9

# 限制返回结果数量
python image_similarity_detector.py -t target.jpg -d ./images -th 0.85 -m 10

# 批量模式：找出目录中所有相互相似的图片对
python image_similarity_detector.py --batch --directory /path/to/images --threshold 0.8 --output results.txt

# 使用中文CLIP模型
python image_similarity_detector.py -t target.jpg -d ./images --model OFA-Sys/chinese-clip-vit-base-patch16

# 保存结果到文件
python image_similarity_detector.py -t target.jpg -d ./images -o similarity_results.txt
"""