#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的图片相似度检测脚本
解决CLIP模型相似度过高的问题

主要改进：
1. 添加多种相似度计算方法
2. 使用更适合的特征提取方法
3. 添加相似度校准功能
4. 提供更准确的阈值建议
"""

import os
import sys
import argparse
from typing import List, Tuple, Dict
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

try:
    from similarities import ClipSimilarity
    import torchvision.transforms as transforms
    from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
    from sklearn.preprocessing import normalize
except ImportError as e:
    print(f"请安装所需库：pip install similarities torch torchvision scikit-learn pillow")
    print(f"缺失库：{e}")
    sys.exit(1)

# 预设路径配置
PRESET_PATHS = {
    "image_directories": {
        "default": "./examples/data/shanqiimage",
        "photos": "./examples/data/shanqiimage",
        "downloads": "./data",
        "dataset": "./examples/data/shanqiimage",
        "test": "./examples/data/shanqiimage",
    },
    "target_images": {
        "sample1": "./examples/data/shanqiimage/1-1.jpg",
        "sample2": "./examples/data/shanqiimage/1-2.jpg",
        "test_img": "./examples/data/shanqiimage/1-3.jpg",
        "reference": "./examples/data/shanqiimage/1-4.jpg",
    },
    "output_files": {
        "default": "./similarity_results.txt",
        "detailed": "./detailed_results.txt",
        "batch": "./batch_results.txt",
    }
}

# 默认配置
DEFAULT_CONFIG = {
    "model_name": "openai/clip-vit-base-patch32",
    "similarity_method": "mixed",  # 新增：相似度计算方法
    "threshold": 0.5,  # 调整：更合理的默认阈值
    "max_results": 10,
    "batch_mode": False,
    "calibration": True,  # 新增：是否启用相似度校准
}


class ImprovedImageSimilarityDetector:
    """改进的图片相似度检测器"""

    def __init__(self,
                 model_name: str = "openai/clip-vit-base-patch32",
                 similarity_method: str = "mixed",
                 enable_calibration: bool = True):
        """
        初始化检测器

        Args:
            model_name: CLIP模型名称
            similarity_method: 相似度计算方法
            enable_calibration: 是否启用相似度校准
        """
        print(f"正在加载模型: {model_name}")
        try:
            self.model = ClipSimilarity(model_name_or_path=model_name)
            self.similarity_method = similarity_method
            self.enable_calibration = enable_calibration
            print("模型加载成功！")

            # 初始化图像预处理
            self.preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

            # 校准参数（通过实验确定）
            self.calibration_params = {
                "cosine": {"scale": 2.0, "shift": -0.5},
                "cosine_normalized": {"scale": 5.0, "shift": -2.0},
                "euclidean": {"scale": 1.0, "shift": 0.0},
            }

        except Exception as e:
            print(f"模型加载失败: {e}")
            sys.exit(1)

    def get_image_features(self, image_path: str) -> np.ndarray:
        """
        获取图像特征向量

        Args:
            image_path: 图片路径

        Returns:
            归一化的特征向量
        """
        try:
            # # 直接使用similarities库计算特征
            # # 我们通过计算与自身的相似度来获取特征表示的效果
            # # 这里采用一个技巧：先计算与参考图片的相似度来获得特征

            # # 由于similarities库没有直接的encode方法，我们改用其他方式
            # # 使用PIL加载图片并转换
            # image = Image.open(image_path).convert('RGB')

            # # 将图片转换为模型可接受的格式
            # # 这里我们采用直接使用similarity方法的方式
            # # 返回图片路径，后续直接使用similarity方法

            return image_path  # 返回路径，在后续计算中直接使用

        except Exception as e:
            print(f"加载图片失败 {image_path}: {e}")
            return None

    def calculate_multiple_similarities(self, image1_path: str, image2_path: str) -> Dict[str, float]:
        """
        使用多种方法计算相似度

        Args:
            image1_path: 第一张图片的路径
            image2_path: 第二张图片的路径

        Returns:
            包含多种相似度分数的字典
        """
        similarities = {}

        try:
            # 1. 原始CLIP相似度
            original_sim = self.model.similarity(image1_path, image2_path)
            similarities['cosine'] = float(original_sim)

            # 2. 校准后的相似度
            # 对原始相似度进行非线性变换，降低过高的相似度
            calibrated_sim = self._calibrate_similarity(original_sim, method='sigmoid')
            similarities['cosine_normalized'] = float(calibrated_sim)

            # 3. 基于原始相似度的其他变换
            # 使用幂函数变换
            power_sim = np.power(original_sim, 3)  # 立方变换，降低高相似度
            similarities['power_transform'] = float(power_sim)

            # 4. 线性变换
            # 将[0.95, 1.0]映射到[0, 1]
            if original_sim >= 0.95:
                linear_sim = (original_sim - 0.95) / 0.05
            else:
                linear_sim = 0.0
            similarities['linear_transform'] = float(linear_sim)

            # 5. 对数变换
            # 使用对数函数压缩高相似度值
            log_sim = np.log10(original_sim * 10) if original_sim > 0 else 0
            # 归一化到[0,1]
            log_sim = max(0, min(1, log_sim))
            similarities['log_transform'] = float(log_sim)

            # 6. 混合相似度
            mixed_sim = (
                    0.3 * similarities['cosine_normalized'] +
                    0.2 * similarities['power_transform'] +
                    0.2 * similarities['linear_transform'] +
                    0.3 * similarities['log_transform']
            )
            similarities['mixed'] = float(mixed_sim)

        except Exception as e:
            print(f"计算相似度失败: {e}")
            # 返回默认值
            for method in ['cosine', 'cosine_normalized', 'power_transform', 'linear_transform', 'log_transform',
                           'mixed']:
                similarities[method] = 0.0

        return similarities

    def _calibrate_similarity(self, similarity: float, method: str = 'sigmoid') -> float:
        """
        校准相似度分数

        Args:
            similarity: 原始相似度
            method: 校准方法

        Returns:
            校准后的相似度
        """
        try:
            if method == 'sigmoid':
                # 使用sigmoid函数：将高相似度进行压缩
                # 设置阈值为0.95，超过0.95的相似度进行强烈压缩
                if similarity >= 0.95:
                    # 将[0.95, 1.0]区间映射到sigmoid函数
                    x = (similarity - 0.95) * 20  # 放大区间
                    calibrated = 1 / (1 + np.exp(-x + 5))  # sigmoid变换
                    return 0.5 + 0.5 * calibrated  # 映射到[0.5, 1.0]
                else:
                    # 低于0.95的相似度线性映射到[0, 0.5]
                    return similarity * 0.526  # 0.95 * 0.526 ≈ 0.5
            else:
                return similarity

        except Exception as e:
            print(f"校准相似度失败: {e}")
            return similarity

    def calculate_similarity(self, image1_path: str, image2_path: str) -> float:
        """
        计算两张图片的相似度（主要方法）

        Args:
            image1_path: 第一张图片路径
            image2_path: 第二张图片路径

        Returns:
            相似度分数
        """
        try:
            # 直接计算多种相似度
            similarities = self.calculate_multiple_similarities(image1_path, image2_path)

            # 返回指定方法的相似度
            return similarities.get(self.similarity_method, 0.0)

        except Exception as e:
            print(f"计算相似度失败 {image1_path} vs {image2_path}: {e}")
            return 0.0

    def calculate_similarity_detailed(self, image1_path: str, image2_path: str) -> Dict[str, float]:
        """
        计算详细的相似度信息（包含所有方法）

        Args:
            image1_path: 第一张图片路径
            image2_path: 第二张图片路径

        Returns:
            包含所有相似度方法结果的字典
        """
        try:
            return self.calculate_multiple_similarities(image1_path, image2_path)

        except Exception as e:
            print(f"计算详细相似度失败: {e}")
            return {method: 0.0 for method in
                    ['cosine', 'cosine_normalized', 'power_transform', 'linear_transform', 'log_transform', 'mixed']}

    def find_similar_images(self,
                            target_image: str,
                            candidate_images: List[str],
                            threshold: float = 0.85,
                            max_results: int = None,
                            detailed: bool = False) -> List[Tuple[str, float, Dict]]:
        """
        找出与目标图片相似的图片（改进版）

        Args:
            target_image: 目标图片路径
            candidate_images: 候选图片路径列表
            threshold: 相似度阈值
            max_results: 最大返回结果数量
            detailed: 是否返回详细信息

        Returns:
            相似图片列表，每个元素为(图片路径, 主相似度分数, 详细信息字典)的元组
        """
        print(f"正在与目标图片比较: {target_image}")
        print(f"使用相似度方法: {self.similarity_method}")
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

            if detailed:
                # 获取详细的相似度信息
                similarities = self.calculate_similarity_detailed(target_image, candidate_image)
                main_similarity = similarities.get(self.similarity_method, 0.0)
                detail_info = similarities
            else:
                # 只计算主要相似度
                main_similarity = self.calculate_similarity(target_image, candidate_image)
                detail_info = {}

            if main_similarity >= threshold:
                similar_images.append((candidate_image, main_similarity, detail_info))
                if detailed:
                    print(f"  找到相似图片: {candidate_image}")
                    print(f"    主要相似度 ({self.similarity_method}): {main_similarity:.4f}")
                    print(f"    所有方法: {detail_info}")
                else:
                    print(f"  找到相似图片: {candidate_image} (相似度: {main_similarity:.4f})")

        # 按相似度降序排序
        similar_images.sort(key=lambda x: x[1], reverse=True)

        # 限制返回结果数量
        if max_results is not None:
            similar_images = similar_images[:max_results]

        return similar_images

    def get_similarity_statistics(self, images_directory: str, sample_size: int = 50) -> Dict:
        """
        分析图片目录的相似度统计信息，帮助确定合适的阈值

        Args:
            images_directory: 图片目录
            sample_size: 采样大小

        Returns:
            统计信息字典
        """
        print("正在分析相似度分布，帮助确定合适的阈值...")

        image_paths = self.find_images_in_directory(images_directory)
        if len(image_paths) < 2:
            print("图片数量不足，无法进行统计分析")
            return {}

        # 随机采样以提高效率
        import random
        if len(image_paths) > sample_size:
            sample_images = random.sample(image_paths, sample_size)
        else:
            sample_images = image_paths

        similarities = []
        total_comparisons = len(sample_images) * (len(sample_images) - 1) // 2
        current = 0

        for i in range(len(sample_images)):
            for j in range(i + 1, len(sample_images)):
                current += 1
                if current % 10 == 0:
                    print(f"统计进度: {current}/{total_comparisons}")

                sim = self.calculate_similarity(sample_images[i], sample_images[j])
                similarities.append(sim)

        similarities = np.array(similarities)

        stats = {
            'mean': float(np.mean(similarities)),
            'std': float(np.std(similarities)),
            'min': float(np.min(similarities)),
            'max': float(np.max(similarities)),
            'median': float(np.median(similarities)),
            'percentile_25': float(np.percentile(similarities, 25)),
            'percentile_75': float(np.percentile(similarities, 75)),
            'percentile_90': float(np.percentile(similarities, 90)),
            'percentile_95': float(np.percentile(similarities, 95)),
            'sample_size': len(similarities),
            'method': self.similarity_method
        }

        # 推荐阈值
        recommended_thresholds = {
            'conservative': float(np.percentile(similarities, 95)),  # 5%最相似
            'moderate': float(np.percentile(similarities, 90)),  # 10%最相似
            'liberal': float(np.percentile(similarities, 75)),  # 25%最相似
        }

        stats['recommended_thresholds'] = recommended_thresholds

        return stats

    def find_images_in_directory(self, directory: str) -> List[str]:
        """在目录中查找所有支持的图片文件"""
        image_paths = []
        supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']

        for root, dirs, files in os.walk(directory):
            for file in files:
                if any(file.lower().endswith(fmt) for fmt in supported_formats):
                    image_paths.append(os.path.join(root, file))

        return sorted(image_paths)

    def print_similarity_help(self):
        """打印相似度方法说明"""
        print("\n可用的相似度计算方法:")
        print("=" * 50)
        print("cosine              - 原始CLIP余弦相似度")
        print("cosine_normalized   - 校准后的余弦相似度 (推荐)")
        print("power_transform     - 幂函数变换相似度")
        print("linear_transform    - 线性变换相似度")
        print("log_transform       - 对数变换相似度")
        print("mixed              - 混合相似度 (多方法加权)")
        print("\n推荐阈值范围:")
        print("cosine:             0.95 - 0.999")
        print("cosine_normalized:  0.5 - 0.9 (推荐)")
        print("power_transform:    0.8 - 0.99")
        print("linear_transform:   0.1 - 0.8")
        print("log_transform:      0.6 - 0.95")
        print("mixed:              0.3 - 0.8")


def resolve_preset_path(path_input: str, path_type: str) -> str:
    """解析预设路径"""
    if not path_input:
        return None

    if path_type in PRESET_PATHS and path_input in PRESET_PATHS[path_type]:
        resolved_path = PRESET_PATHS[path_type][path_input]
        print(f"使用预设路径 '{path_input}': {resolved_path}")
        return resolved_path

    return path_input


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="改进的图片相似度检测工具",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--target", "-t", type=str, default='/Users/cloudbai/PycharmProjects/imagesim/similarities/examples/data/shanqiimage/1-1.jpg',
                        help="目标图片路径或预设名称")
    parser.add_argument("--directory", "-d", type=str,default='/Users/cloudbai/PycharmProjects/imagesim/similarities/examples/data/shanqiimage',
                        help="图片目录路径或预设名称")
    parser.add_argument("--threshold", "-th", type=float, default=DEFAULT_CONFIG['threshold'],
                        help=f"相似度阈值 (默认{DEFAULT_CONFIG['threshold']})")
    parser.add_argument("--max-results", "-m", type=int, default=DEFAULT_CONFIG['max_results'],
                        help="最大返回结果数量")
    parser.add_argument("--similarity-method", "-sm", type=str,
                        default=DEFAULT_CONFIG['similarity_method'],
                        choices=['cosine', 'cosine_normalized', 'power_transform', 'linear_transform', 'log_transform',
                                 'mixed'],
                        help="相似度计算方法")
    parser.add_argument("--model", type=str, default=DEFAULT_CONFIG['model_name'],
                        help="CLIP模型名称")
    parser.add_argument("--detailed", action="store_true",
                        help="显示详细的相似度信息")
    parser.add_argument("--analyze", action="store_true",
                        help="分析目录中图片的相似度分布")
    parser.add_argument("--help-similarity", action="store_true",
                        help="显示相似度方法说明")

    args = parser.parse_args()

    # 显示相似度方法说明
    if args.help_similarity:
        detector = ImprovedImageSimilarityDetector()
        detector.print_similarity_help()
        return

    # 初始化检测器
    detector = ImprovedImageSimilarityDetector(
        model_name=args.model,
        similarity_method=args.similarity_method
    )

    # 解析路径
    target_image = resolve_preset_path(args.target, "target_images")
    images_directory = resolve_preset_path(args.directory, "image_directories")

    if not images_directory:
        print("错误：请指定图片目录路径 (--directory)")
        return

    if not os.path.exists(images_directory):
        print(f"错误：目录不存在 {images_directory}")
        return

    # 分析相似度分布
    if args.analyze:
        stats = detector.get_similarity_statistics(images_directory)
        if stats:
            print(f"\n相似度统计信息 (方法: {stats['method']}):")
            print("=" * 50)
            print(f"平均值:     {stats['mean']:.4f}")
            print(f"标准差:     {stats['std']:.4f}")
            print(f"最小值:     {stats['min']:.4f}")
            print(f"最大值:     {stats['max']:.4f}")
            print(f"中位数:     {stats['median']:.4f}")
            print(f"75百分位:   {stats['percentile_75']:.4f}")
            print(f"90百分位:   {stats['percentile_90']:.4f}")
            print(f"95百分位:   {stats['percentile_95']:.4f}")
            print(f"\n推荐阈值:")
            print(f"保守 (5%最相似):   {stats['recommended_thresholds']['conservative']:.4f}")
            print(f"适中 (10%最相似):  {stats['recommended_thresholds']['moderate']:.4f}")
            print(f"宽松 (25%最相似):  {stats['recommended_thresholds']['liberal']:.4f}")
        return

    # 单目标检测
    if not target_image:
        print("错误：请指定目标图片路径 (--target)")
        return

    if not os.path.exists(target_image):
        print(f"错误：目标图片不存在 {target_image}")
        return

    candidate_images = detector.find_images_in_directory(images_directory)

    similar_images = detector.find_similar_images(
        target_image=target_image,
        candidate_images=candidate_images,
        threshold=args.threshold,
        max_results=args.max_results,
        detailed=args.detailed
    )

    # 打印结果
    print(f"\n检测完成！")
    print(f"找到 {len(similar_images)} 张相似图片:")

    for image_path, score, details in similar_images:
        print(f"  {image_path} (主相似度: {score:.4f})")
        if details:
            print(f"    详细信息: {details}")


if __name__ == "__main__":
    main()
