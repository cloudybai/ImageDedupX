# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于pHash算法的图片相似度检测脚本
使用imagehash库实现pHash算法

python pHash_test.py -r  /Users/cloudbai/PycharmProjects/imagesim/similarities/examples/data/shanqiimage/1-1.jpg -d /Users/cloudbai/PycharmProjects/imagesim/similarities/examples/data/shanqiimage -t 0.9 -m dhash
"""

import os
import argparse
from typing import List, Tuple, Dict
from PIL import Image
import imagehash
import numpy as np


class ImageSimilarityDetector:
    """图片相似度检测器"""

    def __init__(self, hash_method='phash', hash_size=8):
        """
        初始化检测器

        Args:
            hash_method: 哈希算法，支持 'phash', 'dhash', 'ahash', 'whash'
            hash_size: 哈希大小，默认8（产生64位哈希）
        """
        self.hash_method = hash_method.lower()
        self.hash_size = hash_size

        # 选择哈希函数
        self.hash_functions = {
            'phash': lambda img: imagehash.phash(img, hash_size=hash_size),
            'dhash': lambda img: imagehash.dhash(img, hash_size=hash_size),
            'ahash': lambda img: imagehash.average_hash(img, hash_size=hash_size),
            'whash': lambda img: imagehash.whash(img, hash_size=hash_size)
        }

        if self.hash_method not in self.hash_functions:
            raise ValueError(f"不支持的哈希方法: {hash_method}")

        self.hash_func = self.hash_functions[self.hash_method]

    def load_images_from_directory(self, image_dir: str) -> List[str]:
        """
        从目录加载所有图片文件

        Args:
            image_dir: 图片目录路径

        Returns:
            图片文件路径列表
        """
        supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        image_paths = []

        if not os.path.exists(image_dir):
            raise ValueError(f"目录不存在: {image_dir}")

        for filename in os.listdir(image_dir):
            if any(filename.lower().endswith(fmt) for fmt in supported_formats):
                image_paths.append(os.path.join(image_dir, filename))

        return sorted(image_paths)

    def validate_image(self, image_path: str) -> bool:
        """
        验证图片是否可以正常读取

        Args:
            image_path: 图片路径

        Returns:
            是否有效
        """
        try:
            with Image.open(image_path) as img:
                img.verify()
            return True
        except Exception as e:
            print(f"警告: 无法读取图片 {image_path}: {e}")
            return False

    def compute_image_hash(self, image_path: str):
        """
        计算图片的哈希值

        Args:
            image_path: 图片路径

        Returns:
            图片哈希值
        """
        try:
            with Image.open(image_path) as img:
                # 转换为RGB模式（如果需要）
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                return self.hash_func(img)
        except Exception as e:
            print(f"计算哈希值时出错 {image_path}: {e}")
            return None

    def calculate_similarity(self, image1_path: str, image2_path: str) -> float:
        """
        计算两张图片的相似度

        Args:
            image1_path: 第一张图片路径
            image2_path: 第二张图片路径

        Returns:
            相似度分数 (0-1之间，1表示完全相同)
        """
        try:
            hash1 = self.compute_image_hash(image1_path)
            hash2 = self.compute_image_hash(image2_path)

            if hash1 is None or hash2 is None:
                return 0.0

            # 计算汉明距离
            hamming_distance = hash1 - hash2

            # 转换为相似度分数 (0-1)
            # 最大可能的汉明距离是hash_size^2
            max_distance = self.hash_size * self.hash_size
            similarity = 1.0 - (hamming_distance / max_distance)

            return max(0.0, similarity)  # 确保不为负数

        except Exception as e:
            print(f"计算相似度时出错 ({image1_path} vs {image2_path}): {e}")
            return 0.0

    def find_similar_images(self,
                            reference_image: str,
                            candidate_images: List[str],
                            threshold: float = 0.8) -> List[Tuple[str, float]]:
        """
        找出与参考图片相似度超过阈值的图片

        Args:
            reference_image: 参考图片路径
            candidate_images: 候选图片路径列表
            threshold: 相似度阈值

        Returns:
            相似图片列表，每个元素为 (图片路径, 相似度分数)
        """
        if not self.validate_image(reference_image):
            raise ValueError(f"参考图片无效: {reference_image}")

        # 计算参考图片的哈希
        reference_hash = self.compute_image_hash(reference_image)
        if reference_hash is None:
            raise ValueError(f"无法计算参考图片的哈希值: {reference_image}")

        similar_images = []

        print(f"正在与参考图片 '{os.path.basename(reference_image)}' 进行比较...")
        print(f"使用算法: {self.hash_method.upper()}")
        print(f"哈希大小: {self.hash_size}x{self.hash_size}")
        print(f"相似度阈值: {threshold}")
        print("-" * 50)

        for i, candidate_path in enumerate(candidate_images):
            # 跳过参考图片本身
            if os.path.abspath(candidate_path) == os.path.abspath(reference_image):
                continue

            if not self.validate_image(candidate_path):
                continue

            try:
                candidate_hash = self.compute_image_hash(candidate_path)
                if candidate_hash is None:
                    continue

                # 计算汉明距离和相似度
                hamming_distance = reference_hash - candidate_hash
                max_distance = self.hash_size * self.hash_size
                similarity = 1.0 - (hamming_distance / max_distance)
                similarity = max(0.0, similarity)

                print(f"[{i + 1}/{len(candidate_images)}] {os.path.basename(candidate_path)}: "
                      f"{similarity:.4f} (汉明距离: {hamming_distance})")

                if similarity >= threshold:
                    similar_images.append((candidate_path, similarity))

            except Exception as e:
                print(f"处理图片时出错 {candidate_path}: {e}")
                continue

        # 按相似度降序排序
        similar_images.sort(key=lambda x: x[1], reverse=True)
        return similar_images

    def batch_detect_similar_images(self,
                                    image_directory: str,
                                    reference_image: str,
                                    threshold: float = 0.8,
                                    output_file: str = None) -> Dict:
        """
        批量检测相似图片

        Args:
            image_directory: 图片目录
            reference_image: 参考图片路径
            threshold: 相似度阈值
            output_file: 输出结果文件路径（可选）

        Returns:
            检测结果字典
        """
        # 加载所有图片
        all_images = self.load_images_from_directory(image_directory)
        print(f"从目录 '{image_directory}' 加载了 {len(all_images)} 张图片")

        if not all_images:
            print("未找到任何图片文件")
            return {"similar_images": [], "total_processed": 0}

        # 检测相似图片
        similar_images = self.find_similar_images(reference_image, all_images, threshold)

        # 生成结果报告
        result = {
            "reference_image": reference_image,
            "threshold": threshold,
            "hash_method": self.hash_method,
            "hash_size": self.hash_size,
            "total_processed": len(all_images),
            "similar_count": len(similar_images),
            "similar_images": similar_images
        }

        # 打印结果
        print("\n" + "=" * 60)
        print("检测结果汇总:")
        print("=" * 60)
        print(f"参考图片: {os.path.basename(reference_image)}")
        print(f"哈希算法: {result['hash_method'].upper()}")
        print(f"哈希大小: {result['hash_size']}x{result['hash_size']}")
        print(f"处理图片总数: {result['total_processed']}")
        print(f"相似图片数量: {result['similar_count']}")
        print(f"相似度阈值: {threshold}")

        if similar_images:
            print(f"\n发现的相似图片:")
            for i, (img_path, score) in enumerate(similar_images, 1):
                print(f"{i:2d}. {os.path.basename(img_path)} (相似度: {score:.4f})")
        else:
            print(f"\n未找到相似度超过 {threshold} 的图片")

        # 保存结果到文件
        if output_file:
            self.save_results_to_file(result, output_file)
            print(f"\n结果已保存到: {output_file}")

        return result

    def save_results_to_file(self, result: Dict, output_file: str):
        """保存结果到文件"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("图片相似度检测结果\n")
                f.write("=" * 50 + "\n")
                f.write(f"参考图片: {result['reference_image']}\n")
                f.write(f"哈希算法: {result['hash_method'].upper()}\n")
                f.write(f"哈希大小: {result['hash_size']}x{result['hash_size']}\n")
                f.write(f"相似度阈值: {result['threshold']}\n")
                f.write(f"处理图片总数: {result['total_processed']}\n")
                f.write(f"相似图片数量: {result['similar_count']}\n\n")

                if result['similar_images']:
                    f.write("相似图片列表:\n")
                    f.write("-" * 30 + "\n")
                    for i, (img_path, score) in enumerate(result['similar_images'], 1):
                        f.write(f"{i:2d}. {os.path.basename(img_path)} (相似度: {score:.4f})\n")
                        f.write(f"    路径: {img_path}\n")
                else:
                    f.write("未找到相似图片\n")

        except Exception as e:
            print(f"保存结果文件时出错: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='基于pHash算法的图片相似度检测工具')
    parser.add_argument('--reference', '-r', required=True, help='参考图片路径')
    parser.add_argument('--directory', '-d', required=True, help='待检测图片目录')
    parser.add_argument('--threshold', '-t', type=float, default=0.8,
                        help='相似度阈值 (0-1之间，默认0.8)')
    parser.add_argument('--hash-method', '-m', default='phash',
                        choices=['phash', 'dhash', 'ahash', 'whash'],
                        help='哈希算法 (默认phash)')
    parser.add_argument('--hash-size', '-s', type=int, default=8,
                        help='哈希大小 (默认8，产生64位哈希)')
    parser.add_argument('--output', '-o', help='输出结果文件路径')

    args = parser.parse_args()

    # 验证参数
    if not os.path.exists(args.reference):
        print(f"错误: 参考图片文件不存在: {args.reference}")
        return

    if not os.path.exists(args.directory):
        print(f"错误: 图片目录不存在: {args.directory}")
        return

    if not 0 <= args.threshold <= 1:
        print(f"错误: 相似度阈值必须在0-1之间，当前值: {args.threshold}")
        return

    if args.hash_size < 1:
        print(f"错误: 哈希大小必须大于0，当前值: {args.hash_size}")
        return

    try:
        # 创建检测器
        detector = ImageSimilarityDetector(
            hash_method=args.hash_method,
            hash_size=args.hash_size
        )

        # 执行相似度检测
        result = detector.batch_detect_similar_images(
            image_directory=args.directory,
            reference_image=args.reference,
            threshold=args.threshold,
            output_file=args.output
        )

        print(f"\n检测完成! 共找到 {result['similar_count']} 张相似图片")

    except Exception as e:
        print(f"执行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 如果直接运行脚本，可以用示例参数
    import sys

    if len(sys.argv) == 1:
        print("基于pHash算法的图片相似度检测工具")
        print("=" * 50)
        print("")
        print("使用示例:")
        print("python phash_similarity_detector.py -r reference.jpg -d ./images -t 0.8")
        print("")
        print("参数说明:")
        print("  -r, --reference   参考图片路径")
        print("  -d, --directory   待检测图片目录")
        print("  -t, --threshold   相似度阈值 (0-1，默认0.8)")
        print("  -m, --hash-method 哈希算法 (phash/dhash/ahash/whash，默认phash)")
        print("  -s, --hash-size   哈希大小 (默认8，产生64位哈希)")
        print("  -o, --output      输出结果文件路径 (可选)")
        print("")
        print("安装依赖:")
        print("pip install Pillow imagehash")
        print("")
        print("算法说明:")
        print("- pHash: 感知哈希，对亮度和对比度变化鲁棒")
        print("- dHash: 差分哈希，对图像缩放敏感度低")
        print("- aHash: 平均哈希，最基础的哈希算法")
        print("- wHash: 小波哈希，对图像变形更鲁棒")
        print("")
        print("相似度计算:")
        print("- 基于汉明距离计算相似度")
        print("- 相似度 = 1 - (汉明距离 / 最大距离)")
        print("- 汉明距离越小，图片越相似")
    else:
        main()


