#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模型图像相似度检测器
解决单一CLIP模型相似度过高的问题
python alternative_model_test.py --target /Users/cloudbai/PycharmProjects/imagesim/similarities/examples/data/shanqiimage/1-1.jpg --directory /Users/cloudbai/PycharmProjects/imagesim/similarities/examples/data/shanqiimage
python alternative_model_test.py --target /Users/cloudbai/PycharmProjects/imagesim/similarities/examples/data/shanqiimage/2-1.jpg --directory /Users/cloudbai/PycharmProjects/imagesim/similarities/examples/data/shanqiimage

主要改进：
1. 集成多种不同类型的模型
2. 使用传统计算机视觉特征作为补充
3. 混合多种相似度计算方法
4. 自适应阈值调整
"""

import os
import sys
import argparse
from typing import List, Tuple, Dict, Optional
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import cv2
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import pearsonr
import warnings

warnings.filterwarnings('ignore')

try:
    from similarities import ClipSimilarity
    import torchvision.transforms as transforms
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import normalize
    import torchvision.models as models
    from sentence_transformers import SentenceTransformer
    import timm
except ImportError as e:
    print(f"请安装所需库：")
    print(f"pip install similarities torch torchvision scikit-learn pillow opencv-python")
    print(f"pip install sentence-transformers timm")
    print(f"缺失库：{e}")
    sys.exit(1)


class MultiModelSimilarityDetector:
    """多模型图像相似度检测器"""

    def __init__(self,
                 enable_clip: bool = True,
                 enable_resnet: bool = True,
                 enable_vit: bool = True,
                 enable_traditional: bool = True,
                 enable_sift: bool = True):
        """
        初始化多模型检测器

        Args:
            enable_clip: 是否启用CLIP模型
            enable_resnet: 是否启用ResNet特征
            enable_vit: 是否启用Vision Transformer
            enable_traditional: 是否启用传统CV特征
            enable_sift: 是否启用SIFT特征匹配
        """
        self.models = {}
        self.feature_extractors = {}

        print("正在初始化多模型检测器...")

        # 1. CLIP模型 - 语义相似度
        if enable_clip:
            try:
                print("加载CLIP模型...")
                # 尝试不同的CLIP模型
                clip_models = [
                    "openai/clip-vit-base-patch32",
                    "openai/clip-vit-large-patch14",
                    "openai/clip-vit-base-patch16"
                ]

                for model_name in clip_models:
                    try:
                        self.models['clip'] = ClipSimilarity(model_name_or_path=model_name)
                        print(f"CLIP模型加载成功: {model_name}")
                        break
                    except:
                        continue

                if 'clip' not in self.models:
                    print("警告: CLIP模型加载失败")
            except Exception as e:
                print(f"CLIP模型初始化失败: {e}")

        # 2. ResNet特征提取器 - 深度卷积特征
        if enable_resnet:
            try:
                print("加载ResNet模型...")
                resnet = models.resnet50(pretrained=True)
                resnet.fc = torch.nn.Identity()  # 移除最后的分类层
                resnet.eval()
                self.models['resnet'] = resnet

                self.resnet_transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                ])
                print("ResNet模型加载成功")
            except Exception as e:
                print(f"ResNet模型加载失败: {e}")

        # 3. Vision Transformer - 注意力机制特征
        if enable_vit:
            try:
                print("加载Vision Transformer...")
                # 使用timm库的预训练ViT
                vit_model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
                vit_model.eval()
                self.models['vit'] = vit_model

                self.vit_transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])
                print("ViT模型加载成功")
            except Exception as e:
                print(f"ViT模型加载失败: {e}")

        # 4. 传统计算机视觉特征
        if enable_traditional:
            try:
                print("初始化传统CV特征提取器...")
                # 颜色直方图、纹理特征等
                self.traditional_enabled = True
                print("传统CV特征提取器初始化成功")
            except Exception as e:
                print(f"传统CV特征初始化失败: {e}")
                self.traditional_enabled = False

        # 5. SIFT特征匹配
        if enable_sift:
            try:
                print("初始化SIFT特征检测器...")
                self.sift = cv2.SIFT_create()
                self.sift_enabled = True
                print("SIFT特征检测器初始化成功")
            except Exception as e:
                print(f"SIFT初始化失败: {e}")
                self.sift_enabled = False

        print("多模型检测器初始化完成")

    def extract_clip_features(self, image_path: str) -> Optional[np.ndarray]:
        """提取CLIP特征"""
        if 'clip' not in self.models:
            return None

        try:
            # 由于similarities库的限制，我们通过计算与固定参考的相似度来近似特征
            # 这里返回图片路径，在计算相似度时直接使用
            return image_path
        except Exception as e:
            print(f"CLIP特征提取失败: {e}")
            return None

    def extract_resnet_features(self, image_path: str) -> Optional[np.ndarray]:
        """提取ResNet特征"""
        if 'resnet' not in self.models:
            return None

        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.resnet_transform(image).unsqueeze(0)

            with torch.no_grad():
                features = self.models['resnet'](image_tensor)
                features = features.squeeze().numpy()
                features = features / np.linalg.norm(features)  # L2归一化

            return features
        except Exception as e:
            print(f"ResNet特征提取失败: {e}")
            return None

    def extract_vit_features(self, image_path: str) -> Optional[np.ndarray]:
        """提取Vision Transformer特征"""
        if 'vit' not in self.models:
            return None

        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.vit_transform(image).unsqueeze(0)

            with torch.no_grad():
                features = self.models['vit'](image_tensor)
                features = features.squeeze().numpy()
                features = features / np.linalg.norm(features)  # L2归一化

            return features
        except Exception as e:
            print(f"ViT特征提取失败: {e}")
            return None

    def extract_traditional_features(self, image_path: str) -> Optional[np.ndarray]:
        """提取传统计算机视觉特征"""
        if not hasattr(self, 'traditional_enabled') or not self.traditional_enabled:
            return None

        try:
            image = cv2.imread(image_path)
            if image is None:
                return None

            features = []

            # 1. 颜色直方图特征
            hist_b = cv2.calcHist([image], [0], None, [32], [0, 256])
            hist_g = cv2.calcHist([image], [1], None, [32], [0, 256])
            hist_r = cv2.calcHist([image], [2], None, [32], [0, 256])

            features.extend(hist_b.flatten())
            features.extend(hist_g.flatten())
            features.extend(hist_r.flatten())

            # 2. LBP纹理特征
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            lbp_hist = self._calculate_lbp_histogram(gray)
            features.extend(lbp_hist)

            # 3. HOG特征
            hog_features = self._calculate_hog_features(gray)
            features.extend(hog_features)

            features = np.array(features)
            features = features / np.linalg.norm(features)  # L2归一化

            return features
        except Exception as e:
            print(f"传统特征提取失败: {e}")
            return None

    def _calculate_lbp_histogram(self, gray_image: np.ndarray, radius: int = 1, n_points: int = 8) -> List[float]:
        """计算LBP直方图"""
        try:
            # LBP实现
            h, w = gray_image.shape
            lbp = np.zeros((h - 2 * radius, w - 2 * radius), dtype=np.uint8)

            for i in range(radius, h - radius):
                for j in range(radius, w - radius):
                    center = gray_image[i, j]
                    binary = 0
                    for k in range(n_points):
                        angle = 2 * np.pi * k / n_points
                        x = int(round(i + radius * np.cos(angle)))
                        y = int(round(j + radius * np.sin(angle)))
                        if gray_image[x, y] >= center:
                            binary += 2 ** k
                    lbp[i - radius, j - radius] = binary

            # 计算直方图
            hist, _ = np.histogram(lbp.ravel(), bins=2 ** n_points, range=(0, 2 ** n_points))
            hist = hist.astype(float)
            hist = hist / (hist.sum() + 1e-7)  # 归一化

            return hist.tolist()
        except:
            return [0.0] * (2 ** n_points)

    def _calculate_hog_features(self, gray_image: np.ndarray) -> List[float]:
        """计算HOG特征"""
        try:
            # 简化的HOG实现
            resized = cv2.resize(gray_image, (64, 128))

            # 计算梯度
            gx = cv2.Sobel(resized, cv2.CV_32F, 1, 0, ksize=1)
            gy = cv2.Sobel(resized, cv2.CV_32F, 0, 1, ksize=1)

            magnitude = np.sqrt(gx ** 2 + gy ** 2)
            orientation = np.arctan2(gy, gx)

            # 简化的HOG描述符
            hog_features = []
            for i in range(0, resized.shape[0], 8):
                for j in range(0, resized.shape[1], 8):
                    cell_mag = magnitude[i:i + 8, j:j + 8]
                    cell_ori = orientation[i:i + 8, j:j + 8]

                    # 计算方向直方图
                    hist, _ = np.histogram(cell_ori.ravel(), bins=9,
                                           range=(-np.pi, np.pi),
                                           weights=cell_mag.ravel())
                    hog_features.extend(hist)

            return hog_features[:100]  # 限制特征长度
        except:
            return [0.0] * 100

    def extract_sift_features(self, image_path: str) -> Optional[np.ndarray]:
        """提取SIFT特征"""
        if not hasattr(self, 'sift_enabled') or not self.sift_enabled:
            return None

        try:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                return None

            keypoints, descriptors = self.sift.detectAndCompute(image, None)

            if descriptors is not None and len(descriptors) > 0:
                # 使用描述符的统计特征作为图像特征
                features = []
                features.extend(np.mean(descriptors, axis=0))  # 均值
                features.extend(np.std(descriptors, axis=0))  # 标准差
                features.extend(np.max(descriptors, axis=0))  # 最大值
                features.extend(np.min(descriptors, axis=0))  # 最小值

                features = np.array(features)
                features = features / np.linalg.norm(features)

                return features
            else:
                return None
        except Exception as e:
            print(f"SIFT特征提取失败: {e}")
            return None

    def calculate_comprehensive_similarity(self, image1_path: str, image2_path: str) -> Dict[str, float]:
        """计算综合相似度"""
        similarities = {}

        # 1. CLIP相似度
        if 'clip' in self.models:
            try:
                clip_sim = self.models['clip'].similarity(image1_path, image2_path)
                similarities['clip'] = float(clip_sim)
            except:
                similarities['clip'] = 0.0

        # 2. ResNet特征相似度
        resnet_feat1 = self.extract_resnet_features(image1_path)
        resnet_feat2 = self.extract_resnet_features(image2_path)
        if resnet_feat1 is not None and resnet_feat2 is not None:
            similarities['resnet'] = float(1 - cosine(resnet_feat1, resnet_feat2))
        else:
            similarities['resnet'] = 0.0

        # 3. ViT特征相似度
        vit_feat1 = self.extract_vit_features(image1_path)
        vit_feat2 = self.extract_vit_features(image2_path)
        if vit_feat1 is not None and vit_feat2 is not None:
            similarities['vit'] = float(1 - cosine(vit_feat1, vit_feat2))
        else:
            similarities['vit'] = 0.0

        # 4. 传统特征相似度
        trad_feat1 = self.extract_traditional_features(image1_path)
        trad_feat2 = self.extract_traditional_features(image2_path)
        if trad_feat1 is not None and trad_feat2 is not None:
            similarities['traditional'] = float(1 - cosine(trad_feat1, trad_feat2))
        else:
            similarities['traditional'] = 0.0

        # 5. SIFT特征相似度
        sift_feat1 = self.extract_sift_features(image1_path)
        sift_feat2 = self.extract_sift_features(image2_path)
        if sift_feat1 is not None and sift_feat2 is not None:
            similarities['sift'] = float(1 - cosine(sift_feat1, sift_feat2))
        else:
            similarities['sift'] = 0.0

        # 6. 加权融合相似度
        # weights = {
        #     'clip': 0.25,  # CLIP权重降低
        #     'resnet': 0.25,  # ResNet特征
        #     'vit': 0.2,  # ViT特征
        #     'traditional': 0.2,  # 传统特征
        #     'sift': 0.1  # SIFT特征
        # }
        weights = {
            'clip': 0,  # CLIP权重降低
            'resnet': 0.1,  # ResNet特征
            'vit': 0.4,  # ViT特征
            'traditional': 0.5,  # 传统特征
            'sift': 0  # SIFT特征
        }

        weighted_sim = 0.0
        total_weight = 0.0

        for method, sim in similarities.items():
            if method in weights and sim > 0:
                weighted_sim += weights[method] * sim
                total_weight += weights[method]

        if total_weight > 0:
            similarities['weighted_fusion'] = weighted_sim / total_weight
        else:
            similarities['weighted_fusion'] = 0.0

        # 7. 自适应融合（根据特征质量动态调整权重）
        adaptive_weights = self._calculate_adaptive_weights(similarities)
        adaptive_sim = sum(adaptive_weights[method] * similarities[method]
                           for method in adaptive_weights.keys())
        similarities['adaptive_fusion'] = adaptive_sim

        return similarities

    def _calculate_adaptive_weights(self, similarities: Dict[str, float]) -> Dict[str, float]:
        """根据特征质量自适应计算权重"""
        base_methods = ['clip', 'resnet', 'vit', 'traditional', 'sift']
        valid_sims = {k: v for k, v in similarities.items() if k in base_methods and v > 0}

        if not valid_sims:
            return {method: 0.0 for method in base_methods}

        # 基于特征区分度的权重调整
        weights = {}

        # 计算每个方法与其他方法的差异度
        for method in valid_sims:
            diff_scores = []
            for other_method in valid_sims:
                if method != other_method:
                    diff_scores.append(abs(valid_sims[method] - valid_sims[other_method]))

            # 差异度越大，权重越高（更有区分性）
            avg_diff = np.mean(diff_scores) if diff_scores else 0
            weights[method] = 0.1 + avg_diff  # 基础权重0.1 + 差异度奖励

        # 归一化权重
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}

        # 为不存在的方法设置0权重
        for method in base_methods:
            if method not in weights:
                weights[method] = 0.0

        return weights

    def find_similar_images(self,
                            target_image: str,
                            candidate_images: List[str],
                            threshold: float = 0.5,
                            method: str = 'adaptive_fusion',
                            max_results: int = None) -> List[Tuple[str, float, Dict]]:
        """查找相似图片"""
        print(f"使用方法: {method}")
        print(f"阈值: {threshold}")

        similar_images = []

        for i, candidate in enumerate(candidate_images):
            if os.path.abspath(candidate) == os.path.abspath(target_image):
                continue

            if (i + 1) % 10 == 0:
                print(f"进度: {i + 1}/{len(candidate_images)}")

            similarities = self.calculate_comprehensive_similarity(target_image, candidate)
            main_similarity = similarities.get(method, 0.0)

            if main_similarity >= threshold:
                similar_images.append((candidate, main_similarity, similarities))
                print(f"  找到相似图片: {os.path.basename(candidate)} ({method}: {main_similarity:.4f})")

        # 按相似度降序排序
        similar_images.sort(key=lambda x: x[1], reverse=True)

        if max_results:
            similar_images = similar_images[:max_results]

        return similar_images

    def get_recommended_thresholds(self) -> Dict[str, Dict[str, float]]:
        """获取推荐阈值"""
        return {
            'clip': {'conservative': 0.95, 'moderate': 0.90, 'liberal': 0.85},
            'resnet': {'conservative': 0.90, 'moderate': 0.80, 'liberal': 0.70},
            'vit': {'conservative': 0.85, 'moderate': 0.75, 'liberal': 0.65},
            'traditional': {'conservative': 0.80, 'moderate': 0.70, 'liberal': 0.60},
            'sift': {'conservative': 0.75, 'moderate': 0.65, 'liberal': 0.55},
            'weighted_fusion': {'conservative': 0.70, 'moderate': 0.55, 'liberal': 0.40},
            'adaptive_fusion': {'conservative': 0.65, 'moderate': 0.50, 'liberal': 0.35}
        }

    def find_images_in_directory(self, directory: str) -> List[str]:
        """查找目录中的图片文件"""
        image_paths = []
        supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']

        for root, dirs, files in os.walk(directory):
            for file in files:
                if any(file.lower().endswith(fmt) for fmt in supported_formats):
                    image_paths.append(os.path.join(root, file))

        return sorted(image_paths)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="多模型图像相似度检测工具")

    parser.add_argument("--target", "-t", type=str, required=True,default='/Users/cloudbai/PycharmProjects/imagesim/similarities/examples/data/shanqiimage/1-1.jpg',
                        help="目标图片路径")
    parser.add_argument("--directory", "-d", type=str, required=True,default='/Users/cloudbai/PycharmProjects/imagesim/similarities/examples/data/shanqiimage',
                        help="图片目录路径")
    parser.add_argument("--threshold", "-th", type=float, default=0.5,
                        help="相似度阈值")
    parser.add_argument("--method", "-m", type=str, default='weighted_fusion',
                        choices=['clip', 'resnet', 'vit', 'traditional', 'sift',
                                 'weighted_fusion', 'adaptive_fusion'],
                        help="相似度计算方法")
    parser.add_argument("--max-results", type=int, default=10,
                        help="最大返回结果数量")
    parser.add_argument("--disable-clip", action="store_true",
                        help="禁用CLIP模型")
    parser.add_argument("--disable-resnet", action="store_true",
                        help="禁用ResNet特征")
    parser.add_argument("--disable-vit", action="store_true",
                        help="禁用ViT特征")
    parser.add_argument("--disable-traditional", action="store_true",
                        help="禁用传统CV特征")
    parser.add_argument("--disable-sift", action="store_true",
                        help="禁用SIFT特征")

    args = parser.parse_args()

    # 检查输入
    if not os.path.exists(args.target):
        print(f"错误：目标图片不存在 {args.target}")
        return

    if not os.path.exists(args.directory):
        print(f"错误：目录不存在 {args.directory}")
        return

    # 初始化检测器
    detector = MultiModelSimilarityDetector(
        enable_clip=not args.disable_clip,
        enable_resnet=not args.disable_resnet,
        enable_vit=not args.disable_vit,
        enable_traditional=not args.disable_traditional,
        enable_sift=not args.disable_sift
    )

    # 查找候选图片
    candidate_images = detector.find_images_in_directory(args.directory)
    print(f"找到 {len(candidate_images)} 张候选图片")

    # 检测相似图片
    similar_images = detector.find_similar_images(
        target_image=args.target,
        candidate_images=candidate_images,
        threshold=args.threshold,
        method=args.method,
        max_results=args.max_results
    )

    # 打印结果
    print(f"\n检测完成！找到 {len(similar_images)} 张相似图片:")
    print("=" * 80)

    for i, (image_path, score, details) in enumerate(similar_images, 1):
        print(f"{i}. {os.path.basename(image_path)}")
        print(f"   主相似度 ({args.method}): {score:.4f}")
        print(f"   详细分数: {details}")
        print("-" * 60)

    # 显示推荐阈值
    print(f"\n推荐阈值 ({args.method}):")
    thresholds = detector.get_recommended_thresholds()
    if args.method in thresholds:
        method_thresholds = thresholds[args.method]
        print(f"  保守: {method_thresholds['conservative']:.3f}")
        print(f"  适中: {method_thresholds['moderate']:.3f}")
        print(f"  宽松: {method_thresholds['liberal']:.3f}")


if __name__ == "__main__":
    main()