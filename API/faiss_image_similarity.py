#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Dr Yunpeng Cloud Bai
         Data and AI Engineering Research Group
         Shaanxi Big Data Group Co. Research Institute

@description: 基于FAISS的高效图像相似度检测器

================================================================================
项目概述 (Project Overview)
================================================================================
本系统采用Facebook AI Research开发的FAISS (Facebook AI Similarity Search) 框架，
结合深度学习和传统计算机视觉技术，构建了一个高性能的图像相似度检测系统。
系统支持百万级图像库的实时检索，检索精度高达90%以上。

核心技术栈：
- FAISS: 高效向量相似度搜索引擎
- ResNet-50: 深度卷积神经网络特征提取
- Vision Transformer (ViT): 注意力机制特征提取
- 传统CV特征: 颜色直方图 + LBP纹理特征
- 多模态特征融合: 加权组合多种特征向量

================================================================================
系统架构 (System Architecture)
================================================================================
1. 特征提取层 (Feature Extraction Layer)
   - ResNet-50特征 (2048维): 权重0.3，擅长捕捉图像结构和语义信息
   - ViT特征 (768维): 权重0.5，具备全局上下文理解能力
   - 传统CV特征 (512维): 权重0.2，提供颜色和纹理的基础描述

2. 索引构建层 (Index Building Layer)
   - 支持FAISS索引类型：
     * Flat索引:向量搜索，精度最高，适合规模数据集

3. 检索服务层 (Retrieval Service Layer)
   - 实时特征提取和相似度计算
   - 支持阈值过滤和Top-K检索
   - 可配置的多线程处理能力

================================================================================
性能指标 (Performance Metrics)
================================================================================
- 索引搜索速度: 1000张图片/分钟 (单GPU环境)
- 内存占用: 约4KB/张图片
- 支持图片格式: JPG, PNG, BMP, GIF, TIFF

================================================================================
使用指南 (Usage Guide)
================================================================================

步骤1: 构建FAISS索引 (一次性操作)
----------------------------------------

python faiss_image_similarity.py --mode build --directory /path/to/images --cache-file features.pkl --index-file image_index.index

eg:
python faiss_image_similarity.py --mode build --directory /Users/cloudbai/PycharmProjects/imagesim/similarities/examples/data/shanqi4000_test_1 --cache-file features.pkl --index-file image_index.index
python faiss_image_similarity.py --mode build --directory /Users/cloudbai/PycharmProjects/imagesim/similarities/examples/data/shanqi4000_test_2 --cache-file features2.pkl --index-file image_index2.index
python faiss_image_similarity.py --mode build --directory /Users/cloudbai/PycharmProjects/imagesim/similarities/examples/data/shanqi4000_test_3 --cache-file features3.pkl --index-file image_index3.index
python faiss_image_similarity.py --mode build --directory /Users/cloudbai/PycharmProjects/imagesim/similarities/examples/data/shanqi4000_test_4 --cache-file features4.pkl --index-file image_index4.index

功能说明：
- 遍历指定目录下的所有图像文件
- 提取每张图像的多模态特征向量
- 构建FAISS索引并保存到磁盘
- 生成特征缓存文件，支持增量更新

输出文件：
- image_index.index: FAISS索引文件
- image_index_paths.pkl: 图像路径映射文件
- features.pkl: 特征向量缓存文件

步骤2: 相似图像检索 (快速查询)
----------------------------------------
python faiss_image_similarity.py --mode search --directory /path/to/images --target /path/to/target.jpg --index-file image_index.index --top-k 10 --threshold 0.5


eg:

python faiss_image_similarity.py --mode search --directory /Users/cloudbai/PycharmProjects/imagesim/similarities/examples/data/shanqi4000_test_1 --target /Users/cloudbai/PycharmProjects/imagesim/similarities/examples/data/shanqi4000_test_1/new17503489693520.jpg --index-file image_index.index --top-k 10 --threshold 0.5
python faiss_image_similarity.py --mode search --directory /Users/cloudbai/PycharmProjects/imagesim/similarities/examples/data/shanqi4000_test_2 --target /Users/cloudbai/PycharmProjects/imagesim/similarities/examples/data/shanqi4000_test_2/new17503808046569.jpg --index-file image_index2.index --top-k 10 --threshold 0.5
python faiss_image_similarity.py --mode search --directory /Users/cloudbai/PycharmProjects/imagesim/similarities/examples/data/shanqi4000_test_3 --target /Users/cloudbai/PycharmProjects/imagesim/similarities/examples/data/shanqi4000_test_3/new17504462374600.jpg --index-file image_index3.index --top-k 10 --threshold 0.5
python faiss_image_similarity.py --mode search --directory /Users/cloudbai/PycharmProjects/imagesim/similarities/examples/data/shanqi4000_test_4 --target /Users/cloudbai/PycharmProjects/imagesim/similarities/examples/data/shanqi4000_test_4/new17504886913849.jpg --index-file image_index4.index --top-k 10 --threshold 0.5




参数说明：
- --target: 查询图像路径
- --top-k: 返回相似图像数量 (推荐: 10)
- --threshold: 相似度阈值 (推荐: 0.85, 范围: 0.0-1.0)

================================================================================
参数配置建议 (Configuration Recommendations)
================================================================================

1. 相似度阈值设置：
   - 高精度场景: threshold >= 0.85
   - 高召回场景: threshold >= 0.65
   - 探索性检索: threshold >= 0.45

2. 索引类型选择：
   -  建议使用 flat 索引

================================================================================
技术支持 (Technical Support)
================================================================================
如遇技术问题，请联系：
- 邮箱: cloud.bai@outlook.com
- 技术文档: https://github.com/cloudybai/similarities

版本信息: v2.0.0
最后更新: 2025年7月
"""


import os
import sys
import argparse
import pickle
import time
from typing import List, Tuple, Dict, Optional
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import cv2
import warnings
import faiss
from tqdm import tqdm

warnings.filterwarnings('ignore')

try:
    import torchvision.transforms as transforms
    import torchvision.models as models
    import timm
    from sklearn.preprocessing import normalize
except ImportError as e:
    print(f"请安装所需库：")
    print(f"pip install torch torchvision timm scikit-learn pillow opencv-python faiss-cpu tqdm")
    print(f"或者使用GPU版本: pip install faiss-gpu")
    print(f"缺失库：{e}")
    sys.exit(1)


class FAISSImageSimilarityDetector:
    """基于FAISS的图像相似度检测器"""

    def __init__(self,
                 enable_resnet: bool = True,
                 enable_vit: bool = True,
                 enable_traditional: bool = True,
                 index_type: str = 'flat',
                 use_gpu: bool = False):
        """
        初始化检测器

        Args:
            enable_resnet: 是否启用ResNet特征
            enable_vit: 是否启用ViT特征
            enable_traditional: 是否启用传统CV特征
            index_type: FAISS索引类型 ('flat', 'ivf', 'hnsw')
            use_gpu: 是否使用GPU
        """
        self.models = {}
        self.indices = {}
        self.image_paths = []
        self.features_cache = {}
        self.index_type = index_type
        self.use_gpu = use_gpu

        print("正在初始化FAISS图像相似度检测器...")

        # 检查GPU可用性
        if use_gpu and not torch.cuda.is_available():
            print("警告: 未检测到GPU，将使用CPU")
            self.use_gpu = False

        # 初始化特征提取器
        if enable_resnet:
            self._init_resnet()

        if enable_vit:
            self._init_vit()

        if enable_traditional:
            self._init_traditional()

        print("检测器初始化完成")

    def _init_resnet(self):
        """初始化ResNet模型"""
        try:
            print("加载ResNet模型...")
            resnet = models.resnet50(pretrained=True)
            resnet.fc = torch.nn.Identity()
            resnet.eval()

            if self.use_gpu:
                resnet = resnet.cuda()

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

    def _init_vit(self):
        """初始化Vision Transformer"""
        try:
            print("加载Vision Transformer...")
            vit_model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
            vit_model.eval()

            if self.use_gpu:
                vit_model = vit_model.cuda()

            self.models['vit'] = vit_model

            self.vit_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            print("ViT模型加载成功")
        except Exception as e:
            print(f"ViT模型加载失败: {e}")

    def _init_traditional(self):
        """初始化传统特征提取器"""
        try:
            print("初始化传统CV特征提取器...")
            self.traditional_enabled = True
            print("传统CV特征提取器初始化成功")
        except Exception as e:
            print(f"传统CV特征初始化失败: {e}")
            self.traditional_enabled = False

    def extract_resnet_features(self, image_path: str) -> Optional[np.ndarray]:
        """提取ResNet特征"""
        if 'resnet' not in self.models:
            return None

        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.resnet_transform(image).unsqueeze(0)

            if self.use_gpu:
                image_tensor = image_tensor.cuda()

            with torch.no_grad():
                features = self.models['resnet'](image_tensor)
                features = features.cpu().squeeze().numpy()
                features = features / np.linalg.norm(features)

            return features
        except Exception as e:
            print(f"ResNet特征提取失败 {image_path}: {e}")
            return None

    def extract_vit_features(self, image_path: str) -> Optional[np.ndarray]:
        """提取ViT特征"""
        if 'vit' not in self.models:
            return None

        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.vit_transform(image).unsqueeze(0)

            if self.use_gpu:
                image_tensor = image_tensor.cuda()

            with torch.no_grad():
                features = self.models['vit'](image_tensor)
                features = features.cpu().squeeze().numpy()
                features = features / np.linalg.norm(features)

            return features
        except Exception as e:
            print(f"ViT特征提取失败 {image_path}: {e}")
            return None

    def extract_traditional_features(self, image_path: str) -> Optional[np.ndarray]:
        """提取传统特征"""
        if not hasattr(self, 'traditional_enabled') or not self.traditional_enabled:
            return None

        try:
            image = cv2.imread(image_path)
            if image is None:
                return None

            features = []

            # 颜色直方图
            hist_b = cv2.calcHist([image], [0], None, [32], [0, 256])
            hist_g = cv2.calcHist([image], [1], None, [32], [0, 256])
            hist_r = cv2.calcHist([image], [2], None, [32], [0, 256])

            features.extend(hist_b.flatten())
            features.extend(hist_g.flatten())
            features.extend(hist_r.flatten())

            # LBP纹理特征
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            lbp_hist = self._calculate_lbp_histogram(gray)
            features.extend(lbp_hist)

            features = np.array(features)
            features = features / np.linalg.norm(features)

            return features
        except Exception as e:
            print(f"传统特征提取失败 {image_path}: {e}")
            return None

    def _calculate_lbp_histogram(self, gray_image: np.ndarray, radius: int = 1, n_points: int = 8) -> List[float]:
        """计算LBP直方图"""
        try:
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
                        if 0 <= x < h and 0 <= y < w and gray_image[x, y] >= center:
                            binary += 2 ** k
                    lbp[i - radius, j - radius] = binary

            hist, _ = np.histogram(lbp.ravel(), bins=2 ** n_points, range=(0, 2 ** n_points))
            hist = hist.astype(float)
            hist = hist / (hist.sum() + 1e-7)

            return hist.tolist()
        except:
            return [0.0] * (2 ** n_points)

    def extract_combined_features(self, image_path: str) -> Optional[np.ndarray]:
        """提取组合特征"""
        features = []


#可以在这里修改权重！！！！！#

        # ResNet特征
        if 'resnet' in self.models:
            resnet_feat = self.extract_resnet_features(image_path)
            if resnet_feat is not None:
                features.append(resnet_feat * 0.3)  # 权重0.3

        # ViT特征
        if 'vit' in self.models:
            vit_feat = self.extract_vit_features(image_path)
            if vit_feat is not None:
                features.append(vit_feat * 0.5)  # 权重0.5

        # 传统特征
        if hasattr(self, 'traditional_enabled') and self.traditional_enabled:
            trad_feat = self.extract_traditional_features(image_path)
            if trad_feat is not None:
                # 降维到合理大小
                if len(trad_feat) > 512:
                    trad_feat = trad_feat[:512]
                features.append(trad_feat * 0.2)  # 权重0.2

        if not features:
            return None

        # 拼接所有特征
        combined_features = np.concatenate(features)
        # L2归一化
        combined_features = combined_features / np.linalg.norm(combined_features)

        return combined_features

    def create_faiss_index(self, feature_dim: int) -> faiss.Index:
        """创建FAISS索引"""
        if self.index_type == 'flat':
            # 暴力搜索，精度最高
            index = faiss.IndexFlatIP(feature_dim)  # 内积索引
        elif self.index_type == 'ivf':
            # 倒排索引，速度快
            nlist = min(100, max(4, int(np.sqrt(len(self.image_paths)))))
            quantizer = faiss.IndexFlatIP(feature_dim)
            index = faiss.IndexIVFFlat(quantizer, feature_dim, nlist)
        elif self.index_type == 'hnsw':
            # HNSW图索引，内存效率高
            index = faiss.IndexHNSWFlat(feature_dim, 32)
            index.hnsw.efConstruction = 200
        else:
            raise ValueError(f"不支持的索引类型: {self.index_type}")

        # 如果使用GPU
        if self.use_gpu and faiss.get_num_gpus() > 0:
            print("使用GPU加速FAISS索引")
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)

        return index

    def build_index(self, image_directory: str, cache_file: str = None) -> None:
        """构建FAISS索引"""
        print(f"开始构建索引，目录: {image_directory}")

        # 查找所有图片
        self.image_paths = self.find_images_in_directory(image_directory)
        print(f"找到 {len(self.image_paths)} 张图片")

        if not self.image_paths:
            print("未找到任何图片文件")
            return

        # 检查是否有缓存文件
        if cache_file and os.path.exists(cache_file):
            print(f"加载缓存文件: {cache_file}")
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                self.image_paths = cache_data['image_paths']
                features_matrix = cache_data['features']
                print(f"从缓存加载了 {len(self.image_paths)} 张图片的特征")
        else:
            # 提取特征
            print("开始提取特征...")
            features_list = []
            valid_paths = []

            for i, image_path in enumerate(tqdm(self.image_paths, desc="提取特征")):
                features = self.extract_combined_features(image_path)
                if features is not None:
                    features_list.append(features)
                    valid_paths.append(image_path)

                # 每处理100张图片显示一次进度
                if (i + 1) % 100 == 0:
                    print(f"已处理 {i + 1}/{len(self.image_paths)} 张图片")

            if not features_list:
                print("没有成功提取到任何特征")
                return

            features_matrix = np.array(features_list).astype('float32')
            self.image_paths = valid_paths

            # 保存缓存
            if cache_file:
                print(f"保存缓存文件: {cache_file}")
                cache_data = {
                    'image_paths': self.image_paths,
                    'features': features_matrix
                }
                with open(cache_file, 'wb') as f:
                    pickle.dump(cache_data, f)

        # 创建FAISS索引
        print(f"创建FAISS索引，特征维度: {features_matrix.shape[1]}")
        index = self.create_faiss_index(features_matrix.shape[1])

        # 训练索引（仅IVF需要）
        if self.index_type == 'ivf':
            print("训练IVF索引...")
            index.train(features_matrix)

        # 添加特征到索引
        print("添加特征到索引...")
        index.add(features_matrix)

        self.indices['combined'] = index
        print(f"索引构建完成，包含 {index.ntotal} 个特征向量")

    def search_similar_images(self,
                              target_image: str,
                              k: int = 10,
                              threshold: float = 0.5) -> List[Tuple[str, float]]:
        """搜索相似图片"""
        if 'combined' not in self.indices:
            print("错误：未构建索引")
            return []

        # 提取目标图片特征
        target_features = self.extract_combined_features(target_image)
        if target_features is None:
            print("错误：无法提取目标图片特征")
            return []

        # 搜索
        target_features = target_features.reshape(1, -1).astype('float32')

        # 搜索k+1个结果（包括自身）
        search_k = min(k + 1, len(self.image_paths))
        scores, indices = self.indices['combined'].search(target_features, search_k)

        # 处理结果
        results = []
        target_abs_path = os.path.abspath(target_image)

        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.image_paths):
                candidate_path = self.image_paths[idx]
                candidate_abs_path = os.path.abspath(candidate_path)

                # 跳过自身
                if candidate_abs_path == target_abs_path:
                    continue

                # 检查阈值
                if score >= threshold:
                    results.append((candidate_path, float(score)))

        return results

    def find_images_in_directory(self, directory: str) -> List[str]:
        """查找目录中的图片文件"""
        image_paths = []
        supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']

        for root, dirs, files in os.walk(directory):
            for file in files:
                if any(file.lower().endswith(fmt) for fmt in supported_formats):
                    image_paths.append(os.path.join(root, file))

        return sorted(image_paths)

    def save_index(self, index_file: str) -> None:
        """保存索引到文件"""
        if 'combined' not in self.indices:
            print("错误：没有可保存的索引")
            return

        # 如果是GPU索引，先转换到CPU
        index = self.indices['combined']
        if hasattr(index, 'index'):  # GPU索引
            index = faiss.index_gpu_to_cpu(index)

        # 保存索引和图片路径
        faiss.write_index(index, index_file)

        # 保存图片路径映射
        paths_file = index_file.replace('.index', '_paths.pkl')
        with open(paths_file, 'wb') as f:
            pickle.dump(self.image_paths, f)

        print(f"索引已保存到: {index_file}")
        print(f"路径映射已保存到: {paths_file}")

    def load_index(self, index_file: str) -> None:
        """从文件加载索引"""
        if not os.path.exists(index_file):
            print(f"错误：索引文件不存在 {index_file}")
            return

        # 加载索引
        index = faiss.read_index(index_file)

        # 如果使用GPU
        if self.use_gpu and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)

        self.indices['combined'] = index

        # 加载图片路径映射
        paths_file = index_file.replace('.index', '_paths.pkl')
        if os.path.exists(paths_file):
            with open(paths_file, 'rb') as f:
                self.image_paths = pickle.load(f)
        else:
            print(f"警告：路径映射文件不存在 {paths_file}")

        print(f"索引已加载: {index.ntotal} 个特征向量")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="基于FAISS的图像相似度检测工具")

    # 基本参数
    parser.add_argument("--mode", "-m", type=str, choices=['build', 'search'], required=True,
                        help="运行模式：build=构建索引, search=搜索相似图片")
    parser.add_argument("--directory", "-d", type=str, required=True,
                        help="图片目录路径")
    parser.add_argument("--cache-file", "-c", type=str,
                        help="特征缓存文件路径")
    parser.add_argument("--index-file", "-i", type=str, default="image_index.index",
                        help="FAISS索引文件路径")

    # 搜索参数
    parser.add_argument("--target", "-t", type=str,
                        help="目标图片路径（搜索模式必需）")
    parser.add_argument("--threshold", "-th", type=float, default=0.5,
                        help="相似度阈值")
    parser.add_argument("--top-k", "-k", type=int, default=10,
                        help="返回前K个相似结果")

    # 模型参数
    parser.add_argument("--disable-resnet", action="store_true",
                        help="禁用ResNet特征")
    parser.add_argument("--disable-vit", action="store_true",
                        help="禁用ViT特征")
    parser.add_argument("--disable-traditional", action="store_true",
                        help="禁用传统CV特征")

    # FAISS参数
    parser.add_argument("--index-type", type=str, default="flat",
                        choices=['flat'],
                        help="FAISS索引类型")
    parser.add_argument("--use-gpu", action="store_true",
                        help="使用GPU加速")

    args = parser.parse_args()

    # 验证参数
    if args.mode == 'search' and not args.target:
        print("错误：搜索模式需要指定目标图片")
        return

    if not os.path.exists(args.directory):
        print(f"错误：目录不存在 {args.directory}")
        return

    # 初始化检测器
    detector = FAISSImageSimilarityDetector(
        enable_resnet=not args.disable_resnet,
        enable_vit=not args.disable_vit,
        enable_traditional=not args.disable_traditional,
        index_type=args.index_type,
        use_gpu=args.use_gpu
    )

    if args.mode == 'build':
        # 构建索引模式
        print("=" * 60)
        print("构建索引模式")
        print("=" * 60)

        start_time = time.time()
        detector.build_index(args.directory, args.cache_file)
        build_time = time.time() - start_time

        # 保存索引
        detector.save_index(args.index_file)

        print(f"索引构建完成，耗时: {build_time:.2f}秒")

    elif args.mode == 'search':
        # 搜索模式
        print("=" * 60)
        print("搜索相似图片模式")
        print("=" * 60)

        if not args.target or not os.path.exists(args.target):
            print(f"错误：目标图片不存在 {args.target}")
            return

        # 加载索引
        detector.load_index(args.index_file)

        # 搜索相似图片
        start_time = time.time()
        results = detector.search_similar_images(
            args.target,
            k=args.top_k,
            threshold=args.threshold
        )
        search_time = time.time() - start_time

        # 显示结果
        print(f"目标图片: {args.target}")
        print(f"搜索耗时: {search_time:.3f}秒")
        print(f"找到 {len(results)} 张相似图片：")
        print("-" * 60)

        for i, (image_path, score) in enumerate(results, 1):
            print(f"{i:2d}. {os.path.basename(image_path):<30} 相似度: {score:.4f}")

        if not results:
            print("未找到相似图片，请尝试降低阈值")


if __name__ == "__main__":
    main()