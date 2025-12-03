#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
改进版FAISS图像相似度检测器 - 支持增量更新（只增不减版本）
基于原版本优化，添加了增量索引构建功能，在增量更新时只处理新增和修改的图片，不删除已有索引
"""

import os
import sys
import argparse
import pickle
import time
import hashlib
from typing import List, Tuple, Dict, Optional, Set
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import cv2
import warnings
import faiss
from tqdm import tqdm
from pathlib import Path

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


def get_file_hash(file_path: str) -> str:
    """获取文件的MD5哈希值"""
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except:
        return ""


class IncrementalFAISSDetector:
    """支持增量更新的FAISS图像相似度检测器（只增不减版本）"""

    def __init__(self,
                 enable_resnet: bool = True,
                 enable_vit: bool = True,
                 enable_traditional: bool = True,
                 index_type: str = 'flat',
                 use_gpu: bool = False,
                 preserve_deleted: bool = True):
        """
        初始化检测器
        
        Args:
            preserve_deleted: 是否保留已删除文件的索引记录（默认True，实现只增不减）
        """
        self.models = {}
        self.indices = {}
        self.image_paths = []
        self.image_hashes = {}  # 存储图片文件的哈希值
        self.features_cache = {}  # 特征缓存
        self.deleted_paths = set()  # 记录已删除的图片路径（但保留在索引中）
        self.preserve_deleted = preserve_deleted  # 新增：是否保留已删除文件的索引
        self.index_type = index_type
        self.use_gpu = use_gpu
        self.metadata = {
            'version': '3.0.0',  # 版本号更新
            'created_time': None,
            'last_updated': None,
            'total_images': 0,
            'active_images': 0,  # 新增：当前存在的图片数量
            'deleted_images': 0,  # 新增：已删除但保留在索引中的图片数量
            'feature_dim': None,
            'preserve_deleted': preserve_deleted
        }

        print("正在初始化增量FAISS图像相似度检测器（只增不减版本）...")

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

    # def _init_traditional(self):
    #     """初始化传统特征提取器"""
    #     try:
    #         print("初始化传统CV特征提取器...")
    #         self.traditional_enabled = True
    #         print("传统CV特征提取器初始化成功")
    #     except Exception as e:
    #         print(f"传统CV特征初始化失败: {e}")
    #         self.traditional_enabled = False

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

    def find_images_in_directory(self, directory: str) -> List[str]:
        """查找目录中的图片文件"""
        image_paths = []
        supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']
        for root, dirs, files in os.walk(directory):
            for file in files:
                if any(file.lower().endswith(fmt) for fmt in supported_formats):
                    image_paths.append(os.path.join(root, file))
        return sorted(image_paths)

    def get_new_and_modified_images(self, current_images: List[str]) -> Tuple[List[str], List[str]]:
        """获取新增和修改的图片"""
        new_images = []
        modified_images = []

        for img_path in current_images:
            if img_path not in self.image_hashes:
                # 新图片
                new_images.append(img_path)
            else:
                # 检查是否修改
                current_hash = get_file_hash(img_path)
                if current_hash != self.image_hashes[img_path]:
                    modified_images.append(img_path)

        return new_images, modified_images

    def get_missing_images(self, current_images: List[str]) -> List[str]:
        """获取当前目录中缺失但在索引中存在的图片（用于标记为已删除）"""
        current_set = set(current_images)
        missing_images = []

        for img_path in self.image_paths:
            if img_path not in current_set and img_path not in self.deleted_paths:
                missing_images.append(img_path)

        return missing_images

    def create_faiss_index(self, feature_dim: int) -> faiss.Index:
        """创建FAISS索引"""
        if self.index_type == 'flat':
            index = faiss.IndexFlatIP(feature_dim)  # 内积索引
        else:
            raise ValueError(f"不支持的索引类型: {self.index_type}")

        # 如果使用GPU
        if self.use_gpu and faiss.get_num_gpus() > 0:
            print("使用GPU加速FAISS索引")
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)

        return index

    def build_or_update_index(self, image_directory: str, cache_file: str = None, force_rebuild: bool = False) -> None:
        """构建或更新FAISS索引（支持增量更新，只增不减）"""
        print(f"开始构建/更新索引，目录: {image_directory}")
        print(f"保留已删除文件模式: {'开启' if self.preserve_deleted else '关闭'}")

        # 查找当前所有图片
        current_images = self.find_images_in_directory(image_directory)
        print(f"当前目录包含 {len(current_images)} 张图片")

        if not current_images and not self.image_paths:
            print("未找到任何图片文件")
            return

        # 加载已有的缓存和索引
        existing_data_loaded = False
        if cache_file and os.path.exists(cache_file) and not force_rebuild:
            try:
                print(f"加载现有缓存文件: {cache_file}")
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.image_paths = cache_data.get('image_paths', [])
                    self.image_hashes = cache_data.get('image_hashes', {})
                    self.features_cache = cache_data.get('features_cache', {})
                    self.deleted_paths = set(cache_data.get('deleted_paths', []))
                    self.metadata = cache_data.get('metadata', self.metadata)
                    existing_data_loaded = True
                    print(f"从缓存加载了 {len(self.image_paths)} 张图片的数据")
                    print(f"其中已标记删除: {len(self.deleted_paths)} 张")
            except Exception as e:
                print(f"加载缓存失败: {e}，将重新构建")
                force_rebuild = True

        if force_rebuild or not existing_data_loaded:
            print("执行完整索引构建...")
            self._build_full_index(current_images, cache_file)
        else:
            print("执行增量更新（只增不减模式）...")
            self._update_index_incrementally(current_images, cache_file)

    def _build_full_index(self, current_images: List[str], cache_file: str = None) -> None:
        """构建完整索引"""
        print("开始完整索引构建...")

        # 重置所有数据
        self.image_paths = []
        self.image_hashes = {}
        self.features_cache = {}
        self.deleted_paths = set()

        # 提取所有特征
        features_list = []
        valid_paths = []

        for img_path in tqdm(current_images, desc="提取特征"):
            features = self.extract_combined_features(img_path)
            if features is not None:
                features_list.append(features)
                valid_paths.append(img_path)
                # 缓存特征和哈希
                self.features_cache[img_path] = features
                self.image_hashes[img_path] = get_file_hash(img_path)

        if not features_list:
            print("没有成功提取到任何特征")
            return

        features_matrix = np.array(features_list).astype('float32')
        self.image_paths = valid_paths

        # 创建索引
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

        # 更新元数据
        self.metadata.update({
            'created_time': time.time(),
            'last_updated': time.time(),
            'total_images': len(self.image_paths),
            'active_images': len(self.image_paths),
            'deleted_images': 0,
            'feature_dim': features_matrix.shape[1]
        })

        # 保存缓存
        if cache_file:
            self._save_cache(cache_file)

        print(f"完整索引构建完成，包含 {index.ntotal} 个特征向量")

    def _update_index_incrementally(self, current_images: List[str], cache_file: str = None) -> None:
        """增量更新索引（只增不减版本）"""
        # 分析变化
        new_images, modified_images = self.get_new_and_modified_images(current_images)
    def _rebuild_index_from_cache(self) -> None:
        """从缓存重建索引"""
        if not self.features_cache:
            print("没有缓存特征，无法重建索引")
            return

        print("从缓存重建索引...")

        # 确保image_paths与features_cache一致
        valid_paths = []
        features_list = []

        for img_path in self.image_paths:
            if img_path in self.features_cache:
                valid_paths.append(img_path)
                features_list.append(self.features_cache[img_path])

        if not features_list:
            print("没有有效的特征数据")
            return

        self.image_paths = valid_paths
        features_matrix = np.array(features_list).astype('float32')

        # 创建新索引
        index = self.create_faiss_index(features_matrix.shape[1])

        # 训练索引（仅IVF需要）
        if self.index_type == 'ivf':
            index.train(features_matrix)

        # 添加特征
        index.add(features_matrix)

        self.indices['combined'] = index
        print(f"索引重建完成，包含 {index.ntotal} 个特征向量")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="增量FAISS图像相似度检测工具（只增不减版本）")

    # 基本参数
    parser.add_argument("--mode", "-m", type=str, choices=['build', 'update', 'search', 'stats', 'cleanup'], required=True,
                        help="运行模式：build=构建索引, update=更新索引, search=搜索相似图片, stats=显示统计, cleanup=清理已删除记录")
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
    parser.add_argument("--include-deleted", action="store_true",
                        help="搜索结果中包含已删除的图片")

    # 构建参数
    parser.add_argument("--force-rebuild", action="store_true",
                        help="强制重建整个索引")
    parser.add_argument("--no-preserve-deleted", action="store_true",
                        help="关闭保留已删除文件模式（恢复原始行为）")

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

    if args.mode in ['build', 'update', 'search'] and not os.path.exists(args.directory):
        print(f"错误：目录不存在 {args.directory}")
        return

    # 初始化检测器
    detector = IncrementalFAISSDetector(
        enable_resnet=not args.disable_resnet,
        enable_vit=not args.disable_vit,
        enable_traditional=not args.disable_traditional,
        index_type=args.index_type,
        use_gpu=args.use_gpu,
        preserve_deleted=not args.no_preserve_deleted  # 新增参数
    )

    if args.mode in ['build', 'update']:
        # 构建或更新索引模式
        print("=" * 60)
        print(f"{'构建' if args.mode == 'build' else '更新'}索引模式")
        print("=" * 60)

        start_time = time.time()
        force_rebuild = args.force_rebuild or args.mode == 'build'
        detector.build_or_update_index(args.directory, args.cache_file, force_rebuild)
        build_time = time.time() - start_time

        # 保存索引
        detector.save_index(args.index_file)

        print(f"索引{'构建' if force_rebuild else '更新'}完成，耗时: {build_time:.2f}秒")

        # 显示统计信息
        detector.print_statistics()

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

        if 'combined' not in detector.indices:
            print("错误：索引加载失败，请先构建索引")
            return

        # 搜索相似图片
        start_time = time.time()
        results = detector.search_similar_images(
            args.target,
            k=args.top_k,
            threshold=args.threshold,
            include_deleted=args.include_deleted
        )
        search_time = time.time() - start_time

        # 显示结果
        print(f"目标图片: {args.target}")
        print(f"搜索耗时: {search_time:.3f}秒")
        print(f"相似度阈值: {args.threshold}")
        print(f"包含已删除图片: {'是' if args.include_deleted else '否'}")
        print(f"找到 {len(results)} 张相似图片：")
        print("-" * 80)

        if results:
            for i, (image_path, score, is_deleted) in enumerate(results, 1):
                # 相对路径打印（相对目标图片所在目录）
                try:
                    base_dir = os.path.dirname(os.path.abspath(args.target))
                    rel_path = os.path.relpath(image_path, base_dir)
                except Exception:
                    rel_path = image_path
                
                status = " [已删除]" if is_deleted else ""
                print(f"{i:2d}. {rel_path:<50} 相似度: {score:.4f}{status}")
        else:
            print("未找到相似图片")
            print("建议尝试：")
            print(f"  - 降低阈值（当前: {args.threshold}，建议尝试: {max(0.3, args.threshold - 0.2):.1f}）")
            print(f"  - 增加返回数量（当前: {args.top_k}，建议尝试: {args.top_k + 10}）")
            if not args.include_deleted:
                print(f"  - 包含已删除图片（使用 --include-deleted 参数）")

        print("-" * 80)

    elif args.mode == 'stats':
        # 统计信息模式
        print("=" * 60)
        print("索引统计信息模式")
        print("=" * 60)

        # 加载现有缓存获取统计信息
        if args.cache_file and os.path.exists(args.cache_file):
            try:
                with open(args.cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    detector.image_paths = cache_data.get('image_paths', [])
                    detector.image_hashes = cache_data.get('image_hashes', {})
                    detector.features_cache = cache_data.get('features_cache', {})
                    detector.deleted_paths = set(cache_data.get('deleted_paths', []))
                    detector.metadata = cache_data.get('metadata', detector.metadata)
                detector.print_statistics()
            except Exception as e:
                print(f"无法加载缓存文件: {e}")
        else:
            print("缓存文件不存在，请先构建索引")

    elif args.mode == 'cleanup':
        # 清理已删除记录模式
        print("=" * 60)
        print("清理已删除记录模式")
        print("=" * 60)

        # 加载现有数据
        if args.cache_file and os.path.exists(args.cache_file):
            try:
                with open(args.cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    detector.image_paths = cache_data.get('image_paths', [])
                    detector.image_hashes = cache_data.get('image_hashes', {})
                    detector.features_cache = cache_data.get('features_cache', {})
                    detector.deleted_paths = set(cache_data.get('deleted_paths', []))
                    detector.metadata = cache_data.get('metadata', detector.metadata)
                
                # 执行清理
                cleaned_count = detector.cleanup_deleted_records(args.cache_file)
                
                # 重新保存索引
                if cleaned_count > 0:
                    detector.save_index(args.index_file)
                    print(f"清理完成，移除了 {cleaned_count} 条已删除记录")
                    detector.print_statistics()
                
            except Exception as e:
                print(f"清理失败: {e}")
        else:
            print("缓存文件不存在，无法执行清理")


def show_help():
    """显示详细帮助信息"""
    help_text = """
=================================================================
     增量FAISS图像相似度检测器 - 详细使用说明（只增不减版本）
=================================================================

功能特性：
  ✓ 支持增量索引更新，避免全量重建
  ✓ 智能检测图片变化（新增/修改/删除）
  ✓ 多模态特征融合（ResNet + ViT + 传统CV）
  ✓ 高效的特征缓存机制
  ✓ 只增不减模式：保留已删除文件的索引记录
  ✓ 可选的清理功能

基本用法：
  1. 首次构建索引：
     python improved_faiss_detector_ver3.py --mode build \\
       --directory /path/to/images \\
       --cache-file features.pkl \\
       --index-file image_index.index
     
     python improved_faiss_detector_ver3.py --mode build \\
       --directory /path/to/images \\
       --cache-file features.pkl \\
       --index-file image_index.index  
       

  2. 增量更新索引（只增不减）：
     python improved_faiss_detector_ver3.py --mode update \\
       --directory /path/to/images \\
       --cache-file features.pkl \\
       --index-file image_index.index

  3. 搜索相似图片：
     python improved_faiss_detector_ver3.py --mode search \\
       --target /path/to/target.jpg \\
       --index-file image_index.index \\
       --directory /path/to/images \\
       --threshold 0.5 \\
       --top-k 10

  4. 查看统计信息：
     python improved_faiss_detector_ver3.py --mode stats \\
       --cache-file features.pkl

  5. 清理已删除记录（可选）：
     python improved_faiss_detector_ver3.py --mode cleanup \\
       --cache-file features.pkl \\
       --index-file image_index.index

参数说明：
  --mode: 运行模式 [build|update|search|stats|cleanup]
  --directory: 图片目录路径
  --cache-file: 特征缓存文件（强烈推荐使用）
  --index-file: FAISS索引文件路径
  --target: 目标图片路径（搜索模式）
  --threshold: 相似度阈值 [0.0-1.0]，默认0.5
  --top-k: 返回相似图片数量，默认10
  --include-deleted: 搜索结果包含已删除图片
  --force-rebuild: 强制重建整个索引
  --no-preserve-deleted: 关闭只增不减模式

新功能说明：
  - 只增不减模式：默认开启，已删除的图片会被标记但保留在索引中
  - 统计信息：显示活跃图片数、已删除但保留的图片数等
  - 清理功能：可选择性清理已删除记录，释放存储空间
  - 搜索过滤：可选择是否在搜索结果中包含已删除的图片

阈值建议：
  - 高精度匹配: >= 0.85
  - 平衡精度召回: >= 0.65  
  - 高召回检索: >= 0.45

使用场景：
  - 数据库备份：保留所有历史图片索引
  - 版本控制：跟踪图片的历史变化
  - 增量同步：只处理新增和修改的图片

技术支持：
  如遇问题请检查：
  1. 依赖库是否完整安装
  2. 图片目录是否存在且包含支持格式的图片
  3. 缓存文件是否有读写权限

=================================================================
    """
    print(help_text)


if __name__ == "__main__":
    # 检查是否需要显示帮助
    if len(sys.argv) == 1 or (len(sys.argv) == 2 and sys.argv[1] in ['-h', '--help']):
        show_help()
    else:
        main()
        stats = self.get_statistics()
        print("\n" + "=" * 50)
        print("           索引统计信息")
        print("=" * 50)
        print(f"总图片记录:     {stats['total_images']} 张")
        print(f"活跃图片:       {stats['active_images']} 张")
        print(f"已删除但保留:   {stats['deleted_images']} 张")
        print(f"特征维度:       {stats['feature_dimension']}")
        print(f"索引大小:       {stats['index_size']} 个特征向量")
        print(f"保留删除模式:   {'开启' if stats['preserve_deleted_mode'] else '关闭'}")
        
        if stats['created_time']:
            created_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stats['created_time']))
            print(f"创建时间:       {created_time}")
        
        if stats['last_updated']:
            updated_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stats['last_updated']))
            print(f"最后更新:       {updated_time}")
        
        print("=" * 50)

    def cleanup_deleted_records(self, cache_file: str = None) -> int:
        """
        清理已删除记录的功能（可选使用）
        返回清理的记录数量
        """
        if not self.deleted_paths:
            print("没有已删除的记录需要清理")
            return 0

        deleted_count = len(self.deleted_paths)
        print(f"开始清理 {deleted_count} 条已删除记录...")

        # 从各种数据结构中移除已删除的记录
        for deleted_path in self.deleted_paths:
            if deleted_path in self.image_hashes:
                del self.image_hashes[deleted_path]
            if deleted_path in self.features_cache:
                del self.features_cache[deleted_path]
            if deleted_path in self.image_paths:
                self.image_paths.remove(deleted_path)

        # 清空已删除路径集合
        self.deleted_paths.clear()

        # 重建索引（因为我们移除了一些记录）
        self._rebuild_index_from_cache()

        # 保存缓存
        if cache_file:
            self._save_cache(cache_file)

        # 更新元数据
        self._update_metadata_counters()

        print(f"清理完成，移除了 {deleted_count} 条记录")
        return deleted_count
