#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FAISS图像相似度检测服务
基于原始的faiss_image_similarity.py封装的RESTful API服务

启动方式:
python faiss_service.py --config config.json

API端点:
- POST /api/v1/build_index - 构建索引
- POST /api/v1/search - 搜索相似图片
- GET /api/v1/status - 获取服务状态
- GET /api/v1/health - 健康检查
"""

import os
import json
import logging
import threading
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import argparse

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import werkzeug
from werkzeug.utils import secure_filename

# 导入原始的检测器类
from faiss_image_similarity import FAISSImageSimilarityDetector


@dataclass
class ServiceConfig:
    """服务配置类"""
    host: str = "0.0.0.0"
    port: int = 8080
    debug: bool = False
    max_file_size: int = 16 * 1024 * 1024  # 16MB
    upload_folder: str = "/tmp/image_uploads"
    index_folder: str = "./indices"
    cache_folder: str = "./cache"
    allowed_extensions: set = None
    enable_cors: bool = True
    log_level: str = "INFO"

    def __post_init__(self):
        if self.allowed_extensions is None:
            self.allowed_extensions = {'jpg', 'jpeg', 'png', 'bmp', 'gif', 'tiff', 'webp'}


class ImageSimilarityService:
    """图像相似度检测服务"""

    def __init__(self, config: ServiceConfig):
        self.config = config
        self.app = Flask(__name__)
        self.detector = None
        self.index_status = {}
        self.service_stats = {
            'start_time': datetime.now().isoformat(),
            'total_searches': 0,
            'total_builds': 0,
            'current_indices': {}
        }

        self._setup_logging()
        self._setup_directories()
        self._setup_flask()
        self._register_routes()

    def _setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _setup_directories(self):
        """创建必要的目录"""
        for folder in [self.config.upload_folder, self.config.index_folder, self.config.cache_folder]:
            os.makedirs(folder, exist_ok=True)

    def _setup_flask(self):
        """设置Flask应用"""
        self.app.config['MAX_CONTENT_LENGTH'] = self.config.max_file_size
        self.app.config['UPLOAD_FOLDER'] = self.config.upload_folder

        if self.config.enable_cors:
            CORS(self.app)

    def _allowed_file(self, filename: str) -> bool:
        """检查文件扩展名是否允许"""
        return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in self.config.allowed_extensions

    def _register_routes(self):
        """注册API路由"""

        @self.app.route('/')
        def index():
            """根路径 - 服务信息"""
            return jsonify({
                'service': 'FAISS Image Similarity Service',
                'status': 'running',
                'version': '1.0.0',
                'endpoints': {
                    'health': '/api/v1/health',
                    'status': '/api/v1/status',
                    'build_index': '/api/v1/build_index',
                    'search': '/api/v1/search',
                    'indices': '/api/v1/indices'
                },
                'documentation': 'See API endpoints above for available operations'
            })

        @self.app.route('/api/v1/health', methods=['GET'])
        def health_check():
            """健康检查"""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'service': 'FAISS Image Similarity Service'
            })

        @self.app.route('/api/v1/status', methods=['GET'])
        def get_status():
            """获取服务状态"""
            return jsonify({
                'status': 'running',
                'stats': self.service_stats,
                'indices': self.index_status,
                'config': {
                    'max_file_size': self.config.max_file_size,
                    'allowed_extensions': list(self.config.allowed_extensions)
                }
            })

        @self.app.route('/api/v1/build_index', methods=['POST'])
        def build_index():
            """构建索引API"""
            try:
                data = request.get_json()
                if not data:
                    return jsonify({'error': '请提供JSON数据'}), 400

                # 验证必需参数
                required_params = ['index_name', 'image_directory']
                for param in required_params:
                    if param not in data:
                        return jsonify({'error': f'缺少必需参数: {param}'}), 400

                index_name = data['index_name']
                image_directory = data['image_directory']

                # 验证目录存在
                if not os.path.exists(image_directory):
                    return jsonify({'error': f'图片目录不存在: {image_directory}'}), 400

                # 获取可选参数
                model_config = data.get('model_config', {})
                cache_file = data.get('cache_file')

                # 在后台线程中构建索引
                thread = threading.Thread(
                    target=self._build_index_async,
                    args=(index_name, image_directory, model_config, cache_file)
                )
                thread.daemon = True
                thread.start()

                return jsonify({
                    'message': f'开始构建索引: {index_name}',
                    'status': 'building',
                    'index_name': index_name
                })

            except Exception as e:
                self.logger.error(f"构建索引错误: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/v1/search', methods=['POST'])
        def search_similar():
            """搜索相似图片API"""
            try:
                # 检查是否有文件上传
                if 'image' not in request.files:
                    return jsonify({'error': '请上传图片文件'}), 400

                file = request.files['image']
                if file.filename == '':
                    return jsonify({'error': '未选择文件'}), 400

                if not self._allowed_file(file.filename):
                    return jsonify({'error': '不支持的文件格式'}), 400

                # 获取其他参数
                index_name = request.form.get('index_name', 'default')
                top_k = int(request.form.get('top_k', 10))
                threshold = float(request.form.get('threshold', 0.5))

                # 检查索引是否存在
                if index_name not in self.index_status or self.index_status[index_name]['status'] != 'ready':
                    return jsonify({'error': f'索引 {index_name} 不存在或未准备就绪'}), 400

                # 保存上传的文件
                filename = secure_filename(file.filename)
                timestamp = str(int(time.time()))
                safe_filename = f"{timestamp}_{filename}"
                file_path = os.path.join(self.config.upload_folder, safe_filename)
                file.save(file_path)

                try:
                    # 加载对应的检测器和索引
                    detector = self._get_detector(index_name)
                    if not detector:
                        return jsonify({'error': f'无法加载检测器: {index_name}'}), 500

                    # 搜索相似图片
                    results = detector.search_similar_images(file_path, top_k, threshold)

                    # 更新统计
                    self.service_stats['total_searches'] += 1

                    # 格式化结果
                    formatted_results = []
                    for img_path, score in results:
                        formatted_results.append({
                            'image_path': img_path,
                            'similarity_score': float(score),
                            'filename': os.path.basename(img_path)
                        })

                    return jsonify({
                        'results': formatted_results,
                        'total_found': len(formatted_results),
                        'query_image': safe_filename,
                        'parameters': {
                            'index_name': index_name,
                            'top_k': top_k,
                            'threshold': threshold
                        }
                    })

                finally:
                    # 清理上传的临时文件
                    if os.path.exists(file_path):
                        os.remove(file_path)

            except Exception as e:
                self.logger.error(f"搜索错误: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/v1/indices', methods=['GET'])
        def list_indices():
            """列出所有可用的索引"""
            return jsonify({
                'indices': self.index_status,
                'total': len(self.index_status)
            })

        @self.app.route('/api/v1/indices/<index_name>', methods=['DELETE'])
        def delete_index(index_name: str):
            """删除指定索引"""
            try:
                if index_name not in self.index_status:
                    return jsonify({'error': f'索引 {index_name} 不存在'}), 404

                # 删除索引文件
                index_file = os.path.join(self.config.index_folder, f"{index_name}.index")
                paths_file = os.path.join(self.config.index_folder, f"{index_name}_paths.pkl")

                for file_path in [index_file, paths_file]:
                    if os.path.exists(file_path):
                        os.remove(file_path)

                # 从状态中移除
                del self.index_status[index_name]
                if index_name in self.service_stats['current_indices']:
                    del self.service_stats['current_indices'][index_name]

                return jsonify({'message': f'索引 {index_name} 已删除'})

            except Exception as e:
                self.logger.error(f"删除索引错误: {e}")
                return jsonify({'error': str(e)}), 500

    def _build_index_async(self, index_name: str, image_directory: str,
                           model_config: dict, cache_file: Optional[str]):
        """异步构建索引"""
        try:
            self.logger.info(f"开始构建索引: {index_name}")

            # 更新状态
            self.index_status[index_name] = {
                'status': 'building',
                'start_time': datetime.now().isoformat(),
                'image_directory': image_directory,
                'progress': 0
            }

            # 初始化检测器
            detector = FAISSImageSimilarityDetector(
                enable_resnet=model_config.get('enable_resnet', True),
                enable_vit=model_config.get('enable_vit', True),
                enable_traditional=model_config.get('enable_traditional', True),
                index_type=model_config.get('index_type', 'flat'),
                use_gpu=model_config.get('use_gpu', False)
            )

            # 设置缓存文件路径
            if not cache_file:
                cache_file = os.path.join(self.config.cache_folder, f"{index_name}_features.pkl")

            # 构建索引
            detector.build_index(image_directory, cache_file)

            # 保存索引
            index_file = os.path.join(self.config.index_folder, f"{index_name}.index")
            detector.save_index(index_file)

            # 更新状态
            self.index_status[index_name] = {
                'status': 'ready',
                'build_time': datetime.now().isoformat(),
                'image_directory': image_directory,
                'index_file': index_file,
                'total_images': len(detector.image_paths),
                'feature_dim': detector.indices['combined'].d if 'combined' in detector.indices else 0
            }

            # 更新统计
            self.service_stats['total_builds'] += 1
            self.service_stats['current_indices'][index_name] = {
                'created': datetime.now().isoformat(),
                'images': len(detector.image_paths)
            }

            self.logger.info(f"索引构建完成: {index_name}")

        except Exception as e:
            self.logger.error(f"构建索引失败 {index_name}: {e}")
            self.index_status[index_name] = {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def _get_detector(self, index_name: str) -> Optional[FAISSImageSimilarityDetector]:
        """获取检测器实例"""
        try:
            if index_name not in self.index_status:
                return None

            index_info = self.index_status[index_name]
            if index_info['status'] != 'ready':
                return None

            # 创建新的检测器实例
            detector = FAISSImageSimilarityDetector()

            # 加载索引
            index_file = index_info['index_file']
            detector.load_index(index_file)

            return detector

        except Exception as e:
            self.logger.error(f"加载检测器失败 {index_name}: {e}")
            return None

    def run(self):
        """启动服务"""
        self.logger.info(f"启动FAISS图像相似度检测服务")
        self.logger.info(f"服务地址: http://{self.config.host}:{self.config.port}")

        # 扫描现有索引
        self._scan_existing_indices()

        self.app.run(
            host=self.config.host,
            port=self.config.port,
            debug=self.config.debug,
            threaded=True
        )

    def _scan_existing_indices(self):
        """扫描现有的索引文件"""
        try:
            if not os.path.exists(self.config.index_folder):
                return

            for filename in os.listdir(self.config.index_folder):
                if filename.endswith('.index'):
                    index_name = filename[:-6]  # 移除.index后缀
                    index_file = os.path.join(self.config.index_folder, filename)
                    paths_file = os.path.join(self.config.index_folder, f"{index_name}_paths.pkl")

                    if os.path.exists(paths_file):
                        # 尝试加载检测器以验证索引
                        try:
                            detector = FAISSImageSimilarityDetector()
                            detector.load_index(index_file)

                            self.index_status[index_name] = {
                                'status': 'ready',
                                'index_file': index_file,
                                'total_images': len(detector.image_paths),
                                'loaded_at': datetime.now().isoformat()
                            }

                            self.service_stats['current_indices'][index_name] = {
                                'loaded': datetime.now().isoformat(),
                                'images': len(detector.image_paths)
                            }

                            self.logger.info(f"发现现有索引: {index_name}")

                        except Exception as e:
                            self.logger.warning(f"无法加载索引 {index_name}: {e}")

        except Exception as e:
            self.logger.error(f"扫描现有索引失败: {e}")


def load_config(config_file: str) -> ServiceConfig:
    """从文件加载配置"""
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)

        # 转换allowed_extensions为set
        if 'allowed_extensions' in config_dict:
            config_dict['allowed_extensions'] = set(config_dict['allowed_extensions'])

        return ServiceConfig(**config_dict)
    except FileNotFoundError:
        print(f"配置文件不存在: {config_file}")
        return ServiceConfig()
    except json.JSONDecodeError as e:
        print(f"配置文件格式错误: {e}")
        return ServiceConfig()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="FAISS图像相似度检测服务")
    parser.add_argument("--config", "-c", type=str, default="config.json",
                        help="配置文件路径")
    parser.add_argument("--host", type=str, help="服务主机地址")
    parser.add_argument("--port", type=int, help="服务端口")
    parser.add_argument("--debug", action="store_true", help="调试模式")

    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    # 命令行参数覆盖配置文件
    if args.host:
        config.host = args.host
    if args.port:
        config.port = args.port
    if args.debug:
        config.debug = True

    # 创建并启动服务
    service = ImageSimilarityService(config)
    service.run()


if __name__ == "__main__":
    main()