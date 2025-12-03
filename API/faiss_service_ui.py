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
- GET / - Web UI界面
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
        def serve_ui():
            """提供Web UI界面"""
            return self._get_web_ui_html()

        @self.app.route('/favicon.ico')
        def favicon():
            """处理favicon请求"""
            return '', 204

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

    def _get_web_ui_html(self) -> str:
        """返回Web UI的HTML内容"""
        return '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FAISS图像相似度检测服务</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            overflow: hidden;
        }

        .header {
            background: linear-gradient(135deg, #4f46e5, #7c3aed);
            color: white;
            padding: 30px;
            text-align: center;
        }

        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 700;
        }

        .header p {
            font-size: 1.1em;
            opacity: 0.9;
        }

        .status-bar {
            background: #f8fafc;
            padding: 15px 30px;
            border-bottom: 1px solid #e2e8f0;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .status-indicator {
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .status-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #10b981;
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        .main-content {
            padding: 30px;
        }

        .tabs {
            display: flex;
            margin-bottom: 30px;
            background: #f1f5f9;
            border-radius: 12px;
            padding: 6px;
        }

        .tab {
            flex: 1;
            padding: 12px 20px;
            background: transparent;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 500;
            transition: all 0.3s ease;
        }

        .tab.active {
            background: white;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            color: #4f46e5;
        }

        .tab-content {
            display: none;
        }

        .tab-content.active {
            display: block;
            animation: fadeIn 0.3s ease;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .card {
            background: white;
            border-radius: 16px;
            padding: 30px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
            margin-bottom: 20px;
            border: 1px solid #e2e8f0;
        }

        .form-group {
            margin-bottom: 20px;
        }

        .form-group label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #374151;
        }

        .form-control {
            width: 100%;
            padding: 12px 16px;
            border: 2px solid #e2e8f0;
            border-radius: 10px;
            font-size: 16px;
            transition: all 0.3s ease;
        }

        .form-control:focus {
            outline: none;
            border-color: #4f46e5;
            box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1);
        }

        .file-upload {
            position: relative;
            display: inline-block;
            width: 100%;
        }

        .file-upload input[type="file"] {
            position: absolute;
            opacity: 0;
            width: 100%;
            height: 100%;
            cursor: pointer;
        }

        .file-upload-label {
            display: block;
            padding: 20px;
            border: 2px dashed #cbd5e1;
            border-radius: 12px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            background: #f8fafc;
        }

        .file-upload-label:hover {
            border-color: #4f46e5;
            background: #f0f4ff;
        }

        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }

        .btn-primary {
            background: linear-gradient(135deg, #4f46e5, #7c3aed);
            color: white;
        }

        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(79, 70, 229, 0.3);
        }

        .btn-secondary {
            background: #f1f5f9;
            color: #64748b;
        }

        .btn-secondary:hover {
            background: #e2e8f0;
        }

        .results {
            margin-top: 30px;
        }

        .result-item {
            display: flex;
            align-items: center;
            padding: 15px;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            margin-bottom: 15px;
            transition: all 0.3s ease;
        }

        .result-item:hover {
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            transform: translateY(-2px);
        }

        .result-info {
            flex: 1;
        }

        .result-filename {
            font-weight: 600;
            margin-bottom: 5px;
        }

        .result-score {
            color: #10b981;
            font-weight: 500;
        }

        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }

        .spinner {
            width: 40px;
            height: 40px;
            border: 4px solid #f3f4f6;
            border-top: 4px solid #4f46e5;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .alert {
            padding: 15px 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            font-weight: 500;
        }

        .alert-success {
            background: #dcfce7;
            color: #166534;
            border: 1px solid #bbf7d0;
        }

        .alert-error {
            background: #fef2f2;
            color: #dc2626;
            border: 1px solid #fecaca;
        }

        .indices-list {
            display: grid;
            gap: 15px;
        }

        .index-item {
            background: #f8fafc;
            padding: 20px;
            border-radius: 12px;
            border: 1px solid #e2e8f0;
        }

        .index-name {
            font-weight: 600;
            font-size: 18px;
            margin-bottom: 10px;
            color: #1e293b;
        }

        .index-info {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
            margin-bottom: 15px;
        }

        .index-stat {
            text-align: center;
            padding: 10px;
            background: white;
            border-radius: 8px;
        }

        .index-stat-value {
            font-size: 20px;
            font-weight: 700;
            color: #4f46e5;
        }

        .index-stat-label {
            font-size: 12px;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 FAISS 图像相似度检测</h1>
            <p>智能图像搜索与相似度分析服务</p>
        </div>

        <div class="status-bar">
            <div class="status-indicator">
                <div class="status-dot"></div>
                <span id="service-status">正在连接...</span>
            </div>
            <div id="service-info"></div>
        </div>

        <div class="main-content">
            <div class="tabs">
                <button class="tab active" onclick="switchTab('search')">🔍 搜索图片</button>
                <button class="tab" onclick="switchTab('build')">🏗️ 构建索引</button>
                <button class="tab" onclick="switchTab('manage')">📋 管理索引</button>
            </div>

            <!-- 搜索图片 -->
            <div id="search-tab" class="tab-content active">
                <div class="card">
                    <h3>搜索相似图片</h3>
                    <form id="search-form">
                        <div class="form-group">
                            <label>选择图片</label>
                            <div class="file-upload">
                                <input type="file" id="search-file" accept="image/*" required>
                                <label for="search-file" class="file-upload-label">
                                    📁 点击选择图片文件<br>
                                    <small>支持 JPG, PNG, BMP, GIF, TIFF, WebP</small>
                                </label>
                            </div>
                        </div>

                        <div class="form-group">
                            <label>索引名称</label>
                            <select class="form-control" id="search-index" required>
                                <option value="">请选择索引</option>
                            </select>
                        </div>

                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                            <div class="form-group">
                                <label>返回数量</label>
                                <input type="number" class="form-control" id="search-topk" value="10" min="1" max="100">
                            </div>
                            <div class="form-group">
                                <label>相似度阈值</label>
                                <input type="number" class="form-control" id="search-threshold" value="0.5" min="0" max="1" step="0.1">
                            </div>
                        </div>

                        <button type="submit" class="btn btn-primary">🔍 开始搜索</button>
                    </form>

                    <div class="loading" id="search-loading">
                        <div class="spinner"></div>
                        <p>正在搜索相似图片...</p>
                    </div>

                    <div id="search-results" class="results"></div>
                </div>
            </div>

            <!-- 构建索引 -->
            <div id="build-tab" class="tab-content">
                <div class="card">
                    <h3>构建图片索引</h3>
                    <form id="build-form">
                        <div class="form-group">
                            <label>索引名称</label>
                            <input type="text" class="form-control" id="build-name" placeholder="输入索引名称" required>
                        </div>

                        <div class="form-group">
                            <label>图片目录路径</label>
                            <input type="text" class="form-control" id="build-directory" placeholder="输入图片文件夹路径" required>
                        </div>

                        <div class="form-group">
                            <label>模型配置</label>
                            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; margin-top: 10px;">
                                <label style="display: flex; align-items: center; gap: 8px;">
                                    <input type="checkbox" id="enable-resnet" checked>
                                    启用 ResNet
                                </label>
                                <label style="display: flex; align-items: center; gap: 8px;">
                                    <input type="checkbox" id="enable-vit" checked>
                                    启用 ViT
                                </label>
                                <label style="display: flex; align-items: center; gap: 8px;">
                                    <input type="checkbox" id="enable-traditional" checked>
                                    启用传统特征
                                </label>
                                <label style="display: flex; align-items: center; gap: 8px;">
                                    <input type="checkbox" id="use-gpu">
                                    使用 GPU
                                </label>
                            </div>
                        </div>

                        <button type="submit" class="btn btn-primary">🏗️ 开始构建</button>
                    </form>

                    <div class="loading" id="build-loading">
                        <div class="spinner"></div>
                        <p>正在构建索引，请耐心等待...</p>
                    </div>

                    <div id="build-results"></div>
                </div>
            </div>

            <!-- 管理索引 -->
            <div id="manage-tab" class="tab-content">
                <div class="card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
                        <h3>索引管理</h3>
                        <button class="btn btn-secondary" onclick="loadIndices()">🔄 刷新</button>
                    </div>
                    <div id="indices-list" class="indices-list"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const API_BASE = window.location.protocol + '//' + window.location.host + '/api/v1';

        // 页面加载时初始化
        document.addEventListener('DOMContentLoaded', function() {
            checkServiceStatus();
            loadIndices();

            // 绑定表单事件
            document.getElementById('search-form').addEventListener('submit', handleSearch);
            document.getElementById('build-form').addEventListener('submit', handleBuild);

            // 文件选择提示
            document.getElementById('search-file').addEventListener('change', function(e) {
                const label = document.querySelector('.file-upload-label');
                if (e.target.files.length > 0) {
                    label.innerHTML = `📄 ${e.target.files[0].name}<br><small>点击重新选择</small>`;
                }
            });
        });

        // 切换标签页
        function switchTab(tabName) {
            // 隐藏所有标签内容
            document.querySelectorAll('.tab-content').forEach(tab => {
                tab.classList.remove('active');
            });

            // 移除所有按钮的激活状态
            document.querySelectorAll('.tab').forEach(btn => {
                btn.classList.remove('active');
            });

            // 显示选中的标签
            document.getElementById(tabName + '-tab').classList.add('active');
            event.target.classList.add('active');
        }

        // 检查服务状态
        async function checkServiceStatus() {
            try {
                const response = await fetch(`${API_BASE}/health`);
                const data = await response.json();

                document.getElementById('service-status').textContent = '服务正常运行';

                // 获取详细状态
                const statusResponse = await fetch(`${API_BASE}/status`);
                const statusData = await statusResponse.json();

                document.getElementById('service-info').innerHTML = 
                    `总搜索: ${statusData.stats.total_searches} | 总索引: ${statusData.stats.total_builds}`;

            } catch (error) {
                document.getElementById('service-status').textContent = '服务连接失败';
                console.error('服务状态检查失败:', error);
            }
        }

        // 加载索引列表
        async function loadIndices() {
            try {
                const response = await fetch(`${API_BASE}/indices`);
                const data = await response.json();

                // 更新搜索页面的索引选择器
                const searchSelect = document.getElementById('search-index');
                searchSelect.innerHTML = '<option value="">请选择索引</option>';

                // 更新管理页面的索引列表
                const indicesList = document.getElementById('indices-list');
                indicesList.innerHTML = '';

                Object.entries(data.indices).forEach(([name, info]) => {
                    // 添加到搜索选择器
                    if (info.status === 'ready') {
                        const option = document.createElement('option');
                        option.value = name;
                        option.textContent = `${name} (${info.total_images || 0} 图片)`;
                        searchSelect.appendChild(option);
                    }

                    // 添加到管理列表
                    const indexItem = document.createElement('div');
                    indexItem.className = 'index-item';
                    indexItem.innerHTML = `
                        <div class="index-name">${name}</div>
                        <div class="index-info">
                            <div class="index-stat">
                                <div class="index-stat-value">${info.status === 'ready' ? '✅' : info.status === 'building' ? '🔄' : '❌'}</div>
                                <div class="index-stat-label">状态</div>
                            </div>
                            <div class="index-stat">
                                <div class="index-stat-value">${info.total_images || 0}</div>
                                <div class="index-stat-label">图片数量</div>
                            </div>
                            <div class="index-stat">
                                <div class="index-stat-value">${info.feature_dim || 0}</div>
                                <div class="index-stat-label">特征维度</div>
                            </div>
                        </div>
                        ${info.status === 'ready' ? 
                            `<button class="btn btn-secondary" onclick="deleteIndex('${name}')">🗑️ 删除</button>` : 
                            ''
                        }
                    `;
                    indicesList.appendChild(indexItem);
                });

            } catch (error) {
                console.error('加载索引列表失败:', error);
                showAlert('加载索引列表失败', 'error');
            }
        }

        // 处理搜索
        async function handleSearch(e) {
            e.preventDefault();

            const formData = new FormData();
            const fileInput = document.getElementById('search-file');
            const indexName = document.getElementById('search-index').value;
            const topK = document.getElementById('search-topk').value;
            const threshold = document.getElementById('search-threshold').value;

            if (!fileInput.files[0]) {
                showAlert('请选择图片文件', 'error');
                return;
            }

            if (!indexName) {
                showAlert('请选择索引', 'error');
                return;
            }

            formData.append('image', fileInput.files[0]);
            formData.append('index_name', indexName);
            formData.append('top_k', topK);
            formData.append('threshold', threshold);

            // 显示加载状态
            document.getElementById('search-loading').style.display = 'block';
            document.getElementById('search-results').innerHTML = '';

            try {
                const response = await fetch(`${API_BASE}/search`, {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (response.ok) {
                    displaySearchResults(data);
                } else {
                    showAlert(data.error || '搜索失败', 'error');
                }

            } catch (error) {
                console.error('搜索失败:', error);
                showAlert('搜索请求失败', 'error');
            } finally {
                document.getElementById('search-loading').style.display = 'none';
            }
        }

        // 显示搜索结果
        function displaySearchResults(data) {
            const resultsDiv = document.getElementById('search-results');

            if (data.results.length === 0) {
                resultsDiv.innerHTML = '<p style="text-align: center; color: #64748b; padding: 20px;">未找到相似图片</p>';
                return;
            }

            resultsDiv.innerHTML = `
                <h4>找到 ${data.total_found} 张相似图片</h4>
                ${data.results.map(result => `
                    <div class="result-item">
                        <div class="result-info">
                            <div class="result-filename">${result.filename}</div>
                            <div class="result-score">相似度: ${(result.similarity_score * 100).toFixed(2)}%</div>
                            <div style="color: #64748b; font-size: 14px;">${result.image_path}</div>
                        </div>
                    </div>
                `).join('')}
            `;
        }

        // 处理构建索引
        async function handleBuild(e) {
            e.preventDefault();

            const indexName = document.getElementById('build-name').value;
            const imageDirectory = document.getElementById('build-directory').value;
            const enableResnet = document.getElementById('enable-resnet').checked;
            const enableVit = document.getElementById('enable-vit').checked;
            const enableTraditional = document.getElementById('enable-traditional').checked;
            const useGpu = document.getElementById('use-gpu').checked;

            const requestData = {
                index_name: indexName,
                image_directory: imageDirectory,
                model_config: {
                    enable_resnet: enableResnet,
                    enable_vit: enableVit,
                    enable_traditional: enableTraditional,
                    use_gpu: useGpu
                }
            };

            // 显示加载状态
            document.getElementById('build-loading').style.display = 'block';
            document.getElementById('build-results').innerHTML = '';

            try {
                const response = await fetch(`${API_BASE}/build_index`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(requestData)
                });

                const data = await response.json();

                if (response.ok) {
                    showAlert(`开始构建索引: ${indexName}`, 'success');
                    // 定期检查构建状态
                    checkBuildStatus(indexName);
                } else {
                    showAlert(data.error || '构建失败', 'error');
                }

            } catch (error) {
                console.error('构建索引失败:', error);
                showAlert('构建请求失败', 'error');
            } finally {
                document.getElementById('build-loading').style.display = 'none';
            }
        }

        // 检查构建状态
        function checkBuildStatus(indexName) {
            const interval = setInterval(async () => {
                try {
                    const response = await fetch(`${API_BASE}/indices`);
                    const data = await response.json();

                    if (data.indices[indexName]) {
                        const status = data.indices[indexName].status;

                        if (status === 'ready') {
                            showAlert(`索引 ${indexName} 构建完成！`, 'success');
                            loadIndices(); // 刷新索引列表
                            clearInterval(interval);
                        } else if (status === 'error') {
                            showAlert(`索引 ${indexName} 构建失败`, 'error');
                            clearInterval(interval);
                        }
                    }
                } catch (error) {
                    console.error('检查构建状态失败:', error);
                    clearInterval(interval);
                }
            }, 5000); // 每5秒检查一次
        }

        // 删除索引
        async function deleteIndex(indexName) {
            if (!confirm(`确定要删除索引 "${indexName}" 吗？此操作不可恢复。`)) {
                return;
            }

            try {
                const response = await fetch(`${API_BASE}/indices/${indexName}`, {
                    method: 'DELETE'
                });

                const data = await response.json();

                if (response.ok) {
                    showAlert(`索引 ${indexName} 已删除`, 'success');
                    loadIndices(); // 刷新索引列表
                } else {
                    showAlert(data.error || '删除失败', 'error');
                }

            } catch (error) {
                console.error('删除索引失败:', error);
                showAlert('删除请求失败', 'error');
            }
        }

        // 显示提示信息
        function showAlert(message, type) {
            const alertDiv = document.createElement('div');
            alertDiv.className = `alert alert-${type}`;
            alertDiv.textContent = message;

            // 插入到当前活动的标签页开头
            const activeTab = document.querySelector('.tab-content.active .card');
            activeTab.insertBefore(alertDiv, activeTab.firstChild);

            // 3秒后自动移除
            setTimeout(() => {
                alertDiv.remove();
            }, 3000);
        }
    </script>
</body>
</html>'''

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