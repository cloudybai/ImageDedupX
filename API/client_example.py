#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FAISS图像相似度检测服务客户端示例
展示如何使用API进行图像索引构建和相似图片搜索
"""

import requests
import json
import time
import os
from typing import Dict, List, Optional


class FAISSImageClient:
    """FAISS图像相似度检测服务客户端"""

    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url.rstrip('/')
        self.api_base = f"{self.base_url}/api/v1"

    def health_check(self) -> Dict:
        """健康检查"""
        try:
            response = requests.get(f"{self.api_base}/health")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}

    def get_status(self) -> Dict:
        """获取服务状态"""
        try:
            response = requests.get(f"{self.api_base}/status")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}

    def build_index(self,
                    index_name: str,
                    image_directory: str,
                    model_config: Optional[Dict] = None,
                    cache_file: Optional[str] = None) -> Dict:
        """构建索引"""
        if model_config is None:
            model_config = {
                'enable_resnet': True,
                'enable_vit': True,
                'enable_traditional': True,
                'index_type': 'flat',
                'use_gpu': False
            }

        data = {
            'index_name': index_name,
            'image_directory': image_directory,
            'model_config': model_config
        }

        if cache_file:
            data['cache_file'] = cache_file

        try:
            response = requests.post(
                f"{self.api_base}/build_index",
                json=data,
                headers={'Content-Type': 'application/json'}
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}

    def search_similar(self,
                       image_path: str,
                       index_name: str = "default",
                       top_k: int = 10,
                       threshold: float = 0.5) -> Dict:
        """搜索相似图片"""
        if not os.path.exists(image_path):
            return {"error": f"图片文件不存在: {image_path}"}

        try:
            with open(image_path, 'rb') as f:
                files = {'image': f}
                data = {
                    'index_name': index_name,
                    'top_k': str(top_k),
                    'threshold': str(threshold)
                }

                response = requests.post(
                    f"{self.api_base}/search",
                    files=files,
                    data=data
                )
                response.raise_for_status()
                return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}

    def list_indices(self) -> Dict:
        """列出所有索引"""
        try:
            response = requests.get(f"{self.api_base}/indices")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}

    def delete_index(self, index_name: str) -> Dict:
        """删除索引"""
        try:
            response = requests.delete(f"{self.api_base}/indices/{index_name}")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}

    def wait_for_index_ready(self, index_name: str, max_wait: int = 3600) -> bool:
        """等待索引构建完成"""
        start_time = time.time()

        while time.time() - start_time < max_wait:
            status = self.get_status()
            if "error" in status:
                print(f"获取状态失败: {status['error']}")
                return False

            if index_name in status.get('indices', {}):
                index_status = status['indices'][index_name]['status']

                if index_status == 'ready':
                    print(f"索引 {index_name} 构建完成")
                    return True
                elif index_status == 'error':
                    print(f"索引 {index_name} 构建失败")
                    return False
                elif index_status == 'building':
                    print(f"索引 {index_name} 正在构建中...")

            time.sleep(10)  # 等待10秒后重新检查

        print(f"等待索引构建超时: {index_name}")
        return False


def main():
    """示例使用"""
    # 创建客户端
    client = FAISSImageClient("http://localhost:8080")

    print("=" * 60)
    print("FAISS图像相似度检测服务客户端示例")
    print("=" * 60)

    # 1. 健康检查
    print("\n1. 健康检查:")
    health = client.health_check()
    print(json.dumps(health, indent=2, ensure_ascii=False))

    # 2. 获取服务状态
    print("\n2. 服务状态:")
    status = client.get_status()
    print(json.dumps(status, indent=2, ensure_ascii=False))

    # 3. 构建索引示例（请修改为实际的图片目录）
    image_directory = "/Users/cloudbai/PycharmProjects/imagesim/similarities/examples/data/shanqi4000_test_2"  # 修改为实际路径

    if os.path.exists(image_directory):
        print(f"\n3. 构建索引:")
        build_result = client.build_index(
            index_name="test_index",
            image_directory=image_directory,
            model_config={
                'enable_resnet': True,
                'enable_vit': True,
                'enable_traditional': True,
                'index_type': 'flat',
                'use_gpu': False
            }
        )
        print(json.dumps(build_result, indent=2, ensure_ascii=False))

        # 等待索引构建完成
        if "error" not in build_result:
            print("\n等待索引构建完成...")
            if client.wait_for_index_ready("test_index"):
                # 4. 搜索相似图片示例
                target_image = "/Users/cloudbai/PycharmProjects/imagesim/similarities/examples/data/shanqi4000_test_2/new17503808046569.jpg"  # 修改为实际路径

                if os.path.exists(target_image):
                    print(f"\n4. 搜索相似图片:")
                    search_result = client.search_similar(
                        image_path=target_image,
                        index_name="test_index",
                        top_k=5,
                        threshold=0.5
                    )
                    print(json.dumps(search_result, indent=2, ensure_ascii=False))
                else:
                    print(f"目标图片不存在: {target_image}")
    else:
        print(f"图片目录不存在: {image_directory}")

    # 5. 列出所有索引
    print(f"\n5. 列出所有索引:")
    indices = client.list_indices()
    print(json.dumps(indices, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()