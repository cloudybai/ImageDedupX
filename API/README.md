# FAISS图像相似度检测服务

基于原始 [similarities](https://github.com/cloudybai/similarities) 项目封装的RESTful API服务，提供高性能的图像相似度检测功能。

## 功能特性

- 🚀 **高性能检索**: 基于FAISS向量搜索引擎，支持百万级图像库实时检索
- 🧠 **多模态特征**: 融合ResNet-50、Vision Transformer和传统CV特征
- 🔧 **RESTful API**: 标准化的Web API接口，易于集成
- 📦 **Docker部署**: 开箱即用的容器化部署方案
- 🎯 **高精度匹配**: 检索精度可达90%以上
- ⚡ **异步处理**: 支持后台异步索引构建

## 系统架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Client    │    │   Mobile App    │    │  Other Services │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────▼──────────────┐
                    │      Nginx (Optional)      │
                    └─────────────┬──────────────┘
                                 │
                    ┌─────────────▼──────────────┐
                    │   FAISS Image Service      │
                    │   - Flask REST API         │
                    │   - Multi-modal Features   │
                    │   - FAISS Index            │
                    └─────────────┬──────────────┘
                                 │
                    ┌─────────────▼──────────────┐
                    │    File Storage            │
                    │   - Images                 │
                    │   - Indices                │
                    │   - Cache                  │
                    └────────────────────────────┘
```

## 快速开始

### 方式1: Docker部署（推荐）

1. **克隆项目**
```bash
git clone https://github.com/cloudybai/similarities.git
cd similarities
```

2. **准备配置文件**
```bash
# 将配置文件复制到项目根目录
cp config.json.example config.json
# 根据需要修改配置
```

3. **启动服务**
```bash
# 构建并启动
docker-compose up -d

# 查看日志
docker-compose logs -f faiss-image-service
```

4. **验证服务**
```bash
curl http://localhost:8080/api/v1/health
```

### 方式2: 本地部署

1. **安装依赖**
```bash
pip install -r requirements.txt
```

2. **准备原始检测器**
```bash
# 确保faiss_image_similarity.py在同一目录
```

3. **启动服务**
```bash
python faiss_service.py --config config.json
```

## API使用说明

### 1. 健康检查
```bash
GET /api/v1/health
```

### 2. 构建索引
```bash
POST /api/v1/build_index
Content-Type: application/json

{
  "index_name": "my_index",
  "image_directory": "/path/to/images",
  "model_config": {
    "enable_resnet": true,
    "enable_vit": true,
    "enable_traditional": true,
    "index_type": "flat",
    "use_gpu": false
  },
  "cache_file": "/path/to/cache.pkl"
}
```

### 3. 搜索相似图片
```bash
POST /api/v1/search
Content-Type: multipart/form-data

form-data:
- image: [图片文件]
- index_name: "my_index"
- top_k: 10
- threshold: 0.5
```

### 4. 获取服务状态
```bash
GET /api/v1/status
```

### 5. 列出所有索引
```bash
GET /api/v1/indices
```

### 6. 删除索引
```bash
DELETE /api/v1/indices/{index_name}
```

## Python客户端使用

```python
from client_example import FAISSImageClient

# 创建客户端
client = FAISSImageClient("http://localhost:8080")

# 构建索引
result = client.build_index(
    index_name="test_index",
    image_directory="/path/to/images"
)

# 等待索引构建完成
client.wait_for_index_ready("test_index")

# 搜索相似图片
results = client.search_similar(
    image_path="/path/to/query.jpg",
    index_name="test_index",
    top_k=5,
    threshold=0.7
)

print(f"找到 {len(results['results'])} 张相似图片")
```

## 配置说明

### config.json 参数详解

```json
{
  "host": "0.0.0.0",              // 服务绑定地址
  "port": 8080,                   // 服务端口
  "debug": false,                 // 调试模式
  "max_file_size": 16777216,      // 最大文件大小(16MB)
  "upload_folder": "./uploads",   // 上传临时目录
  "index_folder": "./indices",    // 索引存储目录
  "cache_folder": "./cache",      // 缓存目录
  "allowed_extensions": [         // 支持的图片格式
    "jpg", "jpeg", "png", "bmp", "gif", "tiff", "webp"
  ],
  "enable_cors": true,            // 启用CORS
  "log_level": "INFO"             // 日志级别
}
```

### 模型配置参数

```json
{
  "enable_resnet": true,          // 启用ResNet特征 (权重0.3)
  "enable_vit": true,             // 启用ViT特征 (权重0.5)
  "enable_traditional": true,     // 启用传统CV特征 (权重0.2)
  "index_type": "flat",           // FAISS索引类型
  "use_gpu": false                // 是否使用GPU加速
}
```

## 性能调优

### 1. 相似度阈值设置
- **高精度场景**: threshold >= 0.85
- **高召回场景**: threshold >= 0.65  
- **探索性检索**: threshold >= 0.45

### 2. 硬件优化
- **CPU**: 推荐8核以上，支持AVX2指令集
- **内存**: 建议16GB以上，约4KB/张图片
- **GPU**: 支持CUDA的NVIDIA显卡（可选）

### 3. 索引类型选择
- **Flat索引**: 精度最高，适合中小规模数据集
- **IVF索引**: 速度较快，适合大规模数据集
- **HNSW索引**: 内存效率高，适合内存受限环境

## 生产环境部署

### 使用Nginx + Docker Compose

```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  faiss-service:
    build: .
    volumes:
      - /data/images:/app/data:ro
      - ./indices:/app/indices
      - ./cache:/app/cache
    environment:
      - WORKERS=4
    restart: always
    
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - faiss-service
    restart: always
```

### 监控和日志

```bash
# 查看服务状态
curl http://localhost:8080/api/v1/status

# 查看容器日志
docker-compose logs -f faiss-image-service

# 监控资源使用
docker stats faiss-image-service
```

## 故障排除

### 常见问题

1. **内存不足**
   - 减少图片数量或降低特征维度
   - 增加系统内存或使用内存映射

2. **索引构建失败**
   - 检查图片目录权限
   - 确保图片格式支持
   - 查看详细错误日志

3. **搜索速度慢**
   - 考虑使用GPU加速
   - 调整FAISS索引类型
   - 增加系统资源

4. **特征提取失败**
   - 检查深度学习模型是否正确加载
   - 确保图片文件完整性
   - 验证依赖库版本

### 调试模式

```bash
# 启用调试模式
python faiss_service.py --debug --config config.json

# 查看详细日志
export LOG_LEVEL=DEBUG
python faiss_service.py --config config.json
```

## 贡献指南

1. Fork项目仓库
2. 创建功能分支: `git checkout -b feature/new-feature`
3. 提交更改: `git commit -am 'Add new feature'`
4. 推送分支: `git push origin feature/new-feature`
5. 提交Pull Request

## 许可证

本项目基于MIT许可证开源，详见 [LICENSE](LICENSE) 文件。

## 技术支持

- 📧 邮箱: cloud.bai@outlook.com
- 🔗 项目主页: https://github.com/cloudybai/similarities
- 📖 技术文档: https://github.com/cloudybai/similarities/wiki

---

**版本**: v2.0.0  
**最后更新**: 2025年7月