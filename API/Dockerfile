FROM ubuntu:latest
LABEL authors="cloudbai"

ENTRYPOINT ["top", "-b"]

# Dockerfile
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 创建必要的目录
RUN mkdir -p uploads indices cache

# 设置环境变量
ENV PYTHONPATH=/app
ENV FLASK_APP=faiss_service.py

# 暴露端口
EXPOSE 8080

# 健康检查
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/api/v1/health || exit 1

# 启动命令
CMD ["python", "faiss_service.py", "--config", "config.json"]

# ==========================================
# docker-compose.yml
# ==========================================

version: '3.8'

services:
  faiss-image-service:
    build: .
    ports:
      - "8080:8080"
    volumes:
      - ./data:/app/data
      - ./indices:/app/indices
      - ./cache:/app/cache
      - ./uploads:/app/uploads
      - ./config.json:/app/config.json
    environment:
      - PYTHONPATH=/app
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/api/v1/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  # 可选：添加Redis用于缓存
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  # 可选：添加Nginx反向代理
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - faiss-image-service
    restart: unless-stopped

volumes:
  redis_data: