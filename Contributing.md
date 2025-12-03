# Contributing to ImageDedupX

感谢您对ImageDedupX的关注！我们欢迎各种形式的贡献。

## 🤝 如何贡献

### 报告Bug
如果您发现了bug，请通过GitHub Issues报告：
- 使用Bug报告模板
- 提供详细的复现步骤
- 包含系统环境信息（OS、Python版本、GPU信息）
- 如果可能，提供最小复现示例

### 提出功能建议
我们欢迎新功能建议：
- 在Issues中使用功能请求模板
- 清晰描述功能的使用场景和价值
- 说明为什么需要这个功能
- 如果可能，提供初步的实现思路

### 提交代码
我们欢迎代码贡献！请遵循以下流程：

1. **Fork本仓库**
   ```bash
   # 在GitHub上点击Fork按钮
   ```

2. **克隆您的Fork**
   ```bash
   git clone https://github.com/YOUR_USERNAME/imagedupx.git
   cd imagedupx
   ```

3. **创建功能分支**
   ```bash
   git checkout -b feature/AmazingFeature
   ```

4. **进行开发**
   - 编写代码
   - 添加测试
   - 更新文档

5. **提交更改**
   ```bash
   git add .
   git commit -m 'feat: add some AmazingFeature'
   ```

6. **推送到您的Fork**
   ```bash
   git push origin feature/AmazingFeature
   ```

7. **创建Pull Request**
   - 在GitHub上打开您的Fork
   - 点击"New Pull Request"
   - 填写PR模板
   - 等待review

## 📋 代码规范

### Python代码风格
我们遵循PEP 8 Python代码风格指南：

```python
# 好的示例
def extract_features(image_path: str, normalize: bool = True) -> np.ndarray:
    """
    Extract features from an image.
    
    Args:
        image_path: Path to the input image
        normalize: Whether to normalize the feature vector
        
    Returns:
        Normalized feature vector
        
    Raises:
        FileNotFoundError: If image file does not exist
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Feature extraction logic
    features = self._extract_raw_features(image_path)
    
    if normalize:
        features = features / np.linalg.norm(features)
    
    return features
```

**代码风格要求**：
- 使用4个空格缩进（不使用Tab）
- 行长度不超过100字符
- 使用类型注解（Type Hints）
- 为所有公共API编写docstring
- 变量名使用snake_case
- 类名使用PascalCase
- 常量使用UPPER_SNAKE_CASE

### 代码格式化工具
我们推荐使用以下工具：

```bash
# 安装工具
pip install black flake8 isort mypy

# 格式化代码
black imagedupx/

# 检查代码风格
flake8 imagedupx/ --max-line-length=100

# 排序import
isort imagedupx/

# 类型检查
mypy imagedupx/
```

### Commit消息规范
我们采用Conventional Commits规范：

**格式**：`<type>(<scope>): <subject>`

**类型（type）**：
- `feat`: 新功能
- `fix`: Bug修复
- `docs`: 文档更新
- `style`: 代码格式（不影响代码运行）
- `refactor`: 代码重构
- `perf`: 性能优化
- `test`: 测试相关
- `build`: 构建系统或外部依赖
- `ci`: CI配置文件和脚本
- `chore`: 其他不修改src或test的更改

**示例**：
```bash
feat(detector): add distributed indexing support
fix(search): resolve GPU memory leak in similarity search
docs(readme): update installation guide with GPU setup
refactor(features): improve feature extraction pipeline
test(detector): add unit tests for incremental updates
perf(index): optimize FAISS index building speed
```

### Pull Request规范

**PR标题**：
- 使用清晰描述性的标题
- 格式与commit消息类似
- 示例：`feat: add support for video frame deduplication`

**PR描述应包含**：
- 变更类型（Bug修复/新功能/文档更新等）
- 相关Issue编号（如 `Closes #123`）
- 详细的变更说明
- 测试情况说明
- 截图或示例（如适用）

**PR检查清单**：
- [ ] 代码遵循项目规范
- [ ] 已添加/更新单元测试
- [ ] 所有测试通过
- [ ] 已更新相关文档
- [ ] Commit消息符合规范
- [ ] 无新的编译警告
- [ ] 已review自己的代码

### Issue规范

**使用提供的Issue模板**：
- Bug报告：用于报告软件缺陷
- 功能请求：用于建议新功能
- 问题咨询：用于询问使用问题

**Issue最佳实践**：
- 使用清晰的标题
- 提供足够的上下文
- 一个Issue只关注一个问题
- 添加合适的标签
- 如果是Bug，提供复现步骤

## 🧪 测试要求

### 编写测试
所有新功能和Bug修复都应该包含测试：

```python
# tests/test_detector.py
import pytest
from imagedupx import IncrementalFAISSDetector

def test_detector_initialization():
    """测试检测器初始化"""
    detector = IncrementalFAISSDetector(
        enable_resnet=True,
        enable_vit=True,
        use_gpu=False
    )
    assert 'resnet' in detector.models
    assert 'vit' in detector.models

def test_feature_extraction():
    """测试特征提取"""
    detector = IncrementalFAISSDetector()
    features = detector.extract_combined_features('test_image.jpg')
    assert features is not None
    assert features.shape[0] == 3360  # 预期的特征维度

@pytest.mark.parametrize("threshold,expected_count", [
    (0.9, 5),
    (0.7, 10),
    (0.5, 15),
])
def test_search_with_different_thresholds(threshold, expected_count):
    """测试不同阈值下的搜索结果"""
    detector = IncrementalFAISSDetector()
    # ... 测试逻辑
```

### 运行测试
```bash
# 安装测试依赖
pip install pytest pytest-cov

# 运行所有测试
pytest tests/

# 运行特定测试文件
pytest tests/test_detector.py

# 运行测试并生成覆盖率报告
pytest tests/ --cov=imagedupx --cov-report=html

# 查看覆盖率报告
open htmlcov/index.html
```

### 测试要求
- 新功能必须有对应的单元测试
- 测试覆盖率应保持在80%以上
- 测试应该快速、独立、可重复
- 使用fixtures管理测试数据

## 📖 文档要求

### Docstring风格
我们使用Google风格的docstring：

```python
def search_similar_images(
    self,
    target_image: str,
    k: int = 10,
    threshold: float = 0.5
) -> List[Tuple[str, float]]:
    """
    Search for similar images in the index.
    
    This method extracts features from the target image and searches
    for the k most similar images in the FAISS index.
    
    Args:
        target_image: Path to the target/query image
        k: Number of similar images to return (default: 10)
        threshold: Similarity threshold, images with similarity below
            this value will be filtered out (default: 0.5)
    
    Returns:
        A list of tuples containing (image_path, similarity_score),
        sorted by similarity score in descending order
        
    Raises:
        FileNotFoundError: If target image does not exist
        ValueError: If k is less than 1 or threshold is not in [0, 1]
        
    Examples:
        >>> detector = IncrementalFAISSDetector()
        >>> detector.load_index('image_index.index')
        >>> results = detector.search_similar_images(
        ...     'query.jpg',
        ...     k=5,
        ...     threshold=0.7
        ... )
        >>> for path, score in results:
        ...     print(f"{path}: {score:.4f}")
    """
    # Implementation
```

### 更新文档
如果您的PR包含新功能或更改了现有功能：
- 更新README.md中的相关部分
- 在docs/目录下添加或更新详细文档
- 在docstring中添加使用示例
- 更新CHANGELOG.md

## ✅ Review流程

1. **提交PR**后，维护者会在48小时内进行review
2. **反馈修改**：根据review意见修改代码
3. **持续集成**：确保CI检查全部通过
4. **批准合并**：所有讨论解决后，PR将被合并
5. **感谢贡献**：您的名字将被添加到贡献者列表

### Review关注点
- 代码质量和可读性
- 测试覆盖率和质量
- 文档完整性
- 性能影响
- 向后兼容性

## 🎯 开发环境设置

### 推荐的开发环境
```bash
# 1. 克隆仓库
git clone https://github.com/yourusername/imagedupx.git
cd imagedupx

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 安装开发依赖
pip install -e ".[dev]"

# 4. 安装pre-commit hooks
pre-commit install

# 5. 运行测试确保环境正常
pytest tests/
```

### 推荐的IDE配置

**VS Code** (`.vscode/settings.json`):
```json
{
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "editor.formatOnSave": true,
    "editor.rulers": [100]
}
```

**PyCharm**:
- 启用PEP 8检查
- 配置Black作为代码格式化工具
- 设置pytest作为默认测试运行器

## 🐛 调试技巧

### 日志记录
```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

logger.debug("Detailed debug information")
logger.info("General information")
logger.warning("Warning message")
logger.error("Error message")
```

### 性能分析
```python
import cProfile
import pstats

# 性能分析
profiler = cProfile.Profile()
profiler.enable()

# 运行代码
detector.build_index(directory)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # 打印前20个最耗时的函数
```

## 💬 交流方式

### 提问前
- 查阅README和文档
- 搜索现有的Issues
- 尝试在StackOverflow搜索

### 获取帮助
- **GitHub Issues**: 技术问题和bug报告
- **Discussions**: 一般性讨论和想法交流
- **Email**: your.email@example.com（紧急问题）

### 行为准则
- 尊重所有贡献者
- 建设性的反馈
- 友好、包容的态度
- 专注于技术讨论

## 🎁 贡献者认可

我们重视每一位贡献者的付出：
- 您的名字将出现在README的贡献者列表中
- 重大贡献会在Release Notes中特别提及
- 定期评选"月度贡献者"

## 📜 许可证

通过贡献代码，您同意您的贡献将在MIT许可证下发布。

---

再次感谢您对ImageDedupX的贡献！每一个贡献都让这个项目变得更好。🎉

如有任何问题，请随时通过Issues联系我们。

**Happy Contributing!** 🚀
