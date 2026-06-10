# PenShot 代码规范

## Python 版本
- 最低支持: Python 3.9
- 推荐使用: Python 3.10+

## 代码风格

### 格式化工具
```bash
# 使用 Black 自动格式化
black src/ tests/

# 使用 isort 排序导入
isort src/ tests/

# 使用 flake8 检查
flake8 src/ tests/ --max-line-length=88
```

### 命名规范

```python
# 类名: PascalCase
class ScriptParser:
    pass

# 函数/方法: snake_case
def parse_script():
    pass

# 常量: UPPER_SNAKE_CASE
MAX_RETRY_COUNT = 3

# 私有成员: 单下划线前缀
_internal_cache = {}

# 特殊方法: 双下划线 (保持 Python 内置风格)
def __init__(self):
    pass
```

### 类型提示

```python
from typing import List, Dict, Optional, Union, Callable, Awaitable

# 函数签名必须包含类型提示
async def process_script(
    script: str,
    max_fragments: int = 10,
    callback: Optional[Callable[[str], Awaitable[None]]] = None
) -> Dict[str, Union[List[dict], str]]:
    """处理剧本"""
    pass

# 复杂类型使用 TypeAlias
from typing import TypeAlias
FragmentList: TypeAlias = List[Dict[str, Union[str, float]]]
```

### 文档字符串 (Google 风格)

```python
def function_name(param1: str, param2: int) -> bool:
    """
    函数功能描述
    
    Args:
        param1: 参数1说明
        param2: 参数2说明
        
    Returns:
        返回值说明
        
    Raises:
        ValueError: 异常情况说明
        
    Examples:
        >>> function_name("test", 123)
        True
    """
    pass
```

## 项目结构规范

```
video-shot-agent/
├── data/                       # 数据目录
│   ├── checkpoints/            # 检查点数据
│   ├── embedding/              # 嵌入向量数据
│   ├── memory/                 # 记忆数据
│   ├── models/                 # 模型文件
│   ├── output/                 # 输出目录
│   └── template/               # 模板文件
├── docs/                       # 文档
├── examples/                   # 示例代码
├── logs/                       # 日志文件
├── scripts/                    # 脚本工具
├── src/penshot/                # 源代码主目录
│   ├── api/                    # API 接口
│   ├── app/                    # 应用核心
│   ├── config/                 # 配置管理
│   ├── neopen/                 # NeoOpen 核心模块
│   │   ├── agent/              # 智能体模块
│   │   ├── cache/              # 缓存模块
│   │   ├── client/             # 客户端模块
│   │   ├── config/             # 配置模块
│   │   ├── knowledge/          # 知识模块
│   │   ├── prompts/            # Prompts 模块
│   │   ├── task/               # 任务模块
│   │   ├── tools/              # 工具模块
│   │   ├── shot_config.py      # 分镜配置
│   │   ├── shot_context.py     # 分镜上下文
│   │   └── shot_language.py    # 分镜语言
│   ├── utils/                  # 工具函数
│   ├── http_server.py          # HTTP 服务器
│   ├── logger.py               # 日志记录器
│   ├── mcp_http_server.py      # MCP HTTP 服务器
│   └── mcp_server.py           # MCP 服务器
├── tests/                      # 测试目录
│   ├── api/                    # API 测试
│   ├── benchmarks/             # 基准测试
│   ├── e2e/                    # 端到端测试
│   ├── fixtures/               # 测试夹具
│   ├── helpers/                # 测试辅助函数
│   ├── integration/            # 集成测试
│   ├── knowledge/              # 知识库测试
│   ├── planner/                # 规划器测试
│   ├── task/                   # 任务测试
│   ├── unit/                   # 单元测试
│   ├── workflow/               # 工作流测试
│   ├── conftest.py             # 测试配置
│   ├── test_config.py          # 配置测试
├── .env.example                # 环境变量示例
├── .gitignore                  # Git 忽略文件
├── .pre-commit-config.yaml     # Pre-commit 配置
├── CHANGELOG.md                # 变更日志
├── CODE_OF_CONDUCT.md          # 行为准则
├── CONTRIBUTING.md             # 贡献指南
├── CONTRIBUTORS.md             # 贡献者名单
├── README.md                   # 英文说明文档
├── README_zh.md                # 中文说明文档
├── SECURITY.md                 # 安全政策
├── docker-compose.yml          # Docker Compose 配置
├── main.py                     # 主入口文件
└── pyproject.toml              # 项目配置文件
```

## 错误处理

```python
# 使用自定义异常
class PenShotError(Exception):
    """基础异常"""
    pass

class ConfigurationError(PenShotError):
    """配置错误"""
    pass

class AgentExecutionError(PenShotError):
    """Agent 执行错误"""
    pass

# 错误处理模式
async def safe_operation():
    try:
        result = await risky_operation()
        return result
    except SpecificError as e:
        error(f"具体错误: {e}")
        raise AgentExecutionError from e
    except Exception as e:
        exception("未预期错误")
        raise PenShotError(f"操作失败: {e}") from e
```

## 异步编程

```python
# 优先使用异步实现
async def fetch_data() -> dict:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()

# 提供同步包装
def fetch_data_sync() -> dict:
    return asyncio.run(fetch_data())

# 并发控制
from asyncio import Semaphore

semaphore = Semaphore(10)

async def limited_operation():
    async with semaphore:
        return await operation()
```

## 日志规范

```python
from penshot.logger import debug, error, info

# 日志级别使用
debug("调试信息，仅在开发环境")
info("正常流程信息")
warning("警告信息，不影响运行")
error("错误信息，需要关注")
exception("异常信息，包含堆栈")
```

## 测试规范

```python
# tests/unit/test_example.py
import pytest
from penshot.module import Function

class TestFunction:
    @pytest.fixture
    def setup_data(self):
        """测试数据准备"""
        return {"key": "value"}
    
    @pytest.mark.asyncio
    async def test_async_function(self, setup_data):
        """测试异步函数"""
        result = await Function.process(setup_data)
        assert result["success"] is True
    
    def test_sync_function(self):
        """测试同步函数"""
        result = Function.sync_process("input")
        assert result == "expected"
    
    @pytest.mark.parametrize("input,expected", [
        ("a", 1),
        ("b", 2),
    ])
    def test_parametrized(self, input, expected):
        """参数化测试"""
        assert len(input) == expected
```

## Git 提交规范

```python
# Conventional Commits 格式
<type>(<scope>): <subject>

# Type 类型
feat:     新功能
fix:      修复
docs:     文档
style:    格式
refactor: 重构
test:     测试
chore:    构建/工具

# 示例
feat(agent): 添加新的剧本解析器
fix(memory): 修复 Redis 连接泄漏
docs(api): 更新 REST API 文档
test(parser): 增加边界情况测试
```

## 性能要求

- LLM 调用超时: 30秒
- 单任务处理时间: < 60秒
- 并发支持: 5-10 任务
- 内存使用: < 2GB
- 响应时间 (API): < 500ms (不含 LLM)

## 安全检查

```python
# 不在代码中硬编码密钥
# 使用环境变量
import os
API_KEY = os.getenv("PENSHOT_API_KEY")

# 验证输入
def validate_script(script: str) -> bool:
    if len(script) > 100000:  # 限制长度
        return False
    # 检查危险内容
    return True

# 使用参数化查询避免注入
# 使用 HTTPS 进行 API 调用
```

## 代码审查检查项

- 代码通过所有 CI 检查
- 添加/更新了测试
- 更新了相关文档
- 没有引入 breaking changes (或已充分文档化)
- 错误处理完善
- 日志记录适当
- 类型提示完整
- 性能影响评估