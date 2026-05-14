"""
@FileName: workflow_checkpointer.py
@Description: 工作流记忆器，状态检查点
@Author: HiPeng
@Github: https://github.com/neopen/story-shot-agent
@Time: 2026/5/14 20:55
"""
import asyncio
import glob
import os
import sqlite3
import threading
import time
import weakref
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional

from langgraph.checkpoint.sqlite import SqliteSaver

from penshot import ShotConfig
from penshot.logger import debug, info, warning, error


class CheckpointMode(str, Enum):
    """检查点存储模式"""
    MEMORY = "memory"  # 内存模式（开发测试）
    SQLITE = "sqlite"  # SQLite 模式（本地生产）
    POSTGRES = "postgres"  # PostgreSQL 模式（分布式）


class CheckpointState(str, Enum):
    """检查点状态"""
    ACTIVE = "active"  # 活跃中
    CLOSED = "closed"  # 已关闭
    CORRUPTED = "corrupted"  # 已损坏


@dataclass
class CheckpointStats:
    """检查点统计信息"""
    script_id: str
    task_id: str
    mode: str
    state: CheckpointState = CheckpointState.ACTIVE
    total_checkpoints: int = 0
    last_checkpoint_time: Optional[datetime] = None
    storage_path: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    total_size_bytes: int = 0
    is_thread_safe: bool = False


class CheckpointConnectionPool:
    """
    检查点连接池
    管理 SQLite 连接，支持线程池操作
    """

    def __init__(self, db_path: str, max_connections: int = 5,
                 idle_timeout: int = 300):
        self.db_path = db_path
        self.max_connections = max_connections
        self.idle_timeout = idle_timeout
        self._connections: List[sqlite3.Connection] = []
        self._in_use: Dict[int, sqlite3.Connection] = {}
        self._lock = threading.RLock()
        self._idle_check_thread: Optional[threading.Thread] = None
        self._stop_idle_check = threading.Event()

        # 确保目录存在
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        # 启动空闲连接清理线程
        self._start_idle_checker()

    def _start_idle_checker(self):
        """启动空闲连接检查线程"""

        def idle_checker():
            while not self._stop_idle_check.is_set():
                time.sleep(60)  # 每分钟检查一次
                self._cleanup_idle_connections()

        self._idle_check_thread = threading.Thread(
            target=idle_checker,
            daemon=True,
            name=f"CheckpointPool-{os.path.basename(self.db_path)}"
        )
        self._idle_check_thread.start()

    def _cleanup_idle_connections(self):
        """清理空闲连接"""
        with self._lock:
            now = time.time()
            for conn in self._connections[:]:
                if hasattr(conn, '_last_used') and now - conn._last_used > self.idle_timeout:
                    try:
                        conn.close()
                        self._connections.remove(conn)
                        debug(f"已清理空闲连接: {conn}")
                    except Exception as e:
                        warning(f"清理空闲连接失败: {e}")

    def _create_connection(self) -> sqlite3.Connection:
        """创建新的数据库连接"""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)

        # 优化性能
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-20000")  # 20MB cache
        conn.execute("PRAGMA temp_store=MEMORY")

        conn._last_used = time.time()
        return conn

    def get_connection(self) -> sqlite3.Connection:
        """获取连接（线程安全）"""
        with self._lock:
            if self._connections:
                conn = self._connections.pop()
                conn._last_used = time.time()
            else:
                conn = self._create_connection()

            thread_id = threading.get_ident()
            self._in_use[thread_id] = conn
            return conn

    def return_connection(self, conn: sqlite3.Connection):
        """归还连接（线程安全）"""
        with self._lock:
            thread_id = threading.get_ident()
            if thread_id in self._in_use:
                del self._in_use[thread_id]

            if conn in self._in_use.values():
                return  # 连接还被其他线程使用

            # 检查连接是否有效
            try:
                conn.cursor().execute("SELECT 1").fetchone()
                conn._last_used = time.time()
                if len(self._connections) < self.max_connections:
                    self._connections.append(conn)
                else:
                    conn.close()
            except Exception:
                # 连接已损坏，直接关闭
                try:
                    conn.close()
                except Exception:
                    pass

    def close_all(self):
        """关闭所有连接"""
        self._stop_idle_check.set()

        with self._lock:
            for conn in self._connections:
                try:
                    conn.close()
                except Exception:
                    pass
            self._connections.clear()

            for conn in self._in_use.values():
                try:
                    conn.close()
                except Exception:
                    pass
            self._in_use.clear()

        if self._idle_check_thread:
            self._idle_check_thread.join(timeout=5)

        info("检查点连接池已关闭")

    def get_stats(self) -> Dict[str, Any]:
        """获取连接池统计信息"""
        with self._lock:
            return {
                "total_connections": len(self._connections) + len(self._in_use),
                "idle_connections": len(self._connections),
                "in_use_connections": len(self._in_use),
                "max_connections": self.max_connections,
                "db_path": self.db_path
            }


class WorkflowCheckpointer:
    """
    工作流检查点类 - 增强版
    支持：线程池操作、多种存储器自动选择、上下文管理器、异步清理
    """

    def __init__(self, script_id: str, task_id: str, config: ShotConfig):
        self.script_id = script_id
        self.task_id = task_id
        self.config = config

        # 连接池实例（懒加载）
        self._connection_pool: Optional[CheckpointConnectionPool] = None
        self._checkpointer = None
        self._state = CheckpointState.ACTIVE
        self._stats = CheckpointStats(
            script_id=script_id,
            task_id=task_id,
            mode=config.checkpoint_mode or "sqlite"
        )

        # 确定存储模式
        self._mode = self._determine_mode()
        self._stats.mode = self._mode.value

        # 设置存储路径
        self._setup_storage_path()

        # 弱引用回调，用于自动清理
        self._finalizer = weakref.finalize(self, self._auto_cleanup, self._stats)

        info(f"WorkflowCheckpointer 初始化: script={script_id}, task={task_id}, mode={self._mode.value}")

    def _determine_mode(self) -> CheckpointMode:
        """自动确定合适的存储模式"""
        mode = getattr(self.config, 'checkpoint_mode', 'auto')

        if mode == 'auto':
            # 自动选择：优先 SQLite，降级到 Memory
            try:
                # 检查是否可以写入 SQLite
                test_path = os.path.join(self.config.checkpoint_dir, ".write_test")
                os.makedirs(os.path.dirname(test_path), exist_ok=True)
                with open(test_path, 'w') as f:
                    f.write('test')
                os.remove(test_path)
                return CheckpointMode.SQLITE
            except Exception:
                warning("SQLite 不可用，降级到内存模式")
                return CheckpointMode.MEMORY

        elif mode == 'memory':
            return CheckpointMode.MEMORY
        elif mode == 'sqlite':
            return CheckpointMode.SQLITE
        elif mode == 'postgres':
            return CheckpointMode.POSTGRES
        else:
            return CheckpointMode.SQLITE

    def _setup_storage_path(self):
        """设置存储路径"""
        if self._mode == CheckpointMode.SQLITE:
            checkpoint_dir = self.config.checkpoint_dir
            self.script_checkpoint_dir = os.path.join(checkpoint_dir, self.script_id)
            os.makedirs(self.script_checkpoint_dir, exist_ok=True)
            self.db_path = os.path.join(self.script_checkpoint_dir, f"task_{self.task_id}.db")
            self._stats.storage_path = self.db_path
        elif self._mode == CheckpointMode.MEMORY:
            self._stats.storage_path = "memory"
        elif self._mode == CheckpointMode.POSTGRES:
            self._stats.storage_path = self.config.postgres_uri

    def _get_connection_pool(self) -> CheckpointConnectionPool:
        """获取或创建连接池（线程安全）"""
        if self._connection_pool is None:
            if self._mode == CheckpointMode.SQLITE:
                self._connection_pool = CheckpointConnectionPool(
                    self.db_path,
                    max_connections=getattr(self.config, 'checkpoint_max_connections', 5),
                    idle_timeout=getattr(self.config, 'checkpoint_idle_timeout', 300)
                )
        return self._connection_pool

    def create_checkpointer(self):
        """
        创建检查点存储器

        根据配置选择合适的存储器：
        - 开发/测试: MemorySaver
        - 生产环境: SqliteSaver (本地持久化)
        - 分布式部署: PostgresSaver (需要 PostgreSQL)
        """
        if self._checkpointer is not None:
            return self._checkpointer

        try:
            if self._mode == CheckpointMode.MEMORY:
                from langgraph.checkpoint.memory import MemorySaver
                self._checkpointer = MemorySaver()
                info("使用 MemorySaver（内存模式）")

            elif self._mode == CheckpointMode.SQLITE:
                # 使用连接池管理的连接
                pool = self._get_connection_pool()
                conn = pool.get_connection()

                # 初始化数据库表（如果需要）
                self._init_sqlite_schema(conn)

                self._checkpointer = SqliteSaver(conn)
                info(f"使用 SqliteSaver，数据库路径: {self.db_path}")

            elif self._mode == CheckpointMode.POSTGRES:
                from langgraph.checkpoint.postgres import PostgresSaver
                import psycopg

                postgres_uri = getattr(self.config, 'postgres_uri', None)
                if not postgres_uri:
                    warning("未配置 PostgreSQL URI，回退到 SQLite")
                    self._mode = CheckpointMode.SQLITE
                    return self.create_checkpointer()

                conn = psycopg.connect(postgres_uri)
                self._checkpointer = PostgresSaver(conn)
                info("使用 PostgresSaver（PostgreSQL 模式）")

            else:
                return self.create_checkpointer()

            self._stats.is_thread_safe = self._checkpointer_is_thread_safe()
            return self._checkpointer

        except Exception as e:
            error(f"创建检查点失败: {e}")
            self._state = CheckpointState.CORRUPTED
            # 降级到内存模式
            warning("降级到内存模式")
            self._mode = CheckpointMode.MEMORY
            return self.create_checkpointer()

    def _checkpointer_is_thread_safe(self) -> bool:
        """检查 checkpointer 是否线程安全"""
        if self._mode == CheckpointMode.SQLITE:
            # SqliteSaver 使用连接池保证线程安全
            return True
        elif self._mode == CheckpointMode.MEMORY:
            return True  # MemorySaver 是线程安全的
        elif self._mode == CheckpointMode.POSTGRES:
            return True
        return False

    def _init_sqlite_schema(self, conn: sqlite3.Connection):
        """初始化 SQLite 数据库表"""
        try:
            cursor = conn.cursor()

            # 创建检查点表（如果不存在）
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    thread_id TEXT NOT NULL,
                    checkpoint_id TEXT NOT NULL,
                    checkpoint_data BLOB NOT NULL,
                    parent_checkpoint_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (thread_id, checkpoint_id)
                )
            """)

            # 创建元数据表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS checkpoint_metadata (
                    script_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    checkpoint_id TEXT NOT NULL,
                    metadata_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (script_id, task_id, checkpoint_id)
                )
            """)

            # 创建索引
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_checkpoints_created 
                ON checkpoints(created_at)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_metadata_task 
                ON checkpoint_metadata(script_id, task_id)
            """)

            conn.commit()
            debug("SQLite 数据库表初始化完成")
        except Exception as e:
            warning(f"初始化数据库表失败: {e}")

    def get_thread_id(self) -> str:
        """获取线程ID（用于 LangGraph 的 config）"""
        return f"{self.script_id}_{self.task_id}"

    def get_config(self) -> Dict[str, Any]:
        """获取 LangGraph 运行配置"""
        return {
            "configurable": {
                "thread_id": self.get_thread_id()
            }
        }

    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """获取数据库连接（上下文管理器）"""
        if self._mode != CheckpointMode.SQLITE:
            raise RuntimeError("只有 SQLite 模式支持直接获取连接")

        pool = self._get_connection_pool()
        conn = pool.get_connection()
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            pool.return_connection(conn)

    @asynccontextmanager
    async def async_get_connection(self) -> AsyncGenerator[sqlite3.Connection, None]:
        """异步获取数据库连接（上下文管理器）"""
        if self._mode != CheckpointMode.SQLITE:
            raise RuntimeError("只有 SQLite 模式支持直接获取连接")

        # 在线程池中执行同步操作
        loop = asyncio.get_event_loop()
        pool = await loop.run_in_executor(None, self._get_connection_pool)
        conn = await loop.run_in_executor(None, pool.get_connection)

        try:
            yield conn
            await loop.run_in_executor(None, conn.commit)
        except Exception as e:
            await loop.run_in_executor(None, conn.rollback)
            raise e
        finally:
            await loop.run_in_executor(None, pool.return_connection, conn)

    async def cleanup_old_checkpoints_async(self, keep_count: int = 10):
        """
        异步清理旧的检查点文件

        Args:
            keep_count: 保留最近的文件数量
        """
        if self._mode != CheckpointMode.SQLITE:
            return

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._cleanup_old_checkpoints, keep_count)

    def _cleanup_old_checkpoints(self, keep_count: int = 10):
        """清理旧的检查点文件（同步版本）"""
        try:
            pattern = os.path.join(self.script_checkpoint_dir, "task_*.db")
            files = glob.glob(pattern)

            # 按修改时间排序
            files.sort(key=os.path.getmtime, reverse=True)

            # 删除多余的文件
            removed_count = 0
            for old_file in files[keep_count:]:
                try:
                    # 检查是否还有其他任务在使用这个文件
                    if self._is_file_in_use(old_file):
                        debug(f"文件正在使用，跳过清理: {old_file}")
                        continue

                    os.remove(old_file)
                    removed_count += 1
                    debug(f"已清理旧检查点: {old_file}")
                except Exception as e:
                    warning(f"清理检查点失败: {old_file}, {e}")

            if removed_count > 0:
                info(f"已清理 {removed_count} 个旧检查点文件")

        except Exception as e:
            warning(f"清理检查点时出错: {e}")

    def _is_file_in_use(self, file_path: str) -> bool:
        """检查文件是否正在被使用"""
        try:
            # 尝试以独占模式打开文件
            with open(file_path, 'r+b') as f:
                try:
                    import fcntl
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    fcntl.flock(f, fcntl.LOCK_UN)
                    return False
                except (ImportError, OSError):
                    # Windows 或无法使用 fcntl
                    return False
        except (IOError, OSError):
            return True
        return False

    async def save_metadata(self, key: str, value: Any):
        """异步保存元数据"""
        if self._mode != CheckpointMode.SQLITE:
            return

        import json
        loop = asyncio.get_event_loop()

        async with self.async_get_connection() as conn:
            def _save():
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO checkpoint_metadata 
                    (script_id, task_id, checkpoint_id, metadata_json)
                    VALUES (?, ?, ?, ?)
                """, (self.script_id, self.task_id, key, json.dumps(value)))
                conn.commit()

            await loop.run_in_executor(None, _save)

        self._stats.last_accessed = datetime.now()

    async def load_metadata(self, key: str) -> Optional[Any]:
        """异步加载元数据"""
        if self._mode != CheckpointMode.SQLITE:
            return None

        import json
        loop = asyncio.get_event_loop()

        async with self.async_get_connection() as conn:
            def _load():
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT metadata_json FROM checkpoint_metadata
                    WHERE script_id = ? AND task_id = ? AND checkpoint_id = ?
                    ORDER BY created_at DESC LIMIT 1
                """, (self.script_id, self.task_id, key))
                row = cursor.fetchone()
                if row:
                    return json.loads(row[0])
                return None

            return await loop.run_in_executor(None, _load)

    def get_stats(self) -> CheckpointStats:
        """获取检查点统计信息"""
        self._stats.last_accessed = datetime.now()

        # 更新文件大小统计
        if self._mode == CheckpointMode.SQLITE and os.path.exists(self.db_path):
            try:
                self._stats.total_size_bytes = os.path.getsize(self.db_path)
            except Exception:
                pass

        # 更新检查点数量
        if self._checkpointer is not None:
            try:
                # 尝试获取 checkpoint 数量
                self._stats.total_checkpoints = self._get_checkpoint_count()
            except Exception:
                pass

        return self._stats

    def _get_checkpoint_count(self) -> int:
        """获取检查点数量"""
        if self._mode == CheckpointMode.SQLITE:
            try:
                with self.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?",
                        (self.get_thread_id(),)
                    )
                    row = cursor.fetchone()
                    return row[0] if row else 0
            except Exception:
                return 0
        return 0

    async def async_get_stats(self) -> CheckpointStats:
        """异步获取检查点统计信息"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_stats)

    def close(self):
        """关闭检查点"""
        if self._state == CheckpointState.CLOSED:
            return

        self._state = CheckpointState.CLOSED

        if self._connection_pool:
            self._connection_pool.close_all()
            self._connection_pool = None

        info(f"检查点已关闭: script={self.script_id}, task={self.task_id}")

    @staticmethod
    def _auto_cleanup(stats: CheckpointStats):
        """自动清理回调（弱引用触发）"""
        info(f"自动清理检查点: {stats.script_id}/{stats.task_id}")

    async def __aenter__(self):
        """异步上下文管理器入口"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        self.close()

    def __enter__(self):
        """同步上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """同步上下文管理器出口"""
        self.close()


# ============================================================================
# 工厂函数
# ============================================================================

class CheckpointerFactory:
    """
    检查点工厂类
    管理多个检查点实例，支持线程池操作
    """

    _instances: Dict[str, WorkflowCheckpointer] = {}
    _lock = threading.RLock()

    @classmethod
    def get_or_create(cls, script_id: str, task_id: str, config: ShotConfig) -> WorkflowCheckpointer:
        """获取或创建检查点实例"""
        key = f"{script_id}_{task_id}"

        with cls._lock:
            if key not in cls._instances:
                cls._instances[key] = WorkflowCheckpointer(script_id, task_id, config)
            return cls._instances[key]

    @classmethod
    def close_all(cls):
        """关闭所有检查点"""
        with cls._lock:
            for key, checkpointer in cls._instances.items():
                try:
                    checkpointer.close()
                except Exception as e:
                    warning(f"关闭检查点失败: {key}, {e}")
            cls._instances.clear()

    @classmethod
    def get_stats(cls) -> Dict[str, Any]:
        """获取所有检查点的统计信息"""
        with cls._lock:
            return {
                "total_instances": len(cls._instances),
                "instances": {
                    key: cp.get_stats().__dict__
                    for key, cp in cls._instances.items()
                }
            }


# ============================================================================
# 便捷函数
# ============================================================================

def create_checkpointer(script_id: str, task_id: str, config: ShotConfig) -> WorkflowCheckpointer:
    """创建检查点实例（便捷函数）"""
    return CheckpointerFactory.get_or_create(script_id, task_id, config)


def close_all_checkpointers():
    """关闭所有检查点"""
    CheckpointerFactory.close_all()
