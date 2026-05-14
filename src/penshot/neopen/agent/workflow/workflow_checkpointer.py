"""
@FileName: workflow_checkpointer.py
@Description: 工作流记忆器，状态检查点
@Author: HiPeng
@Github: https://github.com/neopen/story-shot-agent
@Time: 2026/5/14 20:55
"""
import glob
import os
import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver

from penshot import ShotConfig
from penshot.logger import debug, info, warning


class WorkflowCheckpointer:
    """
    工作流检查点类，用于保存和恢复工作流运行状态。
    """

    def __init__(self, script_id: str, task_id: str, config: ShotConfig):
        self.script_id = script_id
        self.task_id = task_id
        self.config = config

        if self.config.checkpoint_mode in ['sqlite', 'auto']:
            checkpoint_dir = self.config.checkpoint_dir

            # 为每个任务创建独立的数据库文件
            # db_path = os.path.join(checkpoint_dir, f"{self.script_id}_{self.task_id}.db")

            # 使用 script_id 作为子目录，实现数据隔离
            self.script_checkpoint_dir = os.path.join(checkpoint_dir, self.script_id)
            os.makedirs(self.script_checkpoint_dir, exist_ok=True)

            # 数据库文件路径
            self.db_path = os.path.join(self.script_checkpoint_dir, f"task_{self.task_id}.db")

    def create_checkpointer(self):
        """
        创建检查点存储器

        根据配置选择合适的存储器：
        - 开发/测试: MemorySaver
        - 生产环境: SqliteSaver (本地持久化)
        - 分布式部署: PostgresSaver (需要 PostgreSQL)
        """
        # 获取配置中的持久化模式，默认为 sqlite
        persistence_mode = self.config.checkpoint_mode

        if persistence_mode == 'memory':
            # 内存模式：适合开发测试，重启后丢失
            from langgraph.checkpoint.memory import MemorySaver
            info("使用 MemorySaver（内存模式）")
            return MemorySaver()

        elif persistence_mode == 'sqlite':
            # SQLite 模式：本地持久化，适合生产部署

            # 创建 SQLite 连接
            # check_same_thread=False 允许在多线程环境中使用（如 FastAPI）
            conn = sqlite3.connect(self.db_path, check_same_thread=False)

            # 创建 SqliteSaver 实例
            checkpointer = SqliteSaver(conn)
            info(f"使用 SqliteSaver，数据库路径: {self.db_path}")
            return checkpointer

        elif persistence_mode == 'postgres':
            # PostgreSQL 模式：适合分布式部署
            from langgraph.checkpoint.postgres import PostgresSaver
            import psycopg
            postgres_uri = self.config.postgres_uri
            if not postgres_uri:
                warning("未配置 PostgreSQL URI，回退到 SQLite")
                return self.create_checkpointer()  # 递归回退
            conn = psycopg.connect(postgres_uri)
            checkpointer = PostgresSaver(conn)
            info("使用 PostgresSaver（PostgreSQL 模式）")
            return checkpointer

        else:
            # 默认使用 SQLite
            return self.create_checkpointer()

    def create_checkpointer_context(self):
        """
        使用上下文管理器创建检查点存储器

        优势：自动管理连接生命周期
        """
        # 使用上下文管理器确保连接正确关闭
        from contextlib import contextmanager

        @contextmanager
        def managed_checkpointer():
            conn = None
            try:
                conn = sqlite3.connect(self.db_path, check_same_thread=False)
                conn.execute("PRAGMA journal_mode=WAL")
                checkpointer = SqliteSaver(conn)
                yield checkpointer
            finally:
                if conn:
                    conn.close()
                    debug("检查点连接已自动关闭")

        return managed_checkpointer()

    def _create_checkpointer_with_cleanup(self):
        """
        创建支持清理和管理的检查点存储器

        特性：
        1. 自动创建检查点目录
        2. 支持按 script_id 清理历史检查点
        3. 支持检查点数量限制
        """
        # 清理旧的检查点文件（可选，保留最近 N 个）
        max_checkpoints = getattr(self.config, 'max_checkpoints_per_script', 10)
        self._cleanup_old_checkpoints(self.script_checkpoint_dir, max_checkpoints)

        # 创建连接
        conn = sqlite3.connect(self.db_path, check_same_thread=False)

        # 启用 WAL 模式以提高并发性能
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")

        checkpointer = SqliteSaver(conn)
        info(f"检查点已创建: {self.db_path}")

        # 保存连接引用以便后续关闭
        self._checkpoint_conn = conn

        return checkpointer

    def _cleanup_old_checkpoints(self, directory: str, keep_count: int):
        """清理旧的检查点文件，保留最近 keep_count 个"""
        try:

            # 匹配 task_*.db 文件
            pattern = os.path.join(directory, "task_*.db")
            files = glob.glob(pattern)

            # 按修改时间排序
            files.sort(key=os.path.getmtime, reverse=True)

            # 删除多余的文件
            for old_file in files[keep_count:]:
                try:
                    os.remove(old_file)
                    debug(f"已清理旧检查点: {old_file}")
                except Exception as e:
                    warning(f"清理检查点失败: {old_file}, {e}")
        except Exception as e:
            warning(f"清理检查点时出错: {e}")

    def close_checkpoint(self):
        """关闭检查点数据库连接（在任务完成时调用）"""
        if hasattr(self, '_checkpoint_conn') and self._checkpoint_conn:
            try:
                self._checkpoint_conn.close()
                debug("检查点数据库连接已关闭")
            except Exception as e:
                warning(f"关闭检查点连接失败: {e}")
