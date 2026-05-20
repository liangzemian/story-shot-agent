"""
@FileName: llama_index_knowledge.py
@Description: 剧本知识库管理模块，提供基于LlamaIndex的结构化剧本知识库管理功能
@Author: HiPeng
@Github: https://github.com/neopen/story-shot-agent
@Time: 2025/12/18
"""

import json
import os
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Any

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.node_parser import SentenceSplitter, SentenceWindowNodeParser
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import Document
from llama_index.core.vector_stores import SimpleVectorStore

from penshot.config.config import settings
from penshot.logger import debug, info, error, warning
from penshot.neopen.agent.script_parser.script_parser_models import ParsedScript
from penshot.neopen.tools.script_parser_tool import parse_script_to_documents, parse_script_file_to_documents


class ScriptKnowledgeBase:
    """
    剧本知识库管理类
    负责剧本的解析、索引创建、检索优化等功能
    支持按 script_id 隔离向量存储
    """

    def __init__(self,
                 embeddings: Optional[BaseEmbedding],
                 script_id: str,
                 storage_dir: str = settings.get_data_paths().get("data_embedding"),
                 chunk_size: int = 512,
                 chunk_overlap: int = 20):
        """
        初始化剧本知识库

        Args:
            embeddings: 嵌入模型
            storage_dir: 存储目录
            chunk_size: 文本块大小
            chunk_overlap: 文本块重叠大小
        """
        self.embeddings = embeddings
        self.storage_dir = storage_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # 按 script_id 隔离的索引和向量存储
        self._indices: Dict[str, Optional[VectorStoreIndex]] = {}  # script_id -> index
        self._vector_stores: Dict[str, SimpleVectorStore] = {}  # script_id -> vector_store
        self._storage_contexts: Dict[str, StorageContext] = {}  # script_id -> storage_context
        self._retrievers: Dict[str, BaseRetriever] = {}  # script_id -> retriever

        # 解析结果缓存
        self.parsed_results: Dict[str, Dict[str, Any]] = {}
        self.document_cache: Dict[str, List[Document]] = {}

        self._parser_tool = None
        self._offline_mode = False

        # 当前激活的 script_id（用于默认操作）
        self._current_script_id: str = script_id

        # 初始化存储
        if self.storage_dir:
            os.makedirs(self.storage_dir, exist_ok=True)
            self._load_all_scripts()

        debug("剧本知识库初始化完成（支持 script_id 隔离）")

    def _get_script_subdir(self, script_id: str) -> str:
        """获取指定剧本的子目录路径"""
        return os.path.join(self.storage_dir, script_id) if self.storage_dir else None

    def _get_vector_store_path(self, script_id: str) -> str:
        """获取指定剧本的向量存储文件路径"""
        subdir = self._get_script_subdir(script_id)
        return os.path.join(subdir, "vector_store.json") if subdir else None

    def _get_parsed_dir(self, script_id: str) -> str:
        """获取指定剧本的解析结果目录"""
        subdir = self._get_script_subdir(script_id)
        return os.path.join(subdir, "parsed_results") if subdir else None

    def _load_all_scripts(self):
        """加载所有已存在的剧本"""
        if not self.storage_dir or not os.path.exists(self.storage_dir):
            return

        for item in os.listdir(self.storage_dir):
            script_subdir = os.path.join(self.storage_dir, item)
            if os.path.isdir(script_subdir):
                # 加载该剧本的解析结果
                parsed_dir = os.path.join(script_subdir, "parsed_results")
                if os.path.exists(parsed_dir):
                    for file in os.listdir(parsed_dir):
                        if file.endswith(".json"):
                            script_id = os.path.splitext(file)[0]
                            file_path = os.path.join(parsed_dir, file)
                            try:
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    self.parsed_results[script_id] = json.load(f)
                                debug(f"加载剧本解析结果: {script_id}")
                            except Exception as e:
                                warning(f"加载解析结果失败: {file_path}, {e}")

                # 延迟加载向量存储（在需要时加载）
                debug(f"发现剧本目录: {item}")

        debug(f"已加载 {len(self.parsed_results)} 个剧本的解析结果")

    def _load_vector_store_for_script(self, script_id: str) -> Optional[SimpleVectorStore]:
        """为指定剧本加载向量存储"""
        vector_store_path = self._get_vector_store_path(script_id)
        if not vector_store_path or not os.path.exists(vector_store_path):
            return None

        try:
            if os.path.getsize(vector_store_path) > 0:
                return SimpleVectorStore.from_persist_path(vector_store_path)
            else:
                warning(f"向量存储文件为空: {vector_store_path}")
                return None
        except Exception as e:
            warning(f"加载向量存储失败: {script_id}, {e}")
            return None

    def _ensure_script_index(self, script_id: str):
        """确保指定剧本的索引已加载"""
        if script_id in self._indices:
            return

        # 尝试加载现有向量存储
        vector_store = self._load_vector_store_for_script(script_id)

        if vector_store:
            # 使用现有存储
            self._vector_stores[script_id] = vector_store
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            self._storage_contexts[script_id] = storage_context

            if self.embeddings:
                try:
                    self._indices[script_id] = VectorStoreIndex.from_vector_store(
                        vector_store,
                        storage_context=storage_context,
                        embed_model=self.embeddings
                    )
                    debug(f"已加载剧本索引: {script_id}")
                except Exception as e:
                    warning(f"加载索引失败: {script_id}, {e}")
                    self._indices[script_id] = None
            else:
                self._indices[script_id] = None
        else:
            # 创建新存储
            self._vector_stores[script_id] = SimpleVectorStore()
            storage_context = StorageContext.from_defaults(vector_store=self._vector_stores[script_id])
            self._storage_contexts[script_id] = storage_context
            self._indices[script_id] = None
            debug(f"创建新剧本存储: {script_id}")

    def set_current_script(self, script_id: str):
        """设置当前操作的剧本ID"""
        self._current_script_id = script_id
        self._ensure_script_index(script_id)
        debug(f"当前剧本切换到: {script_id}")

    def get_current_script(self) -> Optional[str]:
        """获取当前剧本ID"""
        return self._current_script_id

    def _get_index(self, script_id: Optional[str] = None) -> Optional[VectorStoreIndex]:
        """获取指定剧本的索引"""
        target_id = script_id or self._current_script_id
        if not target_id:
            return None
        self._ensure_script_index(target_id)
        return self._indices.get(target_id)

    def _get_vector_store(self, script_id: Optional[str] = None) -> Optional[SimpleVectorStore]:
        """获取指定剧本的向量存储"""
        target_id = script_id or self._current_script_id
        if not target_id:
            return None
        return self._vector_stores.get(target_id)

    def _recreate_vector_store(self, script_id: str):
        """重建指定剧本的损坏向量存储"""
        vector_store_path = self._get_vector_store_path(script_id)
        try:
            if vector_store_path and os.path.exists(vector_store_path):
                backup_path = f"{vector_store_path}_backup_{int(datetime.now().timestamp())}"
                shutil.copy(vector_store_path, backup_path)
                warning(f"已备份损坏的向量存储到: {backup_path}")
                os.remove(vector_store_path)

            # 创建新的向量存储
            self._vector_stores[script_id] = SimpleVectorStore()
            self._storage_contexts[script_id] = StorageContext.from_defaults(vector_store=self._vector_stores[script_id])
            self._indices[script_id] = None
            debug(f"已为剧本 {script_id} 创建新的向量存储")

        except Exception as e:
            error(f"重建向量存储失败: {script_id}, {e}")

    def add_parsed_script(self, parsed_script: ParsedScript, script_id: str = None) -> Dict[str, Any]:
        """
        直接添加已解析的剧本对象到知识库

        Args:
            parsed_script: 已解析的 ParsedScript 对象
            script_id: 剧本唯一标识

        Returns:
            添加结果信息
        """
        try:
            if script_id is None:
                script_id = f"script_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            debug(f"添加已解析剧本: {script_id}")

            # 创建剧本子目录
            script_subdir = self._get_script_subdir(script_id)
            if script_subdir:
                os.makedirs(script_subdir, exist_ok=True)

            # 设置当前剧本
            self.set_current_script(script_id)

            # 直接从 ParsedScript 创建文档
            documents = self._create_documents_from_parsed(parsed_script)

            # 为文档添加剧本ID元数据
            for doc in documents:
                doc.metadata["script_id"] = script_id

            # 缓存解析结果
            self.parsed_results[script_id] = parsed_script.model_dump()
            self.document_cache[script_id] = documents

            # 添加到索引
            self._add_documents_to_index(documents, script_id)

            # 保存存储
            self._save_storage(script_id)
            self._save_parsed_result(script_id, self.parsed_results[script_id])

            info(f"成功添加已解析剧本: {script_id}, 包含{len(documents)}个文档")

            return {
                "status": "success",
                "script_id": script_id,
                "scene_count": len(parsed_script.scenes),
                "character_count": len(parsed_script.characters),
                "document_count": len(documents)
            }

        except Exception as e:
            error(f"添加已解析剧本失败: {str(e)}")
            raise

    def _create_documents_from_parsed(self, parsed_script: ParsedScript) -> List[Document]:
        """从 ParsedScript 对象创建文档"""
        return self._get_parser_tool().create_documents(parsed_script)

    def add_script_text(self, script_text: str, script_id: str = None) -> Dict[str, Any]:
        """
        添加剧本文本到知识库

        Args:
            script_text: 剧本文本
            script_id: 剧本唯一标识

        Returns:
            添加结果信息
        """
        try:
            if script_id is None:
                script_id = f"script_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            debug(f"添加剧本文本: {script_id}")

            # 创建剧本子目录
            script_subdir = self._get_script_subdir(script_id)
            if script_subdir:
                os.makedirs(script_subdir, exist_ok=True)

            # 设置当前剧本
            self.set_current_script(script_id)

            # 解析剧本
            parsed_result, documents = parse_script_to_documents(script_text)

            # 为文档添加剧本ID元数据
            for doc in documents:
                doc.metadata["script_id"] = script_id

            # 缓存解析结果
            self.parsed_results[script_id] = parsed_result.model_dump()
            self.document_cache[script_id] = documents

            # 添加到索引
            self._add_documents_to_index(documents, script_id)

            # 保存存储
            self._save_storage(script_id)
            self._save_parsed_result(script_id, self.parsed_results[script_id])

            info(f"成功添加剧本: {script_id}, 包含{len(documents)}个文档")

            return {
                "status": "success",
                "script_id": script_id,
                "scene_count": parsed_result.stats["scene_count"],
                "character_count": parsed_result.stats["character_count"],
                "document_count": len(documents)
            }

        except Exception as e:
            error(f"添加剧本文本失败: {str(e)}")
            raise

    def add_script_file(self, file_path: str, script_id: str = None) -> Dict[str, Any]:
        """
        添加剧本文件到知识库

        Args:
            file_path: 剧本文件路径
            script_id: 剧本唯一标识

        Returns:
            添加结果信息
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"剧本文件不存在: {file_path}")

            if script_id is None:
                base_name = os.path.basename(file_path)
                script_id = f"script_{os.path.splitext(base_name)[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            debug(f"添加剧本文件: {file_path} as {script_id}")

            # 创建剧本子目录
            script_subdir = self._get_script_subdir(script_id)
            if script_subdir:
                os.makedirs(script_subdir, exist_ok=True)

            # 设置当前剧本
            self.set_current_script(script_id)

            # 解析剧本
            parsed_result, documents = parse_script_file_to_documents(file_path)

            # 为文档添加剧本ID元数据
            for doc in documents:
                doc.metadata["script_id"] = script_id

            # 缓存解析结果
            self.parsed_results[script_id] = parsed_result.model_dump()
            self.document_cache[script_id] = documents

            # 添加到索引
            self._add_documents_to_index(documents, script_id)

            # 保存存储
            self._save_storage(script_id)
            self._save_parsed_result(script_id, self.parsed_results[script_id])

            info(f"成功添加剧本文件: {file_path}, 包含{len(documents)}个文档")

            return {
                "status": "success",
                "script_id": script_id,
                "file_path": file_path,
                "scene_count": parsed_result.stats["scene_count"],
                "character_count": parsed_result.stats["character_count"],
                "document_count": len(documents)
            }

        except Exception as e:
            error(f"添加剧本文件失败: {str(e)}")
            raise

    def remove_script(self, script_id: str) -> bool:
        """
        删除指定剧本的所有数据

        Args:
            script_id: 剧本ID

        Returns:
            是否删除成功
        """
        try:
            # 从内存中移除
            self._indices.pop(script_id, None)
            self._vector_stores.pop(script_id, None)
            self._storage_contexts.pop(script_id, None)
            self._retrievers.pop(script_id, None)
            self.parsed_results.pop(script_id, None)
            self.document_cache.pop(script_id, None)

            # 删除磁盘上的剧本子目录
            script_subdir = self._get_script_subdir(script_id)
            if script_subdir and os.path.exists(script_subdir):
                shutil.rmtree(script_subdir)
                info(f"已删除剧本数据: {script_id}")
            else:
                debug(f"剧本目录不存在: {script_id}")

            # 如果删除的是当前剧本，清空当前ID
            if self._current_script_id == script_id:
                self._current_script_id = None

            return True

        except Exception as e:
            error(f"删除剧本失败: {script_id}, {e}")
            return False

    def query(self, query_text: str, script_id: str,
              search_type: str = None, similarity_top_k: int = None,
              use_rerank: bool = False, rerank_model: str = None) -> Dict[str, Any]:
        """
        查询知识库 - 支持按 script_id 隔离

        Args:
            query_text: 查询文本
            script_id: 指定剧本ID（用于隔离查询）
            search_type: 搜索类型
            similarity_top_k: 相似性检索的文档数量
            use_rerank: 是否使用重排序
            rerank_model: 重排序模型名称

        Returns:
            查询结果
        """
        target_id = script_id or self._current_script_id

        if not target_id:
            debug("未指定剧本ID，跳过查询")
            return {
                "query": query_text,
                "results": [],
                "total_results": 0,
                "error": "未指定剧本ID"
            }

        index = self._get_index(target_id)
        if not index:
            debug(f"剧本 {target_id} 索引未创建，跳过查询")
            return {
                "query": query_text,
                "results": [],
                "total_results": 0,
                "error": f"剧本 {target_id} 索引未创建"
            }

        try:
            debug(f"执行查询: script_id={target_id}, query={query_text[:50]}...")

            # 创建检索器（如需要）
            if target_id not in self._retrievers:
                self.create_retriever_for_script(target_id, search_type, similarity_top_k, use_rerank, rerank_model)

            retriever = self._retrievers.get(target_id)
            if not retriever or not hasattr(retriever, 'retrieve'):
                return {
                    "query": query_text,
                    "results": [],
                    "total_results": 0,
                    "error": "检索器未就绪"
                }

            # 执行检索
            nodes = retriever.retrieve(query_text)

            # 格式化结果（只返回当前剧本的结果）
            results = []
            for i, node in enumerate(nodes):
                # 额外过滤确保只返回当前剧本的结果
                meta = dict(node.node.metadata) if hasattr(node.node, 'metadata') else {}
                if meta.get("script_id") != target_id:
                    continue

                result = {
                    "id": str(node.node.node_id),
                    "score": node.score,
                    "text": node.node.text,
                    "metadata": meta,
                    "rank": i + 1
                }
                results.append(result)

            info(f"查询完成: script_id={target_id}, 返回{len(results)}个结果")

            return {
                "query": query_text,
                "results": results,
                "total_results": len(results),
                "script_id": target_id,
                "search_type": search_type,
                "similarity_top_k": similarity_top_k,
                "use_rerank": use_rerank
            }

        except Exception as e:
            error(f"查询失败: {str(e)}")
            return {
                "query": query_text,
                "results": [],
                "total_results": 0,
                "error": str(e)
            }

    def create_retriever_for_script(self, script_id: str, search_type: str = None,
                                     similarity_top_k: int = None, use_rerank: bool = False,
                                     rerank_model: str = None):
        """为指定剧本创建检索器"""
        index = self._get_index(script_id)
        if not index:
            return

        try:
            if search_type == "mmr":
                retriever = index.as_retriever(
                    retriever_mode="mmr",
                    similarity_top_k=similarity_top_k,
                    mmr_threshold=0.8
                )
            else:
                retriever = index.as_retriever(
                    retriever_mode="similarity",
                    similarity_top_k=similarity_top_k
                )

            # 重排序仅在非离线模式时启用
            retriever_config = settings.get_retriever_config()
            if use_rerank or retriever_config.rerank_enabled:
                try:
                    if retriever_config.rerank_model_local_path and os.path.exists(retriever_config.rerank_model_local_path):
                        model_to_use = retriever_config.rerank_model_local_path
                    else:
                        model_to_use = retriever_config.rerank_model_name or rerank_model

                    from llama_index.core.postprocessor import SentenceTransformerRerank
                    rerank = SentenceTransformerRerank(
                        model=model_to_use,
                        top_n=min(3, similarity_top_k)
                    )
                    retriever = index.as_retriever(
                        retriever_mode=search_type or retriever_config.search_type,
                        similarity_top_k=similarity_top_k or retriever_config.similarity_top_k,
                        node_postprocessors=[rerank]
                    )
                except Exception as e:
                    warning(f"重排序初始化失败: {e}")

            self._retrievers[script_id] = retriever
            debug(f"为剧本 {script_id} 创建检索器成功")

        except Exception as e:
            error(f"创建检索器失败: {script_id}, {e}")

    def query_scene(self, scene_id: str, script_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        根据场景ID查询场景信息

        Args:
            scene_id: 场景ID (如 "scene_001")
            script_id: 指定剧本ID
        """
        target_id = script_id or self._current_script_id
        if not target_id:
            return None

        parsed_result = self.parsed_results.get(target_id)
        if not parsed_result:
            return None

        try:
            for scene in parsed_result.get("scenes", []):
                if scene.get("id") == scene_id or str(scene.get("number")) == scene_id:
                    return scene
            return None
        except Exception as e:
            error(f"查询场景失败: {e}")
            return None

    def query_character(self, character_name: str, script_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        根据角色名称查询角色信息

        Args:
            character_name: 角色名称
            script_id: 指定剧本ID
        """
        target_id = script_id or self._current_script_id
        if not target_id:
            return None

        parsed_result = self.parsed_results.get(target_id)
        if not parsed_result:
            return None

        try:
            for char in parsed_result.get("characters", []):
                if char.get("name") == character_name:
                    return char
            return None
        except Exception as e:
            error(f"查询角色失败: {e}")
            return None

    def get_statistics(self, script_id: Optional[str] = None) -> Dict[str, Any]:
        """
        获取知识库统计信息

        Args:
            script_id: 指定剧本ID，不指定则返回全局统计
        """
        if script_id:
            # 返回指定剧本的统计
            parsed_result = self.parsed_results.get(script_id, {})
            documents = self.document_cache.get(script_id, [])
            return {
                "script_id": script_id,
                "scene_count": parsed_result.get("stats", {}).get("scene_count", 0),
                "character_count": parsed_result.get("stats", {}).get("character_count", 0),
                "document_count": len(documents),
                "has_index": script_id in self._indices,
                "storage_dir": self._get_script_subdir(script_id)
            }
        else:
            # 返回全局统计
            total_scenes = 0
            total_characters = 0
            total_documents = 0

            for sid, parsed_result in self.parsed_results.items():
                total_scenes += parsed_result.get("stats", {}).get("scene_count", 0)
                total_characters += parsed_result.get("stats", {}).get("character_count", 0)
                total_documents += len(self.document_cache.get(sid, []))

            return {
                "script_count": len(self.parsed_results),
                "scene_count": total_scenes,
                "character_count": total_characters,
                "document_count": total_documents,
                "scripts": list(self.parsed_results.keys()),
                "current_script": self._current_script_id,
                "storage_dir": self.storage_dir,
                "offline_mode": self._offline_mode
            }

    def clear_script(self, script_id: str):
        """清空指定剧本的所有数据"""
        self.remove_script(script_id)
        info(f"已清空剧本: {script_id}")

    def clear(self):
        """清空所有剧本"""
        for script_id in list(self.parsed_results.keys()):
            self.remove_script(script_id)
        info("已清空所有剧本")


    def create_retriever(self,
                         script_id: Optional[str] = None,
                         search_type: str = None,
                         similarity_top_k: int = None,
                         use_rerank: bool = False) -> Optional[BaseRetriever]:
        """
        创建检索器（公共接口）

        使用示例:
            retriever = kb.create_retriever(script_id="script_001", search_type="mmr")
            nodes = retriever.retrieve("查询文本")

        Args:
            script_id: 剧本ID，不指定则使用当前剧本
            search_type: 搜索类型 (similarity/mmr)
            similarity_top_k: 返回结果数量
            use_rerank: 是否启用重排序

        Returns:
            检索器实例
        """
        target_id = script_id or self._current_script_id
        if not target_id:
            warning("未指定剧本ID，无法创建检索器")
            return None

        # 确保索引存在
        self._ensure_script_index(target_id)

        # 创建检索器
        self.create_retriever_for_script(
            script_id=target_id,
            search_type=search_type,
            similarity_top_k=similarity_top_k,
            use_rerank=use_rerank
        )

        return self._retrievers.get(target_id)


    def get_retriever(self, script_id: str) -> Optional[BaseRetriever]:
        """
        获取指定剧本的检索器

        Args:
            script_id: 剧本ID

        Returns:
            检索器实例
        """
        # 确保索引存在
        self._ensure_script_index(script_id)

        # 如果检索器不存在，创建默认检索器
        if script_id not in self._retrievers:
            self.create_retriever_for_script(
                script_id=script_id,
                search_type="similarity",
                similarity_top_k=5,
                use_rerank=False
            )

        return self._retrievers.get(script_id)

    def get_current_retriever(self) -> Optional[BaseRetriever]:
        """
        获取当前剧本的检索器
        Returns:
            检索器实例
        """
        target_id = self._current_script_id
        if not target_id:
            warning("未设置当前剧本，无法获取检索器")
            return None

        return self.get_retriever(target_id)


    def query_structured(self, query_type: str, query_value: str,
                         script_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        结构化查询（统一入口）

        Args:
            query_type: 查询类型 (scene, character)
            query_value: 查询值（场景ID或角色名）
            script_id: 剧本ID

        Returns:
            查询结果
        """
        target_id = script_id or self._current_script_id
        if not target_id:
            return None

        if query_type == "scene":
            return self.query_scene(query_value, target_id)
        elif query_type == "character":
            return self.query_character(query_value, target_id)
        else:
            warning(f"不支持的查询类型: {query_type}")
            return None

    def _add_documents_to_index(self, documents: List[Document], script_id: str):
        """添加文档到指定剧本的索引"""
        try:
            self._ensure_script_index(script_id)
            index = self._indices.get(script_id)
            vector_store = self._vector_stores.get(script_id)
            storage_context = self._storage_contexts.get(script_id)

            # 创建节点解析器
            if self.chunk_size > 1000:
                node_parser = SentenceWindowNodeParser(
                    window_size=3,
                    window_metadata_key="window",
                    original_text_metadata_key="original_text"
                )
            else:
                node_parser = SentenceSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap
                )

            # 如果索引不存在，创建新索引
            if not index:
                try:
                    self._indices[script_id] = VectorStoreIndex.from_documents(
                        documents,
                        storage_context=storage_context,
                        transformations=[node_parser],
                        embed_model=self.embeddings,
                        show_progress=True
                    )
                except Exception as e:
                    warning(f"创建索引失败: {script_id}, {e}")
                    self._indices[script_id] = None
            else:
                try:
                    nodes = node_parser.get_nodes_from_documents(documents)
                    index.insert_nodes(nodes)
                except Exception as e:
                    warning(f"向索引添加文档失败: {script_id}, {e}")

            debug(f"已添加{len(documents)}个文档到剧本 {script_id}")

        except Exception as e:
            error(f"添加文档到索引失败: {script_id}, {e}")

    def _save_storage(self, script_id: str):
        """保存指定剧本的向量存储"""
        try:
            vector_store = self._vector_stores.get(script_id)
            if vector_store:
                vector_store_path = self._get_vector_store_path(script_id)
                if vector_store_path:
                    os.makedirs(os.path.dirname(vector_store_path), exist_ok=True)
                    vector_store.persist(persist_path=vector_store_path)
                    debug(f"已保存向量存储: {script_id}")
        except Exception as e:
            warning(f"保存存储失败: {script_id}, {e}")

    def _save_parsed_result(self, script_id: str, parsed_result: Dict[str, Any]):
        """保存指定剧本的解析结果"""
        try:
            parsed_dir = self._get_parsed_dir(script_id)
            if parsed_dir:
                os.makedirs(parsed_dir, exist_ok=True)
                file_path = os.path.join(parsed_dir, f"{script_id}.json")
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(parsed_result, f, ensure_ascii=False, indent=2)
                debug(f"已保存解析结果: {script_id}")
        except Exception as e:
            warning(f"保存解析结果失败: {script_id}, {e}")

    def _get_parser_tool(self):
        """懒加载 ScriptParserTool"""
        if self._parser_tool is None:
            from penshot.neopen.tools.script_parser_tool import ScriptParserTool
            self._parser_tool = ScriptParserTool()
        return self._parser_tool


def create_script_knowledge_base(embeddings, storage_dir=None) -> ScriptKnowledgeBase:
    """
    创建剧本知识库实例

    Args:
        embeddings: 嵌入模型
        storage_dir: 存储目录

    Returns:
        剧本知识库实例
    """
    return ScriptKnowledgeBase(embeddings=embeddings, storage_dir=storage_dir)
