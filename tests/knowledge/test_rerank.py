"""
@FileName: test_rerank.py
@Description: 
@Author: HiPeng
@Time: 2026/5/20 22:12
"""

import asyncio
from penshot.neopen.knowledge.llamaIndex.llama_index_knowledge import ScriptKnowledgeBase
from penshot.neopen.shot_config import ShotConfig


async def test_rerank():
    # 1. 初始化知识库
    config = ShotConfig()
    embeddings = config.get_embed_by_config()
    # 2. 设置当前剧本（替换为你的实际 script_id）
    script_id = "SN1780745829485283341820"  # 从日志中获取

    kb = ScriptKnowledgeBase(
        embeddings=embeddings,
        script_id=script_id
    )
    kb.set_current_script(script_id)

    # 3. 测试查询
    query = "灰色羊毛毯 林然"

    print("=" * 60)
    print("测试重排序效果")
    print("=" * 60)

    # 4. 不使用重排序
    print("\n1. 不使用重排序 (use_rerank=False):")
    result_no_rerank = kb.query(
        query_text=query,
        script_id=script_id,
        similarity_top_k=10,
        use_rerank=False
    )

    for i, r in enumerate(result_no_rerank.get("results", [])[:5]):
        print(f"   {i + 1}. score={r['score']:.4f}: {r['text'][:80]}...")

    # 5. 使用重排序
    print("\n2. 使用重排序 (use_rerank=True):")
    result_with_rerank = kb.query(
        query_text=query,
        script_id=script_id,
        similarity_top_k=10,
        use_rerank=True
    )

    for i, r in enumerate(result_with_rerank.get("results", [])[:5]):
        print(f"   {i + 1}. score={r['score']:.4f}: {r['text'][:80]}...")

    # 6. 对比结果
    print("\n3. 对比分析:")
    print(f"   无重排序返回: {len(result_no_rerank.get('results', []))} 条")
    print(f"   有重排序返回: {len(result_with_rerank.get('results', []))} 条")

    # 检查是否使用了重排序模型
    if result_with_rerank.get("use_rerank"):
        print("    重排序已启用")
    else:
        print("    重排序未启用")

    # 7. 查看检索器配置
    print("\n4. 检索器信息:")
    if script_id in kb._retrievers:
        retriever = kb._retrievers[script_id]
        print(f"   检索器类型: {type(retriever)}")
        if hasattr(retriever, '_node_postprocessors'):
            print(f"   后处理器: {retriever._node_postprocessors}")

    # 8. 检查重排序模型是否加载
    print("\n5. 重排序模型检查:")
    try:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder("BAAI/bge-reranker-large")
        print("    bge-reranker-large 模型可用")
    except Exception as e:
        print(f"   重排序模型不可用: {e}")


if __name__ == "__main__":
    asyncio.run(test_rerank())