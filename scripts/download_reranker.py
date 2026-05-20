"""
@FileName: download_reranker.py
@Description: 自动下载 BAAI/bge-reranker-large 模型
            需要安装 pip install sentence-transformers
@Author: HiPeng
@Time: 2026/5/20 22:30
"""

if __name__ == '__main__':
    import os
    # (可选) 国内用户开启镜像加速
    os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"

    from sentence_transformers import CrossEncoder

    # 1. 定义项目根目录（根据你的实际结构调整）
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 2. 设置模型的本地保存路径 (项目根目录下的 data/models)
    local_model_path = os.path.join(project_root, "data", "models", "bge-reranker-large")

    # 3. 确保目录存在
    os.makedirs(local_model_path, exist_ok=True)

    print(f"模型将下载到: {local_model_path}")

    # 4. 下载并保存模型
    model_name = "BAAI/bge-reranker-large"
    model = CrossEncoder(model_name)
    model.save_pretrained(local_model_path)

    print(f" 模型已成功下载并保存到: {local_model_path}")