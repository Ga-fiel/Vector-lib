"""BGE-M3 封装：文字 → 稠密向量 + 稀疏向量（一个模型同时出两路）。"""
import sys
import numpy as np
from FlagEmbedding import BGEM3FlagModel


class Embedder:
    def __init__(self, cfg):
        name = cfg["model"]["local_path"] or cfg["model"]["name"]
        self.max_length = cfg["model"]["max_length"]
        print(f"[embedder] 加载模型: {name} ...", file=sys.stderr)
        self.model = BGEM3FlagModel(
            name,
            use_fp16=False,
            device=cfg["model"].get("device", "cpu"),
        )
        # BGE-M3 返回的稀疏权重 key 是 token 字符串，需转成 token id 给 Qdrant
        self.tokenizer = self.model.tokenizer
        print("[embedder] 模型就绪", file=sys.stderr)

    def encode(self, texts):
        """返回 (dense_list, sparse_list)：
           dense_list  : List[List[float]]，每条一个 1024 维向量
           sparse_list : List[(indices, values)]，每条一组关键词权重
        """
        out = self.model.encode(
            texts,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense = np.asarray(out["dense_vecs"], dtype=np.float32)
        sparse = []
        for item in out["lexical_weights"]:
            # 不同 token 字符串可能映射到同一 token id，Qdrant 要求 indices 唯一，先按 id 聚合
            merged = {}
            for token, weight in item.items():
                tid = self.tokenizer.convert_tokens_to_ids(token)
                if tid is None or tid < 0:
                    continue
                merged[int(tid)] = merged.get(int(tid), 0.0) + float(weight)
            indices = list(merged.keys())
            values = [merged[i] for i in indices]
            sparse.append((indices, values))
        return dense.tolist(), sparse
