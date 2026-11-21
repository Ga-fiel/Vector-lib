"""自测：id 编码 → 两模式检索(主键直取) → 按文档号读原文 → 旧数据迁移 → 库隔离。

用临时目录跑，不污染 qdrant_data 里的真实数据。
"""
import json
import sys
import tempfile
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qdrant_client import QdrantClient, models

from src.config import load_config
from src.embedder import Embedder
from src.store import MemoryStore, CHUNK_BASE

cfg = load_config()
cfg["store"]["path"] = tempfile.mkdtemp(prefix="vm-test-")
# 测试用较小切片，方便短文本也能切成多段验证关联逻辑
cfg["chunk"]["size"] = 120
cfg["chunk"]["overlap"] = 20
print(f"测试数据目录: {cfg['store']['path']}")

embedder = Embedder(cfg)
store = MemoryStore(cfg, embedder)

# [1] 建库
lib_a = store.create_library("记忆库A")
lib_b = store.create_library("Java学习笔记")
print(f"\n[1] 建库: {lib_a} / {lib_b}")

# [2] 写入：长文本切多片，分配文档号，id = 文档号*10000 + 段号
long_text = (
    "Qdrant 本地模式用 path 参数指定存储目录，支持稠密与稀疏向量的混合检索。"
    "BGE-M3 模型一个模型同时输出稠密和稀疏两组向量，适合中文和专业术语场景。"
    "向量检索分为稠密语义检索和稀疏关键词检索两路，RRF 算法把两路结果融合排序。"
    "混合检索既保留了关键词的精确匹配能力，又具备语义理解能力。"
    "在多 agent 协作场景里，记忆库负责沉淀每次任务的知识，供后续 agent 复用。"
    "文档编号把同一次写入的多个切片关联起来，检索时可以拼回完整上下文。"
    "切片之间的重叠 overlap 保证跨段的语义不会被硬切断。"
    "按 id 主键直取让关联段的读取不依赖 payload 索引，本地模式也很快。"
)
w = store.upsert("记忆库A", long_text, memory_type="资料")
print(f"\n[2] 写入: 文档号={w['doc_id']}, 首段id={w['first_point_id']}, 共 {w['total_chunks']} 段")
assert w["total_chunks"] > 2, "测试文本应被切成多段"

# [3] 校验 id 编码 + payload 精简
recs, _ = store.client.scroll(
    collection_name="lib_001", limit=100, with_payload=True, with_vectors=False
)
ids = sorted(int(r.id) for r in recs)
expected = [CHUNK_BASE + i for i in range(w["total_chunks"])]  # 文档1 → 10000,10001,...
print(f"\n[3] id 编码校验: 实际={ids[:5]}... 期望前5={expected[:5]}")
assert ids == expected, f"id 应连续编码为 {expected[:3]}...，实际 {ids[:3]}..."
assert all(
    "doc_id" not in (r.payload or {})
    and "chunk_index" not in (r.payload or {})
    and "library_id" not in (r.payload or {})
    and "source" not in (r.payload or {})
    and "total_chunks" in (r.payload or {})
    for r in recs
), "payload 应精简为 text/total_chunks/memory_type/importance/timestamp，且无 doc_id 等旧字段"
assert store._parse_id(10003) == (1, 3), "id 解码应为 (文档号, 段号)"
print("    payload 字段精简 [OK] / id 前后4位解码 [OK]")

# [4] 模式2（neighbor，默认）：命中段上下各取 1 段。用 top_k=1 确保有扩展段。
print("\n[4] 模式2 neighbor（命中段±1，top_k=1）:")
r2 = store.hybrid_search("记忆库A", "RRF 融合两路结果", mode="neighbor", expand=1, top_k=1)
for r in r2:
    tag = "命中" if r["is_hit"] else "扩展"
    print(f"  [{tag}] id={r['id']} 文档={r['doc_id']} 段{r['chunk_index']}/{r['total_chunks']} | {r['text'][:36]}")
assert any(r["is_hit"] for r in r2), "模式2应有命中段"
assert any(not r["is_hit"] for r in r2), "模式2应有扩展段"
assert all(r["id"].isdigit() and len(r["id"]) == 8 for r in r2), "返回 id 应为8位数字字符串"

# [5] 模式1（full）：命中段所在文档全部段
print("\n[5] 模式1 full（整篇文档全段）:")
r1 = store.hybrid_search("记忆库A", "RRF 融合两路结果", mode="full", top_k=1)
print(f"  返回 {len(r1)} 段 / 全文 {r1[0]['total_chunks']} 段")
assert len(r1) == w["total_chunks"], "模式1应返回整篇全部段"

# [6] read_document：按文档号直取
print("\n[6] read_document（按文档号读 0-1 段）:")
segs = store.read_document("记忆库A", w["doc_id"], 0, 1)
for s in segs:
    print(f"  段{s['chunk_index']} id={s['id']}: {s['text'][:36]}")
assert len(segs) == 2, "应返回 0、1 两段"
assert segs[0]["id"] == "00010000", "第0段 id 应为 00010000"

# [7] 库隔离
store.upsert("Java学习笔记", "Java 的 ArrayList 基于动态数组实现，LinkedList 基于双向链表。")
print("\n[7] 库隔离（Java库搜 Java，不应串入记忆库A内容）:")
for r in store.hybrid_search("lib_002", "Java 集合 数组 链表", top_k=3):
    text = r["text"]
    leaked = "Qdrant" in text or "BGE-M3" in text
    print(f"  {r['id']} | {text[:36]} | {'[串库!]' if leaked else '[OK]'}")

# [8] 不存在的库
print("\n[8] 不存在的库应报错 -> ", end="")
try:
    store.hybrid_search("不存在的库", "test")
    print("没有报错（异常！）")
except ValueError as e:
    print(f"正确报错：{e}")

store.close()

# [9] 旧数据迁移：构造随机 UUID id + source 字段的旧结构，重开 store 触发迁移
print("\n[9] 旧数据迁移校验:")
mig_dir = tempfile.mkdtemp(prefix="vm-mig-")
client = QdrantClient(path=mig_dir)
client.create_collection(
    collection_name="lib_001",
    vectors_config={
        "dense": models.VectorParams(size=cfg["store"]["dense_dim"], distance=models.Distance.COSINE)
    },
    sparse_vectors_config={"sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)},
)
old_points = []
for i in range(3):
    old_points.append(
        models.PointStruct(
            id=str(uuid.uuid4()),
            vector={
                "dense": [0.0] * cfg["store"]["dense_dim"],
                "sparse": models.SparseVector(indices=[1], values=[1.0]),
            },
            payload={
                "text": f"旧数据第{i}段内容",
                "source": "old_doc.md",
                "chunk_index": i,
                "total_chunks": 3,
                "memory_type": "资料",
                "importance": 1.0,
                "timestamp": "2026-01-01T00:00:00Z",
            },
        )
    )
client.upsert(collection_name="lib_001", points=old_points)
client.close()
(Path(mig_dir) / "libraries.json").write_text(
    json.dumps({"lib_001": {"name": "旧库", "created_at": "2026-01-01T00:00:00Z"}}),
    encoding="utf-8",
)
cfg_mig = load_config()
cfg_mig["store"]["path"] = mig_dir
store2 = MemoryStore(cfg_mig, embedder)  # 初始化时自动迁移
mig_recs, _ = store2.client.scroll(
    collection_name="lib_001", limit=100, with_payload=True, with_vectors=False
)
mig_ids = sorted(int(r.id) for r in mig_recs)
print(f"    迁移后 id={mig_ids}，payload 字段={sorted((mig_recs[0].payload or {}).keys())}")
assert mig_ids == [10000, 10001, 10002], f"旧数据应迁移成 10000/10001/10002，实际 {mig_ids}"
assert all("source" not in (r.payload or {}) for r in mig_recs), "迁移后应去掉 source 字段"
store2.close()

print("\n全部自测通过 [OK]")
