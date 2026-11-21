"""Qdrant 封装（方案A：多库 + 映射表 + id 编码关联检索）。

一个库 = 一个 Qdrant collection = qdrant_data/collections/<库id>/ 一个目录。
库id 是安全 ASCII（lib_001），库名是中文可读名（知识库），对应存 libraries.json。

point id 自带分组信息：id = 文档号 * 10000 + 段号（正整数）。
    前 4 位 = 文档号，后 4 位 = 段号；显示时补前导零成 8 位（00010003）。
    段号 = id % 10000，文档号 = id // 10000 —— 从 id 直接解出，payload 不存 doc_id/chunk_index。
payload 只存：text / total_chunks / memory_type / importance / timestamp。

检索两阶段：
    ① 混合检索（dense+sparse）拿命中段的 id；
    ② 从 id 解出文档号/段号，算出关联段 id，用 retrieve(ids=[...]) 主键直取并打包。
全程不依赖 payload 索引（本地模式索引无效），主键直取本地也是 O(1)。
"""
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from qdrant_client import QdrantClient, models

MAPPING_NAME = "libraries.json"
COUNTER_NAME = "doc_counter.json"
ID_PATTERN = re.compile(r"^lib_(\d+)$")
CHUNK_BASE = 10000  # 段号进制：单文档最多 9999 段，文档号最多 9999


class MemoryStore:
    def __init__(self, cfg, embedder):
        self.cfg = cfg
        self.embedder = embedder
        s = cfg["store"]
        self.root = Path(s["path"])
        self.root.mkdir(parents=True, exist_ok=True)
        self.mapping_path = self.root / MAPPING_NAME
        self.counter_path = self.root / COUNTER_NAME
        self.dense_dim = s["dense_dim"]
        r = cfg.get("retrieve", {})
        self.mode = r.get("mode", "neighbor")
        self.expand = int(r.get("expand", 1))
        self.max_chunks = int(r.get("max_chunks", 20))
        print(f"[store] 连接 Qdrant 本地库: {self.root}", file=sys.stderr)
        self.client = QdrantClient(path=str(self.root))
        self.libraries = self._load_mapping()
        self.doc_counter = self._load_counter()
        print(f"[store] 就绪，已有 {len(self.libraries)} 个库", file=sys.stderr)
        self._migrate_legacy()

    # ---------- 映射表 / 文档编号 ----------
    def _load_mapping(self) -> dict:
        if self.mapping_path.exists():
            try:
                return json.loads(self.mapping_path.read_text(encoding="utf-8"))
            except Exception as e:
                raise RuntimeError(f"映射表损坏：{self.mapping_path}：{e}")
        return {}

    def _save_mapping(self):
        tmp = self.mapping_path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(self.libraries, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        tmp.replace(self.mapping_path)

    def _load_counter(self) -> int:
        if self.counter_path.exists():
            try:
                return int(json.loads(self.counter_path.read_text(encoding="utf-8"))["next"])
            except Exception:
                pass
        return 1

    def _save_counter(self):
        self.counter_path.write_text(
            json.dumps({"next": self.doc_counter}), encoding="utf-8"
        )

    def _next_doc_no(self) -> int:
        if self.doc_counter >= CHUNK_BASE:
            raise ValueError("文档编号已达上限 9999，超出 8 位 id 编码范围")
        n = self.doc_counter
        self.doc_counter += 1
        self._save_counter()
        return n

    # ---------- id 编解码 ----------
    @staticmethod
    def _make_id(doc_no: int, chunk: int) -> int:
        return int(doc_no) * CHUNK_BASE + int(chunk)

    @staticmethod
    def _parse_id(pid) -> tuple[int, int]:
        n = int(pid)
        return n // CHUNK_BASE, n % CHUNK_BASE

    @staticmethod
    def _fmt_id(pid) -> str:
        return f"{int(pid):08d}"

    @staticmethod
    def _parse_doc_no(doc_id) -> int:
        """支持 0001 / 1 / 00010000 / doc_00001 几种写法，统一返回文档号。"""
        s = str(doc_id).strip()
        m = re.fullmatch(r"doc_0*(\d+)", s)
        if m:
            n = int(m.group(1))
        else:
            try:
                n = int(s)
            except ValueError:
                raise ValueError(f"文档编号格式不对：{doc_id}")
        if n <= 0:
            raise ValueError(f"文档编号必须为正数：{doc_id}")
        if n >= CHUNK_BASE:
            n = n // CHUNK_BASE  # 传入的是完整 id（如 00010000）
        if n <= 0 or n >= CHUNK_BASE:
            raise ValueError(f"文档编号超出 1~9999 范围：{doc_id}")
        return n

    # ---------- 旧数据迁移 ----------
    def _migrate_legacy(self):
        """旧结构（随机 UUID id / 带 source 字段）重编号为 id 编码方案。

        旧数据的点 id 是 UUID 字符串；新方案 id 是整数。按 source 分组 =
        同一篇原文的多段，重新分配文档号与连续 id，复用原向量，payload 精简。
        先写新点再删旧点，中途失败不会丢数据。
        """
        for lid in self.libraries:
            try:
                self.client.count(collection_name=lid, exact=True)
                scrolled, _ = self.client.scroll(
                    collection_name=lid,
                    limit=10000,
                    with_payload=True,
                    with_vectors=True,
                )
            except Exception:
                continue
            legacy = [p for p in scrolled if not str(p.id).isdigit()]
            if not legacy:
                continue
            groups = {}
            for p in legacy:
                key = (p.payload or {}).get("source") or str(p.id)
                groups.setdefault(key, []).append(p)
            migrated = 0
            for pts in groups.values():
                doc_no = self._next_doc_no()
                pts.sort(key=lambda p: int((p.payload or {}).get("chunk_index", 0)))
                total = len(pts)
                new_points = []
                for i, p in enumerate(pts):
                    pl = p.payload or {}
                    new_points.append(
                        models.PointStruct(
                            id=self._make_id(doc_no, i),
                            vector=p.vector,
                            payload={
                                "text": pl.get("text"),
                                "total_chunks": total,
                                "memory_type": pl.get("memory_type", "资料"),
                                "importance": pl.get("importance", 1.0),
                                "timestamp": pl.get("timestamp")
                                or datetime.now(timezone.utc).isoformat(),
                            },
                        )
                    )
                self.client.upsert(collection_name=lid, points=new_points)
                self.client.delete(
                    collection_name=lid,
                    points_selector=models.PointIdsList(points=[p.id for p in pts]),
                )
                migrated += total
            print(
                f"[store] 迁移 {lid}: {migrated} 段 -> {len(groups)} 个文档（id 已编码）",
                file=sys.stderr,
            )

    # ---------- 库管理 ----------
    def create_library(self, name: str) -> dict:
        name = (name or "").strip()
        if not name:
            raise ValueError("库名不能为空")
        for lid, info in self.libraries.items():
            if info["name"] == name:
                raise ValueError(f"库已存在：{name}（id={lid}）")
        nums = [int(m.group(1)) for lid in self.libraries if (m := ID_PATTERN.match(lid))]
        new_id = f"lib_{(max(nums) + 1) if nums else 1:03d}"
        self.client.create_collection(
            collection_name=new_id,
            vectors_config={
                "dense": models.VectorParams(
                    size=self.dense_dim, distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
        )
        self.libraries[new_id] = {
            "name": name,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self._save_mapping()
        return {"id": new_id, "name": name}

    def list_libraries(self) -> list:
        return [
            {"id": lid, "name": info["name"], "created_at": info.get("created_at")}
            for lid, info in sorted(self.libraries.items())
        ]

    def resolve(self, library: str) -> str:
        """把库名或库id 解析成库id，找不到时报错并列出可用库。"""
        key = (library or "").strip()
        if not key:
            raise ValueError("必须指定库名")
        if key in self.libraries:
            return key
        for lid, info in self.libraries.items():
            if info["name"] == key:
                return lid
        available = "、".join(
            f"{i['name']}({i['id']})" for i in self.list_libraries()
        ) or "（当前没有任何库，先用 create_library 建一个）"
        raise ValueError(f"找不到库：{key}。可用库：{available}")

    # ---------- 写入 ----------
    def upsert(
        self,
        library: str,
        text: str,
        memory_type: str = "资料",
        importance: float = 1.0,
    ) -> dict:
        from .chunker import split_text

        lid = self.resolve(library)
        c = self.cfg["chunk"]
        chunks = split_text(text, c["size"], c["overlap"])
        if not chunks:
            return {"library": library, "library_id": lid, "doc_id": None, "chunks_written": 0}
        if len(chunks) >= CHUNK_BASE:
            raise ValueError(f"单文档段数 {len(chunks)} 超过上限 9999，请拆成多篇写入")
        doc_no = self._next_doc_no()
        total = len(chunks)
        dense, sparse = self.embedder.encode(chunks)
        now = datetime.now(timezone.utc).isoformat()
        points = []
        for i, (d, (idx, val)) in enumerate(zip(dense, sparse)):
            points.append(
                models.PointStruct(
                    id=self._make_id(doc_no, i),
                    vector={
                        "dense": d,
                        "sparse": models.SparseVector(indices=idx, values=val),
                    },
                    payload={
                        "text": chunks[i],
                        "total_chunks": total,
                        "memory_type": memory_type,
                        "importance": importance,
                        "timestamp": now,
                    },
                )
            )
        self.client.upsert(collection_name=lid, points=points)
        return {
            "library": library,
            "library_id": lid,
            "doc_id": f"{doc_no:04d}",   # 文档号，给 read_document 用
            "first_point_id": self._fmt_id(self._make_id(doc_no, 0)),
            "total_chunks": total,
            "chunks_written": total,
        }

    # ---------- 检索（id 编码 + retrieve 主键直取） ----------
    def hybrid_search(
        self,
        library: str,
        query: str,
        top_k: int | None = None,
        mode: str | None = None,
        expand: int | None = None,
        max_chunks: int | None = None,
    ) -> list:
        lid = self.resolve(library)
        r = self.cfg.get("retrieve", {})
        top_k = top_k or r.get("top_k", 5)
        mode = mode or self.mode
        expand = self.expand if expand is None else int(expand)
        max_chunks = max_chunks or self.max_chunks

        # ① 混合检索：dense + sparse 两路 RRF 融合，拿 top_k 个命中段的 id
        dense, sparse = self.embedder.encode([query])
        d, (idx, val) = dense[0], sparse[0]
        res = self.client.query_points(
            collection_name=lid,
            prefetch=[
                models.Prefetch(query=d, using="dense", limit=top_k * 4),
                models.Prefetch(
                    query=models.SparseVector(indices=idx, values=val),
                    using="sparse",
                    limit=top_k * 4,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
        )
        hits = res.points
        if not hits:
            return []

        # 从命中 id 解出文档号/段号，记录每篇文档的段数（从 payload 取）
        hit_ids: set[int] = set()
        hit_score: dict[int, float] = {}
        doc_totals: dict[int, int] = {}
        for p in hits:
            pid = int(p.id)
            hit_ids.add(pid)
            hit_score[pid] = float(p.score)
            doc_no, _ = self._parse_id(pid)
            doc_totals[doc_no] = int((p.payload or {}).get("total_chunks", 1))

        # ② 按模式从 id 算出要取的段 id（neighbor=命中段±expand；full=整篇）
        target_ids: set[int] = set()
        for pid in hit_ids:
            doc_no, ci = self._parse_id(pid)
            total = doc_totals.get(doc_no, 1)
            if mode == "full":
                rng = range(total)
            else:
                rng = range(max(0, ci - expand), min(total, ci + expand + 1))
            for k in rng:
                target_ids.add(self._make_id(doc_no, k))

        # ③ 一次主键直取（本地模式 O(1)，不依赖 payload 索引）
        records = self.client.retrieve(
            collection_name=lid,
            ids=sorted(target_ids),
            with_payload=True,
            with_vectors=False,
        )

        # ④ 单遍历打包：标记 is_hit，不用去重（id 集合天然唯一），再排序 + 截断
        packed = []
        for rec in records:
            pid = int(rec.id)
            doc_no, ci = self._parse_id(pid)
            pl = rec.payload or {}
            packed.append(
                {
                    "id": self._fmt_id(pid),
                    "doc_id": f"{doc_no:04d}",
                    "chunk_index": ci,
                    "total_chunks": pl.get("total_chunks"),
                    "is_hit": pid in hit_ids,
                    "score": hit_score.get(pid),
                    "text": pl.get("text"),
                    "memory_type": pl.get("memory_type"),
                    "importance": pl.get("importance"),
                    "timestamp": pl.get("timestamp"),
                }
            )
        packed.sort(key=lambda x: (x["doc_id"], x["chunk_index"]))
        return packed[:max_chunks]

    # ---------- 按文档号直接读原文（id 推算 + 主键直取） ----------
    def read_document(
        self, library: str, doc_id: str, from_chunk: int = 0, to_chunk: int | None = None
    ) -> list:
        lid = self.resolve(library)
        doc_no = self._parse_doc_no(doc_id)
        first = self.client.retrieve(
            collection_name=lid,
            ids=[self._make_id(doc_no, 0)],
            with_payload=True,
            with_vectors=False,
        )
        if not first:
            raise ValueError(f"找不到文档 {doc_id}（文档号 {doc_no:04d}）")
        total = int((first[0].payload or {}).get("total_chunks", 1))
        lo = max(0, int(from_chunk) if from_chunk else 0)
        hi = min(total - 1, int(to_chunk)) if to_chunk is not None else total - 1
        if lo > hi:
            return []
        ids = [self._make_id(doc_no, i) for i in range(lo, hi + 1)]
        recs = self.client.retrieve(
            collection_name=lid, ids=ids, with_payload=True, with_vectors=False
        )
        segs = sorted(
            [
                {
                    "id": self._fmt_id(r.id),
                    "chunk_index": self._parse_id(r.id)[1],
                    "text": (r.payload or {}).get("text"),
                }
                for r in recs
            ],
            key=lambda x: x["chunk_index"],
        )
        return segs

    def close(self):
        """显式关闭，避免解释器退出时 Qdrant 本地客户端的无害报错。"""
        try:
            self.client.close()
        except Exception:
            pass
