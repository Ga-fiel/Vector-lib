"""MCP Server：把记忆库暴露成 agent 可调用的工具（stdio 传输）。

方案A：多库 + 库名。每个库有 id（lib_001）和中文名（知识库），工具统一用库名操作。
point id 自带分组信息（前4位文档号+后4位段号，如 00010003）；检索靠 id 推算 + 主键直取，不依赖 payload 索引。
检索默认模式2（neighbor）：命中段上下各取 expand 段，拼成连续上下文。

懒加载：Embedder / MemoryStore 不在模块顶部初始化，第一次调用工具时才加载模型（2GB）
和连接 Qdrant，避免 DSH 启动时拉起 MCP 进程就卡在模型加载上。用锁保证并发只初始化一次。
"""
import json
import sys
import threading

from mcp.server.fastmcp import FastMCP
from .config import load_config
# 在主线程完成模块导入；只把重模型和数据库初始化延迟到首次工具调用。
from .embedder import Embedder
from .store import MemoryStore

mcp = FastMCP("vector-memory")

_store = None
_lock = threading.Lock()


def _get_store():
    """第一次调用工具时才加载配置 / 模型 / 数据库，之后复用。线程安全。"""
    global _store
    if _store is None:
        with _lock:
            if _store is None:
                print("[server] 首次调用：加载 BGE-M3 模型与记忆库...", file=sys.stderr, flush=True)
                cfg = load_config()
                embedder = Embedder(cfg)
                _store = MemoryStore(cfg, embedder)
                print("[server] 记忆库就绪", file=sys.stderr, flush=True)
    return _store


def _err(e: Exception) -> str:
    return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)


@mcp.tool()
def create_library(name: str) -> str:
    """新建一个记忆库并取名（如"知识库"），返回库id。同名库已存在会报错。"""
    try:
        info = _get_store().create_library(name)
        return json.dumps({"ok": True, **info}, ensure_ascii=False)
    except Exception as e:
        return _err(e)


@mcp.tool()
def list_libraries() -> str:
    """列出所有记忆库（id + 名字），写入/检索前可先确认有哪些库。"""
    try:
        return json.dumps(
            {"ok": True, "libraries": _get_store().list_libraries()}, ensure_ascii=False
        )
    except Exception as e:
        return _err(e)


@mcp.tool()
def write_memory(
    library: str,
    text: str,
    memory_type: str = "资料",
    importance: float = 1.0,
) -> str:
    """把一段文本写入指定库（库名或库id均可）。内容即原文，切成多段存成一个文档。"""
    try:
        r = _get_store().upsert(library, text, memory_type, importance)
        return json.dumps({"ok": True, **r}, ensure_ascii=False)
    except Exception as e:
        return _err(e)


@mcp.tool()
def search_memory(
    library: str,
    query: str,
    mode: str = "",
    expand: int = -1,
    top_k: int = 0,
    max_chunks: int = 0,
) -> str:
    """混合检索：关键词(稀疏)+语义(稠密)融合。

    mode="neighbor"(默认)：命中段上下各取 expand 段拼连续上下文；
    mode="full"：命中段所在文档的全部段。
    expand<0 表示用配置默认值；max_chunks=0 表示用配置默认值。
    """
    try:
        store = _get_store()
        results = store.hybrid_search(
            library,
            query,
            top_k=top_k or None,
            mode=mode or None,
            expand=expand if expand >= 0 else None,
            max_chunks=max_chunks or None,
        )
        return json.dumps(
            {
                "ok": True,
                "library": library,
                "query": query,
                "mode": mode or store.mode,
                "results": results,
            },
            ensure_ascii=False,
        )
    except Exception as e:
        return _err(e)


@mcp.tool()
def read_document(
    library: str, doc_id: str, from_chunk: int = 0, to_chunk: int = 0
) -> str:
    """按文档编号直接读取原文段落（doc_id 是4位文档号，来自 write_memory 的返回值）。

    靠文档号算出段 id 直接取回，不走全文扫描。
    from_chunk/to_chunk 指定段号区间；to_chunk=0 表示读到末尾。
    """
    try:
        segs = _get_store().read_document(library, doc_id, from_chunk, to_chunk or None)
        return json.dumps(
            {
                "ok": True,
                "library": library,
                "doc_id": doc_id,
                "chunks": segs,
            },
            ensure_ascii=False,
        )
    except Exception as e:
        return _err(e)


if __name__ == "__main__":
    mcp.run()  # stdio
