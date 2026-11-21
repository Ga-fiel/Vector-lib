"""MCP 端到端测试：真正启动 src.server（stdio），连上去握手、列工具、调工具。

用临时数据目录（环境变量 VM_STORE_PATH 重定向），不污染真实 qdrant_data。
"""
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def parse(res):
    for c in res.content:
        if hasattr(c, "text"):
            try:
                return json.loads(c.text)
            except Exception:
                return c.text
    return None


async def main():
    data_dir = tempfile.mkdtemp(prefix="vm-mcp-")
    print(f"临时数据目录: {data_dir}")
    params = StdioServerParameters(
        command=str(ROOT / ".venv" / "Scripts" / "python.exe"),
        args=["-m", "src.server"],
        cwd=str(ROOT),
        env={
            **os.environ,
            "VM_STORE_PATH": data_dir,
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
        },
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("[1] 握手成功 ✅")

            tools = await session.list_tools()
            names = [t.name for t in tools.tools]
            print(f"[2] 工具列表: {names}")
            assert set(names) >= {
                "create_library", "list_libraries", "write_memory",
                "search_memory", "read_document",
            }

            r = parse(await session.call_tool("create_library", {"name": "MCP测试库"}))
            print(f"[3] create_library -> {r}")
            assert r.get("ok") and r.get("id")

            r = parse(await session.call_tool("list_libraries", {}))
            print(f"[4] list_libraries -> {r}")
            assert any(l["name"] == "MCP测试库" for l in r["libraries"])

            r = parse(await session.call_tool("write_memory", {
                "library": "MCP测试库",
                "text": "MCP 服务端到端测试：混合检索要走稠密加稀疏两路融合排序。",
            }))
            print(f"[5] write_memory -> {r}")
            assert r.get("ok") and r["chunks_written"] >= 1
            doc_id = r["doc_id"]

            r = parse(await session.call_tool("search_memory", {
                "library": "MCP测试库", "query": "稠密 稀疏 融合", "mode": "neighbor",
            }))
            print(f"[6] search_memory(neighbor) -> {json.dumps(r, ensure_ascii=False)[:300]}")
            assert r.get("ok") and r["results"] and r["mode"] == "neighbor"

            # read_document：按 doc_id 读原文
            r = parse(await session.call_tool("read_document", {
                "library": "MCP测试库", "doc_id": doc_id,
            }))
            print(f"[7] read_document -> {json.dumps(r, ensure_ascii=False)[:200]}")
            assert r.get("ok") and r["chunks"]

            # 错误路径：不存在的库
            r = parse(await session.call_tool("search_memory", {
                "library": "不存在的库", "query": "x",
            }))
            print(f"[8] 错误库 -> {r}")
            assert r.get("ok") is False

            print("\nMCP 端到端测试全部通过 ✅")


if __name__ == "__main__":
    asyncio.run(main())
