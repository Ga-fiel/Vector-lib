r"""轻量 MCP stdio 分阶段诊断，不调用任何会加载模型的工具。

直接发送 JSON-RPC，逐阶段记录本地时间、相对耗时和子进程 stderr。
用法：.venv\Scripts\python.exe -u tests\diagnose_mcp_stdio.py
"""
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
START = time.monotonic()


def log(stage, message):
    now = datetime.now().astimezone().isoformat(timespec="milliseconds")
    print(
        f"[{now}] [+{time.monotonic() - START:8.3f}s] [{stage}] {message}",
        flush=True,
    )


def send(stdin, stage, payload):
    log(stage, f"发送: {json.dumps(payload, ensure_ascii=False)}")
    stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
    stdin.flush()
    log(stage, "写入并 flush 完成")


def main():
    data_dir = tempfile.mkdtemp(prefix="vm-mcp-light-")
    log("setup", f"临时 VM_STORE_PATH={data_dir}；不会调用工具或加载模型")
    env = {
        **os.environ,
        "VM_STORE_PATH": data_dir,
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
    }
    command = [str(ROOT / ".venv" / "Scripts" / "python.exe"), "-u", "-m", "src.server"]
    log("spawn", f"启动子进程: {command}，cwd={ROOT}")
    proc = subprocess.Popen(
        command,
        cwd=str(ROOT),
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    log("spawn", f"Popen 返回，pid={proc.pid}（仅表示进程已创建，不代表导入完成）")

    replies = queue.Queue()
    stderr_lines = queue.Queue()

    def read_stdout():
        try:
            for line in proc.stdout:
                log("stdout", f"收到原始行: {line.rstrip()}")
                replies.put(line)
        except Exception as exc:
            replies.put(exc)
        finally:
            replies.put(None)

    def read_stderr():
        for line in proc.stderr:
            log("server.stderr", line.rstrip())
        stderr_lines.put(None)

    stdout_thread = threading.Thread(target=read_stdout, daemon=True, name="mcp-stdout")
    stderr_thread = threading.Thread(target=read_stderr, daemon=True, name="mcp-stderr")
    stdout_thread.start()
    stderr_thread.start()

    def receive(stage, timeout=20):
        log(stage, f"等待响应，超时={timeout}s")
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"{stage} 等待 {timeout}s 超时；poll={proc.poll()}；上一阶段之后未收到响应"
                )
            item = replies.get(timeout=remaining)
            if item is None:
                raise RuntimeError(f"{stage}: stdout 已关闭，returncode={proc.poll()}")
            if isinstance(item, Exception):
                raise RuntimeError(f"{stage}: 读取 stdout 失败: {item}") from item
            response = json.loads(item)
            if response.get("id") is not None:
                log(stage, f"解析 JSON-RPC 响应: {json.dumps(response, ensure_ascii=False)}")
                return response
            log(stage, f"跳过通知/请求: {json.dumps(response, ensure_ascii=False)}")

    try:
        send(
            proc.stdin,
            "initialize.request",
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {"name": "mcp-light-diagnostic", "version": "1.0"},
                },
            },
        )
        initialized = receive("initialize.response")
        if "error" in initialized:
            raise RuntimeError(f"initialize 返回错误: {initialized['error']}")
        log("initialize.response", "握手响应成功")

        send(
            proc.stdin,
            "notifications/initialized",
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
        )
        log("notifications/initialized", "通知已发送（通知按协议没有响应）")

        send(
            proc.stdin,
            "tools/list.request",
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        )
        listed = receive("tools/list.response")
        if "error" in listed:
            raise RuntimeError(f"tools/list 返回错误: {listed['error']}")
        names = [tool["name"] for tool in listed["result"]["tools"]]
        log("tools/list.response", f"工具列表: {names}")
        expected = {
            "create_library",
            "list_libraries",
            "write_memory",
            "search_memory",
            "read_document",
        }
        missing = expected - set(names)
        if missing:
            raise AssertionError(f"缺少工具: {sorted(missing)}")
        log("result", "PASS：spawn、initialize、notifications/initialized、tools/list 全部通过")
    except Exception as exc:
        log("result", f"FAIL：{type(exc).__name__}: {exc}")
        return 1
    finally:
        log("shutdown", "关闭 stdin，让 server 自然退出；不终止进程")
        if proc.stdin and not proc.stdin.closed:
            proc.stdin.close()
        try:
            returncode = proc.wait(timeout=10)
            log("shutdown", f"server 自然退出，returncode={returncode}")
        except subprocess.TimeoutExpired:
            log("shutdown", f"WARN：关闭 stdin 后 10s 仍未退出，pid={proc.pid}；按约束不杀进程")
        stdout_thread.join(timeout=2)
        stderr_thread.join(timeout=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
