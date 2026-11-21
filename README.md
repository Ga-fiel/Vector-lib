# 向量记忆库（MCP Server 模式 · 方案A：多库）

BGE-M3 一个模型同时出稠密+稀疏向量，Qdrant 本地模式存储，**多库隔离**，通过 MCP 工具供 agent 调用。

## 核心概念

| 概念 | 说明 |
|---|---|
| 库（library） | 一个独立图书馆，如 `记忆库A`、`Java学习笔记` |
| 库id | 安全 ASCII 标识（`lib_001`），同时是 Qdrant collection 名和磁盘目录名 |
| 库名 | 你读取/指定用的中文名，存在映射表里 |
| 映射表 | `qdrant_data/libraries.json`，记录 `lib_001 ↔ 记忆库A` |

```text
qdrant_data/
├── libraries.json                  ← 映射表（id ↔ 中文名）
└── collections/
    ├── lib_001/                    ← 记忆库A
    └── lib_002/                    ← Java学习笔记
```

**备份 = 复制整个 `qdrant_data/` 目录**（映射表和数据一起，不会错位）。

## 目录结构

```
vector-memory/
├── .venv/          虚拟环境
├── config.yaml     配置（模型/存储 /切分/检索）
├── requirements.txt
├── models/         模型权重目录（配置 local_path 时）
├── qdrant_data/    数据库本体 + libraries.json 映射表
├── src/
│   ├── config.py   读配置
│   ├── chunker.py  文本切分
│   ├── embedder.py BGE-M3 向量化
│   ├── store.py    建库/映射表/写入/两阶段检索/按编号读原文
│   └── server.py   MCP Server 入口
└── tests/test_memory.py  自测脚本（用临时目录，不污染数据）
```

## 安装（别人拿到仓库后照做）

需要 Python 3.10+（推荐 3.12）。

### 1. 装依赖

在项目目录打开 PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. 下载模型（最关键，仓库里没有）

BGE-M3 模型约 2GB，**不在仓库里**（被 `.gitignore` 排除），必须自己下载，否则运行会报“找不到模型”。这是最容易卡住的一步，建议用 hf-mirror 镜像：

```powershell
$env:HF_ENDPOINT="https://hf-mirror.com"
.\.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-m3', local_dir='./models/bge-m3', allow_patterns=['*.json','*.model','*.bin','*.txt','*.pt'])"
```

> `*.bin` 包含约 2GB 的 `pytorch_model.bin`，`*.pt` 包含 `colbert_linear.pt` / `sparse_linear.pt`，缺一个都会加载失败。上面命令已把这些全部包含，一次下全。

### 3.（可选）改配置

有 GPU 就把 `config.yaml` 里的 `device: cpu` 改成 `cuda`，其余保持默认即可。

## 自测

```bash
.venv\Scripts\python.exe tests/test_memory.py
```

验证：建库、写入分配文档号(8位 id 编码)、payload 精简、模式2(neighbor) 命中段+扩展段、模式1(full) 整篇返回、read_document 按文档号直取、库隔离、错误库名提示、旧数据自动迁移。

MCP 端到端测试（真正启动 server、经 stdio 握手并调用 5 个工具，用临时目录不污染数据）：

```bash
.venv\Scripts\python.exe tests/test_mcp.py
```

> 依赖注意：`mcp` 必须用 **1.x**（`mcp<2`），2.x 把 `FastMCP` 改名了，`src/server.py` 只兼容 1.x。`requirements.txt` 已 pin `mcp>=1.0,<2`。

## 接入 agent（MCP）

MCP Server 入口：`src/server.py`，stdio 传输。agent 侧配置示例：

```json
{
  "mcpServers": {
    "vector-memory": {
      "command": "<你的项目路径>\\.venv\\Scripts\\python.exe",
      "args": ["-m", "src.server"],
      "cwd": "<你的项目路径>"
    }
  }
}
```

agent 可用的五个工具：

| 工具 | 作用 |
|---|---|
| `create_library(name)` | 新建库并取名，返回库id |
| `list_libraries()` | 列出所有库（id + 名字） |
| `write_memory(library, text, ...)` | 写入一段文本，切成多片存成一个文档，返回 4 位文档号 `doc_id` |
| `search_memory(library, query, mode=, expand=, top_k=, max_chunks=)` | 混合检索 + 上下文打包 |
| `read_document(library, doc_id, from_chunk=, to_chunk=)` | 按 4 位文档号算 id 直取原文段落 |

## 检索的两种模式

**id 自带分组信息**：每个 point 的 id 是 8 位整数编码 `文档号×10000 + 段号`（如 `00010003` = 文档1的第3段）。前 4 位是文档号、后 4 位是段号，从 id 直接解出，payload 只存原文 `text` 和 `total_chunks`（总段数）等标签，不再单独存 doc_id/chunk_index，也不声明 payload 索引。

检索分两阶段：①混合向量检索拿命中段的 id；②从 id 解出文档号/段号，算出要取的关联段 id，用 `retrieve(ids=[...])` 主键直取一次性拼成连续上下文。**主键直取是 O(1)，不依赖 payload 索引，Qdrant 本地模式也快**（这正是改用 id 编码的原因——本地模式 payload 索引无效）。

单文档最多 9999 段，文档号最多 9999，超限会报错保护 8 位结构。

| 模式 | 行为 |
|---|---|
| `neighbor`（默认） | 命中段上下各取 `expand` 段（默认各1段），拼出连续上下文，段标 `is_hit` 区分命中/扩展 |
| `full` | 命中段所在文档的全部段一次性返回（完整原文） |

配置在 `config.yaml` 的 `retrieve`：`mode`、`expand`、`max_chunks`（返回段数上限，防规模过大）。

## 查询原理

```text
① 关键词 → BGE-M3 → 稠密向量 + 稀疏向量
                      ├─ 稀疏路：词面匹配 → 命中段
                      └─ 稠密路：语义相似 → 命中段
                            ↓ RRF 融合
② 从命中 id 解出文档号/段号，算出关联段 id（neighbor=±expand / full=整篇）
                       └─ retrieve(ids=[...]) 主键直取 → 标记 is_hit / 排序 / 拼接
只在指定的那个库里搜，库之间天然隔离。
```

> 关于旧数据：早期版本的随机 UUID id / 带 source 字段的数据，会在 `MemoryStore` 初始化时自动迁移成新的 8 位 id 编码（按 source 分组还原文档边界，复用原向量），无需手动处理。迁移需要能以独占方式打开 qdrant_data（本地模式单进程独占）。

## 设计原则

业务逻辑（embedder/chunker/store）与接口（MCP）分离。换模型改 `embedder.py`，换库/切服务模式改 `store.py` 连接参数，MCP 层不动。
