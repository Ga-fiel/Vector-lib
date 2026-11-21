# 向量记忆库（Vector-lib）

一个给 AI agent 用的**本地长期记忆库**：把会话、资料、笔记等文本切片后存成向量，之后既能按**关键词**又能按**语义**检索，还能按文档编号直接取回原文。通过 MCP 协议暴露成标准工具，供多个 agent 共用，作为多 agent 协作的记忆底座。

- **Embedding 模型**：BGE-M3 —— 一个模型同时产出稠密向量（语义）和稀疏向量（关键词），无需再拼 BM25。
- **存储**：Qdrant 本地模式（单机、零运维、数据私有，直接在磁盘上）。
- **接口**：MCP Server（stdio 传输），agent 通过 5 个工具读写记忆。
- **隔离**：支持多个库，库与库之间天然隔离（可理解为多个独立的"笔记本"）。

---

## 一、项目解释

### 1.1 它能解决什么问题

普通聊天里 agent 之间没有共享记忆。这个项目提供一层持久化的"记忆"：

- agent 把有意义的内容 `write_memory` 写进某个库；
- 之后任何 agent 都能用 `search_memory` 按关键词或语义把这个内容找回来；
- 找到命中段后，还能把**前后文**一起打包返回，或按文档编号 `read_document` 取回整段原文。

### 1.2 检索原理（为什么会同时支持关键词和语义）

```text
查询 → BGE-M3 → 稠密向量 + 稀疏向量
                ├─ 稀疏路：词面精确匹配（如 "::"、"valueOf" 这种术语）
                └─ 稠密路：语义相似（词不一样但意思相近）
                      ↓ RRF 融合排序
从命中 id 解出「文档号 / 段号」→ 算出关联段 id → retrieve(ids) 主键直取 → 拼接上下文
只在指定的那个库里搜，库之间互不干扰。
```

**id 自带分组信息**：每个切片（point）的 id 是 8 位整数编码 `文档号×10000 + 段号`，例如 `00010003` 表示"文档 1 的第 3 段"。前 4 位是文档号、后 4 位是段号。payload 只存原文 `text`、`total_chunks`（总段数）和少量标签，不单独存 doc_id、也不建 payload 索引。

这样做的好处：取上下文时用 `retrieve(ids=[...])` 主键直取是 O(1)，不依赖 payload 索引（Qdrant 本地模式下 payload 索引本就无效），所以又快又省。

- 单文档最多 9999 段，文档号最多 9999，超限会报错保护 8 位结构。

### 1.3 多库与映射表

| 概念 | 说明 |
|---|---|
| 库（library） | 一个独立的记忆空间，如 `记忆库A`、`Java学习笔记` |
| 库id | 安全的 ASCII 标识（`lib_001`），同时是 Qdrant collection 名和磁盘目录名 |
| 库名 | 你读写时用的中文名，存在映射表里 |
| 映射表 | `qdrant_data/libraries.json`，记录 `lib_001 ↔ 记忆库A` 的对应关系 |

```text
qdrant_data/
├── libraries.json          ← 映射表（id ↔ 中文名）
└── collections/
    ├── lib_001/            ← 记忆库A
    └── lib_002/            ← Java学习笔记
```

**备份 = 复制整个 `qdrant_data/` 目录**（映射表和数据一起复制，不会错位）。

### 1.4 目录结构

```text
Vector-lib/
├── .venv/              虚拟环境（不入库）
├── config.yaml         配置（模型 / 存储 / 切分 / 检索）
├── requirements.txt    Python 依赖清单
├── models/             BGE-M3 模型权重（不入库，需自行下载到 models/bge-m3）
├── qdrant_data/        数据库本体 + libraries.json 映射表（不入库，运行时自动创建）
├── src/
│   ├── config.py       读取配置
│   ├── chunker.py      文本切分
│   ├── embedder.py     BGE-M3 向量化
│   ├── store.py        建库 / 映射表 / 写入 / 两阶段检索 / 按编号读原文
│   └── server.py       MCP Server 入口
└── tests/
    ├── test_memory.py  核心逻辑自测（临时目录，不污染数据）
    ├── test_mcp.py     MCP 端到端自测（临时目录，不污染数据）
    └── diagnose_mcp_stdio.py  MCP 启动链路诊断脚本
```

### 1.5 设计原则

业务逻辑（embedder / chunker / store）与接口（MCP）分离：

- 换模型：改 `embedder.py`
- 换存储 / 切到 Qdrant 服务模式：改 `store.py` 的连接参数
- MCP 层不动

> 关于旧数据：早期版本的随机 UUID id / 带 `source` 字段的数据，会在 `MemoryStore` 初始化时自动迁移成新的 8 位 id 编码（按 source 分组还原文档边界，复用原向量），无需手动处理。迁移需要能以独占方式打开 qdrant_data（本地模式单进程独占）。

---

## 二、快速开始

需要 Python 3.10+（推荐 3.12）。以下都在项目目录下的 PowerShell 中执行。

### 2.1 安装依赖

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> 依赖注意：`mcp` 必须用 **1.x**（`mcp<2`），2.x 把 `FastMCP` 改名了，`src/server.py` 只兼容 1.x。`requirements.txt` 已 pin `mcp>=1.0,<2`。

### 2.2 下载模型（最关键，仓库里没有）

BGE-M3 模型约 2GB，**不在仓库里**（被 `.gitignore` 排除），必须自己下载，否则运行会报"找不到模型"。这是最容易卡住的一步，建议用 hf-mirror 镜像：

```powershell
$env:HF_ENDPOINT="https://hf-mirror.com"
.\.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-m3', local_dir='./models/bge-m3', allow_patterns=['*.json','*.model','*.bin','*.txt','*.pt'])"
```

> `*.bin` 包含约 2GB 的 `pytorch_model.bin`，`*.pt` 包含 `colbert_linear.pt` / `sparse_linear.pt`，缺一个都会加载失败。上面命令已把这些全部包含，一次下全。

### 2.3（可选）改配置

有 GPU 就把 `config.yaml` 里的 `device: cpu` 改成 `cuda`，其余保持默认即可。

### 2.4 自测确认能跑

核心自测：

```powershell
.\.venv\Scripts\python.exe tests/test_memory.py
```

验证：建库、写入分配文档号（8 位 id 编码）、payload 精简、neighbor 模式命中段+扩展段、full 模式整篇返回、read_document 按文档号直取、库隔离、错误库名提示、旧数据自动迁移。

MCP 端到端自测（真正启动 server、经 stdio 握手并调用 5 个工具，用临时目录不污染数据）：

```powershell
.\.venv\Scripts\python.exe tests/test_mcp.py
```

全部通过即说明模型和依赖都装好了。

### 2.5 接入 agent（MCP）

MCP Server 入口：`src/server.py`，stdio 传输。agent 侧配置示例（把路径换成你自己的项目路径）：

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

---

## 三、使用场景

### 3.1 五个工具

| 工具 | 作用 |
|---|---|
| `create_library(name)` | 新建库并取名，返回库 id。同名库已存在会报错 |
| `list_libraries()` | 列出所有库（id + 名字），写入 / 检索前可先确认有哪些库 |
| `write_memory(library, text, memory_type, importance)` | 把一段文本写入指定库，自动切成多段存成一个文档，返回 4 位文档号 `doc_id` 和首段 id |
| `search_memory(library, query, mode, expand, top_k, max_chunks)` | 混合检索（关键词+语义）并把上下文打包返回 |
| `read_document(library, doc_id, from_chunk, to_chunk)` | 按 4 位文档号算出段 id 直取原文段落 |

> `library` 参数既可以传中文库名（如 `知识库`），也可以传库 id（如 `lib_001`）。

### 3.2 典型流程：把资料变成可检索的记忆

1. **建库**（一次性）：
   `create_library("Java学习笔记")` → 返回 `{"id": "lib_001", ...}`

2. **写入**：
   `write_memory("Java学习笔记", "……Lambda 表达式的正文……", memory_type="资料", importance=1.0)`
   → 返回 `doc_id="0001"`、`first_point_id="00010000"`、`total_chunks=6`

3. **检索**：
   `search_memory("Java学习笔记", "方法引用 :: 静态方法 valueOf")`
   → 返回命中段（带 `score`、`is_hit=true`）及其上下文段（`is_hit=false`），拼成连续上下文。

4. **取原文**：
   `read_document("Java学习笔记", "0001", from_chunk=2, to_chunk=3)` → 按编号直取第 2~3 段原文。

### 3.3 两种检索模式

| 模式 | 行为 | 适合场景 |
|---|---|---|
| `neighbor`（默认） | 命中段上下各取 `expand` 段（默认各 1 段），拼出连续上下文，用 `is_hit` 区分命中 / 扩展 | 只想快速定位答案，并看一眼前后文 |
| `full` | 命中段所在文档的全部段一次性返回（完整原文） | 想把整篇资料完整交给 agent 阅读 |

配置在 `config.yaml` 的 `retrieve`：`mode`、`expand`、`max_chunks`（返回段数上限，防止规模过大）。

### 3.4 适合用它做什么

- **多 agent 共享记忆**：多个 agent 读写同一个库，前一个 agent 查到的资料，后一个能直接检索到。
- **长文档问答**：把文档写进去，用关键词+语义混合检索定位，再取连续上下文。
- **术语 / 专名检索**：中英文专业术语、代码片段、符号（如 `::`、`valueOf`）用稀疏向量精确匹配，比纯语义检索更准。
- **会话沉淀**：把有价值的对话结论写进记忆库，之后按语义 recall，而不依赖上下文窗口。

### 3.5 注意事项

- Qdrant 本地模式是**单进程独占**的：同一时间只能有一个进程打开 `qdrant_data`。别同时跑多个 server 测试。
- 数据在 `qdrant_data/`，模型在 `models/`，两者都不入库；迁移 / 备份时按需单独处理。
- 首次调用工具会加载约 2GB 模型，需要几秒到十几秒（取决于磁盘 / 缓存），之后再复用，不会重复加载。
