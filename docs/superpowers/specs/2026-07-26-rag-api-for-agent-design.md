# RAG API for Agent — 设计文档

> 日期: 2026-07-26
> 目的: 在现有 MiniRAG engine 之上构建 FastAPI 层，供 AI Agent 调用

## 背景

现有项目已有完整的 RAG engine（`minirag.py` + `retrieval.py` + `embedding.py` + `vectordb.py` + `chunking.py`），以及 Streamlit 人类界面（`app.py` + `ui/`）。

需要新增 FastAPI 接口层，不是给人用，而是给 AI Agent 调用。Agent 会自己决定调哪个端点、用什么策略、要不要重试。

## 核心设计决策

### 1. 检索和生成分开

Agent 不能接受"一步到位"的黑盒。它需要在检索后、生成前介入判断：
- 信息够不够？
- 要不要换策略再搜一次？
- 要不要修改 query？

因此提供 `retrieve`（只检索）和 `generate`（只生成）两个独立端点，同时保留 `ask`（一步到位）给简单场景。

### 2. 策略缓存 + 共享底层索引

四种检索策略（default / mmr / hyde / hybrid）中，前三种底层都是同一个 `Retriever`（FAISS），hybrid 多一个 `BM25Retriever`。设计为：

- 启动时创建一个 `Retriever` 和一个 `BM25Retriever` 作为共享底层
- 包装不同策略时复用同一个底层索引
- 入库时只需对底层调一次 `ingest`，所有策略自动同步

### 3. 结构化错误 + Agent 行动建议

Agent 需要根据错误类型决定下一步。每个错误响应包含 `action` 字段：

| action | 含义 | Agent 动作 |
|--------|------|-----------|
| `retry` | 临时故障 | 等 2 秒重试 |
| `reindex` | 索引缺失 | 先调 /documents 入库 |
| `fallback` | 当前策略/模型不可用 | 换策略或模型 |
| `give_up` | 彻底失败 | 告知用户 |

## 端点设计

### POST /api/rag/retrieve

纯检索，不生成。

**请求：**
```json
{
  "query": "string (必填)",
  "top_k": "int (默认 5, 范围 1-20)",
  "strategy": "string (默认 'default', 可选: default/mmr/hyde/hybrid)",
  "similarity_threshold": "float (默认 0.5, 范围 0.0-1.0)"
}
```

**响应（成功）：**
```json
{
  "success": true,
  "data": {
    "query": "string",
    "total_found": "int",
    "sufficient": "bool (实际检索到的结果 >= top_k 判定为充分)",
    "strategy_used": "string",
    "results": [
      {
        "content": "string",
        "score": "float",
        "metadata": "dict",
        "chunk_index": "int"
      }
    ]
  },
  "error": null
}
```

### POST /api/rag/generate

只生成，不检索。Agent 自己传入上下文。

**请求：**
```json
{
  "query": "string (必填)",
  "context_chunks": ["string (必填, 至少一条)"],
  "system_prompt_extra": "string (可选, 追加到 system prompt 末尾)"
}
```

**响应（成功）：**
```json
{
  "success": true,
  "data": {
    "answer": "string",
    "sources_used": "int",
    "has_answer": "bool"
  },
  "error": null
}
```

### POST /api/rag/ask

一步到位，检索 + 生成。

**请求：** 同 `/retrieve`

**响应（成功）：**
```json
{
  "success": true,
  "data": {
    "answer": "string",
    "sources": [
      {"content": "string", "score": "float", "metadata": "dict"}
    ],
    "has_answer": "bool"
  },
  "error": null
}
```

### POST /api/rag/documents

文档入库。

**请求：**
```json
{
  "documents": [
    {"content": "string", "metadata": {"key": "value"}}
  ]
}
```

**响应（成功）：**
```json
{
  "success": true,
  "data": {
    "message": "string",
    "document_count": "int",
    "chunk_count": "int"
  },
  "error": null
}
```

### GET /api/rag/status

查询引擎当前状态。

**响应（成功）：**
```json
{
  "success": true,
  "data": {
    "is_indexed": "bool",
    "chunk_count": "int",
    "current_strategy": "string",
    "available_strategies": ["default", "mmr", "hyde", "hybrid"],
    "llm_model": "string",
    "embed_model": "string",
    "similarity_threshold": "float"
  },
  "error": null
}
```

## 错误响应格式

所有端点统一错误格式：

```json
{
  "success": false,
  "data": null,
  "error": {
    "code": "EMBEDDING_ERROR | RETRIEVAL_ERROR | GENERATION_ERROR | VALIDATION_ERROR | INTERNAL_ERROR",
    "message": "string (人类可读)",
    "action": "retry | reindex | fallback | give_up",
    "detail": "string (可选, 额外上下文)"
  }
}
```

### Engine 异常 → action 映射

| Engine 异常 | code | action | 条件 |
|------------|------|--------|------|
| EmbeddingError | EMBEDDING_ERROR | retry | 默认 |
| RetrievalError | RETRIEVAL_ERROR | reindex | "索引为空" in message |
| RetrievalError | RETRIEVAL_ERROR | retry | 其他情况 |
| GenerationError | GENERATION_ERROR | fallback | 空内容 / 安全过滤 |
| GenerationError | GENERATION_ERROR | retry | 网络错误 |
| 其他 Exception | INTERNAL_ERROR | give_up | 未知错误 |

## 生命周期管理

使用 FastAPI `lifespan`：

1. **启动时**：检查 DASHSCOPE_API_KEY，创建 RAGConfig，创建共享底层 Retriever + BM25Retriever
2. **首次使用某策略时**：用共享底层包装策略，缓存
3. **入库时**：对共享底层调 ingest，所有已缓存策略自动同步
4. **关闭时**：无需清理（全部内存数据）

## 架构图

```
Agent
  │ HTTP
  ▼
FastAPI (api.py)
  ├─ POST /retrieve  ─┐
  ├─ POST /generate   ├─ 错误处理中间层 ─┐
  ├─ POST /ask       ─┘                  │
  ├─ POST /documents  ───────────────────┤
  ├─ GET  /status     ───────────────────┤
  └─ lifespan         ───────────────────┤
         │                               │
         ▼                               ▼
   策略缓存 (app.state)            异常 → action 映射
         │
         ▼
   Engine 层 (minirag.py + retrieval.py)
```

## 文件改动

| 文件 | 操作 | 说明 |
|------|------|------|
| api.py | 重写 | FastAPI 应用 + 5 端点 + 生命周期 + 错误处理 |
| minirag.py | 不动 | Engine 层不需要改 |
| retrieval.py | 不动 | 检索策略不需要改 |
| requirements.txt | 可能更新 | 确认 fastapi, uvicorn 已存在 |

## Agent 典型调用流程

```
1. GET /status → 确认引擎就绪
2. POST /retrieve (strategy=default) → 检索
3. 判断 sufficient 为 false → POST /retrieve (strategy=hyde) → 换策略重试
4. 判断 sufficient 为 true → POST /generate → 生成答案
5. 返回答案给最终用户

如果任何一步返回 success=false：
6. 读 error.action 决定重试/降级/放弃
```
