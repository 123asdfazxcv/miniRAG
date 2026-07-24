# 🔬 MiniRAG — 从零构建的 RAG 实验平台

一个**教学级** Retrieval-Augmented Generation (RAG) 系统，7 天从手写向量检索到完整 Streamlit 实验平台。适合学习 RAG 原理、面试准备、快速原型验证。

## ✨ 特性

- **4 种检索策略**: Default (向量) / MMR (多样性) / HyDE (假设文档) / Hybrid (向量+BM25)
- **全渠道文档入库**: PDF / TXT / 粘贴文本 / URL 抓取 / 预设知识库
- **策略对比实验室**: 同 query 多策略并排对比 + 检索重叠分析
- **RAGAS 评估**: 单条实时评估 + 批量评估 + 雷达图可视化
- **生产级工程**: 配置管理 / 异常体系 / 日志 / 类型标注 / 来源追溯
- **Streamlit 全功能界面**: 5 Tab 实验平台，开箱即用

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
# 复制 .env.example 为 .env，填入你的 DashScope API Key
cp .env.example .env
```

> 获取 Key: [阿里云 DashScope 控制台](https://dashscope.console.aliyun.com/apiKey)

### 3. 启动界面

```bash
streamlit run app.py
```

浏览器访问 `http://localhost:8501`

## 📖 项目结构

```
RAG/
├── app.py               # Streamlit 主入口 (5 Tab 路由)
├── ui/                  # 界面模块
│   ├── chat.py          #   Tab 1: 智能问答
│   ├── documents.py     #   Tab 2: 文档管理
│   ├── lab.py           #   Tab 3: 策略实验室
│   ├── eval.py          #   Tab 4: RAGAS 评估
│   ├── settings.py      #   Tab 5: 设置
│   └── components.py    #   复用 UI 组件
├── minirag.py           # 核心引擎: MiniRAG, Retriever, Generator
├── retrieval.py         # 高级检索: MMR, HyDE, BM25, Hybrid
├── RAGAS.py             # 评估引擎: Faithfulness, Relevancy, Recall
├── embedding.py         # DashScope Embedding 封装
├── chunking.py          # 文档切割器 (固定大小 / 按句子)
├── vectordb.py          # FAISS + ChromaDB 向量存储
├── demo_knowledge.txt   # 预设知识库 (LightSpeed Tech 公司文档)
├── requirements.txt     # 依赖清单
└── .env.example         # 环境变量模板
```

## 🧪 学习路径 (Day 1-7)

| Day | 内容 | 核心文件 |
|-----|------|---------|
| 1-2 | Embedding + 语义搜索 + 文档切割 | `embedding.py`, `chunking.py` |
| 3 | FAISS / ChromaDB 向量数据库 | `vectordb.py` |
| 4 | 端到端 RAG 系统 (Ingest → Retrieve → Generate) | `minirag.py` |
| 5 | 检索优化: MMR / HyDE / BM25 / Hybrid | `retrieval.py` |
| 6 | RAGAS 评估 + LLM Reranker + MultiQuery | `RAGAS.py` |
| 7 | Streamlit 全功能实验平台 | `app.py`, `ui/` |

## 🎯 检索策略对比

| 策略 | 原理 | 适用场景 |
|------|------|---------|
| **Default** | 纯向量相似度检索 | 通用，基线 |
| **MMR** | 最大边际相关性，平衡相关度与多样性 | Top-K 结果重复时 |
| **HyDE** | LLM 生成假设答案再检索 | 短查询、query-document 语义鸿沟 |
| **Hybrid** | 向量 + BM25 双路融合 | 需要精确关键词匹配 |

## 🛠 技术栈

- **Embedding**: 阿里云 DashScope (text-embedding-v1/v2/v3)
- **LLM**: 通义千问 (qwen-turbo / qwen-plus / qwen-max)
- **向量库**: FAISS (内存) + ChromaDB (可选持久化)
- **分词**: jieba (BM25 中文分词)
- **前端**: Streamlit
- **评估**: 自实现 RAGAS (Faithfulness / Answer Relevancy / Context Recall)
- **可视化**: Plotly (雷达图)

## 📄 License

MIT
