# RAG 实验平台 — Streamlit UI 设计规格

**日期**: 2026-07-24
**状态**: 已确认
**项目**: C:\Users\wangzongyi\RAG

---

## 1. 目标

将现有基础 Streamlit 界面（app.py）升级为功能完整的 RAG 实验平台，支持：
- 多策略检索问答（Default / MMR / HyDE / Hybrid）
- 全渠道文档入库（PDF / TXT / 粘贴 / URL / 预设）
- 多策略并排对比实验室
- RAGAS 实时 + 批量评估
- 参数/模型/API 集中配置

---

## 2. 架构

### 2.1 文件结构

```
RAG/
├── app.py                  # 入口：页面配置 + Tab 路由 + 全局 Session State
├── ui/
│   ├── __init__.py         # 空
│   ├── chat.py             # Tab 1: 聊天问答
│   ├── documents.py        # Tab 2: 文档上传/管理
│   ├── lab.py              # Tab 3: 策略对比实验室
│   ├── eval.py             # Tab 4: RAGAS 评估
│   ├── settings.py         # Tab 5: 参数配置/API 设置
│   └── components.py       # 复用 UI 组件
├── minirag.py              # 不修改
├── retrieval.py            # 不修改
├── embedding.py            # 不修改
├── chunking.py             # 不修改
├── vectordb.py             # 不修改
└── RAGAS.py                # 不修改
```

### 2.2 设计原则

- **单入口**：`streamlit run app.py` 即可启动，所有 Tab 在一个页面内
- **组件化**：每个 Tab 的 UI 逻辑在 `ui/` 下独立模块，app.py 只做路由
- **状态集中**：全局 Session State 在 app.py 初始化，各 Tab 模块通过参数接收
- **不修改核心引擎**：minirag.py、retrieval.py 等现有文件不动
- **采用方案 C 架构**：单入口 + 组件化导入

---

## 3. 全局设计

### 3.1 页面配置

- 标题：RAG 实验平台
- 图标：🔬
- 布局：wide
- 侧边栏：默认展开

### 3.2 侧边栏（全局共享）

侧边栏显示全局状态面板：

- API 连接状态指示灯
- 已索引 chunk 数量
- 当前检索策略
- 当前 LLM 模型
- [清空全部索引] 按钮（带二次确认）

### 3.3 Session State 初始化

```python
{
    "engine": EnhancedMiniRAG | None,   # 当前 RAG 引擎实例
    "strategy": str,                     # 当前策略名
    "messages": list[dict],              # 聊天历史
    "indexed_docs": list[dict],          # 已索引文档元信息
    "config": RAGConfig,                # 当前配置
    "lab_history": list[dict],           # 实验室对比历史（最近 5 次）
}
```

### 3.4 Tab 结构

5 个 Tab：智能问答 | 文档管理 | 策略实验室 | RAGAS 评估 | 设置

---

## 4. Tab 1: 智能问答（chat.py）

### 4.1 功能清单

| 功能 | 实现方式 |
|------|---------|
| 策略选择 | st.selectbox: default/mmr/hyde/hybrid |
| 策略参数 | 根据策略展开：MMR→lambda 滑块, Hybrid→alpha 滑块 |
| Top-K | st.slider(1, 10, 5) |
| 相似度阈值 | st.slider(0.0, 1.0, 0.5) |
| 多轮对话 | st.chat_message + st.chat_input |
| 来源展示 | st.expander 内嵌来源卡片，颜色编码相关度 |
| Token 统计 | 每条回答下显示 token 用量 + 耗时 |
| 导出对话 | st.download_button 导出 Markdown |
| 预设问题 | 根据已索引文档生成 3 个推荐问题 |

### 4.2 来源卡片颜色编码

- 绿色（#4CAF50）：score > 0.5
- 橙色（#FF9800）：0.3 < score <= 0.5
- 红色（#f44336）：score <= 0.3

### 4.3 数据流

```
用户输入 -> engine.ask(question, top_k) -> RAGResponse
-> 展示 answer + sources + token_usage + elapsed_ms
-> 追加到 st.session_state.messages
```

---

## 5. Tab 2: 文档管理（documents.py）

### 5.1 功能清单

| 区域 | 功能 | 实现方式 |
|------|------|---------|
| PDF 上传 | 上传 PDF 并解析 | st.file_uploader + pdfplumber |
| TXT 上传 | 上传 TXT 文件 | st.file_uploader |
| 粘贴文本 | 文本框输入 | st.text_area + 文档名输入 |
| URL 抓取 | 抓取网页文本 | st.text_input + requests/httpx |
| 预设知识库 | 一键加载 demo | st.button 加载 demo_knowledge.txt |
| 测试文档 | 加载猫/汽车测试 | st.button 加载 TEST_DOCUMENTS |
| 文档列表 | 已索引文档表格 | st.dataframe |
| 文档删除 | 按文档粒度移除 | 重索引剩余文档 |
| 切割配置 | chunk_size + overlap | st.slider |
| 切割预览 | 选中文档查看切割结果 | st.expander |

### 5.2 数据流

```
文档输入 -> 解析文本 -> TextChunker 切割 -> engine.ingest([Document]) -> 更新索引
```

---

## 6. Tab 3: 策略实验室（lab.py）

### 6.1 功能清单

| 功能 | 实现方式 |
|------|---------|
| 策略多选 | st.multiselect: 至少选 2 个 |
| 查询输入 | st.text_input + "对比"按钮 |
| 结果矩阵 | st.columns(N) 并排展示每个策略的结果 |
| 检索重叠分析 | 计算各策略检索结果交集/差集，文字展示 |
| 历史记录 | st.session_state 保留最近 5 次对比 |

### 6.2 结果矩阵布局

每个策略一列，展示：耗时、检索到的 chunks（含分数）、LLM 生成的回答。

### 6.3 实现要点

- 每个策略创建独立的 EnhancedMiniRAG 实例并独立索引
- 以 query 为输入，多个 engine 同时执行 ask

---

## 7. Tab 4: RAGAS 评估（eval.py）

### 7.1 实时单条评估

| 功能 | 实现方式 |
|------|---------|
| 问题输入 | st.text_input 或从聊天 Tab 传入 |
| 答案输入 | st.text_area，自动填入最新回答 |
| 评估指标 | Faithfulness / Answer Relevancy / Context Recall |
| 分数展示 | st.progress 或 st.metric 展示 0-1 分数 |
| 颜色编码 | 绿(>0.7) / 橙(0.4-0.7) / 红(<0.4) |

### 7.2 批量评估

| 功能 | 实现方式 |
|------|---------|
| 测试集上传 | st.file_uploader 接受 JSON/CSV |
| 模板下载 | st.download_button 提供示例模板 |
| 执行评估 | st.button + st.progress 进度条 |
| 汇总报告 | 雷达图（plotly）+ 统计表格 |
| 分项明细 | st.dataframe 可排序/筛选 |
| 导出报告 | st.download_button 导出 CSV/JSON |

### 7.3 数据流

```
测试集 -> engine.ask(question) x N -> RAGAS metrics -> 汇总展示
```

---

## 8. Tab 5: 设置（settings.py）

### 8.1 功能清单

| 区域 | 配置项 | 实现方式 |
|------|--------|---------|
| API | API Key 输入 | st.text_input(type="password") |
| API | 连接状态 | 指示灯 |
| Embedding | 模型选择 | selectbox: v1/v2/v3 |
| LLM | 模型选择 | selectbox: qwen-turbo/plus/max |
| LLM | Temperature | slider(0.0, 1.0, 0.2) |
| LLM | Max Tokens | slider(256, 4096, 1024) |
| 默认值 | Top-K | slider(1, 10, 5) |
| 默认值 | 相似度阈值 | slider(0.0, 1.0, 0.5) |
| 默认值 | 默认策略 | selectbox |
| 界面 | 主题 | toggle 亮色/暗色 |

---

## 9. 复用组件（components.py）

| 组件 | 用途 | 参数 |
|------|------|------|
| render_source_card(source) | 来源文档卡片 | content, score, doc_id |
| render_status_badge(label, value, color) | 状态徽章 | 标签、值、颜色 |
| render_token_stats(token_usage, elapsed_ms) | Token 统计条 | usage dict, 耗时 |
| render_strategy_selector() | 策略下拉+参数 | 回调更新 engine |
| score_color(score) | 分数到颜色映射 | float to str |

---

## 10. 边界情况与错误处理

- **API Key 未配置**：侧边栏显示断开，禁用问答/实验室/评估 Tab
- **索引为空时提问**：提示"请先在文档管理中添加文档"
- **PDF 解析失败**：toast 错误信息，不阻塞界面
- **URL 抓取超时**：5 秒超时，提示用户手动粘贴
- **LLM 调用失败**：chat message 中展示错误信息，不丢失对话历史
- **评估失败**：RAGAS 依赖缺失时给出安装提示

---

## 11. 不包含的功能（YAGNI）

- 用户登录/权限管理
- 数据库持久化
- 多用户并发
- 图谱可视化
- 文档版本管理
