# RAG 实验平台 — Streamlit UI 实现计划

> **Goal:** 将基础 app.py 升级为功能完整的 5-Tab RAG 实验平台

**Architecture:** 单入口 `app.py` 做 Tab 路由 + 全局 Session State，`ui/` 目录下 6 个模块。不修改现有引擎文件。

**Tech Stack:** Python 3.x, Streamlit, DashScope SDK, FAISS, pdfplumber, plotly (雷达图)

## Global Constraints
- 不修改 minirag.py, retrieval.py, embedding.py, chunking.py, vectordb.py, RAGAS.py
- 策略名使用: "default", "mmr", "hyde", "hybrid"
- 错误使用 st.warning/st.error/st.toast，不崩溃页面

---

## 文件创建清单

```
RAG/
├── app.py              # 重写: 页面配置 + Tab 路由 + 全局 Session State + 侧边栏
├── ui/
│   ├── __init__.py     # 新建: 空文件
│   ├── components.py   # 新建: score_color, render_source_card, render_token_stats
│   ├── chat.py         # 新建: Tab 1 智能问答
│   ├── documents.py    # 新建: Tab 2 文档管理
│   ├── lab.py          # 新建: Tab 3 策略实验室
│   ├── eval.py         # 新建: Tab 4 RAGAS 评估
│   └── settings.py     # 新建: Tab 5 设置
└── requirements.txt    # 修改: 添加 plotly
```

---

### Task 1: ui/__init__.py + ui/components.py

- [ ] 创建 `ui/__init__.py`（空文件）
- [ ] 创建 `ui/components.py`，包含:

```python
"""RAG 实验平台 — 复用 UI 组件"""
import streamlit as st

def score_color(score: float) -> str:
    if score > 0.5: return "#4CAF50"
    elif score > 0.3: return "#FF9800"
    else: return "#f44336"

def render_source_card(source: dict) -> None:
    score = source.get("score", 0.0)
    color = score_color(score)
    doc_id = source.get("doc_id", "?")
    st.markdown(f"""
    <div style="background:#f0f2f6;border-left:4px solid {color};border-radius:6px;
    padding:10px 14px;margin:6px 0;font-size:0.9em;">
        <span style="background:{color};color:white;padding:2px 8px;border-radius:4px;font-size:0.78em;">
        相关度: {score:.4f}</span>
        <span style="color:#888;font-size:0.78em;margin-left:8px;">Chunk #{doc_id}</span>
        <p style="margin-top:6px;line-height:1.5;">{source['content']}</p>
    </div>""", unsafe_allow_html=True)

def render_token_stats(token_usage: dict, elapsed_ms: float) -> None:
    cols = st.columns(3)
    cols[0].metric("📥 Input", token_usage.get("input_tokens", token_usage.get("input", 0)))
    cols[1].metric("📤 Output", token_usage.get("output_tokens", token_usage.get("output", 0)))
    cols[2].metric("⏱️ 耗时", f"{elapsed_ms:.0f}ms")
```

- [ ] 验证: `python -c "from ui.components import score_color, render_source_card, render_token_stats; print('OK')"`

---

### Task 2: ui/settings.py

- [ ] 创建 `ui/settings.py`，包含 `render_settings_tab() -> None`
- 3 个子 Tab: API 配置 / 模型选择 / 默认参数
- API Key 输入 + 写回 .env
- Embedding 模型选择 (v1/v2/v3)
- LLM 模型选择 (qwen-turbo/plus/max) + Temperature + Max Tokens
- 默认 Top-K / 阈值 / 策略
- 应用按钮：更新 config → 置空 engine → rerun

- [ ] 验证: `python -c "from ui.settings import render_settings_tab; print('OK')"`

---

### Task 3: ui/documents.py

- [ ] 创建 `ui/documents.py`，包含 `render_documents_tab() -> None`
- 3 个子 Tab: 上传文档 / 预设知识库 / 已索引文档
- 上传: PDF(pdfplumber) / TXT / 粘贴文本 / URL 抓取
- 预设: LightSpeed Tech 知识库 / 猫汽车测试文档
- 列表: DataFrame 展示 + 清空按钮
- 每个上传按钮调用 `engine.ingest([Document])`

- [ ] 验证: `python -c "from ui.documents import render_documents_tab; print('OK')"`

---

### Task 4: ui/chat.py

- [ ] 创建 `ui/chat.py`，包含 `render_chat_tab() -> None`
- 参数区 (expander): 策略选择 + Top-K + 阈值 + 策略专属参数 (MMR λ / Hybrid α)
- 推荐问题按钮 (根据已索引文档生成)
- 对话历史 (st.chat_message)
- 来源展开 (st.expander + render_source_card)
- Token 统计 (render_token_stats)
- 导出对话 (st.download_button → Markdown)
- 清空对话按钮
- 错误处理: try/except 包裹 engine.ask()

- [ ] 验证: `python -c "from ui.chat import render_chat_tab; print('OK')"`

---

### Task 5: ui/lab.py

- [ ] 创建 `ui/lab.py`，包含 `render_lab_tab() -> None`
- 策略多选 (st.multiselect) + Top-K
- 查询输入 + 对比按钮
- 结果矩阵: st.columns(N) 并排展示
- 每个策略: chunks + 回答 + token 统计
- 检索重叠分析: Jaccard 相似度
- 历史记录: 保留最近 5 次

- [ ] 验证: `python -c "from ui.lab import render_lab_tab; print('OK')"`

---

### Task 6: ui/eval.py

- [ ] 创建 `ui/eval.py`，包含 `render_eval_tab() -> None`
- 2 个子 Tab: 实时评估 / 批量评估
- 实时: 问题/回答/上下文/参考答案 4 个输入框 → RAGASEvaluator.evaluate()
- 批量: 上传 CSV/JSON 测试集 + 下载模板 → 进度条 + 雷达图(plotly) + 导出报告
- 自动填入最近一次问答

- [ ] 验证: `python -c "from ui.eval import render_eval_tab; print('OK')"`

---

### Task 7: 重写 app.py

- [ ] 备份: `cp app.py app.py.bak`
- [ ] 重写 app.py:

```python
"""RAG 实验平台 — Streamlit 全功能界面
启动: streamlit run app.py
"""
import sys
from pathlib import Path
_PROJECT_DIR = Path(__file__).resolve().parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

import streamlit as st
from minirag import RAGConfig, init

st.set_page_config(page_title="RAG 实验平台", page_icon="🔬", layout="wide", initial_sidebar_state="expanded")

# 全局 Session State
def init_session():
    defaults = {
        "engine": None, "strategy": "default", "messages": [],
        "indexed_docs": [], "chunk_count": 0, "config": RAGConfig(),
        "lab_history": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

# API 初始化
try:
    init()
    api_ok = True
except RuntimeError:
    api_ok = False

# 侧边栏: 状态面板
def render_sidebar():
    with st.sidebar:
        st.markdown("## 🔬 RAG 实验平台")
        st.divider()
        st.subheader("🔌 连接状态")
        if api_ok:
            st.success("🟢 DashScope API 已连接")
        else:
            st.error("🔴 API Key 未配置")
        st.divider()
        st.subheader("📊 索引状态")
        col1, col2 = st.columns(2)
        col1.metric("📄 文档", len(st.session_state.get("indexed_docs", [])))
        col2.metric("🧩 Chunks", st.session_state.get("chunk_count", 0))
        st.divider()
        st.subheader("🎯 当前配置")
        strategy_names = {"default": "Default", "mmr": "MMR", "hyde": "HyDE", "hybrid": "Hybrid"}
        st.caption(f"策略: {strategy_names.get(st.session_state.strategy, '?')}")
        st.caption(f"LLM: {st.session_state.config.llm_model}")
        st.caption(f"Top-K: {st.session_state.config.top_k}")
        st.divider()
        if st.button("🗑️ 清空全部索引", use_container_width=True):
            st.session_state.engine = None
            st.session_state.indexed_docs = []
            st.session_state.chunk_count = 0
            st.session_state.messages = []
            st.rerun()

# 主界面
def main():
    render_sidebar()
    st.markdown("## 🔬 RAG 实验平台")
    st.caption("多策略检索增强生成 — 从文档到答案的全流程实验环境")

    from ui.chat import render_chat_tab
    from ui.documents import render_documents_tab
    from ui.lab import render_lab_tab
    from ui.eval import render_eval_tab
    from ui.settings import render_settings_tab

    tabs = st.tabs(["💬 智能问答", "📁 文档管理", "🔬 策略实验室", "📊 RAGAS 评估", "⚙️ 设置"])
    with tabs[0]: render_chat_tab()
    with tabs[1]: render_documents_tab()
    with tabs[2]: render_lab_tab()
    with tabs[3]: render_eval_tab()
    with tabs[4]: render_settings_tab()

if __name__ == "__main__":
    main()
```

- [ ] 验证: `streamlit run app.py` 正常启动，5 个 Tab 可切换

---

### Task 8: 更新 requirements.txt

- [ ] 追加 `plotly>=5.18.0` 到 requirements.txt
- [ ] `pip install plotly`

---

## 验证总清单

- [ ] `streamlit run app.py` 正常启动
- [ ] 侧边栏: API状态 + 索引统计 + 当前配置
- [ ] Tab 1: 加载文档 → 切换策略 → 提问 → 来源展示 → 导出对话
- [ ] Tab 2: PDF/TXT/粘贴/URL/预设 → 清空索引
- [ ] Tab 3: 多选策略 → 对比查询 → 重叠分析
- [ ] Tab 4: 单条评估 + 批量评估(CSV)
- [ ] Tab 5: API Key / 模型 / 参数修改
- [ ] 错误场景有友好提示
