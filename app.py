"""
================================================================================
RAG 实验平台 — Streamlit 全功能界面
================================================================================

5 个 Tab: 智能问答 | 文档管理 | 策略实验室 | RAGAS 评估 | 设置

启动: streamlit run app.py
"""

import sys
from pathlib import Path

_PROJECT_DIR = Path(__file__).resolve().parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

import streamlit as st
from minirag import RAGConfig, init

# ============================================================================
# 页面配置
# ============================================================================

st.set_page_config(
    page_title="RAG 实验平台",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# 全局样式
# ============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 1.8em;
        font-weight: bold;
        margin-bottom: 0.2em;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        font-size: 1em;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# Session State 初始化
# ============================================================================

def init_session() -> None:
    """初始化全局 Session State。"""
    defaults = {
        "engine": None,
        "strategy": "default",
        "messages": [],
        "indexed_docs": [],
        "chunk_count": 0,
        "config": RAGConfig(),
        "lab_history": [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session()

# ============================================================================
# API 初始化
# ============================================================================

try:
    init()
    api_ok = True
except RuntimeError:
    api_ok = False

# ============================================================================
# 全局侧边栏
# ============================================================================

def render_sidebar() -> None:
    """渲染全局侧边栏：状态面板 + 快捷操作。"""
    with st.sidebar:
        st.markdown("## 🔬 RAG 实验平台")
        st.divider()

        st.subheader("🔌 连接状态")
        if api_ok:
            st.success("🟢 DashScope API 已连接")
        else:
            st.error("🔴 API Key 未配置")
            st.caption("请在 ⚙️ 设置 中配置 API Key")

        st.divider()

        st.subheader("📊 索引状态")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📄 文档", len(st.session_state.get("indexed_docs", [])))
        with col2:
            st.metric("🧩 Chunks", st.session_state.get("chunk_count", 0))

        st.divider()

        st.subheader("🎯 当前配置")
        strategy_names = {
            "default": "Default (向量)",
            "mmr": "MMR (多样性)",
            "hyde": "HyDE (假设文档)",
            "hybrid": "Hybrid (混合)",
        }
        st.caption(f"策略: {strategy_names.get(st.session_state.get('strategy', 'default'), '?')}")
        st.caption(f"LLM: {st.session_state.config.llm_model}")
        st.caption(f"Embed: {st.session_state.config.embed_model}")
        st.caption(f"Top-K: {st.session_state.config.top_k}")
        st.caption(f"阈值: {st.session_state.config.similarity_threshold}")

        st.divider()

        if st.button("🗑️ 清空全部索引", use_container_width=True):
            st.session_state.engine = None
            st.session_state.indexed_docs = []
            st.session_state.chunk_count = 0
            st.session_state.messages = []
            st.success("已清空")
            st.rerun()

        st.caption("MiniRAG v1.0 | 2026-07-24")


# ============================================================================
# 主界面 — Tab 路由
# ============================================================================

def main() -> None:
    render_sidebar()

    st.markdown('<p class="main-header">🔬 RAG 实验平台</p>', unsafe_allow_html=True)
    st.caption("多策略检索增强生成 — 从文档到答案的全流程实验环境")

    from ui.chat import render_chat_tab
    from ui.documents import render_documents_tab
    from ui.lab import render_lab_tab
    from ui.eval import render_eval_tab
    from ui.settings import render_settings_tab

    tabs = st.tabs([
        "💬 智能问答",
        "📁 文档管理",
        "🔬 策略实验室",
        "📊 RAGAS 评估",
        "⚙️ 设置",
    ])

    with tabs[0]:
        render_chat_tab()

    with tabs[1]:
        render_documents_tab()

    with tabs[2]:
        render_lab_tab()

    with tabs[3]:
        render_eval_tab()

    with tabs[4]:
        render_settings_tab()


if __name__ == "__main__":
    main()
