"""
RAG 实验平台 — Tab 2: 智能问答
基于 EnhancedMiniRAG 的问答界面，支持 4 种检索策略
"""
import streamlit as st
from ui.components import render_source_card, render_token_stats

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

STRATEGY_OPTIONS = ["default", "mmr", "hyde", "hybrid"]
STRATEGY_LABELS = {
    "default": "Default (向量)",
    "mmr": "MMR (多样性)",
    "hyde": "HyDE (假设文档)",
    "hybrid": "Hybrid (混合)",
}


# ---------------------------------------------------------------------------
# 公开入口
# ---------------------------------------------------------------------------

def render_chat_tab() -> None:
    """渲染问答 Tab。"""
    _init_session()

    st.header("💬 智能问答")

    # ---- 参数区 ----
    _render_params()

    # ---- 建议问题 ----
    suggestions = _generate_suggested_questions()
    if suggestions:
        st.divider()
        _render_suggestions(suggestions)

    # ---- 聊天记录 ----
    st.divider()
    _render_chat_history()

    # ---- 输入区 ----
    _render_chat_input()

    # ---- 底部操作 ----
    st.divider()
    _render_bottom_actions()


# ---------------------------------------------------------------------------
# Session 初始化
# ---------------------------------------------------------------------------

def _init_session() -> None:
    """确保必要的 session state 键存在。"""
    from minirag import RAGConfig

    defaults = {
        "strategy": "default",
        "engine": None,
        "messages": [],
        "config": RAGConfig(top_k=5),
        "prev_strategy": "default",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ---------------------------------------------------------------------------
# 参数区
# ---------------------------------------------------------------------------

def _render_params() -> None:
    """渲染检索参数配置。"""
    with st.expander("⚙️ 检索参数", expanded=True):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.selectbox(
                "检索策略",
                options=STRATEGY_OPTIONS,
                format_func=lambda x: STRATEGY_LABELS[x],
                key="strategy",
            )
        # 策略变更时重置引擎
        if st.session_state.strategy != st.session_state.prev_strategy:
            st.session_state.engine = None
            st.session_state.prev_strategy = st.session_state.strategy

        with col2:
            st.session_state.config.top_k = st.slider(
                "Top-K", 1, 10, st.session_state.config.top_k,
            )

        with col3:
            st.session_state.config.similarity_threshold = st.slider(
                "相似度阈值", 0.0, 1.0,
                st.session_state.config.similarity_threshold, 0.05,
            )

        with col4:
            strategy = st.session_state.strategy
            if strategy == "mmr":
                if "mmr_lambda" not in st.session_state:
                    st.session_state.mmr_lambda = 0.7
                st.session_state.mmr_lambda = st.slider(
                    "MMR λ", 0.0, 1.0, st.session_state.mmr_lambda, 0.05,
                    key="mmr_lambda_slider",
                )
            elif strategy == "hybrid":
                if "hybrid_alpha" not in st.session_state:
                    st.session_state.hybrid_alpha = 0.3
                st.session_state.hybrid_alpha = st.slider(
                    "Hybrid α", 0.0, 1.0, st.session_state.hybrid_alpha, 0.05,
                    key="hybrid_alpha_slider",
                )


# ---------------------------------------------------------------------------
# 建议问题
# ---------------------------------------------------------------------------

def _generate_suggested_questions() -> list[str]:
    """根据已索引文档生成最多 4 个推荐问题。"""
    docs = st.session_state.get("indexed_docs", [])
    if not docs:
        return []

    questions = ["请总结文档的主要内容"]
    for doc in docs[:3]:
        name = doc if isinstance(doc, str) else ""
        if name and len(name) < 30:
            questions.append(f"关于「{name}」，有哪些重要信息？")
    return questions[:4]


def _render_suggestions(suggestions: list[str]) -> None:
    """渲染建议问题按钮。"""
    st.caption("💡 试试这些问题：")
    cols = st.columns(len(suggestions))
    for i, question in enumerate(suggestions):
        if cols[i].button(question, key=f"chat_suggest_{i}", use_container_width=True):
            _process_query(question)


# ---------------------------------------------------------------------------
# 聊天记录
# ---------------------------------------------------------------------------

def _render_chat_history() -> None:
    """渲染聊天消息历史。"""
    for msg in st.session_state.messages:
        role = msg["role"]
        if role == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        elif role == "assistant":
            with st.chat_message("assistant"):
                if msg.get("error"):
                    st.error(msg["error"])
                else:
                    st.markdown(msg["content"])
                if msg.get("sources"):
                    with st.expander("📚 查看来源"):
                        for src in msg["sources"]:
                            render_source_card(src)
                if msg.get("token_usage"):
                    render_token_stats(msg["token_usage"], msg.get("elapsed_ms", 0))


# ---------------------------------------------------------------------------
# 输入区
# ---------------------------------------------------------------------------

def _render_chat_input() -> None:
    """渲染聊天输入框。"""
    if prompt := st.chat_input("输入你的问题..."):
        docs = st.session_state.get("indexed_docs", [])
        if not docs:
            st.warning("请先上传文档才能提问！")
            return
        _process_query(prompt)


def _process_query(prompt: str) -> None:
    """处理用户提问：发送消息 → 调用引擎 → 保存结果。"""
    st.session_state.messages.append({"role": "user", "content": prompt})

    try:
        engine = _get_engine()
        result = engine.ask(prompt)
        st.session_state.messages.append({
            "role": "assistant",
            "content": result.answer,
            "sources": [
                {"content": s.content, "score": s.score, "doc_id": s.doc_id}
                for s in result.sources
            ],
            "token_usage": result.token_usage,
            "elapsed_ms": result.elapsed_ms,
        })
    except Exception as e:
        st.session_state.messages.append({
            "role": "assistant",
            "content": "抱歉，处理您的问题时出现错误。",
            "error": str(e),
        })

    st.rerun()


# ---------------------------------------------------------------------------
# 底部操作
# ---------------------------------------------------------------------------

def _render_bottom_actions() -> None:
    """渲染底部操作按钮。"""
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ 清空对话", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.session_state.messages:
            md = _export_chat_markdown()
            st.download_button(
                label="📥 导出对话",
                data=md,
                file_name="rag_chat_export.md",
                mime="text/markdown",
                use_container_width=True,
            )


def _export_chat_markdown() -> str:
    """将对话历史导出为 Markdown 文本。"""
    lines = ["# RAG 问答记录\n"]
    for i, msg in enumerate(st.session_state.messages, 1):
        role_label = "**用户**" if msg["role"] == "user" else "**助手**"
        lines.append(f"## {i}. {role_label}\n")
        lines.append(msg["content"])
        lines.append("")
        if msg.get("sources"):
            lines.append("### 参考来源\n")
            for j, src in enumerate(msg["sources"], 1):
                lines.append(
                    f"{j}. (相关度: {src['score']:.4f}) {src['content'][:100]}..."
                )
            lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 引擎管理
# ---------------------------------------------------------------------------

def _get_engine():
    """惰性创建或返回 EnhancedMiniRAG 实例。"""
    from retrieval import EnhancedMiniRAG
    from minirag import RAGConfig, init

    init()

    if st.session_state.engine is None:
        strategy = st.session_state.get("strategy", "default")
        config = st.session_state.get("config", RAGConfig(top_k=5))
        mmr_lambda = st.session_state.get("mmr_lambda", 0.7)
        hybrid_alpha = st.session_state.get("hybrid_alpha", 0.3)

        st.session_state.engine = EnhancedMiniRAG(
            config,
            strategy=strategy,
            mmr_lambda=mmr_lambda,
            hybrid_alpha=hybrid_alpha,
        )

    return st.session_state.engine
