"""
RAG 实验平台 -- Tab: 策略实验室
同时对比多种检索策略在同一问题上的表现
"""
import time
import streamlit as st
from ui.components import render_source_card, render_token_stats, score_color
from minirag import RAGConfig, init, Retriever, MiniRAG
from retrieval import MMRRetriever, HyDERetriever, BM25Retriever, HybridRetriever
from embedding import Embedder

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
# Session 初始化
# ---------------------------------------------------------------------------

def _init_session() -> None:
    """确保必要的 session state 键存在。"""
    if "lab_history" not in st.session_state:
        st.session_state.lab_history = []


# ---------------------------------------------------------------------------
# 基类 Retriever 获取
# ---------------------------------------------------------------------------

def _get_base_retriever() -> Retriever:
    """从当前 engine 的 retriever 中解包出最底层的 Retriever 实例。"""
    retriever = st.session_state.engine.retriever
    if isinstance(retriever, MMRRetriever):
        return retriever._base
    elif isinstance(retriever, HyDERetriever):
        return retriever._base
    elif isinstance(retriever, HybridRetriever):
        return retriever._vec
    return retriever


# ---------------------------------------------------------------------------
# 核心对比逻辑
# ---------------------------------------------------------------------------

def _run_comparison(query: str, strategies: list[str], top_k: int) -> list[dict]:
    """对每种策略独立检索+生成，返回结果列表。

    复用当前 engine 已索引的 chunks / FAISS index，避免重复索引。
    """
    init()
    base = _get_base_retriever()
    config = RAGConfig(top_k=top_k)

    results = []
    for strategy in strategies:
        t0 = time.perf_counter()

        # 1. 创建新的基础 Retriever 并复制索引数据
        embedder = Embedder(model=base.embedder.model)
        new_retriever = Retriever(embedder, config)
        new_retriever._chunks = list(base._chunks)
        new_retriever._chunk_meta = list(base._chunk_meta)
        new_retriever._index = base._index          # FAISS index 只读共享
        new_retriever._indexed_count = base._indexed_count

        # 2. 按策略包装 retriever
        if strategy == "mmr":
            final_retriever = MMRRetriever(new_retriever, lambda_param=0.7)
        elif strategy == "hyde":
            final_retriever = HyDERetriever(new_retriever, config)
        elif strategy == "hybrid":
            bm25_ret = BM25Retriever()
            bm25_ret._chunks = list(base._chunks)
            bm25_ret._chunk_meta = list(base._chunk_meta)
            bm25_ret._bm25.build_index(base._chunks)
            final_retriever = HybridRetriever(new_retriever, bm25_ret, alpha=0.3)
        else:
            final_retriever = new_retriever

        # 3. 创建 MiniRAG 并替换 retriever
        engine = MiniRAG(config)
        engine.retriever = final_retriever

        # 4. 查询
        response = engine.ask(query)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        results.append({
            "strategy": strategy,
            "answer": response.answer,
            "sources": [
                {"content": s.content, "score": s.score, "doc_id": s.doc_id}
                for s in response.sources
            ],
            "token_usage": response.token_usage,
            "elapsed_ms": elapsed_ms,
        })

    return results


# ---------------------------------------------------------------------------
# 重合分析
# ---------------------------------------------------------------------------

def _compute_overlap(results: list[dict]) -> list[dict]:
    """对每对策略计算 source doc_id 的 Jaccard 相似度。"""
    pairs = []
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            a_ids = {s["doc_id"] for s in results[i]["sources"]}
            b_ids = {s["doc_id"] for s in results[j]["sources"]}
            intersection = len(a_ids & b_ids)
            union = len(a_ids | b_ids)
            jaccard = intersection / union if union > 0 else 0.0
            pairs.append({
                "a": results[i]["strategy"],
                "b": results[j]["strategy"],
                "jaccard": jaccard,
                "intersection": intersection,
                "union": union,
            })
    return pairs


# ---------------------------------------------------------------------------
# 公开入口
# ---------------------------------------------------------------------------

def render_lab_tab() -> None:
    """渲染策略实验室 Tab。"""
    _init_session()

    st.header("🔬 策略实验室")
    st.caption("同一个问题，不同策略怎么检索？并排对比一目了然。")

    # ---- 配置区 ----
    col1, col2 = st.columns(2)
    with col1:
        strategies = st.multiselect(
            "选择检索策略",
            options=STRATEGY_OPTIONS,
            default=STRATEGY_OPTIONS,
            format_func=lambda x: STRATEGY_LABELS[x],
            key="lab_strategies",
        )
    with col2:
        top_k = st.slider("Top-K", 1, 10, 5, key="lab_top_k")

    # ---- 输入区 ----
    query = st.text_input(
        "输入查询问题",
        key="lab_query",
        placeholder="输入你想对比的问题...",
    )

    engine_ready = (
        st.session_state.get("engine") is not None
        and st.session_state.engine.retriever.is_indexed
    )
    compare_clicked = st.button(
        "🔍 开始对比",
        type="primary",
        disabled=len(strategies) < 2,
        key="lab_compare_btn",
    )

    if compare_clicked:
        if not query.strip():
            st.warning("请输入查询问题")
        elif not engine_ready:
            st.warning("请先在「文档管理」中索引文档")
        else:
            with st.spinner(f"正在对比 {len(strategies)} 种策略..."):
                results = _run_comparison(query, strategies, top_k)
            st.session_state.lab_history.insert(0, {
                "query": query,
                "strategies": strategies,
                "top_k": top_k,
                "results": results,
            })
            st.session_state.lab_history = st.session_state.lab_history[:5]
            st.rerun()

    # ---- 展示最新对比结果 ----
    if not st.session_state.lab_history:
        st.info('请选择查询问题，点击“开始对比”查看不同策略的表现')
        return

    latest = st.session_state.lab_history[0]
    results = latest["results"]
    n = len(results)

    st.divider()
    st.subheader("📊 对比结果")

    # ---- 并排卡片 ----
    columns = st.columns(n)
    for idx, (col, r) in enumerate(zip(columns, results)):
        with col:
            label = STRATEGY_LABELS.get(r["strategy"], r["strategy"])
            st.markdown(f"**{label}**")
            st.caption(f"⏱️ {r['elapsed_ms']:.0f}ms")

            if r["sources"]:
                with st.expander(f"📚 来源 ({len(r['sources'])} 条)", expanded=False):
                    for src in r["sources"]:
                        render_source_card(src)

            st.markdown("##### 回答")
            st.markdown(r["answer"])

    # ---- 回答对比 ----
    st.divider()
    st.subheader("📝 回答对比")
    for r in results:
        label = STRATEGY_LABELS.get(r["strategy"], r["strategy"])
        with st.expander(f"{label} — {r['elapsed_ms']:.0f}ms"):
            st.markdown(r["answer"])
            render_token_stats(r["token_usage"], r["elapsed_ms"])

    # ---- 重合分析 ----
    st.divider()
    st.subheader("🔗 来源重合分析 (Jaccard)")

    overlaps = _compute_overlap(results)
    for ov in overlaps:
        a_label = STRATEGY_LABELS.get(ov["a"], ov["a"])
        b_label = STRATEGY_LABELS.get(ov["b"], ov["b"])
        j = ov["jaccard"]

        if j > 0.5:
            color = "#f44336"   # 红: 高度重合
            tag = "高重合"
        elif j > 0.2:
            color = "#FF9800"   # 橙: 中等重合
            tag = "中重合"
        else:
            color = "#4CAF50"   # 绿: 低重合（策略多样化）
            tag = "低重合"

        st.markdown(
            f'<span style="color:{color};font-weight:bold;">{tag}</span> '
            f"<code>{a_label}</code> vs <code>{b_label}</code>: "
            f"Jaccard = {j:.3f} "
            f"({ov['intersection']}/{ov['union']} 条重合)",
            unsafe_allow_html=True,
        )

    # ---- 历史记录 ----
    if len(st.session_state.lab_history) > 1:
        st.divider()
        st.subheader("📋 历史对比")
        for idx, entry in enumerate(st.session_state.lab_history[1:], 1):
            with st.expander(
                f"#{idx} 「{entry['query'][:40]}...」 "
                f"({', '.join(STRATEGY_LABELS.get(s, s) for s in entry['strategies'])})"
            ):
                hist_cols = st.columns(len(entry["results"]))
                for col, r in zip(hist_cols, entry["results"]):
                    with col:
                        label = STRATEGY_LABELS.get(r["strategy"], r["strategy"])
                        st.markdown(f"**{label}** ({r['elapsed_ms']:.0f}ms)")
                        st.markdown(r["answer"][:200] + ("..." if len(r["answer"]) > 200 else ""))
