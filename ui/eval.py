"""
RAG 实验平台 — Tab: RAGAS 评估
实时评估 + 批量评估
"""
import json

import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# 公开入口
# ---------------------------------------------------------------------------

def render_eval_tab() -> None:
    """渲染 RAGAS 评估 Tab。"""
    _init_session_eval()

    tab_single, tab_batch = st.tabs(["🔍 实时评估", "📦 批量评估"])

    with tab_single:
        _render_single_eval()
    with tab_batch:
        _render_batch_eval()


# ---------------------------------------------------------------------------
# Session 初始化
# ---------------------------------------------------------------------------

def _init_session_eval() -> None:
    """确保 eval tab 需要的 session state 键存在。"""
    from minirag import RAGConfig, init

    defaults: dict = {
        "eval_result": None,
        "eval_batch_results": [],
        "config": RAGConfig(top_k=5),
        "engine": None,
        "messages": [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ---------------------------------------------------------------------------
# Tab 1: 实时评估
# ---------------------------------------------------------------------------

def _render_single_eval() -> None:
    st.subheader("🔍 单条评估")
    st.caption("输入问题、回答、参考答案和检索上下文，评估 RAG 系统的质量。")

    # ---- 自动填入按钮 ----
    col_btn, _ = st.columns([1, 3])
    with col_btn:
        if st.button("📋 自动填入最近一次问答", use_container_width=True):
            _auto_fill_last_qa()

    # ---- 输入字段 ----
    question = st.text_input(
        "问题",
        key="eval_question",
        placeholder="例如：光速科技有哪些核心产品？",
    )

    col1, col2 = st.columns(2)
    with col1:
        answer = st.text_area(
            "LLM 回答",
            height=120,
            key="eval_answer",
            placeholder="填入 LLM 生成的回答...",
        )
    with col2:
        context = st.text_area(
            "检索上下文",
            height=120,
            key="eval_context",
            placeholder="填入检索到的文档内容（多个文档用空行分隔）...",
        )

    reference = st.text_area(
        "参考答案 (Ground Truth)",
        height=80,
        key="eval_reference",
        placeholder="填入人工标注的参考答案...",
    )

    # ---- 评估按钮 ----
    all_filled = all([question, answer, context, reference])
    if st.button("🔍 评估", type="primary", disabled=not all_filled, use_container_width=True):
        with st.spinner("正在评估..."):
            try:
                result = _run_single_eval(question, answer, context, reference)
                st.session_state.eval_result = result
            except Exception as e:
                st.error(f"评估失败: {e}")
                return

    # ---- 展示结果 ----
    result = st.session_state.get("eval_result")
    if result:
        st.divider()
        st.subheader("📊 评估结果")
        _render_single_eval_result(result)


def _auto_fill_last_qa() -> None:
    """从 session_state.messages 自动填入最近一次问答。"""
    messages = st.session_state.get("messages", [])
    if not messages:
        st.warning("暂无可填入的问答记录，请先在「智能问答」中进行对话。")
        return

    last_user = None
    last_assistant = None
    for msg in reversed(messages):
        if msg["role"] == "assistant" and last_assistant is None:
            last_assistant = msg
        if msg["role"] == "user" and last_user is None:
            last_user = msg

    if last_user:
        st.session_state.eval_question = last_user["content"]
    if last_assistant:
        st.session_state.eval_answer = last_assistant.get("content", "")
        if last_assistant.get("sources"):
            st.session_state.eval_context = "\n\n".join(
                s.get("content", "") for s in last_assistant["sources"]
            )
    st.success("已自动填入最近一次问答")


def _run_single_eval(question: str, answer: str, context: str, reference: str) -> dict:
    """执行单条 RAGAS 评估。"""
    from RAGAS import RAGASEvaluator

    evaluator = RAGASEvaluator()
    return evaluator.evaluate(
        question=question,
        answer=answer,
        context=context,
        reference=reference,
    )


def _render_single_eval_result(result: dict) -> None:
    """渲染单条评估结果：4 个 metric + 进度条 + JSON 详情。"""
    cr_score = result["context_recall"].get("score", 0.0)
    faith_score = result["faithfulness"].get("score", 0.0)
    ar_score = result["answer_relevancy"].get("score", 0.0)
    overall = result["overall"]

    cols = st.columns(4)
    with cols[0]:
        st.metric("⭐ 综合", f"{overall:.2%}")
        st.progress(min(overall, 1.0))
    with cols[1]:
        st.metric("🎯 Faithfulness", f"{faith_score:.2%}")
        st.progress(min(faith_score, 1.0))
    with cols[2]:
        st.metric("📝 Answer Relevancy", f"{ar_score:.2%}")
        st.progress(min(ar_score, 1.0))
    with cols[3]:
        st.metric("🔍 Context Recall", f"{cr_score:.2%}")
        st.progress(min(cr_score, 1.0))

    with st.expander("📋 详细结果（JSON）"):
        st.json(result)


# ---------------------------------------------------------------------------
# Tab 2: 批量评估
# ---------------------------------------------------------------------------

def _render_batch_eval() -> None:
    st.subheader("📦 批量评估")
    st.caption("上传 CSV 或 JSON 测试集，批量评估 RAG 系统的质量。")

    # ---- 下载模板 ----
    csv_template = _generate_csv_template()
    st.download_button(
        label="📥 下载 CSV 模板",
        data=csv_template,
        file_name="ragas_test_template.csv",
        mime="text/csv",
    )

    # ---- 文件上传 ----
    uploaded = st.file_uploader(
        "上传测试集（CSV 或 JSON）",
        type=["csv", "json"],
        key="eval_test_upload",
    )
    if uploaded is None:
        return

    # ---- 解析文件 ----
    test_cases = _parse_test_file(uploaded)
    if test_cases is None:
        return

    st.info(f"已加载 {len(test_cases)} 条测试用例")

    # ---- 预览 ----
    st.dataframe(pd.DataFrame(test_cases), use_container_width=True)

    # ---- 批量评估按钮 ----
    if st.button("🚀 开始批量评估", type="primary", use_container_width=True):
        docs = st.session_state.get("indexed_docs", [])
        if not docs:
            st.warning("请先在「知识库」中上传文档。")
            return

        engine = _get_engine()
        with st.spinner("正在批量评估..."):
            try:
                results = _run_batch_eval(test_cases, engine)
                st.session_state.eval_batch_results = results
            except Exception as e:
                st.error(f"批量评估失败: {e}")
                return

    # ---- 展示批量结果 ----
    results = st.session_state.get("eval_batch_results", [])
    if results:
        st.divider()
        st.subheader("📊 批量评估结果")
        _render_batch_results(results)


def _generate_csv_template() -> str:
    """生成包含示例行的 CSV 模板。"""
    return (
        "question,reference\n"
        '"光速科技有哪些产品？","光速科技的主要产品包括天机智能营销平台（AI驱动的精准营销）、'
        '星河数据中台（企业数据整合和分析）和星盾安全网关（网络安全防护）。"\n'
        '"光速科技的AI平台有什么功能？","天机智能营销平台具备用户画像分析、智能推荐引擎、'
        '营销自动化等功能，帮助企业提升营销转化率20%以上。"\n'
    )


def _parse_test_file(uploaded) -> list[dict] | None:
    """解析上传的 CSV 或 JSON 测试文件。"""
    try:
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
            required_cols = {"question", "reference"}
            if not required_cols.issubset(df.columns):
                st.error(
                    f"CSV 文件缺少必要列：{required_cols}，当前列：{list(df.columns)}"
                )
                return None
            return df[["question", "reference"]].to_dict("records")
        elif uploaded.name.endswith(".json"):
            data = json.load(uploaded)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and "test_cases" in data:
                return data["test_cases"]
            else:
                st.error("JSON 文件格式不正确：需要数组或包含 test_cases 键的对象。")
                return None
        else:
            st.error("不支持的文件格式，请上传 CSV 或 JSON。")
            return None
    except Exception as e:
        st.error(f"文件解析失败: {e}")
        return None


def _run_batch_eval(test_cases: list[dict], engine) -> list[dict]:
    """执行批量 RAGAS 评估。

    对每条 test_case 调用 engine.ask() → 构建 context → evaluator.evaluate()。
    """
    from RAGAS import RAGASEvaluator

    evaluator = RAGASEvaluator()
    results = []
    progress = st.progress(0, text="准备评估...")
    total = len(test_cases)

    for i, tc in enumerate(test_cases):
        question = tc["question"]
        reference = tc.get("reference", "")
        progress.progress(
            (i + 1) / total,
            text=f"正在评估 {i + 1}/{total}: {question[:30]}...",
        )

        resp = engine.ask(question)
        context = "\n\n".join(
            f"[{s.metadata.get('category', '?')}] {s.content}"
            for s in resp.sources
        )

        eval_result = evaluator.evaluate(
            question=question,
            answer=resp.answer,
            context=context,
            reference=reference,
        )

        results.append({
            "question": question,
            "answer": resp.answer[:200],
            "context_recall": eval_result["context_recall"].get("score", 0.0),
            "faithfulness": eval_result["faithfulness"].get("score", 0.0),
            "answer_relevancy": eval_result["answer_relevancy"].get("score", 0.0),
            "overall": eval_result["overall"],
            "full_result": eval_result,
        })

    progress.empty()
    return results


def _render_batch_results(results: list[dict]) -> None:
    """渲染批量评估结果：汇总指标、雷达图、详细表格、CSV 导出。"""
    n = len(results)
    avg_cr = sum(r["context_recall"] for r in results) / n
    avg_faith = sum(r["faithfulness"] for r in results) / n
    avg_ar = sum(r["answer_relevancy"] for r in results) / n
    avg_overall = sum(r["overall"] for r in results) / n

    # ---- 汇总指标 ----
    st.subheader("📈 平均指标")
    cols = st.columns(4)
    with cols[0]:
        st.metric("⭐ 综合", f"{avg_overall:.2%}")
    with cols[1]:
        st.metric("🔍 Context Recall", f"{avg_cr:.2%}")
    with cols[2]:
        st.metric("🎯 Faithfulness", f"{avg_faith:.2%}")
    with cols[3]:
        st.metric("📝 Answer Relevancy", f"{avg_ar:.2%}")

    # ---- 雷达图 ----
    st.subheader("📊 雷达图")
    try:
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=[avg_cr, avg_faith, avg_ar],
            theta=["Context Recall", "Faithfulness", "Answer Relevancy"],
            fill="toself",
            name="平均分数",
            line_color="#FF6B6B",
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            height=400,
            showlegend=True,
        )
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.info("💡 安装 plotly 后可查看雷达图：`pip install plotly`")

    # ---- 详细结果表 ----
    with st.expander("📋 详细评估结果"):
        df = pd.DataFrame([{
            "问题": r["question"],
            "综合": f'{r["overall"]:.2%}',
            "Context Recall": f'{r["context_recall"]:.2%}',
            "Faithfulness": f'{r["faithfulness"]:.2%}',
            "Answer Relevancy": f'{r["answer_relevancy"]:.2%}',
        } for r in results])
        st.dataframe(df, use_container_width=True)

    # ---- CSV 导出 ----
    csv_data = pd.DataFrame([{
        "question": r["question"],
        "answer": r["answer"],
        "context_recall": r["context_recall"],
        "faithfulness": r["faithfulness"],
        "answer_relevancy": r["answer_relevancy"],
        "overall": r["overall"],
    } for r in results])

    st.download_button(
        label="📥 导出结果为 CSV",
        data=csv_data.to_csv(index=False).encode("utf-8"),
        file_name="ragas_batch_eval_results.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------------
# 引擎管理
# ---------------------------------------------------------------------------

def _get_engine():
    """惰性创建或返回 EnhancedMiniRAG 实例。"""
    from minirag import RAGConfig, init
    from retrieval import EnhancedMiniRAG

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
