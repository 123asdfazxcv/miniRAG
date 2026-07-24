"""RAG 实验平台 — 复用 UI 组件"""
import streamlit as st


def score_color(score: float) -> str:
    """分数→CSS颜色: 绿(>0.5) / 橙(>0.3) / 红(<=0.3)"""
    if score > 0.5:
        return "#4CAF50"
    elif score > 0.3:
        return "#FF9800"
    else:
        return "#f44336"


def render_source_card(source: dict) -> None:
    """渲染单条来源文档卡片。source需含: content, score, doc_id"""
    score = source.get("score", 0.0)
    color = score_color(score)
    doc_id = source.get("doc_id", "?")
    st.markdown(f"""
    <div style="
        background: #f0f2f6;
        border-left: 4px solid {color};
        border-radius: 6px;
        padding: 10px 14px;
        margin: 6px 0;
        font-size: 0.9em;
    ">
        <span style="background:{color};color:white;
        padding:2px 8px;border-radius:4px;font-size:0.78em;">
        相关度: {score:.4f}</span>
        <span style="color:#888;font-size:0.78em;margin-left:8px;">
        Chunk #{doc_id}</span>
        <p style="margin-top:6px;line-height:1.5;">{source['content']}</p>
    </div>
    """, unsafe_allow_html=True)


def render_token_stats(token_usage: dict, elapsed_ms: float) -> None:
    """渲染 Token 用量 + 耗时统计。"""
    input_t = token_usage.get("input_tokens", token_usage.get("input", 0))
    output_t = token_usage.get("output_tokens", token_usage.get("output", 0))
    total_t = token_usage.get("total_tokens", token_usage.get("total", 0))
    cols = st.columns(3)
    cols[0].metric("📥 Input Tokens", input_t)
    cols[1].metric("📤 Output Tokens", output_t)
    cols[2].metric("⏱️ 耗时", f"{elapsed_ms:.0f}ms")
