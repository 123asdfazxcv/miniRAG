"""
RAG 实验平台 — Tab 5: 设置
API Key 管理、模型选择、默认参数配置
"""
import os
import streamlit as st
from pathlib import Path
from minirag import RAGConfig


def render_settings_tab() -> None:
    """渲染设置 Tab。"""
    st.header("⚙️ 设置")

    tab_api, tab_model, tab_defaults = st.tabs(["🔑 API 配置", "🧠 模型选择", "📐 默认参数"])

    # ---- API 配置 ----
    with tab_api:
        st.subheader("DashScope API Key")
        current_key = os.environ.get("DASHSCOPE_API_KEY", "")
        new_key = st.text_input(
            "API Key",
            value=current_key[:8] + "****" if current_key else "",
            type="password",
            placeholder="sk-...",
            help="获取地址: https://dashscope.console.aliyun.com/apiKey",
        )
        if new_key and new_key != current_key and "****" not in new_key:
            os.environ["DASHSCOPE_API_KEY"] = new_key
            import dashscope
            dashscope.api_key = new_key
            env_path = Path(__file__).resolve().parent.parent / ".env"
            env_path.write_text(f"DASHSCOPE_API_KEY={new_key}\n", encoding="utf-8")
            st.success("✅ API Key 已更新")
            st.rerun()
        if current_key:
            st.success(f"✅ 已配置 ({current_key[:8]}****)")
        else:
            st.error("❌ 未配置 API Key")

    # ---- 模型选择 ----
    with tab_model:
        st.subheader("Embedding 模型")
        embed_model = st.selectbox(
            "选择 Embedding 模型",
            options=["text-embedding-v1", "text-embedding-v2", "text-embedding-v3"],
            index=1,
            key="settings_embed_model",
        )
        st.subheader("LLM 模型")
        llm_model = st.selectbox(
            "选择 LLM 模型",
            options=["qwen-turbo", "qwen-plus", "qwen-max"],
            index=0,
            key="settings_llm_model",
        )
        temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05, key="settings_temperature")
        max_tokens = st.slider("Max Tokens", 256, 4096, 1024, 128, key="settings_max_tokens")
        if st.button("✅ 应用模型设置", use_container_width=True):
            st.session_state.config.embed_model = embed_model
            st.session_state.config.llm_model = llm_model
            st.session_state.config.llm_temperature = temperature
            st.session_state.config.llm_max_tokens = max_tokens
            st.session_state.engine = None
            st.success("模型设置已应用，下次问答生效")
            st.rerun()

    # ---- 默认参数 ----
    with tab_defaults:
        st.subheader("检索默认参数")
        default_top_k = st.slider("默认 Top-K", 1, 10, 5, key="settings_default_top_k")
        default_threshold = st.slider("默认相似度阈值", 0.0, 1.0, 0.5, 0.05, key="settings_default_threshold")
        default_strategy = st.selectbox(
            "默认检索策略",
            options=["default", "mmr", "hyde", "hybrid"],
            index=0,
            key="settings_default_strategy",
            format_func=lambda x: {
                "default": "Default (向量)", "mmr": "MMR (多样性)",
                "hyde": "HyDE (假设文档)", "hybrid": "Hybrid (混合)"
            }[x],
        )
        if st.button("✅ 应用默认参数", use_container_width=True):
            st.session_state.config.top_k = default_top_k
            st.session_state.config.similarity_threshold = default_threshold
            st.session_state.strategy = default_strategy
            st.session_state.engine = None
            st.success("默认参数已更新")
            st.rerun()
