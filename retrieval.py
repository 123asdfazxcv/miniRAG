import os
import logging
import math
import numpy as np
import sys
import faiss
import jieba
from dashscope import Generation
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional
from dataclasses import dataclass,field



_PROJECT_DIR = Path(__file__).resolve().parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.append(str(_PROJECT_DIR))
from minirag import (
RAGConfig,RAGError,RetrievalError,EmbeddingError,GenerationError,
Retriever,Document,RetrievedChunk,RAGResponse,Generator,MiniRAG,init,_setup_logger,TEST_DOCUMENTS
)
from embedding import Embedder
logger = logging.getLogger("MiniRAG.Day5")
def mmr_rerank(
        candidates:list[RetrievedChunk],
        query_vec:np.ndarray,
        lambda_param:float = 0.7,
        top_k:int = 3
)->list[RetrievedChunk]:
    if top_k >= len(candidates):
        return candidates
    texts = [c.content for c in candidates]
    scores = np.array([c.score for c in candidates])
    def _text_similarity(text_a:str,text_b:str)->float:
        set_a = set(text_a)
        set_b = set(text_b)
        if not set_b or not set_a:
            return 0.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union
    selected_indices:list[int] = []
    first_idx = int(np.argmax(scores))
    selected_indices.append(first_idx)
    for _ in range(1,min(top_k,len(candidates))):
        best_idx = -1
        best_mmr = -float('inf')
        for i in range(len(candidates)):
            if i in selected_indices:
                continue
            relevance = float(scores[i])
            max_sim_to_selected = max(
                _text_similarity(candidates[i].content,candidates[j].content)
                for j in selected_indices
            )
            mmr_score = lambda_param*relevance - (1- lambda_param)*max_sim_to_selected
            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_idx = i
        if best_idx >= 0 :
            selected_indices.append(best_idx)
    return [candidates[i] for i in selected_indices]
class MMRRetriever:
    def __init__(self,base_retriever:Retriever,lambda_param:float = 0.7,fetch_factor:int = 3):
        self._base = base_retriever
        self.lambda_param = lambda_param
        self.fetch_factor = fetch_factor
        self.embedder = base_retriever.embedder
    @property
    def is_indexed(self):
        return self._base.is_indexed
    def index(self,documents:list[Document]):
        return self._base.index(documents)
    def retrieve(self,query:str,top_k:Optional[int]=None)->list[RetrievedChunk]:
        k = top_k or self._base.config.top_k
        fetch_k = min(k * self.fetch_factor,self._base.chunk)
        candidates = self._base.retrieve(query,top_k=fetch_k)
        if len(candidates) <= k:
            return candidates
        try:
            q_vec = self.embedder.embed(query, text_type="query")
            q_vec = np.array(q_vec, dtype=np.float32)
        except Exception:
            return candidates[:k]
        return mmr_rerank(candidates,q_vec,lambda_param=self.lambda_param,top_k=k)
HYDE_PROMPT = """你是一个知识助手。请根据下面的问题，写一段简短的回答（2-4句话）。

问题：{question}

请用中文回答，只输出回答内容，不要加"回答："之类的前缀。"""
class HyDEGenerator:
    def __init__(self,config:RAGConfig):
        self.config = config
        self._cache:dict[str,str] = {}
    def generate_hypothetical_doc(self,question:str)->str:
        if question in self._cache:
            logger.debug(f"HyDE 缓存命中: '{question[:30]}...'")
            return self._cache[question]
        prompt = HYDE_PROMPT.format(question=question)
        try:
            resp = Generation.call(
                model=self.config.llm_model,
                messages=[{"role":"user","content":prompt}],
                temperature=0.3,
                max_tokens=200,
            )
            if resp.status_code == 200 and resp.output and resp.output.text:
                hypothetical = resp.output.text.strip()
                logger.info(
                    f"HyDE 生成: '{question[:30]}...' → "
                    f"假设答案 {len(hypothetical)} chars"
                )
                self._cache[question] = hypothetical
                return hypothetical
            else:
                logger.warning(f"HyDE 生成失败 (status={resp.status_code})，退回原 query")
                return question
        except Exception as e:
            logger.warning(f"HyDE 生成异常: {e}，退回原 query")
            return question
class HyDERetriever:
    def __init__(self,base_retriever:Retriever,config:RAGConfig):
        self._base = base_retriever
        self._hyde = HyDEGenerator(config)
        self.embedder = base_retriever.embedder
    @property
    def is_indexed(self):
        return self._base.is_indexed
    def index(self,documents:list[Document]):
        return self._base.index(documents)
    def retrieve(self,query:str,top_k:Optional[int]=None)->list[RetrievedChunk]:
        hypothetical = self._hyde.generate_hypothetical_doc(query)
        if hypothetical == query:
            return self._base.retrieve(query,top_k=top_k)
        k = top_k or self._base.config.top_k
        try:
            hyde_vec = self.embedder.embed(hypothetical,text_type="document")
        except Exception:
            logger.warning("HyDE embedding 失败，退回原 query")
            return self._base.retrieve(query,top_k=top_k)
        hyde_vec = np.array([hyde_vec], dtype=np.float32)
        faiss.normalize_L2(hyde_vec)
        scores,indices = self._base._index.search(hyde_vec,k=k)
        results = []
        for score,idx in zip(scores[0],indices[0]):
            if idx == -1:
                continue
            if float(score) < self._base.config.similarity_threshold:
                continue
            meta = self._base._chunk_meta[idx]
            results.append(RetrievedChunk(
                content=self._base._chunks[idx],
                score = score,
                doc_id=str(idx),
                chunk_index=int(idx),
                metadata=meta.get("doc_metadata",{})
            ))
        logger.debug(f"HyDE 检索 '{query[:30]}...' → {len(results)}/{k} 条结果")
        return results
class BM25Scorer:
    def __init__(self,k1:float = 1.5,b:float=0.75):
        self.k1 = k1
        self.b=b
        self._docs:list[list[str]]=[]
        self._doc_lens:list[int]=[]
        self._N:int = 0
        self._avgdl:float=0.0
        self._inverted_index:dict[str,dict[str,int]]=defaultdict(dict)
        self._df:dict[str,int]=defaultdict(int)
        self._built:bool=False
    @staticmethod
    def _tokenize(text:str)->list[str]:
        words = jieba.lcut(text)
        return [w.strip() for w in words if len(w.strip())>=2]
    def build_index(self,documents:list[str])->None:
        self._docs=[]
        self._doc_lens=[]
        self._inverted_index.clear()
        self._df.clear()
        self._N = 0
        for doc_id,text in enumerate(documents):
            tokens = self._tokenize(text)
            self._docs.append(tokens)
            self._doc_lens.append(len(tokens))
            self._N += 1
            tf_in_doc:dict[str,int]=defaultdict(int)
            for token in tokens:
                tf_in_doc[token] += 1
            for token,freq in tf_in_doc.items():
                self._inverted_index[token][doc_id]=freq
                self._df[token] +=1
        self._avgdl=sum(self._doc_lens)/max(self._N,1)
        self._built = True
        logger.info(
            f"BM25 索引构建完成: {self._N} 篇文档, "
            f"{len(self._inverted_index)} 个词, "
            f"平均长度 {self._avgdl:.1f} 词/篇"
        )
    def _idf(self,term:str)->float:
        df = self._df.get(term,0)
        if df == 0:
            return 0.0
        return math.log((self._N-df+0.5)/(df + 0.5)+1)
    def _score_one(self,query_tokens:list[str],doc_id:int)->float:
        doc_len = self._doc_lens[doc_id]
        score = 0.0
        for term in query_tokens:
            tf=self._inverted_index.get(term,{}).get(doc_id,0)
            if tf == 0:
                continue
            idf = self._idf(term)
            numerator = tf*(self.k1+1)
            denominator =tf+self.k1*(1-self.b+(self.b*doc_len)/self._avgdl)
            score += idf*numerator/denominator
        return score
    def search(self,query:str,top_k:int=3)->list[tuple[int,float]]:
        query_tokens=self._tokenize(query)
        if not self._built:
            raise RuntimeError("索引未构建，请先调用 build_index()")
        if not query_tokens:
            return []
        doc_scores = []
        for doc_id in range(self._N):
            score = self._score_one(query_tokens,doc_id)
            if score>0:
                doc_scores.append((doc_id,score))
        doc_scores.sort(key=lambda x :x[1],reverse = True)
        return doc_scores[:top_k]
class BM25Retriever:
    def __init__(self):
        self._bm25=BM25Scorer()
        self._chunks:list[str]=[]
        self._chunk_meta:list[dict]=[]
    @property
    def is_indexed(self)->bool:
        return self._bm25._built
    def index(self,documents:list[Document])->int:
        self._chunks=[]
        self._chunk_meta=[]
        for doc_idx,doc in enumerate(documents):
            if len(doc.content) <= 300:
                chunks_for_doc = [doc.content]
            else:
                import re
                sentences = re.split((r'(?<=[。？！\n])'),doc.content)
                chunks_for_doc = []
                current=""
                for sentence in sentences:
                    if len(sentence) + len(current) > 300:
                        chunks_for_doc.append(current.strip())
                        current = sentence
                    else:
                        current += sentence
                if current.strip():
                    chunks_for_doc.append(current.strip())
            for chunk in chunks_for_doc:
                if chunk.strip():
                    self._chunks.append(chunk)
                    self._chunk_meta.append({
                        "doc_index":doc_idx,
                        "doc_metadata":doc.metadata
                    })
        self._bm25.build_index(self._chunks)
        return len(self._chunks)
    def retrieve(self,query:str,top_k:int=3)->list[RetrievedChunk]:
        results = self._bm25.search(query,top_k)
        return [
            RetrievedChunk(
                content=self._chunks[doc_id],
                score=float(score),
                doc_id=str(doc_id),
                chunk_index=doc_id,
                metadata=self._chunk_meta[doc_id].get("doc_metadata",{})
            )
            for doc_id,score in results
        ]
def _minmax_normalize(scores:list[float])->list[float]:
    if not scores:
        return []
    mn,mx = min(scores),max(scores)
    if mx == mn:
        return [0.5]*len(scores)
    return [(s-mn)/(mx-mn) for s in scores]
class HybridRetriever:
    def __init__(self,vector_retriever:Retriever,bm25_retriever:BM25Retriever,alpha:float=0.3):
        self._vec = vector_retriever
        self._bm25 = bm25_retriever
        self.alpha = alpha
    @property
    def is_indexed(self):
        return self._vec.is_indexed and self._bm25.is_indexed
    def index(self,documents:list[Document])->int:
        n_vec = self._vec.index(documents)
        n_bm25 =self._bm25.index(documents)
        logger.info(f"混合索引: 向量 {n_vec} chunks, BM25 {n_bm25} chunks")
        return max(n_bm25,n_vec)
    def retrieve(self,query:str,top_k:Optional[int]=None)->list[RetrievedChunk]:
        k = top_k or self._vec.config.top_k
        fetch_k = min(k*3,max(self._vec.chunk,1))
        vec_results = self._vec.retrieve(query,top_k=fetch_k)
        bm25_results = self._bm25.retrieve(query,top_k=fetch_k)
        vec_scores:dict[str,float] = {}
        for r in vec_results:
            vec_scores[str(r.chunk_index)] = r.score
        bm25_scores:dict[str,float]={}
        bm25_max = max([s.score for s in bm25_results],default=1.0)
        for r in bm25_results:
            bm25_scores[str(r.chunk_index)] = r.score/max(bm25_max,1.0)
        all_chunk_ids = set(vec_scores.keys())|set(bm25_scores.keys())
        candidates:list[tuple[int,float,float]]=[]
        for cid_str in all_chunk_ids:
            cid = int(cid_str)
            if cid < len(self._vec._chunks):
                vs = vec_scores.get(cid_str,0)
                bs = bm25_scores.get(cid_str,0)
                candidates.append((cid,vs,bs))
        if not candidates:
            return []
        vec_raw = [c[1] for c in candidates]
        bm25_raw = [c[2] for c in candidates]
        vec_norm = _minmax_normalize(vec_raw)
        bm25_norm = _minmax_normalize(bm25_raw)
        fused:list[tuple[int,float]] = []
        for doc_id,(cid,_,_) in enumerate(candidates):
            final_score = self.alpha*bm25_norm[doc_id] + (1-self.alpha)*vec_norm[doc_id]
            fused.append((cid,final_score))
        fused.sort(key=lambda x:x[1],reverse=True)
        top = fused[:k]
        results = []
        for cid,score in top:
            meta=self._vec._chunk_meta[cid]
            results.append(RetrievedChunk(
                content=self._vec._chunks[cid],
                score =score,
                doc_id=str(cid),
                chunk_index=cid,
                metadata=meta.get("doc_metadata",{})
            ))
        logger.debug(
            f"混合检索 '{query[:30]}...' → "
            f"{len(vec_results)}向量 + {len(bm25_results)}BM25 → {len(results)}融合结果"
        )
        return results
class EnhancedMiniRAG(MiniRAG):
    STRATEGIES = ("default","mmr","hyde","hybrid","rerank","full")
    def __init__(self,config:Optional[RAGConfig] = None,strategy:str = "default",
                 mmr_lambda:float=0.7,hybrid_alpha:float=0.3):
        init()
        self.config = config or RAGConfig()
        _setup_logger(self.config.log_level)
        if strategy not in self.STRATEGIES:
            raise ValueError(f"未知策略 '{strategy}'，可选: {self.STRATEGIES}")
        self.strategy = strategy
        self.embedder = Embedder(model=self.config.embed_model)
        base_retriever=Retriever(self.embedder,self.config)
        if strategy == "mmr":
            self.retriever = MMRRetriever(base_retriever,lambda_param=mmr_lambda)
            logger.info(f"检索策略: MMR (λ={mmr_lambda})")
        elif strategy == "hyde":
            self.retriever = HyDERetriever(base_retriever,self.config)
            logger.info("检索策略: HyDE")
        elif strategy == "hybrid":
            bm25_ret = BM25Retriever()
            self._bm25_retriever = bm25_ret
            self.retriever = HybridRetriever(base_retriever,bm25_ret,alpha=hybrid_alpha)
            logger.info(f"检索策略: Hybrid (α={hybrid_alpha})")
        elif strategy in ("rerank","full"):
            self.retriever = base_retriever
            logger.info(f"检索策略: {strategy} (由子类 Day6MiniRAG 处理)")
        else:
            self.retriever = base_retriever
            logger.info("检索策略: Default (向量检索)")
        self.generator = Generator(self.config)
        logger.info(
            f"EnhancedMiniRAG 初始化完成 "
            f"(embed={self.config.embed_model}, llm={self.config.llm_model}, "
            f"strategy={self.strategy})"
        )
    def ingest(self,documents:list[Document])->int:
        return self.retriever.index(documents)
COMPARISON_QUERIES = [
    # ---- MMR 场景：查询涵盖面广，期望多样化结果 ----
    {
        "query": "猫的品种有哪些？",
        "goal": "MMR: 返回不同品种（布偶/暹罗/橘猫），不是全布偶",
        "test_for": "mmr",
    },
    {
        "query": "汽车的分类",
        "goal": "MMR: 返回不同分类角度（SUV/新能源/手动自动）",
        "test_for": "mmr",
    },
    # ---- HyDE 场景：短查询需要扩展 ----
    {
        "query": "怎么养猫？",
        "goal": "HyDE: 短查询扩展后能召回猫砂盆/猫粮/健康等具体文档",
        "test_for": "hyde",
    },
    {
        "query": "买车要注意什么？",
        "goal": "HyDE: 扩展后能覆盖保险/购置税/车型选择",
        "test_for": "hyde",
    },
    # ---- BM25/Hybrid 场景：精确关键词匹配 ----
    {
        "query": "特斯拉",
        "goal": "BM25: 精确命中含'特斯拉'的文档",
        "test_for": "bm25",
    },
    {
        "query": "布偶猫 暹罗猫",
        "goal": "Hybrid: 双路融合应同时覆盖两个品种",
        "test_for": "hybrid",
    },
    # ---- 综合测试 ----
    {
        "query": "发动机保养 电动车 猫",
        "goal": "混合检索: 应主要召回发动机保养，兼顾电动车",
        "test_for": "hybrid",
    },
]


def print_results(label: str, sources: list[RetrievedChunk], max_display: int = 5):
    """格式化打印检索结果。"""
    print(f"\n  📄 {label} (共 {len(sources)} 条):")
    if not sources:
        print("    ⚠️  无结果")
        return
    for j, src in enumerate(sources[:max_display], 1):
        cat = src.metadata.get("category", "?")
        print(f"    [{j}] [{cat}] score={src.score:.3f} | {src.content[:70]}...")


def run_experiments():
    """Day 5 核心实验：对比四种检索策略。

    实验设计：
      1. Default  — 基线：纯向量检索
      2. MMR      — 多样性重排序
      3. HyDE     — 假设文档增强
      4. Hybrid   — 向量 + BM25 双路融合

    对每个查询，用 4 种策略各检索一次，并排对比结果。
    """
    print("=" * 70)
    print("Day 5: 检索质量优化 — 四种策略对比实验")
    print("=" * 70)

    config = RAGConfig(top_k=3, log_level=logging.WARNING)

    # ---- 初始化 4 个 RAG 实例（每个用不同检索策略）----
    print("\n🔧 初始化 4 种检索策略...")
    rag_default = EnhancedMiniRAG(config, strategy="default")
    rag_mmr = EnhancedMiniRAG(config, strategy="mmr", mmr_lambda=0.7)
    rag_hyde = EnhancedMiniRAG(config, strategy="hyde")
    rag_hybrid = EnhancedMiniRAG(config, strategy="hybrid", hybrid_alpha=0.3)

    rags = {
        "Default (向量)": rag_default,
        "MMR (多样性)": rag_mmr,
        "HyDE (假设文档)": rag_hyde,
        "Hybrid (混合)": rag_hybrid,
    }

    # ---- 索引（4 个实例各索引一次）----
    print(f"\n📦 索引 {len(TEST_DOCUMENTS)} 篇文档...")
    for name, rag in rags.items():
        n = rag.ingest(TEST_DOCUMENTS)
        print(f"  {name}: {n} chunks")

    # ---- 逐个查询对比 ----
    for q_idx, q_info in enumerate(COMPARISON_QUERIES, 1):
        query = q_info["query"]
        goal = q_info["goal"]
        test_for = q_info["test_for"]

        print(f"\n{'=' * 70}")
        print(f"🔍 查询 {q_idx}: \"{query}\"")
        print(f"   目标: {goal}")
        print(f"   主要验证: {test_for.upper()}")
        print(f"{'=' * 70}")

        for name, rag in rags.items():
            resp = rag.ask(query)
            indicator = "⭐" if test_for in name.lower() else "  "
            print(f"\n{indicator} [{name}]")
            if resp.sources:
                for j, src in enumerate(resp.sources, 1):
                    cat = src.metadata.get("category", "?")
                    print(f"    [{j}] [{cat}] score={src.score:.3f} | {src.content[:80]}...")
            else:
                print("    ⚠️  无结果")
            # 显示检索耗时和 token
            print(f"    ⏱️  {resp.elapsed_ms:.0f}ms | tokens={resp.token_usage['total']}")

    # ---- 总结 ----
    print("\n" + "=" * 70)
    print("✅ Day 5 完成！你掌握了三种检索质量优化技术。")
    print()
    print("【你新增的能力】")
    print("  - MMR:  当检索结果太重复时，用 MMR 重排序增加多样性")
    print("  - HyDE: 短查询先用 LLM 扩展，桥接 query-document 语义鸿沟")
    print("  - BM25: 精确关键词匹配，弥补向量检索'近义不近词'的盲区")
    print("  - Hybrid: 向量 + BM25 双路融合，取长补短")
    print()
    print("【面试能答的问题】")
    print("  Q: 纯向量检索有什么局限？")
    print("  A: 1) Top-K 可能语义重复（MMR 解决）")
    print("     2) 短 query 和长文档向量分布不同（HyDE 解决）")
    print("     3) 可能漏掉精确关键词匹配（BM25/Hybrid 解决）")
    print()
    print("  Q: HyDE 的核心思想是什么？")
    print("  A: 用户 query 不是直接去检索，而是先用 LLM 生成一段假设答案，")
    print("     用假设答案的向量去检索。因为 LLM 生成的文本和知识库文档")
    print("     在语言风格、长度、用词上更接近。")
    print()
    print("  Q: 混合检索怎么融合两种分数？")
    print("  A: 1) Min-Max 归一化把两种分数都压到 [0,1]")
    print("     2) 加权求和: final = α×BM25 + (1-α)×Vector")
    print("     3) α 按场景调：偏语义→小 α，偏关键词→大 α")
    print()
    print("【明日 Day 6 预告】高级检索 + RAGAS 评估")
    print("  - 用 RAGAS 框架量化评估检索质量")
    print("  - Context Recall / Faithfulness / Answer Relevancy")
    print("=" * 70)


# ============================================================================
# PART 7: 快速演示 — 只运行关键对比
# ============================================================================

def quick_demo():
    """快速演示：用 2 个典型查询对比 4 种策略。"""
    print("=" * 70)
    print("Day 5 快速演示: MMR vs HyDE vs Hybrid")
    print("=" * 70)

    config = RAGConfig(top_k=3, log_level=logging.WARNING)

    strategies = {
        "Default": EnhancedMiniRAG(config, strategy="default"),
        "MMR":    EnhancedMiniRAG(config, strategy="mmr", mmr_lambda=0.7),
        "HyDE":   EnhancedMiniRAG(config, strategy="hyde"),
        "Hybrid": EnhancedMiniRAG(config, strategy="hybrid", hybrid_alpha=0.3),
    }

    for name, rag in strategies.items():
        rag.ingest(TEST_DOCUMENTS)

    demo_queries = [
        "猫的品种有哪些？",
        "怎么养猫？",
        "特斯拉",
    ]

    for query in demo_queries:
        print(f"\n{'─' * 60}")
        print(f'🔍 "{query}"')
        print(f"{'─' * 60}")

        for name, rag in strategies.items():
            resp = rag.ask(query)
            sources_str = " | ".join(
                f"{s.metadata.get('category','?')}:{s.content[:30]}..."
                for s in resp.sources
            )
            print(f"  [{name:8s}] {sources_str}")

    print("\n✅ 快速演示完成！运行 run_experiments() 查看完整对比。")


# ============================================================================
# 主入口
# ============================================================================
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        quick_demo()
    else:
        run_experiments()