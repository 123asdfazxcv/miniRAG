import os
import json
import logging
import sys
import time
from dataclasses import dataclass,field
from pathlib import Path
from typing import Optional
import numpy as np
from dashscope import Generation

_PROJECT_DIR = Path(__file__).resolve().parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))
from minirag import (
RAGConfig,RAGError,RetrievalError,EmbeddingError,GenerationError,
Document,RetrievedChunk,RAGResponse,
Retriever,Generator,MiniRAG,init,_setup_logger,TEST_DOCUMENTS
)
from retrieval import (
EnhancedMiniRAG,MMRRetriever,BM25Retriever,HybridRetriever,HyDERetriever,
BM25Scorer,
)
from embedding import Embedder
logger = logging.getLogger("MiniRAG.Day6")
@dataclass()
class RerankerConfig:
    model:str = "qwen-turbo"
    candidates_k:int = 10
    top_k:int = 3
    score_prompt:str = field(default=(
        "你是一个检索相关性评分专家。\n"
        "我会给你一个【用户问题】和一条【文档内容】。\n"
        "请判断这条文档对回答用户问题有多大帮助，给出 1-10 的整数分数。\n\n"
        "评分标准：\n"
        "- 10 分：文档直接包含问题的全部答案\n"
        "- 7-9 分：文档包含问题的部分答案或高度相关信息\n"
        "- 4-6 分：文档与问题在同一领域，但不够具体\n"
        "- 1-3 分：文档与问题基本无关\n\n"
        "【用户问题】\n{query}\n\n"
        "【文档内容】\n{document}\n\n"
        "请只返回一个 JSON 对象，格式：{{\"score\": 整数, \"reason\": \"一句话理由\"}}\n"
        "不要返回其他内容。"
    ),repr=False)
class LLMReranker:
    def __init__(self,config:RerankerConfig):
        self.config = config or RerankerConfig()
    def _score_single(self,query:str,document:str,doc_index:int)->tuple[int,int,float,str]:
        prompt = self.config.score_prompt.format(query=query,document=document)
        try:
            resp = Generation.call(
                model=self.config.model,
                prompt=prompt,
                result_format="message"
            )
            if resp.status_code != 200:
                logger.warning(
                    "Reranker LLM 调用失败: status=%s, msg=%s",
                    resp.status_code, resp.message
                )
                return (doc_index, 0, 0.0, f"API 错误: {resp.message}")
            raw_output = resp.output.choices[0].message.content.strip()
            result = json.loads(raw_output)
            score = int(result.get("score",0))
            reason = result.get("reason","无理由")
            score = max(1,min(10,score))
            return (doc_index, score, score/10.0, reason)
        except json.JSONDecodeError:
            logger.warning("Reranker 解析 JSON 失败，raw=%s", raw_output[:100])
            return (doc_index, 0, 0.0, "JSON 解析失败")
        except Exception as e:
            logger.warning("Reranker 异常: %s", e)
            return (doc_index,0,0.0,str(e))
    def rerank(self,query:str,candidates:list[RetrievedChunk])->list[RetrievedChunk]:
        if not candidates:
            return []

        candidates = candidates[:self.config.candidates_k]
        logger.info("LLM Reranker 开始重排序，候选数=%d，query=「%s」",
                     len(candidates), query[:50])
        scored_results=[]
        for i,chunk in enumerate(candidates):
            idx,score,norm_score,reason = self._score_single(query,chunk.content,i)
            scored_results.append((idx,score,norm_score,reason))
            logger.debug("  [%d/%d] score=%d reason=%s",
                         i + 1, len(candidates), score, reason[:40])
        scored_results.sort(key=lambda x:x[1],reverse=True)
        reranked=[]
        for idx,raw_score,norm_score,reason in scored_results[:self.config.top_k]:
            chunk = candidates[idx]
            chunk.score =norm_score
            chunk.metadata["rerank_score"] = raw_score
            chunk.metadata["rerank_reason"] = reason
            reranked.append(chunk)
        logger.info("LLM Reranker 完成: %d → %d 条", len(candidates), len(reranked))
        return reranked
@dataclass
class MultiQueryConfig:
    model:str = "qwen-turbo"
    num_queries:int = 3
    prompt:str = field(default=(
        "你是一个查询改写专家。用户会问一个问题，请从不同角度生成{num}个语义相同"
        "但措辞不同的查询。\n\n"
        "要求:\n"
        "1.每个查询应该有不同的关键词或表达方式\n"
        "2.覆盖问题的；不同侧重点\n"
        "3.保持原始问题的意图不变\n\n"
        "原始问题：{query}\n\n"
        "请只返回一个JSON数组，格式：[\"查询1\",\"查询2\",\"查询3\"]\n"
        "不要返回其他内容。"
    ),repr=False)
class MultiQueryRetriever:
    def __init__(self,base_retriever:Retriever,config:MultiQueryConfig):
        self.base_retriever = base_retriever
        self.config = config or MultiQueryConfig()
    def _generate_variants(self,query:str)->list[str]:
        prompt = self.config.prompt.format(
            num = self.config.num_queries,query=query
        )
        try:
            resp = Generation.call(
                model=self.config.model,
                prompt=prompt,
                result_format="message",
            )
            if resp.status_code != 200:
                logger.warning("MultiQuery 生成变体失败: %s", resp.message)
                return [query]
            raw_output = resp.output.choices[0].message.content.strip()
            variants = json.loads(raw_output)
            if isinstance(variants,list):
                return [query] + variants
            else:
                return [query]
        except json.JSONDecodeError:
            logger.warning("MultiQuery JSON 解析失败: %s", raw_output[:100])
            return [query]
        except Exception as e:
            logger.warning("MultiQuery 异常: %s", e)
            return [query]
    def search(self,query:str,top_k:int=3)->list[RetrievedChunk]:
        variants = self._generate_variants(query)
        logger.info("MultiQuery: 原始「%s」→ %d 个变体", query[:40], len(variants) - 1)
        all_chunks:dict[str,RetrievedChunk] = {}
        for vi,variant in enumerate(variants):
            results = self.base_retriever.search(variant,top_k=top_k)
            logger.debug(
                "  [%d/%d] 「%s」→ %d 条结果",
                vi + 1, len(variants), variant[:30], len(results)
            )
            for chunk in results:
                chunk_id = chunk.chunk_id
                if chunk_id not in all_chunks or chunk.score > all_chunks[chunk_id].score:
                    all_chunks[chunk_id] = chunk
                    if "matched_queries" not in chunk.metadata:
                        chunk.metadata["matched_queries"] = []
                    chunk.metadata["matched_queries"].append(variant)
        merged = sorted(
            all_chunks.values(),
            key=lambda c:c.score,
            reverse=True,
        )
        logger.info(
            "MultiQuery 合并: %d 个变体 → %d 条去重 → 返回 Top-%d",
            len(variants), len(merged), min(top_k, len(merged))
        )
        return merged[:top_k]
@dataclass
class RAGASConfig:
    model:str = "qwen-turbo"
    context_recall_prompt:str=field(default=(
        "你是一个评估专家。请判断以下【参考答案】中的每个句子，"
        "能否从【检索到的上下文】中推断出来"
        "【参考答案】\n{reference}\n"
        "【检索到的上下文】\n{context}\n"
        "请返回一个JSON对象：\n"
        '{{"sentences":['
        '  {{"sentence":"参考答案第1句","supported":true/false}}'
        '  {{"sentence":"参考答案第2句","supported":true/false}}'
        '],"total":总句数，"supported_count":能被支持的句子数}}\n'
        "说明：supported=true 表示这句话的信息可以从上下文中找到或推断出来。\n"
        "不要返回其他内容。"
    ),repr=False)
    faithfulness_prompt:str = field(default=(
        "你是一个事实核查专家。请判断以下【LLM回答】中的每个陈述，"
        "能否从【检索到的上下文】中找到依据。\n\n"
        "【LLM回答】\n{answer}\n\n"
        "【检索到的上下文】\n{context}\n\n"
        "请返回一个JSON对象：\n"
        '{{"claims":['
        '  {{"claim":"回答中的陈述1","supported":true/false,"evidence":"上下文中的证据或null"}}'
        '  {{"claim":"回答中的陈述2","supported":true/false,"evidence":"上下文中的证据或null"}}'
        '],"total":总陈述数,"supported_count":有依据的陈述数}}\n'
        "说明：supported = true表示这个陈述在上下文中明确出现或可以合理判断。\n"
        "不要返回其他内容。"
    ),repr=False)
    answer_relevancy_prompt:str = field(default=(
        "你是一个问题生成专家。请根据以下【LLM回答】，"
        "反向生成这个回答能回答的问题。\n\n"
        "【LLM回答】\n{answer}\n\n"
        "请返回一个JSON对象：\n"
        '{{"questions":["问题1","问题2","问题3"]}}\n'
        "生成3个问题，这些问题都是这个回答能够回答的。\n"
        "不要返回其他内容。"
    ),repr=False)
    question_relevancy_prompt:str=field(default=(
        "请判断以下两个问题是否在问同一件事（语义相同或高度相关）：\n"
        "问题1：{q1}\n"
        "问题2：{q2}\n"
        "只返回一个JSON：{{\"related\":true/false}}"
    ),repr=False)
class RAGASEvaluator:
    def __init__(self,config:Optional[RAGASConfig]=None):
        self.config = config or RAGASConfig()
    def context_recall(self,reference:str,context:str)->dict:
        prompt = self.config.context_recall_prompt.format(
            reference=reference,context=context
        )
        try:
            resp = Generation.call(
                model=self.config.model,
                prompt=prompt,
                result_format="message"
            )
            if resp.status_code != 200:
                return {"score":0.0,"error":f"API错误:{resp.message}"}
            raw = resp.output.choices[0].message.content.strip()
            result = json.loads(raw)
            total = result.get("total",1)
            supported = result.get("supported_count",0)
            score = supported / total if total > 0 else 0.0
            return {
                "score":round(score,4),
                "supported_count":supported,
                "total":total,
                "details":result.get("sentences",[])
            }
        except Exception as e:
            logger.warning("Context Recall 计算失败: %s", e)
            return {"score": 0.0, "error": str(e)}
    def faithfulness(self,context:str,answer:str)->dict:
        prompt = self.config.faithfulness_prompt.format(
            context=context,answer=answer)
        try:
            resp = Generation.call(
                model=self.config.model,
                prompt=prompt,
                result_format="message"
            )
            if resp.status_code != 200:
                return  {"score": 0.0, "error": f"API 错误: {resp.message}"}
            raw = resp.output.choices[0].message.content.strip()
            result = json.loads(raw)
            total = result.get("total",1)
            supported = result.get("supported_count",0)
            score = supported / total if total > 0 else 0.0
            return {
                "score":round(score,4),
                "supported_count":supported,
                "total":total,
                "details":result.get("claims",[])
            }
        except Exception as e:
            logger.warning("Faithfulness 计算失败: %s", e)
            return {"score": 0.0, "error": str(e)}
    def answer_relevancy(self,question:str,answer:str)->dict:
        gen_prompt = self.config.answer_relevancy_prompt.format(answer=answer)
        try:
            resp = Generation.call(
                model=self.config.model,
                prompt=gen_prompt,
                result_format="message"
            )
            if resp.status_code!=200:
                return {"score":0.0,"error":f"API错误:{resp.message}"}
            raw = resp.output.choices[0].message.content.strip()
            result = json.loads(raw)
            generated_qs = result.get("questions",[])
            if not generated_qs:
                return {"score": 0.0, "error": "未能生成反向问题"}
        except Exception as e:
            logger.warning("反向生成问题失败: %s", e)
            return {"score": 0.0, "error": str(e)}
        related_count = 0
        for gq in generated_qs:
            rel_prompt = self.config.question_relevancy_prompt.format(q1=question,q2=gq)
            try:
                resp2 = Generation.call(
                    model=self.config.model,
                    prompt=rel_prompt,
                    result_format="message"
                )
                if resp2.status_code==200:
                    raw2 = resp2.output.choices[0].message.content.strip()
                    rel_result = json.loads(raw2)
                    if rel_result.get("related",False):
                        related_count += 1
            except Exception:
                pass
        total = len(generated_qs)
        score = related_count / total if total > 0 else 0.0
        return {
            "score":round(score,4),
            "related_count":related_count,
            "total":total,
            "generated_questions":generated_qs
        }
    def evaluate(self,question:str,answer:str,context:str,reference:str)->dict:
        logger.info("开始 RAGAS 评估: 「%s」", question[:50])
        cr = self.context_recall(reference,context)
        faith = self.faithfulness(context,answer)
        ar = self.answer_relevancy(question,answer)
        scores = [cr.get("score",0.0),
                  faith.get("score",0.0),
                  ar.get("score",0.0)]
        overall = round(sum(scores)/len(scores),4)
        return {
            "context_recall":cr,
            "faithfulness":faith,
            "answer_relevancy":ar,
            "overall":overall
        }
class Day6MiniRAG(EnhancedMiniRAG):
    def __init__(self,config:RAGConfig,strategy:str = "default",**kwargs,):
        self.strategy = strategy
        self._day6_reranker:Optional[LLMReranker]=None
        self._day6_multiq_retriever:Optional[MultiQueryRetriever]=None
        if strategy in ("rerank","full"):
            self._day6_reranker = LLMReranker(config)
        if strategy in ("multiq","full"):
            self._day6_multiq_retriever = MultiQueryRetriever(
                base_retriever=None,config=config
            )
        super().__init__(config,strategy=strategy,**kwargs)
    def ingest(self,documents:list[Document])->int:
        count = super().ingest(documents)
        if self._day6_multiq_retriever is not None:
            self._day6_multiq_retriever.base_retriever = self.retriever
        return count
    def _retrieve(self,query:str)->list[RetrievedChunk]:
        if self.strategy in ("default","mmr","hyde","hybrid"):
            return super().retriever(query)
        if self.strategy == "rerank" and self._day6_reranker:
            candidates = self.retriever.search(
                query,top_k=self._day6_reranker.config.candidates_k
            )
            return self._day6_reranker.rerank(query,candidates)
        if self.strategy == "multiq" and self._day6_multiq_retriever:
            return self._day6_multiq_retriever.search(
                query,top_k=self.config.top_k
            )
        if self.strategy == "full" and self._day6_multiq_retriever:
            candidates = self._day6_multiq_retriever.search(
                query,top_k=self._day6_reranker.config.candidates_k
                if self._day6_reranker else self.config.top_k
            )
            if self._day6_reranker and len(candidates)> self.config.top_k:
                return self._day6_reranker.rerank(query,candidates)
            return candidates[:self.config.top_k]
        return super()._retrieve(query)
EVAL_DOCUMENTS = TEST_DOCUMENTS + [
    Document(
        content=(
            "猫是严格的肉食动物，它们的消化系统适合消化动物蛋白。"
            "猫每天需要摄入约 200-300 卡路里的热量，取决于体型和活动量。"
            "猫不能吃巧克力、洋葱、大蒜和葡萄，这些食物对猫有毒。"
            "猫需要随时有清洁的饮水，缺水会导致泌尿系统疾病。"
        ),
        metadata={"category": "宠物饲养", "source": "宠物百科"},
    ),
    Document(
        content=(
            "电动汽车（Electric Vehicle, EV）使用电池和电机驱动，"
            "不需要汽油或柴油。电动汽车的续航里程一般在 300-700 公里之间。"
            "充电方式分为慢充（家用 220V，6-8 小时充满）和快充（充电桩，30 分钟充 80%）。"
            "电动汽车的维护成本比燃油车低约 50%，因为不需要换机油、火花塞等。"
        ),
        metadata={"category": "汽车知识", "source": "汽车百科"},
    ),
    Document(
        content=(
            "深度学习（Deep Learning）是机器学习的一个子领域，"
            "使用多层神经网络（Neural Network）来学习数据的层次化表示。"
            "常见的深度学习架构包括 CNN（卷积神经网络，用于图像）、"
            "RNN（循环神经网络，用于序列）和 Transformer（用于 NLP）。"
            "反向传播（Backpropagation）是训练神经网络的核心算法。"
        ),
        metadata={"category": "技术科普", "source": "AI 百科"},
    ),
]

# ---- 评估测试用例 ----
# 每个用例：问题 + 参考答案（ground truth），参考答案用于计算 Context Recall
EVAL_TEST_CASES = [
    {
        "question": "怎么给猫喂食？",
        "reference": (
            "猫需要每天定时喂食，成年猫每天喂2-3次。"
            "猫是肉食动物，需要以肉类为主的饮食。"
            "不能喂巧克力、洋葱、大蒜和葡萄。"
            "需要提供清洁的饮水。"
        ),
    },
    {
        "question": "电动汽车有什么优点？",
        "reference": (
            "电动汽车使用电力驱动，不需要汽油或柴油。"
            "维护成本比燃油车低约50%，不需要换机油和火花塞。"
            "充电分为慢充（家用220V，6-8小时）和快充（充电桩，30分钟充80%）。"
            "续航里程一般在300-700公里之间。"
        ),
    },
    {
        "question": "什么是深度学习？",
        "reference": (
            "深度学习是机器学习的一个子领域，使用多层神经网络学习数据的层次化表示。"
            "常见架构包括 CNN（卷积神经网络）、RNN（循环神经网络）和 Transformer。"
            "反向传播是训练神经网络的核心算法。"
        ),
    },
    {
        "question": "猫的饮食有什么禁忌？",
        "reference": (
            "猫不能吃巧克力、洋葱、大蒜和葡萄，这些食物对猫有毒。"
            "猫是严格的肉食动物，消化系统适合动物蛋白。"
            "每天需要摄入约200-300卡路里的热量。"
        ),
    },
]
def run_evaluation():
    """Day 6 核心实验：用 RAGAS 评估不同检索策略。

    实验设计：
      对比 6 种策略的 RAGAS 分数：
      1. Default (Day4 基线)  — 纯向量检索
      2. MMR     (Day5)       — 多样性重排序
      3. HyDE    (Day5)       — 假设文档增强
      4. Hybrid  (Day5)       — 向量 + BM25
      5. Rerank  (Day6 新增)  — LLM 重排序
      6. Full    (Day6 最强)  — MultiQuery + Rerank

    每个策略对每个测试用例：
      1. 检索 → 得到 context（检索上下文）
      2. 生成 → 得到 answer（LLM 回答）
      3. 评估 → 计算三个 RAGAS 指标
    """
    print("=" * 70)
    print("Day 6: RAGAS 评估 — 6 种检索策略量化对比")
    print("=" * 70)

    config = RAGConfig(top_k=3, log_level=logging.WARNING)
    evaluator = RAGASEvaluator()

    # ---- 配置 6 种策略 ----
    strategies = {
        "Default (Day4 基线)": {
            "strategy": "default",
        },
        "MMR (Day5 多样性)": {
            "strategy": "mmr",
            "mmr_lambda": 0.7,
        },
        "HyDE (Day5 假设文档)": {
            "strategy": "hyde",
        },
        "Hybrid (Day5 混合)": {
            "strategy": "hybrid",
            "hybrid_alpha": 0.3,
        },
        "Rerank (Day6 LLM重排)": {
            "strategy": "rerank",
        },
        "Full (Day6 最强)": {
            "strategy": "full",
        },
    }

    # ---- 初始化 + 索引（每个策略一个实例）----
    print(f"\n🔧 初始化 6 种检索策略...")
    rags = {}
    for name, kwargs in strategies.items():
        rag = Day6MiniRAG(config, **kwargs)
        n = rag.ingest(EVAL_DOCUMENTS)
        rags[name] = rag
        print(f"  {name}: {n} chunks 已索引")

    # ---- 对每个测试用例，用所有策略评估 ----
    print(f"\n📊 开始评估 {len(EVAL_TEST_CASES)} 个测试用例...\n")

    # 汇总分数表
    all_results = []

    for tc_idx, tc in enumerate(EVAL_TEST_CASES, 1):
        question = tc["question"]
        reference = tc["reference"]

        print(f"{'─' * 70}")
        print(f"📋 测试用例 {tc_idx}: 「{question}」")
        print(f"{'─' * 70}")

        for strategy_name, rag in rags.items():
            # Step 1: 检索 + 生成
            resp = rag.ask(question)

            # Step 2: 构建 context 字符串（多篇文档拼接）
            context = "\n\n".join(
                f"[{s.metadata.get('category', '?')}] {s.content}"
                for s in resp.sources
            )

            # Step 3: RAGAS 评估
            eval_result = evaluator.evaluate(
                question=question,
                answer=resp.answer,
                context=context,
                reference=reference,
            )

            cr = eval_result["context_recall"].get("score", "?")
            faith = eval_result["faithfulness"].get("score", "?")
            ar = eval_result["answer_relevancy"].get("score", "?")
            overall = eval_result["overall"]

            all_results.append({
                "test_case": question,
                "strategy": strategy_name,
                "context_recall": cr,
                "faithfulness": faith,
                "answer_relevancy": ar,
                "overall": overall,
            })

            print(f"  [{strategy_name:25s}] "
                  f"CR={cr} | Faith={faith} | AR={ar} | "
                  f"⭐综合={overall}")

    # ---- 汇总：每个策略的平均分 ----
    print(f"\n{'=' * 70}")
    print("📊 汇总：各策略平均 RAGAS 分数")
    print(f"{'=' * 70}")

    # 按策略汇总分数
    strategy_summary = {}
    for r in all_results:
        s = r["strategy"]
        if s not in strategy_summary:
            strategy_summary[s] = {
                "context_recall": [],
                "faithfulness": [],
                "answer_relevancy": [],
                "overall": [],
            }
        for key in ["context_recall", "faithfulness", "answer_relevancy", "overall"]:
            if isinstance(r[key], (int, float)):
                strategy_summary[s][key].append(r[key])

    # 打印汇总表
    # CR = Context Recall（上下文召回率）
    # Faith = Faithfulness（忠实度）
    # AR = Answer Relevancy（答案相关性）
    print(f"\n{'策略':30s} {'CR':>6s} {'Faith':>6s} {'AR':>6s} {'综合':>6s}")
    print(f"{'─' * 60}")
    for name in strategies:
        summary = strategy_summary.get(name)
        if not summary:
            continue
        avg_cr = np.mean(summary["context_recall"])
        avg_faith = np.mean(summary["faithfulness"])
        avg_ar = np.mean(summary["answer_relevancy"])
        avg_overall = np.mean(summary["overall"])
        print(f"{name:30s} {avg_cr:6.3f} {avg_faith:6.3f} {avg_ar:6.3f} {avg_overall:6.3f}")

    # ---- 结论 ----
    print(f"\n{'=' * 70}")
    print("✅ Day 6 完成！你现在能用数字证明 RAG 系统的质量了。")
    print()
    print("【你新增的能力】")
    print("  - LLM Reranker:    粗排（向量）→ 精排（LLM），两阶段检索")
    print("  - MultiQuery:      一次问题生成多个变体，多路检索合并")
    print("  - Context Recall:  测检索环节有没有丢信息")
    print("  - Faithfulness:    测 LLM 有没有产生幻觉/瞎编")
    print("  - Answer Relevancy: 测 LLM 有没有跑题")
    print()
    print("【面试能答的问题】")
    print("  Q: 怎么评估 RAG 系统的质量？")
    print("  A: 用 RAGAS 框架三个维度：")
    print("     1. Context Recall — 检索到的文档能否覆盖参考答案")
    print("     2. Faithfulness — LLM 的回答是否忠实于检索到的文档")
    print("     3. Answer Relevancy — LLM 的回答是否切题")
    print()
    print("  Q: 向量检索之后为什么还要做重排序？")
    print("  A: 向量相似度 ≠ 语义相关性。粗排（向量检索）速度快，")
    print("     精排（LLM/Cross-Encoder）更准。两阶段各取所长。")
    print()
    print("  Q: Multi-Query 和 HyDE 有什么区别？")
    print("  A: HyDE 是生成「假设答案」去搜，桥接语义鸿沟；")
    print("     MultiQuery 是生成「多个问题变体」去搜，扩大覆盖范围。")
    print("     两者可以叠加。")
    print()
    print("【明日 Day 7 预告】完整项目 + 部署")
    print("  - LangChain 重构（工程化）")
    print("  - Streamlit/Gradio 前端界面")
    print("  - GitHub 开源部署")
    print("=" * 70)

    return all_results


def quick_demo():
    """快速演示：用 2 个策略对比 2 个查询，避免消耗太多 API 调用。"""
    print("=" * 70)
    print("Day 6 快速演示: Rerank vs MultiQuery")
    print("=" * 70)

    config = RAGConfig(top_k=3, log_level=logging.WARNING)

    rags = {
        "Base (Day4基线)": Day6MiniRAG(config, strategy="default"),
        "Rerank (LLM重排)": Day6MiniRAG(config, strategy="rerank"),
        "MultiQ (多查询)": Day6MiniRAG(config, strategy="multiq"),
    }

    for name, rag in rags.items():
        n = rag.ingest(EVAL_DOCUMENTS)
        print(f"  {name}: {n} chunks")

    demo_queries = [
        "怎么养猫？",
        "电动车和燃油车哪个好？",
    ]

    for query in demo_queries:
        print(f"\n{'─' * 60}")
        print(f'🔍 "{query}"')
        print(f"{'─' * 60}")
        for name, rag in rags.items():
            resp = rag.ask(query)
            sources_str = " | ".join(
                f"[{s.metadata.get('category', '?')}] score={s.score:.3f}"
                for s in resp.sources
            )
            print(f"  [{name:20s}] {sources_str}")
            # 也显示回答摘录
            if resp.answer:
                print(f"  {'':20s}  回答: {resp.answer[:80]}...")

    print(f"\n✅ 快速演示完成！运行 run_evaluation() 查看完整 RAGAS 评估。")
    print("   ⚠️  注意：RAGAS 完整评估约消耗 20-30 次 LLM 调用（6策略×4用例×3指标）")


# ============================================================================
# 主入口
# ============================================================================
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        quick_demo()
    else:
        run_evaluation()