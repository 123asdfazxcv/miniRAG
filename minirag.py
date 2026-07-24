import logging
import os
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import faiss
import numpy as np
from dashscope import Generation


from embedding import Embedder

from dotenv import load_dotenv
_setup_done = False
def init()->None:
    global _setup_done
    if _setup_done:
        return
    for _env_dir in [Path(__file__).resolve().parent,Path.cwd()]:
        _env_file = _env_dir/".env"
        if _env_file.exists():
            load_dotenv(_env_file)
            break
    import dashscope
    if not dashscope.api_key:
        dashscope.api_key = os.environ.get("DASHSCOPE_API_KEY","")
    if not dashscope.api_key:
        raise RuntimeError("DASHSCOPE_API_KEY not set")
    _setup_done = True
@dataclass
class RAGConfig:
    embed_model:str = "text-embedding-v2"
    embed_dim:int = 1536
    llm_model:str = "qwen-turbo"
    llm_temperature:float = 0.2
    llm_max_tokens:int = 1024
    top_k:int = 3
    similarity_threshold:float = 0.5
    log_level:int = logging.INFO

class RAGError(Exception):
    def __init__(self,message:str,*,code:str|None = None):
        super().__init__(message)
        self.code = code
class EmbeddingError(RAGError):
    def __init__(self,message:str,*,phase:str,model:str|None = None):
        super().__init__(message,code="EMBEDDING_ERROR")
        self.phase = phase
        self.model = model
class RetrievalError(RAGError):
    def __init__(self,message:str):
        super().__init__(message,code="RETRIEVAL_ERROR")
class GenerationError(RAGError):
    def __init__(self,message:str,*,reason:str,model:str|None = None,status_code:int|None = None):
        super().__init__(message,code="GENERATION_ERROR")
        self.reason = reason
        self.model = model
        self.status_code = status_code

@dataclass
class Document:
    content:str
    metadata:dict = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    content:str
    score:float
    doc_id:str
    chunk_index:int = 0
    metadata:dict = field(default_factory=dict)
@dataclass
class RAGResponse:
    answer:str
    sources:list[RetrievedChunk]
    token_usage : dict
    elapsed_ms : float
logger = logging.getLogger("MiniRAG")
logger.setLevel(logging.DEBUG)
def _setup_logger(level:int = logging.INFO):
    if logger.handlers:
        return
    handle = logging.StreamHandler(sys.stdout)
    handle.setLevel(level)
    handle.setFormatter(logging.Formatter(
        "%(asctime)s|%(levelname)-5s|%(name)s|%(message)s"
    ))
    logger.addHandler(handle)

RAG_SYSTEM_PROMPT = """你是一个严谨的知识助手，只能根据下方提供的「参考文档」回答问题。
严格遵循以下规则：
1. 回答必须基于参考文档中的具体信息，不得使用外部知识
2. 如果参考文档没有覆盖用户的问题，明确回答："根据提供的文档，无法回答此问题"
3. 回答时引用具体的文档内容，让用户知道信息来源
4. 答案应简洁、准确，不添加文档中没有的细节"""
def build_prompt(question:str,sources:list[RetrievedChunk])->dict:
    docs_text_parts=[]
    for i,src in enumerate(sources,1):
        docs_text_parts.append(f"[文档{i}] (相关度：{src.score:.2f}\n{src.content}")
    docs_text = "\n\n".join(docs_text_parts)
    user_message = (
        f"## 参考文档\n\n{docs_text}\n\n"
        f"## 用户问题\n\n{question}\n\n"
        f"请基于以上参考文档回答。如果文档中没有相关信息，请明确说明"
    )
    return{
        "system":RAG_SYSTEM_PROMPT,
        "user":user_message,
    }
class Retriever:
    def __init__(self,embedder:Embedder,config:RAGConfig):
        self.embedder = embedder
        self.config = config
        self._chunks:list[str] = []
        self._chunk_meta:list[dict] = []
        self._index:Optional[faiss.IndexFlatIP] = None
        self._indexed_count:int = 0
    @property
    def is_indexed(self)->bool:
        return self._index is not None and self._indexed_count > 0
    @property
    def chunk(self)->int:
        return self._indexed_count
    def index(self,documents:list[Document])->int:
        if not documents:
            raise ValueError("文档列表不能为空")
        logger.info(f"开始索引{len(documents)}篇文档")
        t0 = time.perf_counter()
        self._chunks = []
        self._chunk_meta = []
        for doc_idx , doc in enumerate(documents):
            chunks_for_doc = self._split_text(doc.content)
            for chunk in chunks_for_doc:
                if chunk.strip():
                    self._chunks.append(chunk)
                    self._chunk_meta.append(
                        {"doc_index": doc_idx,
                         "doc_metadata":doc.metadata}
                    )
        logger.info(f"切割完成：{len(documents)}篇→{len(self._chunks)}chunks")

        try:
            embeddings = self.embedder.embed_batch(self._chunks,text_type="document")
        except Exception as e:
            raise EmbeddingError(
                f"批量向量化失败：{e}",
                phase="index",model=self.config.embed_model,
            )
        try:
            embeddings = np.array(embeddings,dtype=np.float32)
            faiss.normalize_L2(embeddings)
            self._index = faiss.IndexFlatIP(self.config.embed_dim)
            self._index.add(embeddings)
        except Exception as e:
            raise RetrievalError(f"FAISS索引构建失败：{e}")from e
        self._indexed_count = len(self._chunks)
        elapsed = (time.perf_counter() - t0)*1000
        logger.info(
            f"索引完成：{self._indexed_count}chunks"
            f"耗时：{elapsed}ms"
        )
        return self._indexed_count

    def retrieve(self,query:str,top_k:Optional[int] = None):
        if not self.is_indexed:
            raise RetrievalError("索引为空，请先调用index()方法")
        k = top_k or self.config.top_k
        try:
            q_vec = self.embedder.embed(query,text_type="query")
        except Exception as e:
            raise EmbeddingError(
                f"Query向量化失败：{e}",
                phase="query",model=self.config.embed_model,
            )from e

        q_vec = np.array([q_vec],dtype=np.float32)
        faiss.normalize_L2(q_vec)
        scores,indices = self._index.search(q_vec,k)
        results = []
        for score,idx in zip(scores[0],indices[0]):
            if idx == -1 :
                continue
            if float(score) < self.config.similarity_threshold:
                logger.debug(f"chunk{idx}分数{score:.3f}低于阈值，已丢弃")
                continue
            meta = self._chunk_meta[idx]
            results.append(RetrievedChunk(
                content=self._chunks[idx],
                metadata=meta["doc_metadata"],
                score=float(score),
                doc_id=meta["doc_index"],
            ))
        logger.debug(
            f"检索'{query[:30]}...'→{len(results)}/{k}条结果"
            f"(阈值{self.config.similarity_threshold}"
        )
        return results
    @staticmethod
    def _split_text(text:str,max_len=500,overlap=50):
        if len(text) <= max_len:
            return [text]
        import re
        sentances = re.split(r'(?<=[？！。\n])',text)
        chunks = []
        current =""
        for sentance in sentances:
            if len(sentance) > max_len:
                if current.strip():
                    chunks.append(current.strip())
                for start in range(0,len(sentance),max_len - overlap):
                    chunks.append(sentance[start:start + max_len].strip())
                current = ""
                continue
            if len(sentance) + len(current) > max_len:
                chunks.append(current.strip())
                current = current[-overlap:] + sentance if overlap > 0 else sentance
            else:
                current += sentance
        if current:
            chunks.append(current.strip())
        return chunks
class Generator:
    def __init__(self,config:RAGConfig):
        self.config = config
    def generate(self,question:str,sources:list[RetrievedChunk])->tuple[str,dict]:
        prompt = build_prompt(question,sources)
        logger.debug(f"System prompt : {len(prompt['system'])}chars")
        logger.debug(f"User prompt : {len(prompt['user'])}chars")
        try:
            resp = Generation.call(
                model=self.config.llm_model,
                messages=[
                    {"role":"system","content":prompt['system']},
                    {"role":"user","content":prompt['user']},
                ],
                temperature=self.config.llm_temperature,
                max_tokens=self.config.llm_max_tokens,
            )
        except Exception as e:
            raise GenerationError(
                f"LLM调用异常：{e}",
                reason="network_error",model=self.config.llm_model,
            )
        if resp.status_code != 200:
            raise GenerationError(
                f"LLM API返回错误(code = {resp.status_code}):"
                f"{getattr(resp,'message','unknown')}",
                reason="api_error",model=self.config.llm_model,
                status_code=resp.status_code,
            )
        if resp.output is None or not resp.output.text:
            raise GenerationError(
                f"LLM返回空内容 - 可能被内容安全过滤，或模型内错误",
                reason="empty_error",model=self.config.llm_model,
            )
        answer = resp.output.text
        usage = resp.usage or {}
        token_usage = {
            "input":usage.get("input_tokens",0),
            "output":usage.get("output_tokens",0),
            "total":usage.get("total_tokens",0),
        }
        logger.info(
            f"LLM生成完成:{len(answer)}chars,"
            f"tokens:{token_usage['input']},in / {token_usage['output']}out"
        )
        return answer,token_usage
class MiniRAG:
    def __init__(self,config:Optional[RAGConfig]=None):
        init()
        self.config = config
        _setup_logger(self.config.log_level)
        self.embedder = Embedder(model = self.config.embed_model)
        self.retriever = Retriever(self.embedder,self.config)
        self.generator = Generator(config=self.config)
        logger.info(
            f"MiniRAG初始化已完成"
            f"(embed = {self.config.embed_model},llm = {self.config.llm_model})"
        )
    def ingest(self,documents:list[Document]):
        return self.retriever.index(documents)
    def ask(self,question:str,top_k:Optional[int]=None)->RAGResponse:
        t0 = time.perf_counter()
        logger.info(f"查询：{question}")
        try:
            sources = self.retriever.retrieve(question,top_k)
            if not sources:
                return RAGResponse(
                    answer="根据提供的文档，无法回答此问题",
                    sources=[],
                    token_usage={"input":0,"output":0,"total":0},
                    elapsed_ms=(time.perf_counter() - t0)*1000,
                )
            answer , token_usage = self.generator.generate(question,sources)
        except Exception as e:
            logger.error(f"查询失败：'{question[:50]}...'",exc_info=True)
            raise
        elpased = (time.perf_counter() - t0)*1000
        logger.info(f"查询完成：耗时{elpased:0f}ms")
        return RAGResponse(
            answer = answer,
            sources=sources,
            token_usage=token_usage,
            elapsed_ms=elpased,
        )

TEST_DOCUMENTS = [
    # ===== 猫（索引 0-9）=====
    Document(content="猫咪是世界上最受欢迎的宠物之一，它们独立而优雅。", metadata={"category": "猫", "topic": "宠物"}),
    Document(content="养猫需要准备猫砂盆、猫粮和猫抓板。猫砂盆要每天清理保持卫生。", metadata={"category": "猫", "topic": "饲养"}),
    Document(content="布偶猫是一种长毛猫，性格温顺，适合家庭饲养。它们喜欢与人互动。", metadata={"category": "猫", "topic": "品种"}),
    Document(content="橘猫通常体型较大，以爱吃著称，被称为'橘猪'。需要控制饮食防止肥胖。", metadata={"category": "猫", "topic": "品种"}),
    Document(content="猫咪的寿命一般在12-18年，有些能活到20岁以上。定期体检很重要。", metadata={"category": "猫", "topic": "健康"}),
    Document(content="英短蓝猫以其圆脸、短毛和憨厚的外表深受喜爱。性格独立但亲人。", metadata={"category": "猫", "topic": "品种"}),
    Document(content="猫咪每天需要睡眠12-16小时来保持精力。它们是黄昏活动型动物。", metadata={"category": "猫", "topic": "习性"}),
    Document(content="暹罗猫是最古老的宠物猫品种之一，原产于泰国。叫声响亮且喜欢与人交流。", metadata={"category": "猫", "topic": "品种"}),
    Document(content="猫粮主要分为干粮和湿粮，干粮有助于清洁牙齿，湿粮含水量高。", metadata={"category": "猫", "topic": "饲养"}),
    Document(content="无毛猫虽然外表特殊，但性格友好，对温度很敏感。需要定期洗澡和保暖。", metadata={"category": "猫", "topic": "品种"}),
    # ===== 汽车（索引 10-19）=====
    Document(content="特斯拉Model 3是目前全球最畅销的电动汽车之一。续航里程超过500公里。", metadata={"category": "汽车", "topic": "电动车"}),
    Document(content="发动机需要定期更换机油，通常每5000-10000公里一次。按时保养延长发动机寿命。", metadata={"category": "汽车", "topic": "保养"}),
    Document(content="SUV车型因其空间大、通过性好而受到家庭用户的青睐。后备箱容量远超轿车。", metadata={"category": "汽车", "topic": "车型"}),
    Document(content="混合动力汽车结合了燃油发动机和电动机，能显著降低油耗。城市通勤优势明显。", metadata={"category": "汽车", "topic": "新能源"}),
    Document(content="自动驾驶技术分为L1到L5五个等级，目前主流是L2+级别。L3以上法规尚未完善。", metadata={"category": "汽车", "topic": "技术"}),
    Document(content="丰田卡罗拉是全球累计销量最高的车型之一，以可靠性著称。保养成本低、保值率高。", metadata={"category": "汽车", "topic": "车型"}),
    Document(content="汽车轮胎需要定期检查胎压，正常值一般在2.3-2.5bar。胎压异常影响油耗和安全。", metadata={"category": "汽车", "topic": "保养"}),
    Document(content="新能源车使用电池驱动，零排放，是未来交通的趋势。充电便利性是关键考虑因素。", metadata={"category": "汽车", "topic": "新能源"}),
    Document(content="手动挡和自动挡的主要区别在于换挡方式的不同。自动挡操作简便，适合城市驾驶。", metadata={"category": "汽车", "topic": "车型"}),
    Document(content="购买新车时，购置税和保险是需要额外考虑的费用。购置税约为车价的10%。", metadata={"category": "汽车", "topic": "购车"}),
]
TEST_QUERIES = [
    "我想养一只猫，有什么建议？",
    "电动车有什么优缺点？",
    "布偶猫和暹罗猫有什么区别？",
    "发动机怎么保养？",
    "自动驾驶是什么？",
    # 边界测试：文档中没有的信息
    "怎么训练狗坐下？",
]
def run_experiments():
    print("Day 4: 生产级最小 RAG 系统 — 端到端实验")
    rag = MiniRAG(RAGConfig(top_k=3,log_level=logging.WARNING))
    print(f"\n📦 索引 {len(TEST_DOCUMENTS)} 篇文档...")
    n_chunks = rag.ingest(TEST_DOCUMENTS)
    print(f"   切割为 {n_chunks} 个 chunks\n")
    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"{'─' * 70}")
        print(f"🔍 查询 {i}: \"{query}\"")
        print(f"{'─' * 70}")
        resp = rag.ask(query)
        if resp.sources:
            print(f"\n📄 检索来源 (共 {len(resp.sources)} 条):")
            for j, src in enumerate(resp.sources, 1):
                cat = src.metadata.get("category", "?")
                print(f"  [{j}] [{cat}] 相似度={src.score:.3f} | {src.content[:60]}...")
        else:
            print("  ⚠️  未检索到相关文档")
        print(f"\n🤖 LLM 回答:")
        print(f"  {resp.answer}")

        # ---- 输出统计 ----
        print(f"\n📊 统计: "
              f"tokens={resp.token_usage['total']} "
              f"({resp.token_usage['input']} in / {resp.token_usage['output']} out) | "
              f"耗时={resp.elapsed_ms:.0f}ms")
        print()
        # ---- 总结 ----
    print("=" * 70)
    print("✅ Day 4 完成！你刚刚亲手搭建了一个生产级的 RAG 系统。")
    print()
    print("【你掌握的能力】")
    print("  - 端到端 RAG 流程: Ingest → Embed → Retrieve → Prompt → Generate")
    print("  - 生产级工程: 配置管理 / 异常处理 / 日志 / 类型标注 / 来源追溯")
    print("  - 防幻觉: System Prompt 约束 + 检索阈值过滤")
    print()
    print("【面试能答的问题】")
    print("  Q: RAG 的三个阶段是什么？")
    print("  A: Retrieve（检索相关文档）→ Augment（拼入 Prompt）→ Generate（LLM 生成）")
    print()
    print("  Q: 怎么防止 RAG 编造答案？")
    print("  A: 1) System Prompt 明确禁止 2) 检索阈值过滤低相关文档")
    print("     3) 无结果时直接返回\"无法回答\"，不让 LLM 凭空生成")
    print("=" * 70)

# ============================================================================
# 主入口
# ============================================================================
if __name__ == "__main__":
    run_experiments()