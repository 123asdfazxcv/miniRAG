import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal,Optional
from fastapi import FastAPI,HTTPException,UploadFile,File
from pydantic import BaseModel,Field,field_validator
from starlette.responses import JSONResponse

from RAGAS import RAGASEvaluator
from embedding import Embedder
from minirag import RAGConfig,Document,RetrievedChunk,init,_setup_logger,Retriever,Generator
from retrieval import BM25Retriever,MMRRetriever,HyDERetriever,HybridRetriever
from loaders import DocumentLoaderFactory
logger = logging.getLogger("RAG.API")

"""定义请求模型"""
class RAGQuery(BaseModel):
    """检索/一步到位的请求"""
    query:str = Field(...,min_length=1,description="用户问题")
    top_k:int = Field(default=5,ge=1,le=20,description="返回结果的数量")
    strategy:str=Field(default="default",description="检索策略：default/mmr/hyde/hybrid")
    similarity_threshold:Optional[float] = Field(default=None,ge=0.1,le=1.0,description="相似度阈值，不传就用引擎默认值")

class IngestRequest(BaseModel):
    """文档入库请求"""
    documents:list[dict]=Field(...,min_length=1,description="文档列表，每项包含content和可选metadata")

class GenerateRequest(BaseModel):
    """纯生成请求 -Agent 自己传上下文"""
    query:str = Field(...,min_length=1,description="用户问题")
    context_chunks:list[str] = Field(...,min_length=1,description="检索到的上下文chunks")
    system_prompt_extra:Optional[str] = Field(default=None,description="追加到的system prompt的指令")

"""定义响应数据模型"""
class SourceItem(BaseModel):
    """一条检索结果"""
    content:str = Field(description="chunk内容")
    score:float = Field(description="相似度分数")
    metadata:dict = Field(default_factory=dict,description="文档元数据")
    chunk_index:int = Field(default=0,description="chunk序号")

class RetrieveResult(BaseModel):
    """检索响应数据"""
    query:str = Field(description="原始查询")
    total_found:int = Field(description="实际检索到的结果数")
    sufficient:bool = Field(description="结果是否充分(total_found >= top_k)")
    strategy_used:str = Field(description="实际使用的策略")
    results:list[SourceItem] = Field(default_factory=list,description="检索结果列表")

class GenerateResult(BaseModel):
    """生成响应数据"""
    answer:str = Field(description="生成的回答")
    sources_used:int = Field(description="用了几条来源")
    has_answer:bool = Field(description="是否生成了有效内容")

class AskResult(BaseModel):
    """一步到位的响应数据"""
    answer:str = Field(description="生成的回答")
    sources:list[SourceItem] = Field(default_factory=list,description="检索来源")
    has_answer:bool = Field(description="是否生成了有效内容")

class IngestResult(BaseModel):
    """入库响应数据"""
    message:str = Field(description="处理结果描述")
    document_count:int = Field(description="文档数量")
    chunk_count:int = Field(description="切割后的chunk数量")

class StatusResult(BaseModel):
    """状态查询响应"""
    is_indexed:bool = Field(description="是否有索引")
    chunk_count:int = Field(description="chunk数量")
    available_strategies:list[str] = Field(description="可用的检索策略")
    llm_model:str = Field(description="LLM模型")
    embed_model:str = Field(description="Embedding模型")
    similarity_threshold:float = Field(description="当前相似度阈值")

class APIError(BaseModel):
    """结构化错误信息"""
    code:str = Field(description="错误码：EMBEDDING_ERROR/RETRIEVAL_ERROR/GENERATION_ERROR/...")
    message:str = Field(description="人类可读的错误信息")
    action:Literal["retry","reindex","fallback","give_up"] = Field(
        description="建议Agent采取的动作"
    )
    detail:Optional[str] = Field(default=None,description="额外上下文")

class ResponseEnvelope(BaseModel):
    """所有端点的统一响应包装"""
    success:bool = Field(description="是否成功")
    data:Optional[
        RetrieveResult|GenerateResult|AskResult|IngestResult|StatusResult
    ] = Field(default=None,description="成功时的数据")
    error:Optional[APIError] = Field(default=None,description="失败是的错误信息")
class EvaluateRequest(BaseModel):
    question:str = Field(...,description="用户问题")
    reference:str = Field(...,description="标准参考答案")
    top_k:int = Field(default=3,ge=1,le=10)
    strategy:str = Field(default="default",description="检索策略：default/mmr/hyde/hybrid")
    @field_validator("question","reference")
    @classmethod
    def clean_control_chars(cls,v:str)->str:
        if not isinstance(v,str):
            return v
        return v.replace("\r\n"," ").replace("\n"," ").replace("\r"," ").strip()


##############
#生命周期管理
##############

@asynccontextmanager
async def lifespan(app:FastAPI):
    """"FastAPI生命周期：启动时检查配置，创建共享的engine实例"""
    # -----启动-----
    logger.info("RAG API 启动中...")

    # 1.初始化DashScope SDK
    try:
        init()
    except RuntimeError as e:
        logger.critical(f"API Key 未配置:{e}")
        raise

    # 2.创建配置
    config = RAGConfig()
    _setup_logger(config.log_level)
    logger.info(f"配置:llm={config.llm_model},embed={config.embed_model}")

    # 3.创建共享底层
    embedder = Embedder(model = config.embed_model)
    shared_retriever = Retriever(embedder,config)
    shared_bm25 = BM25Retriever()

    # 4.策略缓存
    strategy_cache:dict[str,Retriever|MMRRetriever|HyDERetriever|HybridRetriever] = {}

    # 5.记录已入库的文档
    indexed_documents:list[Document] = []

    # 6.挂到app.state
    app.state.config = config
    app.state.embedder = embedder
    app.state.retriever = shared_retriever
    app.state.bm25_retriever = shared_bm25
    app.state.strategy_cache = strategy_cache
    app.state.indexed_documents = indexed_documents

    logger.info(f"RAG API启动完成(embed={config.embed_model},llm={config.llm_model})")

    yield #这里开始接受请求

    # ----关闭----
    logger.info("RAG API关闭")

##############
#FastAPI应用
##############
app = FastAPI(
    title="RAG API",
    description="MiniRAG 引擎的 FastAPI接口层，专为AI Agent设计",
    version="1.0.0",
    lifespan=lifespan
)

#####################
#端点：状态查询
######################
@app.get("/api/rag/status",response_model=ResponseEnvelope)
def get_status():
    """查询引擎当前状态"""
    retriever = app.state.retriever
    config = app.state.config
    cached_count = len([k for k in app.state.strategy_cache.keys()])
    data = StatusResult(
        is_indexed=retriever.is_indexed,
        chunk_count=retriever.chunk,
        available_strategies=["default","mmr","hyde","hybrid"],
        llm_model=config.llm_model,
        embed_model=config.embed_model,
        similarity_threshold=config.similarity_threshold,
    )
    return ResponseEnvelope(success=True, data=data)
#####################
# 端点：文档入库
#####################
@app.post("/api/rag/documents",response_model=ResponseEnvelope)
def ingest_documents(request:IngestRequest):
    docs = [
        Document(
            content=item["content"],
            metadata=item.get("metadata",{}),
        )
        for item in request.documents
    ]


    chunk_count =app.state.retriever.index(docs,incremental=True)
    if app.state.bm25_retriever:
        app.state.bm25_retriever.index(docs)
    data = IngestResult(
        message=f"已入库 {len(docs)} 篇文档，总 {chunk_count} 个chunks",
        document_count=len(docs),
        chunk_count=chunk_count,
    )
    return ResponseEnvelope(success=True, data=data)


###########################
# 文件上传
##########################
import tempfile
@app.post("/api/rag/upload",response_model=ResponseEnvelope)
async def upload_document(file:UploadFile = File(...),embed_model:Optional[str]=None):
    if not file.filename:
        raise HTTPException(status_code=400,detail="未提供文件名")
    suffix = Path(file.filename).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False,suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    try:
        loader = DocumentLoaderFactory.get_loader(file.filename)
        source_name = file.filename
        docs = loader.load(tmp_path,filename=source_name)
        if not docs:
            return ResponseEnvelope(
                success=True,
                data=IngestResult(
                    message=f"文件'{source_name}'未能提取出内容",
                    document_count=0,chunk_count=0
                )
            )

        if embed_model:
            logger.info(f"切换向量模型：{embed_model}")
            app.state.config.embed_model = embed_model
            app.state.config.embed_dim = Embedder(embed_model).dim
            new_embedder = Embedder(model=embed_model)
            new_retriever=Retriever(new_embedder,app.state.config)
            app.state.embedder = new_embedder
            app.state.retriever = new_retriever
        app.state.strategy_cache.clear()
        chunk_count = app.state.retriever.index(docs,incremental = True)
        if app.state.bm25_retriever:
            app.state.bm25_retriever.index(docs)
        data = IngestResult(
            message=f"已入库文件'{source_name}'；{len(docs)}篇文档，{chunk_count}个chunks",
            document_count=len(docs),chunk_count=chunk_count,
        )
        return ResponseEnvelope(success=True, data=data)
    finally:
        os.unlink(tmp_path)

##########################
# 策略缓存管理
###########################
def _get_or_create_strategy(strategy:str)->Retriever|MMRRetriever|HyDERetriever|HybridRetriever:
    cache = app.state.strategy_cache
    if strategy in cache:
        return cache[strategy]

    config = app.state.config
    embedder = app.state.embedder

    if strategy == "default":
        engine = app.state.retriever
    elif strategy == "mmr":
        engine = MMRRetriever(app.state.retriever)
    elif strategy == "hyde":
        engine = HyDERetriever(app.state.retriever,config)
    elif strategy == "hybrid":
        engine = HybridRetriever(app.state.retriever,app.state.bm25_retriever)
    else:
        raise ValueError(f"未知策略：{strategy}")

    if app.state.indexed_documents:
        engine.index(app.state.indexed_documents)

    cache[strategy] = engine
    return engine
######################
# 检索
######################
@app.post("/api/rag/retrieve",response_model=ResponseEnvelope)
def retrieve(request:RAGQuery):
    engine = _get_or_create_strategy(request.strategy)
    config = app.state.config

    original_threshold = config.similarity_threshold
    if request.similarity_threshold is not None:
        config.similarity_threshold = request.similarity_threshold

    try:
        results = engine.retrieve(request.query,top_k=request.top_k)
    finally:
        config.similarity_threshold = original_threshold

    source_items = [
        SourceItem(
            content=r.content,
            score=r.score,
            metadata=r.metadata,
            chunk_index=r.chunk_index,
        )
        for r in results
    ]

    data = RetrieveResult(
        query=request.query,
        total_found=len(results),
        sufficient=len(results) >= request.top_k,
        strategy_used=request.strategy,
        results=source_items,
    )
    return ResponseEnvelope(success=True, data=data)



#####################
# 端点：生成
####################
@app.post("/api/rag/generate",response_model=ResponseEnvelope)
def generate(request:GenerateRequest):
    genetor = Generator(app.state.config)
    context_chunks = request.context_chunks
    if request.system_prompt_extra:
        context_chunks = [f"[附加指令] {request.system_prompt_extra}"] + context_chunks
    sources = [
        RetrievedChunk(content=chunk,score=0.0,doc_id="",metadata={})
        for chunk in context_chunks
    ]
    answer,_ = genetor.generate(request.query,sources)
    data = GenerateResult(
        answer=answer,
        sources_used=len(sources),
        has_answer=bool(answer.strip())
    )
    return ResponseEnvelope(success=True, data=data)


###################
# 端点：一步到位（检索+生成）
###########################
@app.post("/api/rag/ask",response_model=ResponseEnvelope)
def ask(request:RAGQuery):
    engine = _get_or_create_strategy(request.strategy)
    config = app.state.config
    generator = Generator(config)

    original_threshold = config.similarity_threshold
    if request.similarity_threshold is not None:
        config.similarity_threshold = request.similarity_threshold

    try:
        results = engine.retrieve(request.query,top_k=request.top_k)
    finally:
        config.similarity_threshold = original_threshold

    if not results:
        data = AskResult(
            answer="根据提供的文档，无法回答此问题",
            sources=[],
            has_answer=False,
        )
        return ResponseEnvelope(success=True, data=data)
    answer,_ = generator.generate(request.query,results)
    source_items = [
        SourceItem(
            content=r.content,
            score=r.score,
            metadata=r.metadata,
            chunk_index=r.chunk_index,
        )
        for r in results
    ]

    data = AskResult(
        answer=answer,
        sources=source_items,
        has_answer=bool(answer.strip()),
    )
    return ResponseEnvelope(success=True, data=data)


##################
# 错误处理
###################
from minirag import EmbeddingError,RetrievalError,GenerationError,RAGError

@app.exception_handler(EmbeddingError)
@app.exception_handler(RetrievalError)
@app.exception_handler(GenerationError)
@app.exception_handler(RAGError)
def handle_rag_error(request,exc:RAGError):
    action = _determine_action(exc)
    api_error = APIError(
        code=exc.code or "RAG_ERROR",
        message=str(exc),
        action=action,
    )
    return JSONResponse(
        status_code=200,
        content=ResponseEnvelope(success=False, data=None,error=api_error).model_dump(),
    )

def _determine_action(exc:RAGError):
    if isinstance(exc,EmbeddingError):
        return "retry"
    if isinstance(exc,RetrievalError):
        if "索引为空" in str(exc):
            return "reindex"
        return "retry"
    if isinstance(exc,GenerationError):
        if exc.reason in ("network_error","api_error"):
            return "retry"
        if exc.reason in ("empty_error"):
            return "fallback"
        return "retry"
    return "give_up"

@app.exception_handler(Exception)
def handle_unexpected_error(request,exc:Exception):
    logger.error(f"未预期的错误：{exc}",exc_info=True)
    api_error = APIError(
        code="INTERNAL_ERROR",
        message=f"服务器内部错误:{type(exc).__name__}",
        action="give_up",
    )
    return JSONResponse(
        status_code=200,
        content=ResponseEnvelope(success=False, data=None,error=api_error).model_dump(),
    )

@app.post("/api/rag/evaluate")
def evaluate(request:EvaluateRequest):
    engine = _get_or_create_strategy(request.strategy)
    sources = engine.retrieve(request.question,top_k=request.top_k)
    context = "\n".join([s.content for s in sources]) if sources else ""
    if context:
        gen = Generator(app.state.config)
        answer,_ = gen.generate(request.question,sources)
    else:
        answer = ""
    evaluator = RAGASEvaluator()
    result = evaluator.evaluate(
        question=request.question,
        answer=answer,
        context=context,
        reference=request.reference,
    )
    result["strategy"] = request.strategy
    result["top_k"] = request.top_k
    return JSONResponse(
        content={"success":True,"data":result,"error":None},
    )