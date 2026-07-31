import math
import numpy as np
from dashscope import TextEmbedding
def cosine_Similarity(a:list[float],b:list[float])->float:
    dot = sum(x*y for x,y in zip(a,b))
    norm_a = math.sqrt(sum(i ** 2 for i in a))
    norm_b = math.sqrt(sum(i ** 2 for i in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
class Embedder:
    def __init__(self,model:str="text-embedding-v4"):
        self.model = model
    def embed(self,text:str,text_type:str="query")->np.ndarray:
        resp = TextEmbedding.call(
            model=self.model,
            input=text,
            text_type=text_type
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"Embedding API 调用失败！\n"
                f"  错误码：{resp.code}\n"
                f"  错误信息：{resp.message}\n"
                f"  可能原因：API key 无效/余额不足/网络不通"
            )
        embedding_list = resp.output["embeddings"][0]["embedding"]
        return np.array(embedding_list,dtype=np.float32)
    def embed_batch(self,texts:list[str],text_type:str="document")->np.ndarray:
        batch_size = self.max_batch_size
        all_vectors = []
        for start in range(0,len(texts),batch_size):
            batch = texts[start:start+batch_size]
            resp = TextEmbedding.call(
                model=self.model,
                input=batch,
                text_type=text_type
            )
            if resp.status_code != 200:
                raise RuntimeError(
                    f"Embedding API 批量调用失败！\n"
                    f"  错误码: {resp.code}\n"
                    f"  错误信息: {resp.message}"
                )
            embeddings = sorted(
                resp.output["embeddings"],
                key=lambda x :x["text_index"]
            )
            all_vectors.extend([e["embedding"] for e in embeddings])
        return np.array(all_vectors,dtype=np.float32)
    @property
    def dim(self)->int:
        dims={
            "text-embedding-v1":1536,
            "text-embedding-v2":1536,
            "text-embedding-v3":1024,
            "text-embedding-v4":1024
        }
        return dims.get(self.model,1536)
    @property
    def max_batch_size(self)->int:
        """不同模型的单次调用输入条数上限（未知模型取安全值 10）。"""
        limits = {
            "text-embedding-v1": 25,
            "text-embedding-v2": 25,
            "text-embedding-v3": 10,
            "text-embedding-v4": 10,
        }
        return limits.get(self.model, 10)
class semanticSearch:
    def __init__(self,embedder:Embedder):
        self.embedder = embedder
        self.documents:list[str] = []
        self.embeddings:np.ndarray | None = None
    def index(self,documents:list[str]):
        self.documents = documents
        self.embeddings = self.embedder.embed_batch(documents,text_type="document")
    def search(self,query:str,top_k:int = 3):
        q_vec = self.embedder.embed(query,text_type="query")
        q_norm = np.linalg.norm(q_vec)
        doc_norms = np.linalg.norm(self.embeddings,axis=1)
        dot_products = self.embeddings@q_vec
        scores = dot_products / (q_norm * doc_norms + 1e-10)
        results = []
        q_indices = np.argsort(scores)[::-1][:top_k]
        for i in q_indices:
            results.append((self.documents[i],float(scores[i])))
        return results