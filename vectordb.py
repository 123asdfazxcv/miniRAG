import chromadb
import faiss
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")
from embedding import Embedder


class FaissIndex:
    """FAISS 向量索引，基于内积相似度（IndexFlatIP）。"""

    def __init__(self, embedder: Embedder, dim: int = 1536):
        self.embedder = embedder
        self.dim = dim
        self.documents = []
        self.index = faiss.IndexFlatIP(self.dim)

    def add_documents(self, documents: list[str]):
        self.documents.extend(documents)
        embeddings = self.embedder.embed_batch(documents, text_type="document")
        embeddings = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        return len(self.documents)

    def search(self, query: str, top_k: int = 3):
        q_vec = self.embedder.embed(query, text_type="query")
        q_vec = np.array([q_vec], dtype=np.float32)
        faiss.normalize_L2(q_vec)
        scores, indices = self.index.search(q_vec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                results.append((self.documents[idx], float(score)))
        return results


class ChromaVectorStore:
    """ChromaDB 向量存储，基于余弦距离。"""

    def __init__(self, embedder: Embedder, collection_name: str = "rag_doc"):
        self.embedder = embedder
        self.client = chromadb.Client()
        try:
            self.collection = self.client.get_collection(collection_name)
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )

    def add_documents(self, documents: list[str], metadatas: list[dict] = None):
        embeddings = self.embedder.embed_batch(documents, text_type="document")
        ids = [f"{i + self.collection.count()}" for i in range(len(documents))]
        if metadatas is None:
            metadatas = [{"source": "unknown"} for _ in range(len(documents))]
        self.collection.add(
            embeddings=embeddings, documents=documents, ids=ids, metadatas=metadatas
        )
        return len(documents)

    def search(self, query: str, top_k: int = 3):
        q_vec = self.embedder.embed(query, text_type="query")
        result = self.collection.query(query_embeddings=[q_vec], n_results=top_k)
        results = []
        for doc_id, doc_text, distance in zip(
            result["ids"][0], result["documents"][0], result["distances"][0]
        ):
            results.append((doc_text, 1 - distance))
        return results


def compare_faiss_vs_chromadb(documents: list[str], query: str):
    """FAISS 和 ChromaDB 对比测试。"""
    embedder = Embedder()
    faiss_idx = FaissIndex(embedder)
    faiss_idx.add_documents(documents)
    faiss_results = faiss_idx.search(query)

    chroma_store = ChromaVectorStore(embedder)
    chroma_store.add_documents(documents)
    chroma_results = chroma_store.search(query)

    print("===== FAISS =====")
    for doc, score in faiss_results:
        print(f"[{score:.3f}] {doc[:50]}")
    print("===== ChromaDB =====")
    for doc, score in chroma_results:
        print(f"[{score:.3f}] {doc[:50]}")


if __name__ == "__main__":
    documents = [
        "猫咪是世界上最受欢迎的宠物之一，它们独立而优雅。",
        "布偶猫以其温顺的性格和蓝色的眼睛而闻名，是理想的家庭宠物。",
        "猫咪的胡须是非常敏感的触觉器官，能帮助它们在黑暗中感知周围环境。",
        "英短猫体型圆润，性格沉稳，是非常受欢迎的品种猫。",
        "狗狗是人类最忠诚的朋友，它们需要每天遛弯和充足的运动量。",
        "金毛寻回犬性格温顺，智商高，常被训练为导盲犬或搜救犬。",
        "训练狗狗需要耐心和正向激励，零食奖励比惩罚更有效。",
        "太阳系有八大行星，其中木星是体积最大的，土星拥有美丽的光环。",
        "黑洞是引力极强的天体，连光都无法逃脱，它们由大质量恒星坍缩形成。",
        "银河系是一个棒旋星系，直径约10万光年，包含上千亿颗恒星。",
        "火锅是中国人最爱的聚餐方式之一，麻辣锅底和清汤锅底各有拥趸。",
        "意大利面种类繁多，常见的有意式肉酱面、奶油培根面和青酱面。",
        "寿司起源于日本，以醋饭搭配生鱼片、海鲜或蔬菜，讲究食材新鲜度。",
    ]
    compare_faiss_vs_chromadb(documents, "宠物")
