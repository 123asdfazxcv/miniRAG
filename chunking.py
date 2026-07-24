import re


class TextChunker:
    """文档切割器，支持固定大小切割和按句子边界切割。"""

    def __init__(self, chunk_size: int = 500, overlap: int = 20):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split_by_fixed_size(self, text: str) -> list[str]:
        """按固定大小切割，相邻 chunk 有 overlap 重叠。"""
        chunks = []
        start = 0
        step = self.chunk_size - self.overlap
        while start < len(text):
            chunks.append(text[start:start + self.chunk_size])
            start += step
        return chunks

    def split_by_sentences(self, text: str) -> list[str]:
        """按句子边界切割，尽量保持语义完整。"""
        sentences = re.split(r'(?<=[？。！\n])', text)
        current_chunk = ""
        chunks = []
        for sentence in sentences:
            if len(sentence) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                chunks.append(sentence.strip())
                continue
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = current_chunk[-self.overlap:] + sentence
            else:
                current_chunk += sentence
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks


if __name__ == "__main__":
    CORPUS = (
        "猫咪是世界上最受欢迎的宠物之一，它们独立而优雅。"
        "养猫需要准备猫砂盆、猫粮和猫抓板。"
        "布偶猫是一种长毛猫，性格温顺，适合家庭饲养。"
    )

    chunker = TextChunker(chunk_size=5, overlap=2)
    print("固定大小切割：")
    for i, c in enumerate(chunker.split_by_fixed_size("0123456789ABCDEFGHIJ")):
        print(f"  chunk{i}: '{c}'")

    print("\n" + "=" * 50)
    print("按句子切割：")
    for i, c in enumerate(chunker.split_by_sentences(CORPUS)):
        print(f"  chunk{i}: '{c}'")

