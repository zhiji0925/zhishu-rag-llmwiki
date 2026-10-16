"""Embedding 客户端

支持任意 OpenAI 兼容接口（火山方舟 Seed1.5-Embedding / DashScope / vLLM 私有部署等），
以及无 Key 时打通链路用的确定性 mock。查询侧可配置指令前缀（Seed1.5 官方推荐）。
"""
import hashlib
import math
import time

from config import SETTINGS


class MockEmbedder:
    """确定性哈希向量：无语义能力，仅用于无 Key 时验证链路"""

    def __init__(self, dim: int):
        self.dim = dim

    def _vec(self, text: str):
        v = [0.0] * self.dim
        for i in range(0, len(text), 4):
            h = int(hashlib.md5(text[i:i + 4].encode("utf-8")).hexdigest()[:8], 16)
            v[h % self.dim] += (h % 7) - 3
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, q: str):
        return self._vec(q)


class OpenAICompatEmbedder:
    def __init__(self, settings=SETTINGS, batch_size=10):
        from openai import OpenAI
        if not settings.embedding_api_key:
            raise RuntimeError("EMBEDDING_API_KEY 未配置，请检查 .env")
        self.s = settings
        self.batch = batch_size
        self.client = OpenAI(base_url=settings.embedding_base_url,
                             api_key=settings.embedding_api_key,
                             timeout=60, max_retries=2)

    def _call(self, texts, with_dim=True):
        kwargs = dict(model=self.s.embedding_model, input=texts, encoding_format="float")
        if with_dim:
            kwargs["dimensions"] = self.s.embedding_dim
        try:
            return [d.embedding for d in self.client.embeddings.create(**kwargs).data]
        except Exception as e:
            # 端点不支持 dimensions 参数时降级：请求全量向量后本地 MRL 截断
            if with_dim and ("dimension" in str(e).lower() or "400" in str(e)):
                return self._call(texts, with_dim=False)
            raise

    def _post(self, vec):
        dim = self.s.embedding_dim
        if len(vec) > dim:  # MRL 截断
            vec = vec[:dim]
        n = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / n for x in vec]  # L2 归一化，配合 cosine 检索

    def embed_documents(self, texts):
        out = []
        for i in range(0, len(texts), self.batch):
            batch = texts[i:i + self.batch]
            for attempt in range(4):
                try:
                    vecs = self._call(batch)
                    break
                except Exception as e:
                    if attempt == 3:
                        raise
                    wait = 3 * (attempt + 1)
                    print(f"  embedding 重试 {attempt + 1}/3（等待{wait}s）: {str(e)[:120]}")
                    time.sleep(wait)
            out.extend(self._post(v) for v in vecs)
            if (i // self.batch) % 5 == 0 or i + self.batch >= len(texts):
                print(f"  embedding 进度 {min(i + self.batch, len(texts))}/{len(texts)}")
        return out

    def embed_query(self, q: str):
        ins = self.s.embedding_query_instruct
        text = f"{ins} {q}".strip() if ins else q
        return self._post(self._call([text])[0])


def get_embedder(settings=SETTINGS):
    if settings.embedding_provider == "mock":
        return MockEmbedder(settings.embedding_dim)
    return OpenAICompatEmbedder(settings)
