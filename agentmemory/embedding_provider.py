"""Embedding 提供者：将文本转为向量的抽象层。

内置 HashEmbeddingProvider（零依赖、确定性），以及可选的
OpenAIEmbeddingProvider 和 HuggingFaceEmbeddingProvider。
"""

from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Embedding 提供者抽象基类。

    实现此接口即可将任意文本转换为向量，供 HybridMemory 使用。
    """

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """将文本转为向量。

        Args:
            text: 输入文本

        Returns:
            向量表示（浮点数列表）
        """

    @abstractmethod
    def dimension(self) -> int:
        """返回向量维度。"""


class HashEmbeddingProvider(EmbeddingProvider):
    """基于哈希的确定性 Embedding 提供者。

    不需要任何外部依赖或 API key。适用于：
    - 开发/测试阶段的快速原型
    - 演示和集成测试
    - 不需要语义理解的关键词匹配场景

    使用 MurmurHash-like 算法将文本 token 映射到固定维度向量，
    相同文本始终产生相同向量，相似文本会产生相似（但不等同于语义相似）的向量。

    Args:
        dim: 向量维度，越大越精细但计算越慢。默认 128。
    """

    def __init__(self, dim: int = 128) -> None:
        if dim < 1:
            raise ValueError(f"维度必须 >= 1, got {dim}")
        self._dim = dim

    def embed(self, text: str) -> list[float]:
        """将文本转为确定性向量。

        算法：将文本分词，每个 token 通过 SHA-256 哈希映射到向量各维度，
        最终归一化为单位向量。

        Args:
            text: 输入文本

        Returns:
            长度为 dim 的归一化浮点向量
        """
        vec = [0.0] * self._dim
        tokens = text.lower().split()
        for token in tokens:
            h = hashlib.sha256(token.encode("utf-8")).digest()
            # 用哈希字节填充向量维度
            for i in range(self._dim):
                byte_val = h[i % len(h)]
                # 取多个哈希扩展
                ext = hashlib.sha256(f"{token}:{i}".encode("utf-8")).digest()
                combined = (byte_val ^ ext[0]) / 255.0 * 2.0 - 1.0
                vec[i] += combined

        # L2 归一化
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]

        return vec

    def dimension(self) -> int:
        """返回向量维度"""
        return self._dim


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """基于 OpenAI API 的 Embedding 提供者。

    需要安装 openai 库：`pip install openai`

    Args:
        model: 模型名称，默认 "text-embedding-3-small"
        api_key: OpenAI API key，也可以通过环境变量 OPENAI_API_KEY 设置
        base_url: 自定义 API 地址（用于兼容 API）
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        try:
            import openai  # noqa: F401
        except ImportError:
            raise ImportError(
                "使用 OpenAIEmbeddingProvider 需要安装 openai: pip install openai"
            )
        self._model = model
        kwargs: dict = {}
        if api_key is not None:
            kwargs["api_key"] = api_key
        if base_url is not None:
            kwargs["base_url"] = base_url
        import openai
        self._client = openai.OpenAI(**kwargs)
        self._dim: int | None = None

    def embed(self, text: str) -> list[float]:
        """调用 OpenAI API 获取文本 embedding。

        Args:
            text: 输入文本

        Returns:
            向量表示

        Raises:
            RuntimeError: API 调用失败
        """
        try:
            response = self._client.embeddings.create(
                model=self._model,
                input=text,
            )
            vector = response.data[0].embedding
            self._dim = len(vector)
            return list(vector)
        except Exception as e:
            raise RuntimeError(f"OpenAI embedding 调用失败: {e}") from e

    def dimension(self) -> int:
        """返回向量维度（首次 embed 后确定）。"""
        if self._dim is None:
            # 先调用一次以确定维度
            self.embed("dimension probe")
        return self._dim  # type: ignore[return-value]


class HuggingFaceEmbeddingProvider(EmbeddingProvider):
    """基于 HuggingFace sentence-transformers 的 Embedding 提供者。

    需要安装 sentence-transformers：`pip install sentence-transformers`

    Args:
        model: 模型名称，默认 "all-MiniLM-L6-v2"
    """

    def __init__(self, model: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer  # noqa: F401
        except ImportError:
            raise ImportError(
                "使用 HuggingFaceEmbeddingProvider 需要安装 sentence-transformers: "
                "pip install sentence-transformers"
            )
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model)
        self._dim: int | None = None

    def embed(self, text: str) -> list[float]:
        """使用 sentence-transformers 获取文本 embedding。

        Args:
            text: 输入文本

        Returns:
            向量表示
        """
        embedding = self._model.encode(text)
        vector = embedding.tolist()
        self._dim = len(vector)
        return vector

    def dimension(self) -> int:
        """返回向量维度。"""
        if self._dim is None:
            self.embed("dimension probe")
        return self._dim  # type: ignore[return-value]
