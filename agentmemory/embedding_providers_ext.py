"""扩展 Embedding 提供者：Cohere、Voyage AI 及通用远程端点。

提供 CohereEmbeddingProvider、VoyageEmbeddingProvider 和 RemoteEmbeddingProvider，
均继承自 EmbeddingProvider 抽象基类。
"""

from __future__ import annotations

from typing import Optional

from .embedding_provider import EmbeddingProvider


# ---------------------------------------------------------------------------
# Cohere Embedding Provider
# ---------------------------------------------------------------------------

# Cohere 已知模型 → 向量维度映射
COHERE_MODEL_DIMENSIONS: dict[str, int] = {
    "embed-english-v3.0": 1024,
    "embed-multilingual-v3.0": 1024,
    "embed-english-light-v3.0": 384,
    "embed-multilingual-light-v3.0": 384,
}


class CohereEmbeddingProvider(EmbeddingProvider):
    """基于 Cohere API 的 Embedding 提供者。

    需要安装 cohere 库：``pip install cohere``

    Args:
        api_key: Cohere API key
        model: 模型名称，默认 ``embed-english-v3.0``
        input_type: 输入类型，可选 ``search_document``、``search_query``、
            ``classification``、``clustering`` 等
        dimension: 手动指定向量维度；为 ``None`` 时根据模型自动推断
    """

    def __init__(
        self,
        api_key: str,
        model: str = "embed-english-v3.0",
        input_type: str = "search_document",
        dimension: Optional[int] = None,
    ) -> None:
        try:
            import cohere  # noqa: F401
        except ImportError:
            raise ImportError(
                "使用 CohereEmbeddingProvider 需要安装 cohere: pip install cohere"
            )
        import cohere

        self._client = cohere.Client(api_key=api_key)
        self._model = model
        self._input_type = input_type
        self._dim: int | None = dimension

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def embed(self, text: str) -> list[float]:
        """调用 Cohere API 获取单条文本的 embedding。

        Args:
            text: 输入文本

        Returns:
            向量表示（浮点数列表）

        Raises:
            RuntimeError: API 调用失败
        """
        try:
            response = self._client.embed(
                texts=[text],
                model=self._model,
                input_type=self._input_type,
            )
            vector: list[float] = list(response.embeddings[0])
            self._dim = len(vector)
            return vector
        except Exception as e:
            raise RuntimeError(f"Cohere embedding 调用失败: {e}") from e

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """调用 Cohere API 批量获取 embedding。

        Args:
            texts: 输入文本列表

        Returns:
            与输入等长的向量列表

        Raises:
            RuntimeError: API 调用失败
        """
        try:
            response = self._client.embed(
                texts=texts,
                model=self._model,
                input_type=self._input_type,
            )
            embeddings: list[list[float]] = [list(e) for e in response.embeddings]
            if embeddings:
                self._dim = len(embeddings[0])
            return embeddings
        except Exception as e:
            raise RuntimeError(f"Cohere embedding 批量调用失败: {e}") from e

    def dimension(self) -> int:
        """返回向量维度。

        优先使用手动指定值或已知模型映射；若均不可用则通过一次探测调用获取。
        """
        if self._dim is not None:
            return self._dim
        if self._model in COHERE_MODEL_DIMENSIONS:
            return COHERE_MODEL_DIMENSIONS[self._model]
        # 探测
        self.embed("dimension probe")
        return self._dim  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Voyage AI Embedding Provider
# ---------------------------------------------------------------------------

# Voyage 已知模型 → 向量维度映射
VOYAGE_MODEL_DIMENSIONS: dict[str, int] = {
    "voyage-3": 1024,
    "voyage-3-large": 1024,
    "voyage-3-lite": 512,
    "voyage-code-3": 1024,
    "voyage-finance-2": 1024,
    "voyage-law-2": 1024,
}


class VoyageEmbeddingProvider(EmbeddingProvider):
    """基于 Voyage AI API 的 Embedding 提供者。

    需要安装 voyageai 库：``pip install voyageai``

    Args:
        api_key: Voyage AI API key
        model: 模型名称，默认 ``voyage-3``
        dimension: 手动指定向量维度；为 ``None`` 时根据模型自动推断
    """

    def __init__(
        self,
        api_key: str,
        model: str = "voyage-3",
        dimension: Optional[int] = None,
    ) -> None:
        try:
            import voyageai  # noqa: F401
        except ImportError:
            raise ImportError(
                "使用 VoyageEmbeddingProvider 需要安装 voyageai: pip install voyageai"
            )
        import voyageai

        self._client = voyageai.Client(api_key=api_key)
        self._model = model
        self._dim: int | None = dimension

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def embed(self, text: str) -> list[float]:
        """调用 Voyage AI API 获取单条文本的 embedding。

        Args:
            text: 输入文本

        Returns:
            向量表示（浮点数列表）

        Raises:
            RuntimeError: API 调用失败
        """
        try:
            result = self._client.embed([text], model=self._model)
            vector: list[float] = list(result.embeddings[0])
            self._dim = len(vector)
            return vector
        except Exception as e:
            raise RuntimeError(f"Voyage embedding 调用失败: {e}") from e

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """调用 Voyage AI API 批量获取 embedding。

        Args:
            texts: 输入文本列表

        Returns:
            与输入等长的向量列表

        Raises:
            RuntimeError: API 调用失败
        """
        try:
            result = self._client.embed(texts, model=self._model)
            embeddings: list[list[float]] = [list(e) for e in result.embeddings]
            if embeddings:
                self._dim = len(embeddings[0])
            return embeddings
        except Exception as e:
            raise RuntimeError(f"Voyage embedding 批量调用失败: {e}") from e

    def dimension(self) -> int:
        """返回向量维度。

        优先使用手动指定值或已知模型映射；若均不可用则通过一次探测调用获取。
        """
        if self._dim is not None:
            return self._dim
        if self._model in VOYAGE_MODEL_DIMENSIONS:
            return VOYAGE_MODEL_DIMENSIONS[self._model]
        self.embed("dimension probe")
        return self._dim  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Remote / Generic HTTP Embedding Provider
# ---------------------------------------------------------------------------


class RemoteEmbeddingProvider(EmbeddingProvider):
    """通用远程 Embedding 端点提供者。

    通过 HTTP POST 调用任意兼容的 embedding 服务。
    请求体格式：``{"texts": [...], "model": "..."}``
    响应体格式：``{"embeddings": [[...], ...]}``

    需要安装 httpx 库：``pip install httpx``

    Args:
        endpoint: Embedding 服务的 URL 地址
        api_key: 可选的 API key，作为 ``Authorization: Bearer <key>`` 请求头发送
        model: 模型标识符，默认 ``default``
        dimension: 向量维度，默认 384
    """

    def __init__(
        self,
        endpoint: str,
        api_key: Optional[str] = None,
        model: str = "default",
        dimension: int = 384,
    ) -> None:
        try:
            import httpx  # noqa: F401
        except ImportError:
            raise ImportError(
                "使用 RemoteEmbeddingProvider 需要安装 httpx: pip install httpx"
            )
        self._endpoint = endpoint
        self._model = model
        self._dim = dimension
        self._headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key is not None:
            self._headers["Authorization"] = f"Bearer {api_key}"

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _post(self, texts: list[str]) -> list[list[float]]:
        """发送 POST 请求并返回 embedding 列表。"""
        import httpx

        payload = {"texts": texts, "model": self._model}
        try:
            with httpx.Client(timeout=60.0) as client:
                resp = client.post(
                    self._endpoint,
                    json=payload,
                    headers=self._headers,
                )
                resp.raise_for_status()
                data = resp.json()
            embeddings = data["embeddings"]
            if embeddings:
                self._dim = len(embeddings[0])
            return [list(e) for e in embeddings]
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"远程 embedding 服务返回错误 {e.response.status_code}: {e}"
            ) from e
        except Exception as e:
            raise RuntimeError(f"远程 embedding 调用失败: {e}") from e

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def embed(self, text: str) -> list[float]:
        """通过远程端点获取单条文本的 embedding。

        Args:
            text: 输入文本

        Returns:
            向量表示（浮点数列表）
        """
        embeddings = self._post([text])
        return embeddings[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """通过远程端点批量获取 embedding。

        Args:
            texts: 输入文本列表

        Returns:
            与输入等长的向量列表
        """
        return self._post(texts)

    def dimension(self) -> int:
        """返回向量维度。"""
        return self._dim
