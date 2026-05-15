"""v1.0.0 扩展 Embedding Provider 测试：Cohere、Voyage AI、Remote。"""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from agentmemory.embedding_provider import EmbeddingProvider
from agentmemory.embedding_providers_ext import (
    COHERE_MODEL_DIMENSIONS,
    VOYAGE_MODEL_DIMENSIONS,
    CohereEmbeddingProvider,
    RemoteEmbeddingProvider,
    VoyageEmbeddingProvider,
)


# ============================================================
# 辅助工具
# ============================================================


def _ensure_module_missing(name: str):
    """确保指定模块在 sys.modules 中不存在，返回可用于 mock.patch 的 dict。"""
    saved = sys.modules.pop(name, None)
    return saved


def _restore_module(name: str, saved):
    """恢复被移除的模块。"""
    if saved is not None:
        sys.modules[name] = saved
    else:
        sys.modules.pop(name, None)


# ============================================================
# CohereEmbeddingProvider 测试
# ============================================================


class TestCohereEmbeddingProvider:
    """CohereEmbeddingProvider 单元测试"""

    def test_subclass_of_embedding_provider(self):
        """CohereEmbeddingProvider 是 EmbeddingProvider 的子类"""
        assert issubclass(CohereEmbeddingProvider, EmbeddingProvider)

    def test_raises_import_error_when_cohere_not_installed(self):
        """cohere 未安装时应抛出 ImportError 并包含安装提示"""
        saved = _ensure_module_missing("cohere")
        try:
            with patch.dict(sys.modules, {"cohere": None}):
                with pytest.raises(ImportError, match="cohere"):
                    CohereEmbeddingProvider(api_key="test-key")
        finally:
            _restore_module("cohere", saved)

    def test_import_error_message_contains_pip_install(self):
        """ImportError 消息应包含 pip install 指引"""
        saved = _ensure_module_missing("cohere")
        try:
            with patch.dict(sys.modules, {"cohere": None}):
                with pytest.raises(ImportError, match="pip install cohere"):
                    CohereEmbeddingProvider(api_key="test-key")
        finally:
            _restore_module("cohere", saved)

    def test_cohere_model_dimensions_dict_keys(self):
        """COHERE_MODEL_DIMENSIONS 应包含预期的模型键"""
        expected_keys = {
            "embed-english-v3.0",
            "embed-multilingual-v3.0",
            "embed-english-light-v3.0",
            "embed-multilingual-light-v3.0",
        }
        assert set(COHERE_MODEL_DIMENSIONS.keys()) == expected_keys

    def test_cohere_model_dimensions_dict_values(self):
        """COHERE_MODEL_DIMENSIONS 的维度值应正确"""
        assert COHERE_MODEL_DIMENSIONS["embed-english-v3.0"] == 1024
        assert COHERE_MODEL_DIMENSIONS["embed-multilingual-v3.0"] == 1024
        assert COHERE_MODEL_DIMENSIONS["embed-english-light-v3.0"] == 384
        assert COHERE_MODEL_DIMENSIONS["embed-multilingual-light-v3.0"] == 384

    def test_cohere_model_dimensions_full_models_are_1024(self):
        """完整版模型维度应为 1024"""
        for model in ("embed-english-v3.0", "embed-multilingual-v3.0"):
            assert COHERE_MODEL_DIMENSIONS[model] == 1024

    def test_cohere_model_dimensions_light_models_are_384(self):
        """轻量版模型维度应为 384"""
        for model in ("embed-english-light-v3.0", "embed-multilingual-light-v3.0"):
            assert COHERE_MODEL_DIMENSIONS[model] == 384

    def test_dimension_with_explicit_value(self):
        """手动指定 dimension 时应直接返回该值"""
        mock_cohere = ModuleType("cohere")
        mock_client = MagicMock()
        mock_cohere.Client = MagicMock(return_value=mock_client)
        saved = sys.modules.get("cohere")
        sys.modules["cohere"] = mock_cohere
        try:
            provider = CohereEmbeddingProvider(
                api_key="test-key", dimension=512
            )
            assert provider.dimension() == 512
        finally:
            _restore_module("cohere", saved)

    def test_dimension_from_model_mapping(self):
        """未手动指定 dimension 时应从模型映射表获取"""
        mock_cohere = ModuleType("cohere")
        mock_client = MagicMock()
        mock_cohere.Client = MagicMock(return_value=mock_client)
        saved = sys.modules.get("cohere")
        sys.modules["cohere"] = mock_cohere
        try:
            provider = CohereEmbeddingProvider(
                api_key="test-key", model="embed-english-v3.0"
            )
            assert provider.dimension() == 1024
        finally:
            _restore_module("cohere", saved)

    def test_dimension_from_model_mapping_light(self):
        """轻量模型维度从映射表获取应为 384"""
        mock_cohere = ModuleType("cohere")
        mock_client = MagicMock()
        mock_cohere.Client = MagicMock(return_value=mock_client)
        saved = sys.modules.get("cohere")
        sys.modules["cohere"] = mock_cohere
        try:
            provider = CohereEmbeddingProvider(
                api_key="test-key", model="embed-english-light-v3.0"
            )
            assert provider.dimension() == 384
        finally:
            _restore_module("cohere", saved)

    def test_embed_calls_client(self):
        """embed() 应调用 cohere Client.embed 并返回向量"""
        mock_cohere = ModuleType("cohere")
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1, 0.2, 0.3]]
        mock_client.embed.return_value = mock_response
        mock_cohere.Client = MagicMock(return_value=mock_client)
        saved = sys.modules.get("cohere")
        sys.modules["cohere"] = mock_cohere
        try:
            provider = CohereEmbeddingProvider(api_key="test-key")
            result = provider.embed("hello world")
            assert result == [0.1, 0.2, 0.3]
            mock_client.embed.assert_called_once()
        finally:
            _restore_module("cohere", saved)

    def test_embed_batch_calls_client(self):
        """embed_batch() 应调用 cohere Client.embed 并返回多条向量"""
        mock_cohere = ModuleType("cohere")
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1, 0.2], [0.3, 0.4]]
        mock_client.embed.return_value = mock_response
        mock_cohere.Client = MagicMock(return_value=mock_client)
        saved = sys.modules.get("cohere")
        sys.modules["cohere"] = mock_cohere
        try:
            provider = CohereEmbeddingProvider(api_key="test-key")
            result = provider.embed_batch(["hello", "world"])
            assert result == [[0.1, 0.2], [0.3, 0.4]]
        finally:
            _restore_module("cohere", saved)

    def test_embed_runtime_error_on_failure(self):
        """embed() 失败时应抛出 RuntimeError"""
        mock_cohere = ModuleType("cohere")
        mock_client = MagicMock()
        mock_client.embed.side_effect = Exception("API error")
        mock_cohere.Client = MagicMock(return_value=mock_client)
        saved = sys.modules.get("cohere")
        sys.modules["cohere"] = mock_cohere
        try:
            provider = CohereEmbeddingProvider(api_key="test-key")
            with pytest.raises(RuntimeError, match="Cohere embedding"):
                provider.embed("hello")
        finally:
            _restore_module("cohere", saved)

    def test_known_models_returns_correct_list(self):
        """COHERE_MODEL_DIMENSIONS 应包含 4 个已知模型"""
        models = list(COHERE_MODEL_DIMENSIONS.keys())
        assert len(models) == 4
        assert "embed-english-v3.0" in models
        assert "embed-multilingual-v3.0" in models


# ============================================================
# VoyageEmbeddingProvider 测试
# ============================================================


class TestVoyageEmbeddingProvider:
    """VoyageEmbeddingProvider 单元测试"""

    def test_subclass_of_embedding_provider(self):
        """VoyageEmbeddingProvider 是 EmbeddingProvider 的子类"""
        assert issubclass(VoyageEmbeddingProvider, EmbeddingProvider)

    def test_raises_import_error_when_voyageai_not_installed(self):
        """voyageai 未安装时应抛出 ImportError"""
        saved = _ensure_module_missing("voyageai")
        try:
            with patch.dict(sys.modules, {"voyageai": None}):
                with pytest.raises(ImportError, match="voyageai"):
                    VoyageEmbeddingProvider(api_key="test-key")
        finally:
            _restore_module("voyageai", saved)

    def test_import_error_message_contains_pip_install(self):
        """ImportError 消息应包含 pip install voyageai 指引"""
        saved = _ensure_module_missing("voyageai")
        try:
            with patch.dict(sys.modules, {"voyageai": None}):
                with pytest.raises(ImportError, match="pip install voyageai"):
                    VoyageEmbeddingProvider(api_key="test-key")
        finally:
            _restore_module("voyageai", saved)

    def test_voyage_model_dimensions_dict_keys(self):
        """VOYAGE_MODEL_DIMENSIONS 应包含预期的模型键"""
        expected_keys = {
            "voyage-3",
            "voyage-3-large",
            "voyage-3-lite",
            "voyage-code-3",
            "voyage-finance-2",
            "voyage-law-2",
        }
        assert set(VOYAGE_MODEL_DIMENSIONS.keys()) == expected_keys

    def test_voyage_model_dimensions_dict_values(self):
        """VOYAGE_MODEL_DIMENSIONS 的维度值应正确"""
        assert VOYAGE_MODEL_DIMENSIONS["voyage-3"] == 1024
        assert VOYAGE_MODEL_DIMENSIONS["voyage-3-large"] == 1024
        assert VOYAGE_MODEL_DIMENSIONS["voyage-3-lite"] == 512
        assert VOYAGE_MODEL_DIMENSIONS["voyage-code-3"] == 1024
        assert VOYAGE_MODEL_DIMENSIONS["voyage-finance-2"] == 1024
        assert VOYAGE_MODEL_DIMENSIONS["voyage-law-2"] == 1024

    def test_voyage_lite_model_is_512(self):
        """voyage-3-lite 模型维度应为 512"""
        assert VOYAGE_MODEL_DIMENSIONS["voyage-3-lite"] == 512

    def test_voyage_domain_models_are_1024(self):
        """领域专用模型维度应为 1024"""
        for model in ("voyage-finance-2", "voyage-law-2", "voyage-code-3"):
            assert VOYAGE_MODEL_DIMENSIONS[model] == 1024

    def test_dimension_with_explicit_value(self):
        """手动指定 dimension 时应直接返回该值"""
        mock_voyageai = ModuleType("voyageai")
        mock_client = MagicMock()
        mock_voyageai.Client = MagicMock(return_value=mock_client)
        saved = sys.modules.get("voyageai")
        sys.modules["voyageai"] = mock_voyageai
        try:
            provider = VoyageEmbeddingProvider(
                api_key="test-key", dimension=768
            )
            assert provider.dimension() == 768
        finally:
            _restore_module("voyageai", saved)

    def test_dimension_from_model_mapping(self):
        """未手动指定 dimension 时应从模型映射表获取"""
        mock_voyageai = ModuleType("voyageai")
        mock_client = MagicMock()
        mock_voyageai.Client = MagicMock(return_value=mock_client)
        saved = sys.modules.get("voyageai")
        sys.modules["voyageai"] = mock_voyageai
        try:
            provider = VoyageEmbeddingProvider(
                api_key="test-key", model="voyage-3"
            )
            assert provider.dimension() == 1024
        finally:
            _restore_module("voyageai", saved)

    def test_embed_calls_client(self):
        """embed() 应调用 voyageai Client.embed 并返回向量"""
        mock_voyageai = ModuleType("voyageai")
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.embeddings = [[0.5, 0.6, 0.7]]
        mock_client.embed.return_value = mock_result
        mock_voyageai.Client = MagicMock(return_value=mock_client)
        saved = sys.modules.get("voyageai")
        sys.modules["voyageai"] = mock_voyageai
        try:
            provider = VoyageEmbeddingProvider(api_key="test-key")
            result = provider.embed("test text")
            assert result == [0.5, 0.6, 0.7]
            mock_client.embed.assert_called_once()
        finally:
            _restore_module("voyageai", saved)

    def test_embed_batch_calls_client(self):
        """embed_batch() 应调用 voyageai Client.embed 并返回多条向量"""
        mock_voyageai = ModuleType("voyageai")
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.embeddings = [[0.1, 0.2], [0.3, 0.4]]
        mock_client.embed.return_value = mock_result
        mock_voyageai.Client = MagicMock(return_value=mock_client)
        saved = sys.modules.get("voyageai")
        sys.modules["voyageai"] = mock_voyageai
        try:
            provider = VoyageEmbeddingProvider(api_key="test-key")
            result = provider.embed_batch(["hello", "world"])
            assert result == [[0.1, 0.2], [0.3, 0.4]]
        finally:
            _restore_module("voyageai", saved)

    def test_known_models_returns_correct_count(self):
        """VOYAGE_MODEL_DIMENSIONS 应包含 6 个已知模型"""
        models = list(VOYAGE_MODEL_DIMENSIONS.keys())
        assert len(models) == 6


# ============================================================
# RemoteEmbeddingProvider 测试
# ============================================================


class TestRemoteEmbeddingProvider:
    """RemoteEmbeddingProvider 单元测试"""

    def test_subclass_of_embedding_provider(self):
        """RemoteEmbeddingProvider 是 EmbeddingProvider 的子类"""
        assert issubclass(RemoteEmbeddingProvider, EmbeddingProvider)

    def test_initialization_with_dimension(self):
        """初始化时可指定 dimension"""
        mock_httpx = ModuleType("httpx")
        saved = sys.modules.get("httpx")
        sys.modules["httpx"] = mock_httpx
        try:
            provider = RemoteEmbeddingProvider(
                endpoint="http://localhost:8080/embed",
                dimension=768,
            )
            assert provider.dimension() == 768
        finally:
            _restore_module("httpx", saved)

    def test_initialization_default_dimension(self):
        """未指定 dimension 时默认值为 384"""
        mock_httpx = ModuleType("httpx")
        saved = sys.modules.get("httpx")
        sys.modules["httpx"] = mock_httpx
        try:
            provider = RemoteEmbeddingProvider(
                endpoint="http://localhost:8080/embed"
            )
            assert provider.dimension() == 384
        finally:
            _restore_module("httpx", saved)

    def test_initialization_sets_auth_header(self):
        """提供 api_key 时应设置 Authorization 请求头"""
        mock_httpx = ModuleType("httpx")
        saved = sys.modules.get("httpx")
        sys.modules["httpx"] = mock_httpx
        try:
            provider = RemoteEmbeddingProvider(
                endpoint="http://localhost:8080/embed",
                api_key="my-secret-key",
            )
            assert provider._headers["Authorization"] == "Bearer my-secret-key"
        finally:
            _restore_module("httpx", saved)

    def test_initialization_no_auth_header_without_key(self):
        """未提供 api_key 时不应有 Authorization 请求头"""
        mock_httpx = ModuleType("httpx")
        saved = sys.modules.get("httpx")
        sys.modules["httpx"] = mock_httpx
        try:
            provider = RemoteEmbeddingProvider(
                endpoint="http://localhost:8080/embed"
            )
            assert "Authorization" not in provider._headers
        finally:
            _restore_module("httpx", saved)

    def test_raises_import_error_when_httpx_not_installed(self):
        """httpx 未安装时应抛出 ImportError"""
        saved = _ensure_module_missing("httpx")
        try:
            with patch.dict(sys.modules, {"httpx": None}):
                with pytest.raises(ImportError, match="httpx"):
                    RemoteEmbeddingProvider(endpoint="http://localhost/embed")
        finally:
            _restore_module("httpx", saved)

    def test_embed_via_mocked_httpx(self):
        """embed() 应通过 httpx POST 获取单条向量"""
        mock_response = MagicMock()
        mock_response.json.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}
        mock_response.raise_for_status = MagicMock()

        mock_client_instance = MagicMock()
        mock_client_instance.post.return_value = mock_response
        mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
        mock_client_instance.__exit__ = MagicMock(return_value=False)

        mock_httpx = ModuleType("httpx")
        mock_httpx.Client = MagicMock(return_value=mock_client_instance)
        mock_httpx.HTTPStatusError = type("HTTPStatusError", (Exception,), {})
        saved = sys.modules.get("httpx")
        sys.modules["httpx"] = mock_httpx
        try:
            provider = RemoteEmbeddingProvider(
                endpoint="http://localhost:8080/embed",
                model="test-model",
            )
            result = provider.embed("hello")
            assert result == [0.1, 0.2, 0.3]
            mock_client_instance.post.assert_called_once_with(
                "http://localhost:8080/embed",
                json={"texts": ["hello"], "model": "test-model"},
                headers=provider._headers,
            )
        finally:
            _restore_module("httpx", saved)

    def test_embed_batch_via_mocked_httpx(self):
        """embed_batch() 应通过 httpx POST 获取多条向量"""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "embeddings": [[0.1, 0.2], [0.3, 0.4]]
        }
        mock_response.raise_for_status = MagicMock()

        mock_client_instance = MagicMock()
        mock_client_instance.post.return_value = mock_response
        mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
        mock_client_instance.__exit__ = MagicMock(return_value=False)

        mock_httpx = ModuleType("httpx")
        mock_httpx.Client = MagicMock(return_value=mock_client_instance)
        mock_httpx.HTTPStatusError = type("HTTPStatusError", (Exception,), {})
        saved = sys.modules.get("httpx")
        sys.modules["httpx"] = mock_httpx
        try:
            provider = RemoteEmbeddingProvider(
                endpoint="http://localhost:8080/embed"
            )
            result = provider.embed_batch(["text1", "text2"])
            assert result == [[0.1, 0.2], [0.3, 0.4]]
        finally:
            _restore_module("httpx", saved)

    def test_embed_updates_dimension_from_response(self):
        """embed() 成功后应更新 dimension 为实际返回向量长度"""
        mock_response = MagicMock()
        mock_response.json.return_value = {"embeddings": [[1.0] * 768]}
        mock_response.raise_for_status = MagicMock()

        mock_client_instance = MagicMock()
        mock_client_instance.post.return_value = mock_response
        mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
        mock_client_instance.__exit__ = MagicMock(return_value=False)

        mock_httpx = ModuleType("httpx")
        mock_httpx.Client = MagicMock(return_value=mock_client_instance)
        mock_httpx.HTTPStatusError = type("HTTPStatusError", (Exception,), {})
        saved = sys.modules.get("httpx")
        sys.modules["httpx"] = mock_httpx
        try:
            provider = RemoteEmbeddingProvider(
                endpoint="http://localhost:8080/embed",
                dimension=384,
            )
            provider.embed("probe")
            assert provider.dimension() == 768
        finally:
            _restore_module("httpx", saved)

    def test_embed_runtime_error_on_http_failure(self):
        """embed() 遇到 HTTP 错误时应抛出 RuntimeError"""
        mock_httpx = ModuleType("httpx")
        mock_httpx.HTTPStatusError = type("HTTPStatusError", (Exception,), {})

        mock_response = MagicMock()
        mock_response.status_code = 500
        http_error = mock_httpx.HTTPStatusError()
        http_error.response = mock_response

        mock_client_instance = MagicMock()
        mock_client_instance.post.side_effect = http_error
        mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
        mock_client_instance.__exit__ = MagicMock(return_value=False)

        mock_httpx.Client = MagicMock(return_value=mock_client_instance)
        saved = sys.modules.get("httpx")
        sys.modules["httpx"] = mock_httpx
        try:
            provider = RemoteEmbeddingProvider(
                endpoint="http://localhost:8080/embed"
            )
            with pytest.raises(RuntimeError, match="远程 embedding"):
                provider.embed("hello")
        finally:
            _restore_module("httpx", saved)

    def test_embed_sends_model_in_payload(self):
        """embed() 请求体中应包含 model 字段"""
        mock_response = MagicMock()
        mock_response.json.return_value = {"embeddings": [[0.1]]}
        mock_response.raise_for_status = MagicMock()

        mock_client_instance = MagicMock()
        mock_client_instance.post.return_value = mock_response
        mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
        mock_client_instance.__exit__ = MagicMock(return_value=False)

        mock_httpx = ModuleType("httpx")
        mock_httpx.Client = MagicMock(return_value=mock_client_instance)
        mock_httpx.HTTPStatusError = type("HTTPStatusError", (Exception,), {})
        saved = sys.modules.get("httpx")
        sys.modules["httpx"] = mock_httpx
        try:
            provider = RemoteEmbeddingProvider(
                endpoint="http://localhost:8080/embed",
                model="custom-model",
            )
            provider.embed("test")
            call_args = mock_client_instance.post.call_args
            payload = call_args[1]["json"] if "json" in call_args[1] else call_args[0][1]
            # 位置参数或关键字参数
            if isinstance(payload, dict):
                assert payload["model"] == "custom-model"
        finally:
            _restore_module("httpx", saved)


# ============================================================
# 通用接口一致性测试
# ============================================================


class TestProviderInterfaceConsistency:
    """所有扩展 Provider 的接口一致性测试"""

    @pytest.mark.parametrize(
        "provider_class",
        [CohereEmbeddingProvider, VoyageEmbeddingProvider, RemoteEmbeddingProvider],
    )
    def test_all_providers_subclass_embedding_provider(self, provider_class):
        """所有扩展 Provider 均为 EmbeddingProvider 的子类"""
        assert issubclass(provider_class, EmbeddingProvider)

    @pytest.mark.parametrize(
        "provider_class",
        [CohereEmbeddingProvider, VoyageEmbeddingProvider],
    )
    def test_cohere_voyage_have_embed_method(self, provider_class):
        """Cohere/Voyage Provider 应有 embed 方法"""
        assert hasattr(provider_class, "embed")
        assert callable(getattr(provider_class, "embed"))

    @pytest.mark.parametrize(
        "provider_class",
        [CohereEmbeddingProvider, VoyageEmbeddingProvider],
    )
    def test_cohere_voyage_have_embed_batch_method(self, provider_class):
        """Cohere/Voyage Provider 应有 embed_batch 方法"""
        assert hasattr(provider_class, "embed_batch")
        assert callable(getattr(provider_class, "embed_batch"))

    @pytest.mark.parametrize(
        "provider_class",
        [CohereEmbeddingProvider, VoyageEmbeddingProvider, RemoteEmbeddingProvider],
    )
    def test_all_providers_have_dimension_method(self, provider_class):
        """所有 Provider 应有 dimension 方法"""
        assert hasattr(provider_class, "dimension")
        assert callable(getattr(provider_class, "dimension"))

    @pytest.mark.parametrize(
        "provider_class",
        [CohereEmbeddingProvider, VoyageEmbeddingProvider, RemoteEmbeddingProvider],
    )
    def test_all_providers_have_init(self, provider_class):
        """所有 Provider 应有 __init__ 方法"""
        assert hasattr(provider_class, "__init__")
