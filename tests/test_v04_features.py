"""v0.4.0 新功能测试 — async API、缓存、过滤器、session。"""

import asyncio
import time
import tempfile
import os

import pytest

from agentmemory import (
    HybridMemory,
    HashEmbeddingProvider,
    CachedEmbeddingProvider,
    SearchFilter,
    filter_search_results,
    AsyncHybridMemory,
    MemorySession,
)
from agentmemory.search_filter import SearchFilter


# ============================================================
# CachedEmbeddingProvider 测试
# ============================================================


class TestCachedEmbeddingProvider:
    """CachedEmbeddingProvider LRU 缓存测试"""

    def _make_provider(self, max_cache_size: int = 128):
        inner = HashEmbeddingProvider(dim=64)
        return CachedEmbeddingProvider(inner, max_cache_size=max_cache_size)

    def test_basic_embed(self):
        """基本嵌入计算"""
        prov = self._make_provider()
        vec = prov.embed("hello world")
        assert len(vec) == 64
        assert all(isinstance(v, float) for v in vec)

    def test_cache_hit(self):
        """相同文本返回缓存结果"""
        prov = self._make_provider()
        vec1 = prov.embed("test text")
        vec2 = prov.embed("test text")
        assert vec1 == vec2
        assert prov.cache_stats["hits"] == 1
        assert prov.cache_stats["misses"] == 1

    def test_cache_miss_different_text(self):
        """不同文本产生缓存未命中"""
        prov = self._make_provider()
        prov.embed("text A")
        prov.embed("text B")
        assert prov.cache_stats["misses"] == 2
        assert prov.cache_stats["hits"] == 0

    def test_cache_eviction(self):
        """缓存满时淘汰最久未使用的"""
        prov = self._make_provider(max_cache_size=3)
        prov.embed("a")
        prov.embed("b")
        prov.embed("c")
        assert prov.cache_stats["size"] == 3

        # 添加新条目应淘汰 "a"
        prov.embed("d")
        assert prov.cache_stats["size"] == 3

        # "a" 被淘汰，再次计算应是 miss
        prov.embed("a")
        assert prov.cache_stats["misses"] == 5  # a,b,c,d + a again

    def test_lru_ordering(self):
        """访问后重新排列 LRU 顺序"""
        prov = self._make_provider(max_cache_size=3)
        prov.embed("a")
        prov.embed("b")
        prov.embed("c")
        # 访问 "a" 使其变为最近使用
        prov.embed("a")
        # 添加 "d" 应淘汰 "b"（最久未使用）
        prov.embed("d")
        # "b" 被淘汰
        stats = prov.cache_stats
        assert stats["size"] == 3

    def test_long_text_uses_hash(self):
        """长文本使用 sha256 作为缓存键"""
        prov = self._make_provider()
        long_text = "x" * 500
        vec = prov.embed(long_text)
        assert len(vec) == 64
        # 再次调用应命中缓存
        vec2 = prov.embed(long_text)
        assert vec == vec2
        assert prov.cache_stats["hits"] == 1

    def test_dimension(self):
        """维度传递正确"""
        prov = self._make_provider()
        assert prov.dimension() == 64

    def test_clear_cache(self):
        """清空缓存重置统计"""
        prov = self._make_provider()
        prov.embed("test")
        prov.clear_cache()
        stats = prov.cache_stats
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["size"] == 0

    def test_hit_rate(self):
        """命中率计算正确"""
        prov = self._make_provider()
        prov.embed("a")
        prov.embed("a")
        prov.embed("a")
        assert prov.cache_stats["hit_rate"] == pytest.approx(2 / 3, abs=0.001)

    def test_repr(self):
        """repr 输出"""
        prov = self._make_provider()
        r = repr(prov)
        assert "CachedEmbeddingProvider" in r


# ============================================================
# SearchFilter 测试
# ============================================================


class TestSearchFilter:
    """SearchFilter 过滤器测试"""

    def _make_memory(self, content="test", metadata=None, tags=None, created_at=None):
        from agentmemory.models import Memory
        mem = Memory(
            content=content,
            embedding=[0.1] * 64,
            metadata=metadata or {},
            tags=tags or [],
        )
        if created_at is not None:
            mem.created_at = created_at
        return mem

    def test_metadata_filter_match(self):
        """metadata 键值匹配"""
        f = SearchFilter(metadata_filters={"type": "note"})
        mem = self._make_memory(metadata={"type": "note", "color": "blue"})
        assert f.matches(mem) is True

    def test_metadata_filter_no_match(self):
        """metadata 键值不匹配"""
        f = SearchFilter(metadata_filters={"type": "note"})
        mem = self._make_memory(metadata={"type": "task"})
        assert f.matches(mem) is False

    def test_metadata_filter_missing_key(self):
        """metadata 缺少过滤键"""
        f = SearchFilter(metadata_filters={"type": "note"})
        mem = self._make_memory(metadata={"color": "blue"})
        assert f.matches(mem) is False

    def test_time_range_filter(self):
        """时间范围过滤"""
        now = time.time()
        f = SearchFilter(created_after=now - 100, created_before=now + 100)
        mem = self._make_memory(created_at=now)
        assert f.matches(mem) is True

    def test_time_range_filter_too_old(self):
        """记忆早于时间下限"""
        now = time.time()
        f = SearchFilter(created_after=now + 1)
        mem = self._make_memory(created_at=now)
        assert f.matches(mem) is False

    def test_time_range_filter_too_new(self):
        """记忆晚于时间上限"""
        now = time.time()
        f = SearchFilter(created_before=now - 1)
        mem = self._make_memory(created_at=now)
        assert f.matches(mem) is False

    def test_tags_filter_match(self):
        """标签过滤匹配"""
        f = SearchFilter(tags=["python", "coding"])
        mem = self._make_memory(tags=["Python", "coding", "dev"])
        assert f.matches(mem) is True

    def test_tags_filter_missing_tag(self):
        """缺少标签"""
        f = SearchFilter(tags=["python", "rust"])
        mem = self._make_memory(tags=["python", "coding"])
        assert f.matches(mem) is False

    def test_exclude_tags(self):
        """排除标签"""
        f = SearchFilter(exclude_tags=["archived"])
        mem = self._make_memory(tags=["archived", "old"])
        assert f.matches(mem) is False

    def test_exclude_tags_no_match(self):
        """排除标签不匹配（应通过）"""
        f = SearchFilter(exclude_tags=["archived"])
        mem = self._make_memory(tags=["active"])
        assert f.matches(mem) is True

    def test_content_contains(self):
        """内容包含过滤"""
        f = SearchFilter(content_contains=["python"])
        mem = self._make_memory(content="I love Python programming")
        assert f.matches(mem) is True

    def test_content_contains_no_match(self):
        """内容不包含"""
        f = SearchFilter(content_contains=["rust"])
        mem = self._make_memory(content="I love Python programming")
        assert f.matches(mem) is False

    def test_content_contains_any_match(self):
        """内容包含任一子串"""
        f = SearchFilter(content_contains=["rust", "python"])
        mem = self._make_memory(content="I love Python")
        assert f.matches(mem) is True

    def test_content_not_contains(self):
        """内容排除过滤"""
        f = SearchFilter(content_not_contains=["spam"])
        mem = self._make_memory(content="normal content")
        assert f.matches(mem) is True

    def test_content_not_contains_match(self):
        """内容包含排除词"""
        f = SearchFilter(content_not_contains=["spam"])
        mem = self._make_memory(content="this is spam")
        assert f.matches(mem) is False

    def test_combined_filters(self):
        """组合多个过滤条件"""
        f = SearchFilter(
            metadata_filters={"source": "api"},
            tags=["important"],
            content_contains=["report"],
            created_after=time.time() - 1000,
        )
        mem = self._make_memory(
            content="Q4 report summary",
            metadata={"source": "api"},
            tags=["important", "quarterly"],
        )
        assert f.matches(mem) is True

    def test_combined_filters_partial_fail(self):
        """组合条件中一个不满足"""
        f = SearchFilter(
            metadata_filters={"source": "api"},
            tags=["nonexistent"],
        )
        mem = self._make_memory(
            content="test",
            metadata={"source": "api"},
            tags=["important"],
        )
        assert f.matches(mem) is False

    def test_empty_filter_matches_all(self):
        """空过滤器匹配所有"""
        f = SearchFilter()
        mem = self._make_memory(content="anything", tags=["any"], metadata={"k": "v"})
        assert f.matches(mem) is True


class TestFilterSearchResults:
    """filter_search_results 函数测试"""

    def test_filter_results(self):
        """过滤搜索结果列表"""
        from agentmemory.models import Memory, SearchResult

        mems = [
            Memory(content="python guide", embedding=[0.1] * 64, tags=["python"]),
            Memory(content="rust tutorial", embedding=[0.2] * 64, tags=["rust"]),
            Memory(content="python tips", embedding=[0.3] * 64, tags=["python"]),
        ]
        results = [SearchResult(memory=m, score=0.9) for m in mems]

        f = SearchFilter(tags=["python"])
        filtered = filter_search_results(results, f)
        assert len(filtered) == 2
        assert all("python" in r.memory.tags for r in filtered)

    def test_filter_preserves_order(self):
        """过滤保持原有排序"""
        from agentmemory.models import Memory, SearchResult

        mems = [
            Memory(content="a", embedding=[0.1] * 64, tags=["x"]),
            Memory(content="b", embedding=[0.2] * 64, tags=["y"]),
            Memory(content="c", embedding=[0.3] * 64, tags=["x"]),
        ]
        results = [SearchResult(memory=m, score=0.9 - i * 0.1) for i, m in enumerate(mems)]

        f = SearchFilter(tags=["x"])
        filtered = filter_search_results(results, f)
        assert len(filtered) == 2
        assert filtered[0].memory.content == "a"
        assert filtered[1].memory.content == "c"


# ============================================================
# MemorySession 上下文管理器测试
# ============================================================


class TestMemorySession:
    """MemorySession 上下文管理器测试"""

    def test_session_basic(self):
        """基本 session 使用"""
        hm = HybridMemory(dimension=64, embedding_provider=HashEmbeddingProvider(dim=64))

        with hm.session() as s:
            s.remember("hello")
            s.remember("world")
            assert s.operations_count == 2

    def test_session_search(self):
        """session 中搜索"""
        hm = HybridMemory(dimension=64, embedding_provider=HashEmbeddingProvider(dim=64))

        with hm.session() as s:
            s.remember("Python is great")
            s.remember("Rust is fast")
            results = s.search_text("Python")
            assert len(results) > 0

    def test_session_forget(self):
        """session 中删除"""
        hm = HybridMemory(dimension=64, embedding_provider=HashEmbeddingProvider(dim=64))

        with hm.session() as s:
            mem = s.remember("to be deleted")
            s.forget(mem.id)
            assert s.operations_count == 2

    def test_session_stats(self):
        """session stats 包含操作计数"""
        hm = HybridMemory(dimension=64, embedding_provider=HashEmbeddingProvider(dim=64))

        with hm.session() as s:
            s.remember("a")
            s.remember("b")
            st = s.stats()
            assert st["session_operations"] == 2
            assert st["memory_count"] == 2

    def test_session_auto_save(self):
        """session 退出时自动保存"""
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "test_session")
            hm = HybridMemory(
                dimension=64,
                embedding_provider=HashEmbeddingProvider(dim=64),
                storage_path=path,
            )

            with hm.session() as s:
                s.remember("persistent data")

            # 验证数据已保存
            hm2 = HybridMemory(
                dimension=64,
                embedding_provider=HashEmbeddingProvider(dim=64),
                storage_path=path,
                auto_load=True,
            )
            assert hm2.embedding_store.count() == 1

    def test_session_no_backend(self):
        """无后端时 session 也能正常工作"""
        hm = HybridMemory(dimension=64, embedding_provider=HashEmbeddingProvider(dim=64))

        with hm.session() as s:
            s.remember("no persistence")
            assert s.operations_count == 1


# ============================================================
# HybridMemory 全局过滤器测试
# ============================================================


class TestDefaultFilter:
    """HybridMemory 全局搜索过滤器测试"""

    def test_set_default_filter(self):
        """设置默认过滤器"""
        hm = HybridMemory(dimension=64, embedding_provider=HashEmbeddingProvider(dim=64))
        f = SearchFilter(tags=["important"])
        hm.set_default_filter(f)
        assert hm._default_filter is f

    def test_default_filter_applies_to_search(self):
        """默认过滤器应用到 search"""
        hm = HybridMemory(dimension=64, embedding_provider=HashEmbeddingProvider(dim=64))
        hm.remember("important note", tags=["important"])
        hm.remember("regular note", tags=["regular"])

        hm.set_default_filter(SearchFilter(tags=["important"]))
        results = hm.search_text("note")
        assert all("important" in r.memory.tags for r in results)

    def test_clear_default_filter(self):
        """清除默认过滤器"""
        hm = HybridMemory(dimension=64, embedding_provider=HashEmbeddingProvider(dim=64))
        hm.remember("a", tags=["x"])
        hm.remember("b", tags=["y"])

        hm.set_default_filter(SearchFilter(tags=["x"]))
        assert len(hm.search_text("a")) <= 1

        hm.set_default_filter(None)
        results = hm.search_text("a")
        # 不再有过滤，应该返回所有
        assert len(results) >= 1


# ============================================================
# AsyncHybridMemory 测试
# ============================================================


class TestAsyncHybridMemory:
    """异步 API 测试"""

    def _make_async_memory(self):
        hm = HybridMemory(dimension=64, embedding_provider=HashEmbeddingProvider(dim=64))
        return AsyncHybridMemory(hm, max_workers=2)

    def test_async_remember(self):
        """异步添加记忆"""
        am = self._make_async_memory()

        async def run():
            mem = await am.aremember("async hello")
            assert mem.content == "async hello"
            return mem

        asyncio.run(run())

    def test_async_search(self):
        """异步搜索"""
        am = self._make_async_memory()

        async def run():
            await am.aremember("Python programming")
            await am.aremember("Rust systems")
            results = await am.asearch_text("Python")
            assert len(results) > 0

        asyncio.run(run())

    def test_async_batch_remember(self):
        """异步批量添加"""
        am = self._make_async_memory()

        async def run():
            memories = await am.abatch_remember(["a", "b", "c"])
            assert len(memories) == 3

        asyncio.run(run())

    def test_async_forget(self):
        """异步删除"""
        am = self._make_async_memory()

        async def run():
            mem = await am.aremember("to delete")
            await am.aforget(mem.id)
            all_mems = await am.alist_all()
            assert len(all_mems) == 0

        asyncio.run(run())

    def test_async_save_load(self):
        """异步保存加载"""
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "async_test")

            async def run():
                hm = HybridMemory(
                    dimension=64,
                    embedding_provider=HashEmbeddingProvider(dim=64),
                    storage_path=path,
                )
                am = AsyncHybridMemory(hm)
                await am.aremember("persistent")
                await am.asave()

                hm2 = HybridMemory(
                    dimension=64,
                    embedding_provider=HashEmbeddingProvider(dim=64),
                    storage_path=path,
                    auto_load=True,
                )
                assert hm2.embedding_store.count() == 1

            asyncio.run(run())

    def test_async_stats(self):
        """异步统计"""
        am = self._make_async_memory()

        async def run():
            await am.aremember("a")
            await am.aremember("b")
            st = await am.astats()
            assert st["memory_count"] == 2

        asyncio.run(run())

    def test_async_context_manager(self):
        """异步上下文管理器"""

        async def run():
            hm = HybridMemory(dimension=64, embedding_provider=HashEmbeddingProvider(dim=64))
            async with AsyncHybridMemory(hm) as am:
                await am.aremember("ctx test")
                st = await am.astats()
                assert st["memory_count"] == 1

        asyncio.run(run())

    def test_async_entity_relation(self):
        """异步添加实体和关系"""
        am = self._make_async_memory()

        async def run():
            e1 = await am.aadd_entity("Alice", "person")
            e2 = await am.aadd_entity("Bob", "person")
            rel = await am.aadd_relation(e1.id, e2.id, "knows")
            assert rel.relation_type == "knows"

        asyncio.run(run())

    def test_async_lifecycle(self):
        """异步生命周期管理"""
        am = self._make_async_memory()

        async def run():
            await am.aremember("ttl test", ttl=0.01)
            await asyncio.sleep(0.02)
            expired = await am.acleanup_expired()
            assert len(expired) == 1

        asyncio.run(run())

    def test_async_batch_search(self):
        """异步批量搜索"""
        am = self._make_async_memory()

        async def run():
            await am.aremember("python code")
            await am.aremember("rust code")
            provider = HashEmbeddingProvider(dim=64)
            q1 = provider.embed("python")
            q2 = provider.embed("rust")
            results = await am.abatch_search([q1, q2])
            assert len(results) == 2
            assert len(results[0]) > 0

        asyncio.run(run())

    def test_async_aget_memory(self):
        """异步获取记忆"""
        am = self._make_async_memory()

        async def run():
            mem = await am.aremember("test")
            fetched = await am.aget_memory(mem.id)
            assert fetched is not None
            assert fetched.content == "test"

        asyncio.run(run())

    def test_async_aupdate_memory(self):
        """异步更新记忆"""
        am = self._make_async_memory()

        async def run():
            mem = await am.aremember("original")
            updated = await am.aupdate_memory(mem.id, content="updated")
            assert updated.content == "updated"

        asyncio.run(run())

    def test_async_memory_property(self):
        """访问底层 HybridMemory"""
        hm = HybridMemory(dimension=64, embedding_provider=HashEmbeddingProvider(dim=64))
        am = AsyncHybridMemory(hm)
        assert am.memory is hm
