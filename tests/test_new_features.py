"""HybridMemory 新功能测试：LSH、生命周期、update/merge/forget_where"""

import time
import pytest
from agentmemory import (
    HybridMemory, HashEmbeddingProvider,
    Memory, Entity, Relation,
)


@pytest.fixture
def hm():
    """创建测试用 HybridMemory"""
    return HybridMemory(
        dimension=64,
        embedding_provider=HashEmbeddingProvider(dim=64),
    )


@pytest.fixture
def hm_lsh():
    """创建带 LSH 的 HybridMemory"""
    return HybridMemory(
        dimension=64,
        embedding_provider=HashEmbeddingProvider(dim=64),
        use_lsh=True,
        lsh_tables=4,
        lsh_hyperplanes=8,
    )


class TestLSHIntegration:
    """LSH 索引集成测试"""

    def test_lsh_enabled(self, hm_lsh):
        """LSH 启用"""
        assert hm_lsh.embedding_store.use_lsh is True

    def test_lsh_disabled_by_default(self, hm):
        """默认不启用 LSH"""
        assert hm.embedding_store.use_lsh is False

    def test_lsh_add_and_search(self, hm_lsh):
        """LSH 模式下添加和搜索"""
        hm_lsh.remember("Python 编程语言", tags=["tech"])
        hm_lsh.remember("Rust 系统编程", tags=["tech"])
        hm_lsh.remember("烹饪美食", tags=["food"])

        results = hm_lsh.search_text("编程", top_k=3)
        assert len(results) > 0
        # 编程相关的应排在前面
        assert "编程" in results[0].memory.content

    def test_lsh_large_dataset(self, hm_lsh):
        """LSH 大规模数据测试"""
        # 批量添加 200 条记忆
        contents = [f"记忆内容第{i}条" for i in range(200)]
        hm_lsh.batch_remember(contents)

        assert hm_lsh.stats()["memory_count"] == 200
        results = hm_lsh.search_text("记忆", top_k=5)
        assert len(results) == 5

    def test_lsh_forget(self, hm_lsh):
        """LSH 模式下删除记忆"""
        mem = hm_lsh.remember("test")
        assert hm_lsh.stats()["memory_count"] == 1
        hm_lsh.forget(mem.id)
        assert hm_lsh.stats()["memory_count"] == 0

    def test_stats_shows_lsh(self, hm_lsh):
        """统计信息包含 LSH 状态"""
        stats = hm_lsh.stats()
        assert stats["use_lsh"] is True


class TestUpdateMemory:
    """update_memory 测试"""

    def test_update_content(self, hm):
        """更新内容"""
        mem = hm.remember("旧内容")
        updated = hm.update_memory(mem.id, content="新内容")
        assert updated.content == "新内容"

    def test_update_metadata(self, hm):
        """更新元数据"""
        mem = hm.remember("test", metadata={"a": 1})
        updated = hm.update_memory(mem.id, metadata={"b": 2})
        assert updated.metadata == {"a": 1, "b": 2}

    def test_update_tags(self, hm):
        """更新标签"""
        mem = hm.remember("test", tags=["old"])
        updated = hm.update_memory(mem.id, tags=["new1", "new2"])
        assert updated.tags == ["new1", "new2"]

    def test_update_nonexistent(self, hm):
        """更新不存在的记忆"""
        with pytest.raises(KeyError):
            hm.update_memory("nonexistent", content="x")

    def test_update_recomputes_embedding(self, hm):
        """更新内容后重新计算 embedding"""
        mem = hm.remember("old text")
        old_emb = list(mem.embedding)
        updated = hm.update_memory(mem.id, content="completely different text")
        # embedding 应该改变了
        assert updated.embedding != old_emb


class TestMergeMemories:
    """merge_memories 测试"""

    def test_merge_two(self, hm):
        """合并两条记忆"""
        m1 = hm.remember("记忆一", tags=["tag1"], metadata={"k1": "v1"})
        m2 = hm.remember("记忆二", tags=["tag2"], metadata={"k2": "v2"})

        merged = hm.merge_memories([m1.id, m2.id])
        assert merged.content == "记忆一\n记忆二"
        assert set(merged.tags) == {"tag1", "tag2"}
        assert merged.metadata == {"k1": "v1", "k2": "v2"}
        assert hm.stats()["memory_count"] == 1

    def test_merge_with_custom_content(self, hm):
        """合并时指定新内容"""
        m1 = hm.remember("a")
        m2 = hm.remember("b")
        merged = hm.merge_memories([m1.id, m2.id], new_content="合并结果")
        assert merged.content == "合并结果"

    def test_merge_empty_list(self, hm):
        """空列表"""
        with pytest.raises(ValueError):
            hm.merge_memories([])

    def test_merge_nonexistent(self, hm):
        """包含不存在的 ID"""
        m1 = hm.remember("a")
        with pytest.raises(KeyError):
            hm.merge_memories([m1.id, "nonexistent"])

    def test_merge_deduplicates_tags(self, hm):
        """合并标签去重"""
        m1 = hm.remember("a", tags=["tag1", "tag2"])
        m2 = hm.remember("b", tags=["tag2", "tag3"])
        merged = hm.merge_memories([m1.id, m2.id])
        assert len(merged.tags) == 3
        assert set(merged.tags) == {"tag1", "tag2", "tag3"}


class TestForgetWhere:
    """forget_where 测试"""

    def test_forget_by_content(self, hm):
        """按内容条件删除"""
        hm.remember("苹果很好吃")
        hm.remember("香蕉很甜")
        hm.remember("苹果派做法")

        deleted = hm.forget_where(lambda m: "苹果" in m.content)
        assert len(deleted) == 2
        assert hm.stats()["memory_count"] == 1

    def test_forget_by_tag(self, hm):
        """按标签条件删除"""
        hm.remember("a", tags=["temp"])
        hm.remember("b", tags=["temp"])
        hm.remember("c", tags=["keep"])

        deleted = hm.forget_where(lambda m: m.has_tag("temp"))
        assert len(deleted) == 2
        remaining = hm.list_all()
        assert len(remaining) == 1
        assert remaining[0].content == "c"

    def test_forget_no_match(self, hm):
        """无匹配"""
        hm.remember("test")
        deleted = hm.forget_where(lambda m: "nonexistent" in m.content)
        assert len(deleted) == 0
        assert hm.stats()["memory_count"] == 1


class TestLifecycleIntegration:
    """生命周期集成测试"""

    def test_remember_with_ttl(self, hm):
        """带 TTL 添加记忆"""
        mem = hm.remember("test", ttl=0.1)
        assert hm.lifecycle.time_remaining(mem) is not None
        assert hm.lifecycle.time_remaining(mem) > 0

    def test_remember_with_importance(self, hm):
        """带重要性添加记忆"""
        mem = hm.remember("test", importance=0.9)
        info = hm.get_lifecycle_info(mem.id)
        assert info is not None
        assert info["importance"] == 0.9

    def test_get_memory_records_access(self, hm):
        """get_memory 记录访问"""
        mem = hm.remember("test")
        hm.get_memory(mem.id)
        hm.get_memory(mem.id)
        info = hm.get_lifecycle_info(mem.id)
        assert info["access_count"] == 2

    def test_cleanup_expired(self, hm):
        """清理过期记忆"""
        hm.remember("will expire", ttl=0.1)
        hm.remember("will not expire")
        time.sleep(0.15)
        expired = hm.cleanup_expired()
        assert len(expired) == 1
        assert hm.stats()["memory_count"] == 1

    def test_get_lifecycle_info_nonexistent(self, hm):
        """获取不存在记忆的生命周期信息"""
        assert hm.get_lifecycle_info("nonexistent") is None

    def test_search_records_access(self, hm):
        """搜索记录访问"""
        hm.remember("Python 编程")
        hm.remember("Java 编程")
        results = hm.search_text("编程", top_k=2)
        for r in results:
            info = hm.get_lifecycle_info(r.memory.id)
            assert info["access_count"] >= 1


class TestCLIIntegration:
    """CLI 新命令测试"""

    def test_version_command(self, hm):
        """version 命令"""
        from agentmemory.cli import main
        import io
        import sys
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            main(["version"])
            output = sys.stdout.getvalue()
            assert "agentmemory" in output
            assert "v" in output
        finally:
            sys.stdout = old_stdout

    def test_inspect_command(self):
        """inspect 命令"""
        from agentmemory.cli import main
        import io
        import sys
        import os

        store_path = "/tmp/test_cli_inspect2"
        # 清理旧数据
        import shutil
        if os.path.exists(store_path):
            shutil.rmtree(store_path)

        # 先添加一条记忆并获取完整 ID
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            main(["--store", store_path, "remember", "测试记忆内容", "--tags", "test"])
            output = sys.stdout.getvalue()
            import json
            mem_data = json.loads(output)
            mem_id = mem_data["id"]
        finally:
            sys.stdout = old_stdout

        # inspect 使用完整 ID
        sys.stdout = io.StringIO()
        try:
            main(["--store", store_path, "inspect", mem_id])
            output = sys.stdout.getvalue()
            assert "测试记忆内容" in output
            assert "lifecycle" in output
        finally:
            sys.stdout = old_stdout

    def test_cleanup_command(self):
        """cleanup 命令"""
        from agentmemory.cli import main
        import io
        import sys

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            main(["cleanup"])
            output = sys.stdout.getvalue()
            assert "没有过期记忆" in output
        finally:
            sys.stdout = old_stdout

    def test_search_hybrid_flag(self):
        """search --hybrid 标志"""
        from agentmemory.cli import main
        import io
        import sys

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            main(["search", "test", "--hybrid"])
            output = sys.stdout.getvalue()
            assert "未找到" in output  # 空记忆库
        finally:
            sys.stdout = old_stdout

    def test_stats_shows_dimension(self):
        """stats 显示维度"""
        from agentmemory.cli import main
        import io
        import sys

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            main(["stats"])
            output = sys.stdout.getvalue()
            assert "向量维度" in output
            assert "LSH" in output
        finally:
            sys.stdout = old_stdout
