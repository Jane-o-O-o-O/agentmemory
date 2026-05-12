"""记忆生命周期管理测试"""

import time
import pytest
from agentmemory.models import Memory
from agentmemory.lifecycle import MemoryLifecycle


class TestMemoryLifecycle:
    """生命周期管理器测试"""

    def test_create_lifecycle(self):
        """创建生命周期管理器"""
        lc = MemoryLifecycle()
        assert lc._default_ttl is None

    def test_create_with_ttl(self):
        """带 TTL 创建"""
        lc = MemoryLifecycle(default_ttl=3600)
        assert lc._default_ttl == 3600

    def test_is_expired_no_ttl(self):
        """无 TTL 时永不过期"""
        lc = MemoryLifecycle()
        mem = Memory(content="test")
        assert lc.is_expired(mem) is False

    def test_is_expired_with_ttl(self):
        """TTL 过期检查"""
        lc = MemoryLifecycle(default_ttl=0.1)
        mem = Memory(content="test")
        assert lc.is_expired(mem) is False
        time.sleep(0.15)
        assert lc.is_expired(mem) is True

    def test_custom_ttl(self):
        """自定义 TTL"""
        lc = MemoryLifecycle()
        mem = Memory(content="test")
        lc.set_ttl(mem.id, 0.1)
        assert lc.is_expired(mem) is False
        time.sleep(0.15)
        assert lc.is_expired(mem) is True

    def test_time_remaining(self):
        """剩余时间"""
        lc = MemoryLifecycle(default_ttl=10.0)
        mem = Memory(content="test")
        remaining = lc.time_remaining(mem)
        assert remaining is not None
        assert 9.0 <= remaining <= 10.0

    def test_time_remaining_no_ttl(self):
        """无 TTL 时返回 None"""
        lc = MemoryLifecycle()
        mem = Memory(content="test")
        assert lc.time_remaining(mem) is None

    def test_compute_decay_factor(self):
        """衰减因子计算"""
        lc = MemoryLifecycle(decay_rate=0.001)
        mem = Memory(content="test")
        # 新记忆衰减因子应接近 1
        factor = lc.compute_decay_factor(mem)
        assert 0.99 <= factor <= 1.0

    def test_decay_factor_decreases(self):
        """衰减因子随时间递减"""
        lc = MemoryLifecycle(decay_rate=1.0)  # 快速衰减
        old_mem = Memory(content="old", created_at=time.time() - 100)
        new_mem = Memory(content="new")
        assert lc.compute_decay_factor(new_mem) > lc.compute_decay_factor(old_mem)

    def test_record_access(self):
        """记录访问"""
        lc = MemoryLifecycle()
        mem = Memory(content="test")
        assert lc.get_access_count(mem.id) == 0
        lc.record_access(mem.id)
        lc.record_access(mem.id)
        assert lc.get_access_count(mem.id) == 2

    def test_set_importance(self):
        """设置重要性"""
        lc = MemoryLifecycle()
        mem = Memory(content="test")
        lc.set_importance(mem.id, 0.8)
        assert lc._importance[mem.id] == 0.8

    def test_invalid_importance(self):
        """无效重要性值"""
        lc = MemoryLifecycle()
        with pytest.raises(ValueError, match="importance 必须在 0~1"):
            lc.set_importance("test", 1.5)

    def test_compute_importance_score(self):
        """综合重要性评分"""
        lc = MemoryLifecycle(
            recency_weight=0.3,
            frequency_weight=0.3,
            relevance_weight=0.4,
        )
        mem = Memory(content="test")
        score = lc.compute_importance_score(mem)
        assert 0.0 <= score <= 1.0

    def test_importance_with_access_count(self):
        """有访问记录的重要性评分"""
        lc = MemoryLifecycle(frequency_weight=0.5, recency_weight=0.25, relevance_weight=0.25)
        mem1 = Memory(content="frequent")
        mem2 = Memory(content="rare")

        # mem1 访问 10 次
        for _ in range(10):
            lc.record_access(mem1.id)

        score1 = lc.compute_importance_score(mem1)
        score2 = lc.compute_importance_score(mem2)
        assert score1 > score2

    def test_filter_expired(self):
        """过滤过期记忆"""
        lc = MemoryLifecycle(default_ttl=0.5)
        fresh = Memory(content="fresh")
        # 创建一个很早之前的记忆（已过期）
        expired = Memory(content="expired", created_at=time.time() - 10.0)

        result = lc.filter_expired([fresh, expired])
        assert len(result) == 1
        assert result[0].content == "fresh"

    def test_rank_by_importance(self):
        """按重要性排序"""
        lc = MemoryLifecycle()
        mem1 = Memory(content="less important")
        mem2 = Memory(content="very important")
        lc.set_importance(mem1.id, 0.2)
        lc.set_importance(mem2.id, 0.9)

        ranked = lc.rank_by_importance([mem1, mem2])
        assert ranked[0][0].content == "very important"

    def test_rank_with_relevance_scores(self):
        """带相关性分数的排序"""
        lc = MemoryLifecycle()
        mem1 = Memory(content="a")
        mem2 = Memory(content="b")

        scores = {mem1.id: 0.9, mem2.id: 0.1}
        ranked = lc.rank_by_importance([mem1, mem2], relevance_scores=scores)
        assert ranked[0][0].content == "a"

    def test_cleanup(self):
        """清理过期记忆"""
        lc = MemoryLifecycle(default_ttl=0.1)
        mem = Memory(content="test")
        lc.set_importance(mem.id, 0.5)
        lc.record_access(mem.id)

        time.sleep(0.15)
        result = lc.cleanup([mem])
        assert len(result) == 0
        # 生命周期数据应被清除
        assert lc.get_access_count(mem.id) == 0

    def test_get_lifecycle_info(self):
        """获取生命周期信息"""
        lc = MemoryLifecycle(default_ttl=3600)
        mem = Memory(content="test")
        lc.record_access(mem.id)

        info = lc.get_lifecycle_info(mem)
        assert info["memory_id"] == mem.id
        assert info["is_expired"] is False
        assert info["access_count"] == 1
        assert info["ttl"] == 3600
        assert info["decay_factor"] > 0

    def test_decay_weights_sum_to_one(self):
        """权重总和应为 1"""
        lc = MemoryLifecycle(
            recency_weight=0.3,
            frequency_weight=0.3,
            relevance_weight=0.4,
        )
        total = lc._recency_weight + lc._frequency_weight + lc._relevance_weight
        assert abs(total - 1.0) < 1e-9
