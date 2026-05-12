"""LSH 近似最近邻索引测试"""

import math
import pytest
from agentmemory.lsh_index import LSHIndex


class TestLSHIndex:
    """LSH 索引基本功能测试"""

    def test_create_index(self):
        """创建索引"""
        idx = LSHIndex(dimension=128)
        assert idx.size() == 0

    def test_invalid_dimension(self):
        """无效维度"""
        with pytest.raises(ValueError, match="维度必须 >= 1"):
            LSHIndex(dimension=0)

    def test_invalid_tables(self):
        """无效表数"""
        with pytest.raises(ValueError, match="num_tables 必须 >= 1"):
            LSHIndex(dimension=128, num_tables=0)

    def test_invalid_hyperplanes(self):
        """无效超平面数"""
        with pytest.raises(ValueError, match="num_hyperplanes 必须 >= 1"):
            LSHIndex(dimension=128, num_hyperplanes=0)

    def test_add_and_query(self):
        """添加向量并查询"""
        idx = LSHIndex(dimension=4, num_tables=4, num_hyperplanes=4)
        vec = [1.0, 0.0, 0.0, 0.0]
        idx.add("v1", vec)
        assert idx.size() == 1

        results = idx.query(vec)
        assert "v1" in results

    def test_add_multiple_vectors(self):
        """添加多个向量"""
        idx = LSHIndex(dimension=8, num_tables=4, num_hyperplanes=4)
        for i in range(10):
            vec = [float(i)] * 8
            idx.add(f"v{i}", vec)
        assert idx.size() == 10

    def test_remove_vector(self):
        """删除向量"""
        idx = LSHIndex(dimension=4)
        idx.add("v1", [1.0, 0.0, 0.0, 0.0])
        idx.add("v2", [0.0, 1.0, 0.0, 0.0])
        idx.remove("v1")
        assert idx.size() == 1

    def test_remove_nonexistent(self):
        """删除不存在的向量"""
        idx = LSHIndex(dimension=4)
        with pytest.raises(KeyError):
            idx.remove("nonexistent")

    def test_query_returns_similar_vectors(self):
        """相似向量应被召回"""
        idx = LSHIndex(dimension=64, num_tables=8, num_hyperplanes=8, seed=42)

        # 创建一组向量：一个目标向量和一些随机向量
        target = [1.0] * 32 + [0.0] * 32
        similar = [0.9] * 32 + [0.1] * 32  # 非常相似
        different = [0.0] * 32 + [1.0] * 32  # 完全不同

        idx.add("target", target)
        idx.add("similar", similar)
        idx.add("different", different)

        results = idx.query(target)
        # 目标和相似向量都应被召回
        assert "target" in results
        assert "similar" in results

    def test_get_vector(self):
        """获取向量数据"""
        idx = LSHIndex(dimension=4)
        vec = [1.0, 2.0, 3.0, 4.0]
        idx.add("v1", vec)
        assert idx.get_vector("v1") == vec
        assert idx.get_vector("nonexistent") is None

    def test_dimension_mismatch_add(self):
        """添加维度不匹配的向量"""
        idx = LSHIndex(dimension=4)
        with pytest.raises(ValueError, match="向量维度不匹配"):
            idx.add("v1", [1.0, 2.0])

    def test_dimension_mismatch_query(self):
        """查询维度不匹配"""
        idx = LSHIndex(dimension=4)
        with pytest.raises(ValueError, match="向量维度不匹配"):
            idx.query([1.0, 2.0])

    def test_clear(self):
        """清空索引"""
        idx = LSHIndex(dimension=4)
        idx.add("v1", [1.0, 0.0, 0.0, 0.0])
        idx.add("v2", [0.0, 1.0, 0.0, 0.0])
        idx.clear()
        assert idx.size() == 0

    def test_rebuild(self):
        """重建索引"""
        idx = LSHIndex(dimension=4)
        idx.add("v1", [1.0, 0.0, 0.0, 0.0])
        idx.add("v2", [0.0, 1.0, 0.0, 0.0])
        idx.rebuild()
        assert idx.size() == 2
        results = idx.query([1.0, 0.0, 0.0, 0.0])
        assert "v1" in results

    def test_deterministic_hash(self):
        """相同输入产生相同哈希"""
        idx1 = LSHIndex(dimension=4, seed=42)
        idx2 = LSHIndex(dimension=4, seed=42)
        vec = [0.5, 0.3, 0.7, 0.1]
        key1 = idx1._hash_vector(vec, 0)
        key2 = idx2._hash_vector(vec, 0)
        assert key1 == key2

    def test_query_with_max_candidates(self):
        """限制候选数量"""
        idx = LSHIndex(dimension=4, num_tables=2, num_hyperplanes=2)
        for i in range(20):
            idx.add(f"v{i}", [float(i % 2)] * 4)
        results = idx.query([1.0, 1.0, 1.0, 1.0], max_candidates=5)
        assert len(results) <= 5

    def test_large_scale(self):
        """大规模数据测试"""
        idx = LSHIndex(dimension=32, num_tables=8, num_hyperplanes=12)
        import random
        rng = random.Random(42)
        for i in range(1000):
            vec = [rng.gauss(0, 1) for _ in range(32)]
            idx.add(f"v{i}", vec)

        assert idx.size() == 1000
        query_vec = [rng.gauss(0, 1) for _ in range(32)]
        results = idx.query(query_vec, max_candidates=50)
        assert len(results) > 0
        assert len(results) <= 50
