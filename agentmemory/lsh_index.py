"""LSH（局部敏感哈希）近似最近邻索引。

使用随机投影 LSH 算法，在 O(1) 时间内近似查找最近邻，
比暴力搜索 O(n) 快得多，尤其适合大规模数据（>10k 条记忆）。

v0.5.0: 多探针 LSH — 不仅翻转1位，还翻转2位、3位，
显著提升召回率。自适应回退确保结果不为空。

纯 Python 实现，无外部依赖。
"""

from __future__ import annotations

import math
import random
from itertools import combinations
from typing import Optional


class LSHIndex:
    """基于随机投影的多探针 LSH 近似最近邻索引。

    将高维向量通过随机超平面投影到多个哈希桶中，
    相似向量更可能落入相同桶，从而实现快速近似搜索。

    v0.5.0: 多探针机制 — 在主桶未找到足够候选时，
    自动翻转 2 位、3 位来扩大搜索范围，大幅提升召回率。

    Args:
        dimension: 向量维度
        num_tables: 哈希表数量（越多召回率越高，但内存和时间开销越大）
        num_hyperplanes: 每个表的超平面数量（决定桶的粒度）
        seed: 随机种子（确保可重现性）
        max_probe_bits: 最大翻转位数（默认 3，即翻转 1~3 位）
        min_candidates: 最少候选数（低于此值继续探针）
    """

    def __init__(
        self,
        dimension: int,
        num_tables: int = 8,
        num_hyperplanes: int = 12,
        seed: int = 42,
        max_probe_bits: int = 3,
        min_candidates: int = 10,
    ) -> None:
        if dimension < 1:
            raise ValueError(f"维度必须 >= 1, got {dimension}")
        if num_tables < 1:
            raise ValueError(f"num_tables 必须 >= 1, got {num_tables}")
        if num_hyperplanes < 1:
            raise ValueError(f"num_hyperplanes 必须 >= 1, got {num_hyperplanes}")
        if max_probe_bits < 1:
            raise ValueError(f"max_probe_bits 必须 >= 1, got {max_probe_bits}")

        self._dimension = dimension
        self._num_tables = num_tables
        self._num_hyperplanes = num_hyperplanes
        self._max_probe_bits = min(max_probe_bits, num_hyperplanes)
        self._min_candidates = min_candidates

        # 生成随机超平面：每个表有 num_hyperplanes 个超平面
        rng = random.Random(seed)
        self._hyperplanes: list[list[list[float]]] = []
        for _ in range(num_tables):
            table_planes: list[list[float]] = []
            for _ in range(num_hyperplanes):
                plane = [rng.gauss(0, 1) for _ in range(dimension)]
                table_planes.append(plane)
            self._hyperplanes.append(table_planes)

        # 哈希表：table_idx -> {hash_key -> set[vector_id]}
        self._tables: list[dict[str, set[str]]] = [{} for _ in range(num_tables)]
        # 向量存储：vector_id -> vector
        self._vectors: dict[str, list[float]] = {}
        # 向量ID到桶键的映射（用于删除）
        self._id_to_keys: dict[str, list[str]] = {}
        # 预计算的多探针偏移模式缓存
        self._probe_patterns: dict[int, list[tuple[int, ...]]] = {}

    def _get_probe_patterns(self, num_bits: int) -> list[tuple[int, ...]]:
        """获取翻转 num_bits 位的所有组合（预计算缓存）。

        Args:
            num_bits: 翻转的位数

        Returns:
            所有翻转位索引组合的列表
        """
        if num_bits not in self._probe_patterns:
            self._probe_patterns[num_bits] = list(
                combinations(range(self._num_hyperplanes), num_bits)
            )
        return self._probe_patterns[num_bits]

    def _hash_vector(self, vector: list[float], table_idx: int) -> str:
        """计算向量在指定哈希表中的桶键。

        通过与每个超平面做点积，大于0记为1，小于等于0记为0，
        拼接成二进制字符串作为哈希键。

        Args:
            vector: 输入向量
            table_idx: 哈希表索引

        Returns:
            哈希键字符串
        """
        bits: list[str] = []
        for plane in self._hyperplanes[table_idx]:
            dot = sum(v * p for v, p in zip(vector, plane))
            bits.append("1" if dot > 0 else "0")
        return "".join(bits)

    def _flip_bits(self, key: str, positions: tuple[int, ...]) -> str:
        """翻转哈希键中指定位置的位。

        Args:
            key: 原始哈希键
            positions: 要翻转的位索引

        Returns:
            翻转后的新哈希键
        """
        key_list = list(key)
        for pos in positions:
            key_list[pos] = "0" if key_list[pos] == "1" else "1"
        return "".join(key_list)

    def add(self, vector_id: str, vector: list[float]) -> None:
        """将向量添加到索引。

        Args:
            vector_id: 向量唯一标识
            vector: 向量数据

        Raises:
            ValueError: 向量维度不匹配
        """
        if len(vector) != self._dimension:
            raise ValueError(
                f"向量维度不匹配: 期望 {self._dimension}, 实际 {len(vector)}"
            )

        self._vectors[vector_id] = vector
        keys: list[str] = []

        for i in range(self._num_tables):
            key = self._hash_vector(vector, i)
            keys.append(key)
            if key not in self._tables[i]:
                self._tables[i][key] = set()
            self._tables[i][key].add(vector_id)

        self._id_to_keys[vector_id] = keys

    def remove(self, vector_id: str) -> None:
        """从索引中删除向量。

        Args:
            vector_id: 向量标识

        Raises:
            KeyError: 向量不存在
        """
        if vector_id not in self._vectors:
            raise KeyError(f"向量 {vector_id} 不存在")

        keys = self._id_to_keys.pop(vector_id, [])
        for i, key in enumerate(keys):
            if key in self._tables[i]:
                self._tables[i][key].discard(vector_id)
                if not self._tables[i][key]:
                    del self._tables[i][key]

        del self._vectors[vector_id]

    def query(
        self,
        query_vector: list[float],
        max_candidates: int = 200,
    ) -> set[str]:
        """多探针近似最近邻查询：返回候选向量 ID 集合。

        v0.5.0: 多探针策略 —
        1. 先查主桶
        2. 翻转 1 位查找邻居桶
        3. 翻转 2 位扩大搜索
        4. 翻转 3 位进一步扩大
        在每个阶段检查是否已收集足够候选，满足即提前返回。

        Args:
            query_vector: 查询向量
            max_candidates: 最大候选数量

        Returns:
            候选向量 ID 集合

        Raises:
            ValueError: 向量维度不匹配
        """
        if len(query_vector) != self._dimension:
            raise ValueError(
                f"向量维度不匹配: 期望 {self._dimension}, 实际 {len(query_vector)}"
            )

        candidates: set[str] = set()

        for i in range(self._num_tables):
            key = self._hash_vector(query_vector, i)

            # 主桶
            if key in self._tables[i]:
                candidates.update(self._tables[i][key])

            if len(candidates) >= max_candidates:
                return set(list(candidates)[:max_candidates])

        # 如果主桶已经足够，返回
        if len(candidates) >= self._min_candidates:
            return set(list(candidates)[:max_candidates])

        # 多探针：逐级翻转更多位
        for num_flip in range(1, self._max_probe_bits + 1):
            patterns = self._get_probe_patterns(num_flip)
            for i in range(self._num_tables):
                key = self._hash_vector(query_vector, i)
                for positions in patterns:
                    neighbor_key = self._flip_bits(key, positions)
                    if neighbor_key in self._tables[i]:
                        candidates.update(self._tables[i][neighbor_key])

                    if len(candidates) >= max_candidates:
                        return set(list(candidates)[:max_candidates])

            # 已经收集到足够的候选
            if len(candidates) >= self._min_candidates:
                break

        return candidates

    def get_vector(self, vector_id: str) -> Optional[list[float]]:
        """获取向量数据。

        Args:
            vector_id: 向量标识

        Returns:
            向量数据，不存在返回 None
        """
        return self._vectors.get(vector_id)

    def size(self) -> int:
        """返回索引中的向量数量。"""
        return len(self._vectors)

    def clear(self) -> None:
        """清空索引。"""
        for table in self._tables:
            table.clear()
        self._vectors.clear()
        self._id_to_keys.clear()

    def rebuild(self) -> None:
        """重建索引（在批量删除后使用以清理空桶）。"""
        vectors = dict(self._vectors)
        self.clear()
        for vid, vec in vectors.items():
            self.add(vid, vec)

    def stats(self) -> dict[str, int | float]:
        """返回索引统计信息。

        Returns:
            包含向量数、表数、桶数、平均桶大小等信息
        """
        total_buckets = sum(len(table) for table in self._tables)
        total_items = sum(
            len(bucket)
            for table in self._tables
            for bucket in table.values()
        )
        avg_bucket_size = total_items / total_buckets if total_buckets > 0 else 0.0
        return {
            "num_vectors": len(self._vectors),
            "num_tables": self._num_tables,
            "num_hyperplanes": self._num_hyperplanes,
            "total_buckets": total_buckets,
            "avg_bucket_size": round(avg_bucket_size, 2),
            "max_probe_bits": self._max_probe_bits,
            "min_candidates": self._min_candidates,
        }
