"""LSH（局部敏感哈希）近似最近邻索引。

使用随机投影 LSH 算法，在 O(1) 时间内近似查找最近邻，
比暴力搜索 O(n) 快得多，尤其适合大规模数据（>10k 条记忆）。

纯 Python 实现，无外部依赖。
"""

from __future__ import annotations

import hashlib
import math
import random
from typing import Optional


class LSHIndex:
    """基于随机投影的 LSH 近似最近邻索引。

    将高维向量通过随机超平面投影到多个哈希桶中，
    相似向量更可能落入相同桶，从而实现快速近似搜索。

    Args:
        dimension: 向量维度
        num_tables: 哈希表数量（越多召回率越高，但内存和时间开销越大）
        num_hyperplanes: 每个表的超平面数量（决定桶的粒度）
        seed: 随机种子（确保可重现性）
    """

    def __init__(
        self,
        dimension: int,
        num_tables: int = 8,
        num_hyperplanes: int = 16,
        seed: int = 42,
    ) -> None:
        if dimension < 1:
            raise ValueError(f"维度必须 >= 1, got {dimension}")
        if num_tables < 1:
            raise ValueError(f"num_tables 必须 >= 1, got {num_tables}")
        if num_hyperplanes < 1:
            raise ValueError(f"num_hyperplanes 必须 >= 1, got {num_hyperplanes}")

        self._dimension = dimension
        self._num_tables = num_tables
        self._num_hyperplanes = num_hyperplanes

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
        """近似最近邻查询：返回候选向量 ID 集合。

        通过查找查询向量所在桶及相邻桶中的向量，
        返回最多 max_candidates 个候选 ID。

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
            # 查找主桶
            if key in self._tables[i]:
                candidates.update(self._tables[i][key])

            # 查找相邻桶（翻转每一位）
            for bit_idx in range(self._num_hyperplanes):
                neighbor_key = key[:bit_idx] + ("0" if key[bit_idx] == "1" else "1") + key[bit_idx + 1:]
                if neighbor_key in self._tables[i]:
                    candidates.update(self._tables[i][neighbor_key])

                if len(candidates) >= max_candidates:
                    return set(list(candidates)[:max_candidates])

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
