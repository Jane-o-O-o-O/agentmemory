"""向量量化 — Product Quantization (PQ) 和 Scalar Quantization (SQ) 压缩向量。

减少向量存储的内存占用，支持近似最近邻搜索。
"""

from __future__ import annotations

import math
import random
import struct
from dataclasses import dataclass, field
from typing import Optional

import hashlib


@dataclass
class QuantizationStats:
    """量化统计信息。

    Attributes:
        original_dim: 原始向量维度
        compressed_size_bytes: 每个压缩向量的字节数
        compression_ratio: 压缩比（原始/压缩）
        num_vectors: 已量化的向量数量
        method: 量化方法名称
    """

    original_dim: int
    compressed_size_bytes: int
    compression_ratio: float
    num_vectors: int
    method: str


class ScalarQuantizer:
    """标量量化器 (SQ8)。

    将 float32 向量量化为 uint8，每个维度从 4 字节压缩到 1 字节，
    压缩比约 4x。通过 min/max 归一化保持精度。

    Args:
        dim: 向量维度
    """

    def __init__(self, dim: int) -> None:
        if dim <= 0:
            raise ValueError(f"维度必须为正整数: {dim}")
        self._dim = dim
        self._min: Optional[list[float]] = None
        self._max: Optional[list[float]] = None
        self._fitted = False
        self._vectors_count = 0

    @property
    def dim(self) -> int:
        """向量维度"""
        return self._dim

    @property
    def fitted(self) -> bool:
        """是否已拟合"""
        return self._fitted

    def fit(self, vectors: list[list[float]]) -> ScalarQuantizer:
        """拟合量化参数（计算全局 min/max）。

        Args:
            vectors: 训练向量列表

        Returns:
            self，支持链式调用

        Raises:
            ValueError: 向量列表为空或维度不匹配
        """
        if not vectors:
            raise ValueError("训练向量列表不能为空")

        for v in vectors:
            if len(v) != self._dim:
                raise ValueError(f"向量维度不匹配: 期望 {self._dim}, 实际 {len(v)}")

        self._min = list(vectors[0])
        self._max = list(vectors[0])

        for v in vectors[1:]:
            for i in range(self._dim):
                if v[i] < self._min[i]:
                    self._min[i] = v[i]
                if v[i] > self._max[i]:
                    self._max[i] = v[i]

        # 处理常量维度（min == max）
        for i in range(self._dim):
            if self._max[i] == self._min[i]:
                self._max[i] = self._min[i] + 1.0

        self._fitted = True
        self._vectors_count = len(vectors)
        return self

    def auto_fit(self, vector: list[float]) -> ScalarQuantizer:
        """增量自适应拟合，用于在线场景。

        如果未拟合，使用该向量初始化；否则扩展 min/max 范围。

        Args:
            vector: 新向量

        Returns:
            self
        """
        if len(vector) != self._dim:
            raise ValueError(f"向量维度不匹配: 期望 {self._dim}, 实际 {len(vector)}")

        if not self._fitted:
            self._min = [v - 0.5 for v in vector]
            self._max = [v + 0.5 for v in vector]
            self._fitted = True
        else:
            for i in range(self._dim):
                if vector[i] < self._min[i]:
                    self._min[i] = vector[i]
                if vector[i] > self._max[i]:
                    self._max[i] = vector[i]

        self._vectors_count += 1
        return self

    def quantize(self, vector: list[float]) -> bytes:
        """将浮点向量量化为 uint8 字节序列。

        Args:
            vector: 输入向量

        Returns:
            量化后的字节序列

        Raises:
            RuntimeError: 未拟合
            ValueError: 维度不匹配
        """
        if not self._fitted:
            raise RuntimeError("请先调用 fit() 或 auto_fit() 拟合量化参数")
        if len(vector) != self._dim:
            raise ValueError(f"向量维度不匹配: 期望 {self._dim}, 实际 {len(vector)}")

        result = bytearray(self._dim)
        for i in range(self._dim):
            # 归一化到 [0, 1]
            normalized = (vector[i] - self._min[i]) / (self._max[i] - self._min[i])
            # 量化到 [0, 255]
            quantized = max(0, min(255, round(normalized * 255)))
            result[i] = quantized

        return bytes(result)

    def dequantize(self, data: bytes) -> list[float]:
        """将量化字节序列还原为浮点向量。

        Args:
            data: 量化字节序列

        Returns:
            还原的浮点向量（近似值）

        Raises:
            RuntimeError: 未拟合
            ValueError: 数据长度不匹配
        """
        if not self._fitted:
            raise RuntimeError("请先调用 fit() 拟合量化参数")
        if len(data) != self._dim:
            raise ValueError(f"数据长度不匹配: 期望 {self._dim}, 实际 {len(data)}")

        result = [0.0] * self._dim
        for i in range(self._dim):
            normalized = data[i] / 255.0
            result[i] = self._min[i] + normalized * (self._max[i] - self._min[i])

        return result

    def quantize_batch(self, vectors: list[list[float]]) -> list[bytes]:
        """批量量化向量。

        Args:
            vectors: 向量列表

        Returns:
            量化字节序列列表
        """
        return [self.quantize(v) for v in vectors]

    def dequantize_batch(self, data_list: list[bytes]) -> list[list[float]]:
        """批量反量化。

        Args:
            data_list: 量化字节序列列表

        Returns:
            浮点向量列表
        """
        return [self.dequantize(d) for d in data_list]

    def compressed_size(self) -> int:
        """每个压缩向量的字节数"""
        return self._dim  # 1 byte per dimension

    def stats(self) -> QuantizationStats:
        """获取量化统计信息"""
        original_bytes = self._dim * 4  # float32
        compressed = self.compressed_size()
        return QuantizationStats(
            original_dim=self._dim,
            compressed_size_bytes=compressed,
            compression_ratio=original_bytes / compressed if compressed > 0 else 0,
            num_vectors=self._vectors_count,
            method="SQ8",
        )


class ProductQuantizer:
    """乘积量化器 (PQ)。

    将高维向量分成多个子空间，每个子空间独立聚类，
    用聚类中心索引（uint8）替代原始向量，实现更高压缩比。

    Args:
        dim: 向量维度
        num_subspaces: 子空间数量（dim 必须能被其整除）
        num_centroids: 每个子空间的聚类中心数（最大 256）
    """

    def __init__(
        self,
        dim: int,
        num_subspaces: int = 8,
        num_centroids: int = 256,
    ) -> None:
        if dim <= 0:
            raise ValueError(f"维度必须为正整数: {dim}")
        if num_subspaces <= 0:
            raise ValueError(f"子空间数量必须为正整数: {num_subspaces}")
        if dim % num_subspaces != 0:
            raise ValueError(
                f"维度 {dim} 必须能被子空间数量 {num_subspaces} 整除"
            )
        if not 1 <= num_centroids <= 256:
            raise ValueError(f"聚类中心数必须在 1~256 之间: {num_centroids}")

        self._dim = dim
        self._num_subspaces = num_subspaces
        self._num_centroids = num_centroids
        self._subspace_dim = dim // num_subspaces

        # 每个子空间的聚类中心: list[subspace_idx] -> list[centroid_idx] -> list[float]
        self._centroids: list[list[list[float]]] = []
        self._fitted = False
        self._vectors_count = 0

    @property
    def dim(self) -> int:
        """向量维度"""
        return self._dim

    @property
    def num_subspaces(self) -> int:
        """子空间数量"""
        return self._num_subspaces

    @property
    def num_centroids(self) -> int:
        """每个子空间的聚类中心数"""
        return self._num_centroids

    @property
    def fitted(self) -> bool:
        """是否已拟合"""
        return self._fitted

    def _split_subspace(self, vector: list[float]) -> list[list[float]]:
        """将向量拆分为子空间"""
        return [
            vector[i * self._subspace_dim : (i + 1) * self._subspace_dim]
            for i in range(self._num_subspaces)
        ]

    def fit(
        self,
        vectors: list[list[float]],
        num_iterations: int = 10,
        seed: Optional[int] = None,
    ) -> ProductQuantizer:
        """拟合乘积量化参数。

        使用简化的 k-means 聚类（随机初始化 + 迭代分配）。

        Args:
            vectors: 训练向量列表
            num_iterations: k-means 迭代次数
            seed: 随机种子

        Returns:
            self，支持链式调用

        Raises:
            ValueError: 向量列表为空或维度不匹配
        """
        if not vectors:
            raise ValueError("训练向量列表不能为空")
        for v in vectors:
            if len(v) != self._dim:
                raise ValueError(f"向量维度不匹配: 期望 {self._dim}, 实际 {len(v)}")

        rng = random.Random(seed)
        n = len(vectors)
        k = min(self._num_centroids, n)  # 聚类中心数不超过样本数

        self._centroids = []

        for s in range(self._num_subspaces):
            # 提取子空间数据
            sub_data = [v[s * self._subspace_dim : (s + 1) * self._subspace_dim] for v in vectors]

            # 随机初始化聚类中心
            indices = rng.sample(range(n), k)
            centroids = [list(sub_data[i]) for i in indices]

            # 简化 k-means 迭代
            assignments = [0] * n
            for _ in range(num_iterations):
                # 分配
                for i in range(n):
                    best_j = 0
                    best_dist = float("inf")
                    for j in range(k):
                        dist = sum(
                            (sub_data[i][d] - centroids[j][d]) ** 2
                            for d in range(self._subspace_dim)
                        )
                        if dist < best_dist:
                            best_dist = dist
                            best_j = j
                    assignments[i] = best_j

                # 更新聚类中心
                counts = [0] * k
                new_centroids = [[0.0] * self._subspace_dim for _ in range(k)]
                for i in range(n):
                    j = assignments[i]
                    counts[j] += 1
                    for d in range(self._subspace_dim):
                        new_centroids[j][d] += sub_data[i][d]

                for j in range(k):
                    if counts[j] > 0:
                        centroids[j] = [
                            new_centroids[j][d] / counts[j]
                            for d in range(self._subspace_dim)
                        ]

            self._centroids.append(centroids)

        self._fitted = True
        self._vectors_count = len(vectors)
        return self

    def quantize(self, vector: list[float]) -> bytes:
        """将向量量化为 PQ 编码。

        每个子空间找到最近聚类中心，存储其索引（uint8）。

        Args:
            vector: 输入向量

        Returns:
            PQ 编码（num_subspaces 字节）

        Raises:
            RuntimeError: 未拟合
            ValueError: 维度不匹配
        """
        if not self._fitted:
            raise RuntimeError("请先调用 fit() 拟合量化参数")
        if len(vector) != self._dim:
            raise ValueError(f"向量维度不匹配: 期望 {self._dim}, 实际 {len(vector)}")

        subspaces = self._split_subspace(vector)
        result = bytearray(self._num_subspaces)

        for s in range(self._num_subspaces):
            best_j = 0
            best_dist = float("inf")
            for j in range(len(self._centroids[s])):
                dist = sum(
                    (subspaces[s][d] - self._centroids[s][j][d]) ** 2
                    for d in range(self._subspace_dim)
                )
                if dist < best_dist:
                    best_dist = dist
                    best_j = j
            result[s] = best_j

        return bytes(result)

    def dequantize(self, data: bytes) -> list[float]:
        """将 PQ 编码还原为向量（用聚类中心重建）。

        Args:
            data: PQ 编码

        Returns:
            重建的浮点向量

        Raises:
            RuntimeError: 未拟合
            ValueError: 编码长度不匹配
        """
        if not self._fitted:
            raise RuntimeError("请先调用 fit() 拟合量化参数")
        if len(data) != self._num_subspaces:
            raise ValueError(f"编码长度不匹配: 期望 {self._num_subspaces}, 实际 {len(data)}")

        result: list[float] = []
        for s in range(self._num_subspaces):
            centroid_idx = data[s]
            result.extend(self._centroids[s][centroid_idx])

        return result

    def quantize_batch(self, vectors: list[list[float]]) -> list[bytes]:
        """批量量化向量。

        Args:
            vectors: 向量列表

        Returns:
            PQ 编码列表
        """
        return [self.quantize(v) for v in vectors]

    def dequantize_batch(self, data_list: list[bytes]) -> list[list[float]]:
        """批量反量化。

        Args:
            data_list: PQ 编码列表

        Returns:
            浮点向量列表
        """
        return [self.dequantize(d) for d in data_list]

    def compressed_size(self) -> int:
        """每个压缩向量的字节数"""
        return self._num_subspaces  # 1 byte per subspace

    def stats(self) -> QuantizationStats:
        """获取量化统计信息"""
        original_bytes = self._dim * 4  # float32
        compressed = self.compressed_size()
        return QuantizationStats(
            original_dim=self._dim,
            compressed_size_bytes=compressed,
            compression_ratio=original_bytes / compressed if compressed > 0 else 0,
            num_vectors=self._vectors_count,
            method=f"PQ(subspaces={self._num_subspaces}, centroids={self._num_centroids})",
        )


class CompressedVectorStore:
    """压缩向量存储。

    使用量化器压缩向量，显著减少内存占用。

    Args:
        quantizer: 量化器实例 (ScalarQuantizer 或 ProductQuantizer)
    """

    def __init__(self, quantizer: object) -> None:
        self._quantizer = quantizer
        self._store: dict[str, bytes] = {}  # id -> compressed bytes

    @property
    def size(self) -> int:
        """存储的向量数量"""
        return len(self._store)

    def add(self, vector_id: str, vector: list[float]) -> None:
        """添加压缩向量。

        Args:
            vector_id: 向量 ID
            vector: 原始浮点向量
        """
        compressed = self._quantizer.quantize(vector)  # type: ignore
        self._store[vector_id] = compressed

    def get(self, vector_id: str) -> Optional[list[float]]:
        """获取并解压向量。

        Args:
            vector_id: 向量 ID

        Returns:
            还原的浮点向量，不存在返回 None
        """
        data = self._store.get(vector_id)
        if data is None:
            return None
        return self._quantizer.dequantize(data)  # type: ignore

    def get_compressed(self, vector_id: str) -> Optional[bytes]:
        """获取压缩的原始字节。

        Args:
            vector_id: 向量 ID

        Returns:
            压缩字节，不存在返回 None
        """
        return self._store.get(vector_id)

    def remove(self, vector_id: str) -> bool:
        """删除向量。

        Args:
            vector_id: 向量 ID

        Returns:
            是否成功删除
        """
        if vector_id in self._store:
            del self._store[vector_id]
            return True
        return False

    def contains(self, vector_id: str) -> bool:
        """检查是否包含指定 ID 的向量。"""
        return vector_id in self._store

    def list_ids(self) -> list[str]:
        """列出所有向量 ID。"""
        return list(self._store.keys())

    def memory_usage_bytes(self) -> int:
        """估算压缩存储的内存使用（字节）"""
        if not self._store:
            return 0
        sample = next(iter(self._store.values()))
        return len(sample) * len(self._store)

    def stats(self) -> dict:
        """获取存储统计信息"""
        q_stats = self._quantizer.stats()  # type: ignore
        return {
            "num_vectors": self.size,
            "compressed_bytes_per_vector": q_stats.compressed_size_bytes,
            "total_compressed_bytes": self.memory_usage_bytes(),
            "compression_ratio": q_stats.compression_ratio,
            "method": q_stats.method,
        }
