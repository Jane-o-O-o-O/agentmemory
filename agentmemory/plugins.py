"""插件架构 — 可注册自定义后端、Provider、搜索策略。

提供统一的插件注册和发现机制，允许用户扩展：
- 自定义持久化后端
- 自定义 Embedding Provider
- 自定义搜索策略/重排序器
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Type


class PluginRegistry:
    """插件注册表 — 管理可扩展的组件。

    支持三种插件类型：
    - backend: 持久化后端（JSONBackend, SQLiteBackend, ...）
    - provider: Embedding Provider（Hash, OpenAI, HuggingFace, ...）
    - scorer: 搜索评分策略（WeightedScorer, ...）

    Example:
        >>> registry = PluginRegistry()
        >>> registry.register_backend("chromadb", ChromaDBBackend)
        >>> backend_cls = registry.get_backend("chromadb")
    """

    def __init__(self) -> None:
        self._backends: dict[str, Type] = {}
        self._providers: dict[str, Type] = {}
        self._scorers: dict[str, Type] = {}
        self._search_strategies: dict[str, Callable] = {}

    # --- 后端注册 ---

    def register_backend(self, name: str, backend_cls: Type) -> None:
        """注册持久化后端。

        Args:
            name: 后端名称
            backend_cls: 后端类

        Raises:
            ValueError: 名称已注册
        """
        if name in self._backends:
            raise ValueError(f"后端 '{name}' 已注册")
        self._backends[name] = backend_cls

    def get_backend(self, name: str) -> Optional[Type]:
        """获取已注册的后端类。

        Args:
            name: 后端名称

        Returns:
            后端类，未注册返回 None
        """
        return self._backends.get(name)

    def list_backends(self) -> list[str]:
        """列出所有已注册的后端名称。"""
        return list(self._backends.keys())

    # --- Provider 注册 ---

    def register_provider(self, name: str, provider_cls: Type) -> None:
        """注册 Embedding Provider。

        Args:
            name: Provider 名称
            provider_cls: Provider 类

        Raises:
            ValueError: 名称已注册
        """
        if name in self._providers:
            raise ValueError(f"Provider '{name}' 已注册")
        self._providers[name] = provider_cls

    def get_provider(self, name: str) -> Optional[Type]:
        """获取已注册的 Provider 类。

        Args:
            name: Provider 名称

        Returns:
            Provider 类，未注册返回 None
        """
        return self._providers.get(name)

    def list_providers(self) -> list[str]:
        """列出所有已注册的 Provider 名称。"""
        return list(self._providers.keys())

    # --- Scorer 注册 ---

    def register_scorer(self, name: str, scorer_cls: Type) -> None:
        """注册搜索评分策略。

        Args:
            name: 评分策略名称
            scorer_cls: 评分器类

        Raises:
            ValueError: 名称已注册
        """
        if name in self._scorers:
            raise ValueError(f"Scorer '{name}' 已注册")
        self._scorers[name] = scorer_cls

    def get_scorer(self, name: str) -> Optional[Type]:
        """获取已注册的评分策略类。

        Args:
            name: 评分策略名称

        Returns:
            评分器类，未注册返回 None
        """
        return self._scorers.get(name)

    def list_scorers(self) -> list[str]:
        """列出所有已注册的评分策略名称。"""
        return list(self._scorers.keys())

    # --- 搜索策略注册 ---

    def register_search_strategy(self, name: str, strategy_fn: Callable) -> None:
        """注册自定义搜索策略函数。

        Args:
            name: 策略名称
            strategy_fn: 策略函数

        Raises:
            ValueError: 名称已注册
        """
        if name in self._search_strategies:
            raise ValueError(f"搜索策略 '{name}' 已注册")
        self._search_strategies[name] = strategy_fn

    def get_search_strategy(self, name: str) -> Optional[Callable]:
        """获取已注册的搜索策略函数。

        Args:
            name: 策略名称

        Returns:
            策略函数，未注册返回 None
        """
        return self._search_strategies.get(name)

    def list_search_strategies(self) -> list[str]:
        """列出所有已注册的搜索策略名称。"""
        return list(self._search_strategies.keys())

    # --- 综合查询 ---

    def list_all(self) -> dict[str, list[str]]:
        """列出所有已注册的插件。

        Returns:
            按类别分组的插件名称字典
        """
        return {
            "backends": self.list_backends(),
            "providers": self.list_providers(),
            "scorers": self.list_scorers(),
            "search_strategies": self.list_search_strategies(),
        }

    def unregister(self, name: str) -> bool:
        """取消注册插件（从所有类别中查找并移除）。

        Args:
            name: 插件名称

        Returns:
            是否找到并移除
        """
        for registry in (self._backends, self._providers, self._scorers, self._search_strategies):
            if name in registry:
                del registry[name]
                return True
        return False


# 全局插件注册表
_global_registry = PluginRegistry()


def get_registry() -> PluginRegistry:
    """获取全局插件注册表。

    Returns:
        全局 PluginRegistry 实例
    """
    return _global_registry
