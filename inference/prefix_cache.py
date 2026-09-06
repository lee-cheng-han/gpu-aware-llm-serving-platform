from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock
from typing import Any


@dataclass(frozen=True)
class PrefixCacheSnapshot:
    entries: int
    cached_tokens: int
    max_entries: int
    max_tokens: int
    hits: int
    misses: int
    insertions: int
    evictions: int
    bypasses: int
    exact_hits: int
    partial_hits: int
    matched_tokens: int

    @property
    def hit_rate(self) -> float:
        lookups = self.hits + self.misses
        return self.hits / lookups if lookups else 0.0


@dataclass(frozen=True)
class PrefixCacheEntry:
    token_ids: tuple[int, ...]
    backend_state: Any

    @property
    def token_count(self) -> int:
        return len(self.token_ids)


@dataclass(frozen=True)
class PrefixMatch:
    token_ids: tuple[int, ...]
    backend_state: Any
    exact: bool

    @property
    def matched_tokens(self) -> int:
        return len(self.token_ids)


class _RadixNode:
    def __init__(self) -> None:
        self.children: dict[int, _RadixNode] = {}
        self.entry: PrefixCacheEntry | None = None


class PrefixCache:
    """Thread-safe, bounded LRU cache for immutable backend prefix snapshots."""

    def __init__(self, max_entries: int = 0, max_tokens: int = 0) -> None:
        if max_entries < 0 or max_tokens < 0:
            raise ValueError("prefix cache limits must be non-negative")
        if bool(max_entries) != bool(max_tokens):
            raise ValueError("prefix cache entry and token limits must both be zero or positive")
        self.max_entries = max_entries
        self.max_tokens = max_tokens
        self._entries: OrderedDict[tuple[int, ...], PrefixCacheEntry] = OrderedDict()
        self._root = _RadixNode()
        self._cached_tokens = 0
        self._hits = self._misses = self._insertions = 0
        self._evictions = self._bypasses = 0
        self._exact_hits = self._partial_hits = self._matched_tokens = 0
        self._lock = Lock()

    @property
    def enabled(self) -> bool:
        return self.max_entries > 0

    def get(self, token_ids: tuple[int, ...]) -> Any | None:
        match = self.longest_prefix(token_ids, exact_only=True)
        return match.backend_state if match is not None else None

    def longest_prefix(
        self,
        token_ids: tuple[int, ...],
        *,
        exact_only: bool = False,
    ) -> PrefixMatch | None:
        with self._lock:
            if not self.enabled:
                self._bypasses += 1
                return None
            node = self._root
            entry: PrefixCacheEntry | None = None
            for token_id in token_ids:
                node = node.children.get(token_id)
                if node is None:
                    break
                if node.entry is not None:
                    entry = node.entry
            if exact_only and (entry is None or entry.token_ids != token_ids):
                entry = None
            if entry is None:
                self._misses += 1
                return None
            self._entries.move_to_end(entry.token_ids)
            self._hits += 1
            exact = entry.token_ids == token_ids
            self._exact_hits += int(exact)
            self._partial_hits += int(not exact)
            self._matched_tokens += entry.token_count
            return PrefixMatch(entry.token_ids, entry.backend_state, exact)

    def put(self, token_ids: tuple[int, ...], backend_state: Any) -> bool:
        if not token_ids:
            raise ValueError("a cached prefix must contain at least one token")
        with self._lock:
            if not self.enabled or len(token_ids) > self.max_tokens:
                self._bypasses += 1
                return False
            replaced = self._entries.pop(token_ids, None)
            if replaced is not None:
                self._cached_tokens -= replaced.token_count
            entry = PrefixCacheEntry(token_ids, backend_state)
            self._entries[token_ids] = entry
            node = self._root
            for token_id in token_ids:
                node = node.children.setdefault(token_id, _RadixNode())
            node.entry = entry
            self._cached_tokens += entry.token_count
            self._insertions += 1
            while (
                len(self._entries) > self.max_entries
                or self._cached_tokens > self.max_tokens
            ):
                evicted_key, evicted = self._entries.popitem(last=False)
                self._cached_tokens -= evicted.token_count
                self._evictions += 1
                self._remove_radix_entry(evicted_key)
            return token_ids in self._entries

    def _remove_radix_entry(self, token_ids: tuple[int, ...]) -> None:
        path: list[tuple[_RadixNode, int, _RadixNode]] = []
        node = self._root
        for token_id in token_ids:
            child = node.children.get(token_id)
            if child is None:
                return
            path.append((node, token_id, child))
            node = child
        node.entry = None
        for parent, token_id, child in reversed(path):
            if child.entry is not None or child.children:
                break
            del parent.children[token_id]

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._root = _RadixNode()
            self._cached_tokens = 0

    def snapshot(self) -> PrefixCacheSnapshot:
        with self._lock:
            return PrefixCacheSnapshot(
                len(self._entries),
                self._cached_tokens,
                self.max_entries,
                self.max_tokens,
                self._hits,
                self._misses,
                self._insertions,
                self._evictions,
                self._bypasses,
                self._exact_hits,
                self._partial_hits,
                self._matched_tokens,
            )
