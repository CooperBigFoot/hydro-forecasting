import threading
import time
import polars as pl
from typing import Optional, Dict

class FileCache:
    """
    Worker‑specific Parquet file cache with LRU eviction.
    """
    def __init__(self, max_files: int = 10, max_memory_mb: int = 500) -> None:
        self._cache: Dict[str, pl.DataFrame] = {}
        self._access: Dict[str, float] = {}
        self.max_files = max_files
        self.max_memory_mb = max_memory_mb
        self._current_mem = 0.0
        self._lock = threading.Lock()

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        # remove unpicklable lock
        state.pop('_lock', None)
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        # recreate lock after unpickle
        self._lock = threading.Lock()

    def get_file(self, file_path: str, columns: Optional[list[str]] = None) -> Optional[pl.DataFrame]:
        """
        Return a Polars DataFrame, loading and caching it if necessary.
        """
        with self._lock:
            now = time.time()
            if file_path in self._cache:
                self._access[file_path] = now
                return self._cache[file_path]
            try:
                df = pl.read_parquet(file_path, columns=columns)
                size_mb = df.estimated_size() / (1024 * 1024)
                # evict if needed
                while (len(self._cache) >= self.max_files or 
                       self._current_mem + size_mb > self.max_memory_mb) and self._cache:
                    lru = min(self._access, key=self._access.get)
                    freed = self._cache[lru].estimated_size() / (1024 * 1024)
                    del self._cache[lru]; del self._access[lru]
                    self._current_mem -= freed
                # insert
                self._cache[file_path] = df
                self._access[file_path] = now
                self._current_mem += size_mb
                return df
            except Exception:
                return None
