from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple, Union
import sqlite3

import numpy as np


_BASE_PRAGMAS = """
PRAGMA synchronous=NORMAL;
PRAGMA temp_store=MEMORY;
PRAGMA mmap_size=30000000000;
PRAGMA locking_mode=EXCLUSIVE;
PRAGMA foreign_keys=OFF;
"""

# Conservatively tuned pragmas for read-only access on shared/network filesystems.
_BASE_RO_PRAGMAS = """
PRAGMA query_only=ON;
PRAGMA temp_store=MEMORY;
PRAGMA foreign_keys=OFF;
"""

_NEW_DB_PRAGMAS = """
PRAGMA page_size=32768;
PRAGMA cache_size=-524288;
VACUUM;
"""

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS embedding (
  caption   TEXT PRIMARY KEY,
  emb       BLOB NOT NULL,
  frequency INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS metadata (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
"""


ArrayLike = Union[np.ndarray, Sequence[float], bytes, bytearray, memoryview]


class SQLiteClient:
    """Minimal SQLite client for reading/writing embeddings only."""

    def __init__(
        self,
        db_path: Union[str, Path],
        init_schema: bool = True,
        *,
        read_only: bool = False,
        immutable: bool = False,
        timeout: float = 5.0,
    ) -> None:
        self.db_path = Path(db_path)
        self._conn: Optional[sqlite3.Connection] = None
        self._read_only = read_only
        self._immutable = immutable
        self._timeout = timeout
        self._connect_and_prepare(init_schema=init_schema)

    # ---- lifecycle ----
    def _connect_and_prepare(self, init_schema: bool) -> None:
        new_db = not self.db_path.exists()
        if self._read_only:
            # Use SQLite URI to open in read-only mode. immutable=1 hints that the file will not change.
            uri = f"file:{self.db_path}?mode=ro"
            if self._immutable:
                uri += "&immutable=1"
            self._conn = sqlite3.connect(uri, uri=True, timeout=self._timeout)
        else:
            self._conn = sqlite3.connect(str(self.db_path), timeout=self._timeout)
        self._conn.row_factory = sqlite3.Row
        cur = self._conn.cursor()
        if self._read_only:
            cur.executescript(_BASE_RO_PRAGMAS)
        else:
            cur.executescript(_BASE_PRAGMAS)
            if new_db:
                cur.executescript(_NEW_DB_PRAGMAS)
            if init_schema:
                cur.executescript(_SCHEMA_SQL)
                self._conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "SQLiteClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ---- embeddings ----
    @staticmethod
    def _to_blob(
        array_like: ArrayLike, dtype: Union[np.dtype, str] = np.float32
    ) -> sqlite3.Binary:
        if isinstance(array_like, (bytes, bytearray, memoryview)):
            return sqlite3.Binary(bytes(array_like))
        arr = np.asarray(array_like, dtype=dtype)
        return sqlite3.Binary(arr.tobytes())

    def insert_embeddings(
        self,
        items: Iterable[Tuple[str, ArrayLike]],
        *,
        dtype: Union[np.dtype, str] = np.float32,
    ) -> None:
        conn = self._ensure_conn()
        payload = [
            (caption, self._to_blob(emb, dtype=dtype), frequency)
            for caption, emb, frequency in items
        ]
        # BEGIN IMMEDIATE to reduce writer contention and get better throughput
        conn.execute("BEGIN IMMEDIATE")
        conn.executemany(
            "INSERT OR REPLACE INTO embedding(caption, emb, frequency) VALUES (?, ?, ?)",
            payload,
        )
        conn.commit()

    def insert_metadata(self, items: Iterable[Tuple[str, str]]) -> None:
        conn = self._ensure_conn()
        payload = [(key, value) for key, value in items]
        conn.executemany(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)", payload
        )
        conn.commit()

    def get_embedding(
        self,
        caption: str,
        *,
        dtype: Union[np.dtype, str] = np.float32,
    ) -> Optional[np.ndarray]:
        conn = self._ensure_conn()
        row = conn.execute(
            "SELECT emb FROM embedding WHERE caption=?",
            (caption,),
        ).fetchone()
        if row is None:
            return None
        return np.frombuffer(row[0], dtype=dtype)

    def count_embeddings(self) -> int:
        conn = self._ensure_conn()
        return conn.execute("SELECT COUNT(*) FROM embedding").fetchone()[0]

    # (no extra utilities; write and read only)

    def _ensure_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("Connection is closed")
        return self._conn


__all__ = ["SQLiteClient"]
