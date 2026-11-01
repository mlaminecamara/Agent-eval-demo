"""PostgreSQL + pgvector store with an HNSW index on cosine distance.

Parameters (config.yaml, retrieval.hnsw):
- m=16, ef_construction=64: pgvector defaults; enough for a corpus of a few hundred chunks.
- ef_search=40 (per query, SET LOCAL): candidate list size at query time; must be >= k.
Collection filtering uses a WHERE clause. With an HNSW index a filter is applied after the
index scan, so `hnsw.iterative_scan` (pgvector >= 0.8) is enabled to keep scanning until k rows
match; results are re-sorted by distance afterwards since relaxed_order may return them
slightly out of order.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg import sql

from agent_eval.config import HNSWConfig
from agent_eval.ingest.chunker import Chunk
from agent_eval.retrieval.base import IndexSpec, SearchHit, VectorStore

_COLUMNS = (
    "chunk_id, doc_id, collection, chunk_index, text, n_tokens, page_start, page_end, "
    "section, title, url, source_file"
)


class PgVectorStore(VectorStore):
    def __init__(self, spec: IndexSpec, dsn: str, hnsw: HNSWConfig | None = None) -> None:
        super().__init__(spec)
        self.dsn = dsn
        self.hnsw = hnsw or HNSWConfig()
        self.table = sql.Identifier(spec.table_name)
        self._iterative_scan: bool | None = None
        self._conn: psycopg.Connection[Any] | None = None

    def connect(self) -> psycopg.Connection[Any]:
        if self._conn is None or self._conn.closed:
            # prepare_threshold=None disables server-side prepared statements, which connection
            # poolers in transaction mode (e.g. Supabase's Supavisor on port 6543) do not support.
            conn = psycopg.connect(self.dsn, autocommit=True, prepare_threshold=None)
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            register_vector(conn)
            self._conn = conn
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def ensure_schema(self) -> None:
        conn = self.connect()
        conn.execute(
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {table} (
                    chunk_id    text PRIMARY KEY,
                    doc_id      text NOT NULL,
                    collection  text NOT NULL,
                    chunk_index integer NOT NULL,
                    text        text NOT NULL,
                    n_tokens    integer NOT NULL,
                    page_start  integer,
                    page_end    integer,
                    section     text,
                    title       text NOT NULL,
                    url         text,
                    source_file text NOT NULL,
                    metadata    jsonb NOT NULL DEFAULT '{{}}'::jsonb,
                    embedding   vector({dim}) NOT NULL
                )
                """
            ).format(table=self.table, dim=sql.Literal(self.spec.dimension))
        )
        conn.execute(
            sql.SQL(
                "CREATE INDEX IF NOT EXISTS {idx} ON {table} "
                "USING hnsw (embedding vector_cosine_ops) WITH (m = {m}, ef_construction = {efc})"
            ).format(
                idx=sql.Identifier(f"{self.spec.table_name}_hnsw"),
                table=self.table,
                m=sql.Literal(self.hnsw.m),
                efc=sql.Literal(self.hnsw.ef_construction),
            )
        )
        conn.execute(
            sql.SQL("CREATE INDEX IF NOT EXISTS {idx} ON {table} (collection, doc_id)").format(
                idx=sql.Identifier(f"{self.spec.table_name}_coll"), table=self.table
            )
        )
        # Row-level security with no policies: on Supabase, tables in `public` are otherwise
        # readable and writable through the REST API with the anon key. The table owner (the
        # role this code connects with) bypasses RLS, so ingestion and search are unaffected.
        conn.execute(
            sql.SQL("ALTER TABLE {table} ENABLE ROW LEVEL SECURITY").format(table=self.table)
        )

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        self.ensure_schema()
        conn = self.connect()
        query = sql.SQL(
            "INSERT INTO {table} (" + _COLUMNS + ", embedding) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (chunk_id) DO UPDATE SET text = EXCLUDED.text, "
            "embedding = EXCLUDED.embedding, section = EXCLUDED.section"
        ).format(table=self.table)
        with conn.transaction(), conn.cursor() as cur:
            cur.executemany(
                query,
                [
                    (
                        c.chunk_id,
                        c.doc_id,
                        c.collection,
                        c.chunk_index,
                        c.text,
                        c.n_tokens,
                        c.page_start,
                        c.page_end,
                        c.section,
                        c.title,
                        c.url,
                        c.source_file,
                        np.asarray(v, dtype=np.float32),
                    )
                    for c, v in zip(chunks, vectors, strict=True)
                ],
            )

    def delete_docs(self, doc_ids: list[str]) -> None:
        self.ensure_schema()
        self.connect().execute(
            sql.SQL("DELETE FROM {table} WHERE doc_id = ANY(%s)").format(table=self.table),
            (doc_ids,),
        )

    def _supports_iterative_scan(self, conn: psycopg.Connection[Any]) -> bool:
        if self._iterative_scan is None:
            row = conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            version = (row.fetchone() or ("0",))[0]
            major, minor = (int(x) for x in version.split(".")[:2])
            self._iterative_scan = (major, minor) >= (0, 8)
        return self._iterative_scan

    def search(
        self, query_vector: list[float], k: int, collections: list[str] | None = None
    ) -> list[SearchHit]:
        conn = self.connect()
        vec = np.asarray(query_vector, dtype=np.float32)
        where = sql.SQL("WHERE collection = ANY(%(collections)s)") if collections else sql.SQL("")
        query = sql.SQL(
            "SELECT " + _COLUMNS + ", embedding <=> %(vec)s AS distance FROM {table} {where} "
            "ORDER BY embedding <=> %(vec)s LIMIT %(k)s"
        ).format(table=self.table, where=where)
        with conn.transaction():
            conn.execute(
                sql.SQL("SET LOCAL hnsw.ef_search = {}").format(
                    sql.Literal(max(self.hnsw.ef_search, k))
                )
            )
            if collections and self._supports_iterative_scan(conn):
                conn.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
            rows = conn.execute(query, {"vec": vec, "k": k, "collections": collections}).fetchall()

        rows.sort(key=lambda r: (r[-1], r[0]))
        hits = []
        for rank, r in enumerate(rows, start=1):
            chunk = Chunk(
                chunk_id=r[0],
                doc_id=r[1],
                collection=r[2],
                chunk_index=r[3],
                text=r[4],
                n_tokens=r[5],
                page_start=r[6],
                page_end=r[7],
                section=r[8],
                title=r[9],
                url=r[10],
                source_file=r[11],
            )
            hits.append(SearchHit(chunk=chunk, score=1.0 - float(r[-1]), rank=rank))
        return hits

    def count(self, collections: list[str] | None = None) -> int:
        self.ensure_schema()
        where = sql.SQL("WHERE collection = ANY(%s)") if collections else sql.SQL("")
        row = (
            self.connect()
            .execute(
                sql.SQL("SELECT count(*) FROM {table} {where}").format(
                    table=self.table, where=where
                ),
                (collections,) if collections else None,
            )
            .fetchone()
        )
        return int(row[0]) if row else 0

    def index_definition(self) -> str | None:
        """HNSW index definition as stored by PostgreSQL (used by tests and the README)."""
        row = (
            self.connect()
            .execute(
                "SELECT indexdef FROM pg_indexes WHERE tablename = %s AND indexname = %s",
                (self.spec.table_name, f"{self.spec.table_name}_hnsw"),
            )
            .fetchone()
        )
        return str(row[0]) if row else None
