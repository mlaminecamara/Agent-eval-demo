"""Token-based chunking with overlap, keeping page and section metadata.

Blocks (paragraphs) are packed greedily into chunks of at most `chunk_size` tokens. A block
longer than `chunk_size` is split on token boundaries. The last blocks of a chunk, up to
`overlap` tokens, are repeated at the start of the next chunk so that facts sitting on a chunk
boundary stay retrievable; if the last block alone exceeds `overlap`, its last `overlap`
tokens are repeated instead. Token counts use tiktoken cl100k_base for every provider: it is
exact for OpenAI embeddings and an approximation for chat models.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache

import tiktoken

from agent_eval.ingest.loader import Block, Document

ENCODING = "cl100k_base"


@lru_cache(maxsize=1)
def _encoder() -> tiktoken.Encoding:
    return tiktoken.get_encoding(ENCODING)


def count_tokens(text: str) -> int:
    return len(_encoder().encode(text))


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    collection: str
    chunk_index: int
    text: str
    n_tokens: int
    page_start: int | None
    page_end: int | None
    section: str | None
    title: str
    url: str | None
    source_file: str

    def embedding_text(self) -> str:
        """Text sent to the embedder: a short header gives each chunk its document context."""
        header = self.title if not self.section else f"{self.title} — {self.section}"
        return f"{header}\n{self.text}"

    def covers_page(self, page: int) -> bool:
        if self.page_start is None or self.page_end is None:
            return False
        return self.page_start <= page <= self.page_end


@dataclass
class _Piece:
    text: str
    n_tokens: int
    page: int | None
    section: str | None


def _split_long(block: Block, chunk_size: int) -> list[_Piece]:
    tokens = _encoder().encode(block.text)
    if len(tokens) <= chunk_size:
        return [_Piece(block.text, len(tokens), block.page, block.section)]
    return [
        _Piece(
            _encoder().decode(tokens[i : i + chunk_size]),
            len(tokens[i : i + chunk_size]),
            block.page,
            block.section,
        )
        for i in range(0, len(tokens), chunk_size)
    ]


def chunk_document(doc: Document, chunk_size: int, overlap: int) -> list[Chunk]:
    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise ValueError("need chunk_size > 0 and 0 <= overlap < chunk_size")

    pieces = [p for b in doc.blocks for p in _split_long(b, chunk_size)]
    groups: list[list[_Piece]] = []
    current: list[_Piece] = []
    current_tokens = 0
    sep = 1  # tokens for the newline joining pieces (approximate)

    for piece in pieces:
        if current and current_tokens + sep + piece.n_tokens > chunk_size:
            groups.append(current)
            # Carry trailing pieces (whole paragraphs) as overlap, if they fit.
            carried: list[_Piece] = []
            carried_tokens = 0
            for prev in reversed(current):
                if carried_tokens + prev.n_tokens > overlap:
                    break
                carried.insert(0, prev)
                carried_tokens += prev.n_tokens + sep
            if not carried and overlap > 0:
                # Last paragraph is longer than the overlap: carry its last `overlap` tokens.
                tail = _encoder().encode(current[-1].text)[-overlap:]
                last = current[-1]
                carried = [_Piece(_encoder().decode(tail), len(tail), last.page, last.section)]
                carried_tokens = len(tail) + sep
            if carried_tokens + piece.n_tokens > chunk_size:
                carried, carried_tokens = [], 0
            current, current_tokens = carried, carried_tokens
        current.append(piece)
        current_tokens += piece.n_tokens + sep
    if current:
        groups.append(current)

    chunks: list[Chunk] = []
    for idx, group in enumerate(groups):
        text = "\n".join(p.text for p in group)
        pages = [p.page for p in group if p.page is not None]
        section = next((p.section for p in group if p.section), None)
        digest = hashlib.sha1(f"{doc.doc_id}|{chunk_size}|{overlap}|{idx}|{text}".encode())
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}:{idx:04d}:{digest.hexdigest()[:8]}",
                doc_id=doc.doc_id,
                collection=doc.collection,
                chunk_index=idx,
                text=text,
                n_tokens=count_tokens(text),
                page_start=min(pages) if pages else None,
                page_end=max(pages) if pages else None,
                section=section,
                title=doc.title,
                url=doc.url,
                source_file=doc.source_file,
            )
        )
    return chunks


def chunk_documents(docs: list[Document], chunk_size: int, overlap: int) -> list[Chunk]:
    return [c for d in docs for c in chunk_document(d, chunk_size, overlap)]
