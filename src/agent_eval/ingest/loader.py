"""Load corpus documents (PDF, HTML, Markdown/text) into blocks with page and section labels.

A block is a paragraph-sized piece of text that remembers where it came from (page number for
PDFs, nearest heading for every format). The chunker packs blocks into chunks, so citations can
point to a page and section.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from bs4 import BeautifulSoup
from pypdf import PdfReader


@dataclass
class Block:
    text: str
    page: int | None  # 1-based page number (PDF only)
    section: str | None  # nearest heading seen so far


@dataclass
class Document:
    doc_id: str
    title: str
    source_file: str
    url: str | None = None
    collection: str = "corpus"
    blocks: list[Block] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(b.text for b in self.blocks)


# FAQ-style numbered headings: "1.", "1.3", "8.1." followed by text. Kept short to avoid
# treating numbered list items inside paragraphs as headings.
# The number must contain a dot ("1.", "1.3", "1.3.") so that "1 du CGI)" is not a heading.
_NUMBERED_HEADING = re.compile(r"^\d{1,2}\.(\d{1,2}\.?)*\s+\S.{2,180}$")
_FICHE_HEADING = re.compile(r"^(Fiche \d+|Bon à savoir|À retenir)\b", re.IGNORECASE)


def clean_text(text: str) -> str:
    """Normalise unicode and whitespace; keep line breaks (they separate blocks)."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace(" ", " ").replace(" ", " ").replace("’", "'")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    out = "\n".join(lines)
    # Fix spacing artefacts from PDF extraction: " ," " ." " )" "( ".
    out = re.sub(r" +([,.;:)\]])", r"\1", out)
    out = re.sub(r"([(\[]) +", r"\1", out)
    return re.sub(r"\n{3,}", "\n\n", out).strip()


# Legal texts: "Article 242 nonies B", "Article 1", "Article 3".
_ARTICLE_HEADING = re.compile(r"^Article \d+(\s[a-z]+)*(\s[A-Z])?$")


def _is_heading(line: str) -> bool:
    return bool(
        _NUMBERED_HEADING.match(line) or _FICHE_HEADING.match(line) or _ARTICLE_HEADING.match(line)
    )


def _repeated_lines(pages: list[str]) -> set[str]:
    """Running headers/footers: lines in the first or last two lines of a page that recur
    (digits ignored) on at least half the pages, e.g. the print date and URL Légifrance adds to
    every printed page. Body lines are never candidates, so repeated content such as
    "EN VIGUEUR DEPUIS LE ..." under each article is kept."""
    if len(pages) < 3:
        return set()
    counts: dict[str, int] = {}
    for text in pages:
        lines = [line for line in text.splitlines() if line]
        edges = lines[:2] + lines[-2:]
        for key in {re.sub(r"\d+", "#", line) for line in edges}:
            counts[key] = counts.get(key, 0) + 1
    return {k for k, n in counts.items() if n >= len(pages) / 2}


def _paragraphs(text: str) -> list[str]:
    """Merge PDF lines into paragraphs: a new paragraph starts after a blank line, at a heading,
    at a bullet, or after a line ending with sentence punctuation."""
    paras: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if not line:
            if current:
                paras.append(" ".join(current))
                current = []
            continue
        starts_new = _is_heading(line) or bool(re.match(r"^[•\-–➢✗✓▪◦*]\s?", line))
        if starts_new and current:
            paras.append(" ".join(current))
            current = []
        current.append(line)
        if _is_heading(line) or line.endswith((".", "?", "!", ":", ";")):
            paras.append(" ".join(current))
            current = []
    if current:
        paras.append(" ".join(current))
    return [p for p in paras if p.strip()]


def load_pdf(path: Path) -> list[Block]:
    reader = PdfReader(str(path))
    pages = [clean_text(page.extract_text() or "") for page in reader.pages]
    repeated = _repeated_lines(pages)
    blocks: list[Block] = []
    section: str | None = None
    for page_no, text in enumerate(pages, start=1):
        lines = text.splitlines()
        non_empty = [i for i, ln in enumerate(lines) if ln]
        edges = set(non_empty[:2] + non_empty[-2:])
        kept = [
            ln
            for i, ln in enumerate(lines)
            if not (i in edges and re.sub(r"\d+", "#", ln) in repeated)
        ]
        for para in _paragraphs("\n".join(kept)):
            if _is_heading(para):
                section = para[:200]
            blocks.append(Block(text=para, page=page_no, section=section))
    return blocks


def load_html(path: Path) -> list[Block]:
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "form", "noscript", "aside"]):
        tag.decompose()
    root = soup.find("main") or soup.find("article") or soup.body or soup
    blocks: list[Block] = []
    section: str | None = None
    for el in root.find_all(["h1", "h2", "h3", "h4", "p", "li", "td"]):
        if el.find(["p", "li"]):  # skip containers; their children are visited on their own
            continue
        text = clean_text(el.get_text(" "))
        if not text:
            continue
        if el.name in {"h1", "h2", "h3", "h4"}:
            section = text[:200]
        blocks.append(Block(text=text, page=None, section=section))
    return blocks


def load_text(path: Path) -> list[Block]:
    blocks: list[Block] = []
    section: str | None = None
    for para in re.split(r"\n\s*\n", clean_text(path.read_text(encoding="utf-8"))):
        para = para.strip()
        if not para:
            continue
        if para.startswith("#"):
            section = para.lstrip("# ").splitlines()[0][:200]
        blocks.append(Block(text=para, page=None, section=section))
    return blocks


def load_file(path: Path) -> list[Block]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path)
    if suffix in {".html", ".htm"}:
        return load_html(path)
    if suffix in {".md", ".txt"}:
        return load_text(path)
    raise ValueError(f"Unsupported file type: {path}")


def load_manifest(manifest_path: Path, include_excluded: bool = False) -> list[Document]:
    """Load every document listed in the manifest (only include: true unless asked)."""
    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    base = manifest_path.parent
    docs: list[Document] = []
    for entry in data.get("documents", []):
        if not entry.get("include", True) and not include_excluded:
            continue
        path = base / entry["file"]
        if not path.exists():
            raise FileNotFoundError(f"Manifest lists {entry['file']} but {path} is missing")
        docs.append(
            Document(
                doc_id=entry["id"],
                title=entry.get("title", entry["id"]),
                source_file=entry["file"],
                url=entry.get("url"),
                collection=entry.get("collection", "corpus"),
                blocks=load_file(path),
            )
        )
    return docs
