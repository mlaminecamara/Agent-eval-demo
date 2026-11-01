import itertools
from pathlib import Path

import pytest

from agent_eval.config import PROJECT_ROOT
from agent_eval.ingest.chunker import chunk_document, count_tokens
from agent_eval.ingest.loader import Block, Document, clean_text, load_html, load_manifest, load_pdf

FIXTURES = Path(__file__).parent / "fixtures"
CORPUS = PROJECT_ROOT / "data" / "corpus"
FAQ_PDF = CORPUS / "impots_faq_je-decouvre-la-fe.pdf"


def test_clean_text_fixes_extraction_artefacts():
    assert clean_text("a b   c ,  d ( e )") == "a b c, d (e)"
    assert clean_text("l’entreprise") == "l'entreprise"


def test_load_html_keeps_content_and_sections():
    blocks = load_html(FIXTURES / "sample.html")
    texts = [b.text for b in blocks]
    assert "Texte de la section A, avec une espace insécable." in texts
    assert not any("navigation" in t or "Pied de page" in t or "tracking" in t for t in texts)
    by_text = {b.text: b for b in blocks}
    assert by_text["Élément de liste deux"].section == "Section A"
    assert by_text["Texte de la section B."].section == "Section B"
    assert all(b.page is None for b in blocks)


@pytest.mark.skipif(not FAQ_PDF.exists(), reason="corpus PDF not present")
def test_load_pdf_tracks_pages_and_sections():
    blocks = load_pdf(FAQ_PDF)
    pages = {b.page for b in blocks}
    assert min(pages) == 1 and max(pages) >= 2
    assert any(b.section and b.section.startswith("5.") for b in blocks)
    assert "septembre 2027" in "\n".join(b.text for b in blocks)


def test_manifest_skips_excluded_documents(tmp_path):
    (tmp_path / "a.html").write_bytes((FIXTURES / "sample.html").read_bytes())
    (tmp_path / "manifest.yaml").write_text(
        "documents:\n"
        "  - {id: kept, file: a.html, include: true}\n"
        "  - {id: dropped, file: a.html, include: false}\n",
        encoding="utf-8",
    )
    assert [d.doc_id for d in load_manifest(tmp_path / "manifest.yaml")] == ["kept"]
    all_docs = load_manifest(tmp_path / "manifest.yaml", include_excluded=True)
    assert [d.doc_id for d in all_docs] == ["kept", "dropped"]


def test_repo_manifest_loads_every_document():
    docs = load_manifest(CORPUS / "manifest.yaml")
    assert {"faq-fe", "cgi-ann2-242nonies", "decret-2022-1299"} <= {d.doc_id for d in docs}
    assert all(d.blocks for d in docs)


def test_running_headers_removed_but_article_dates_kept():
    path = CORPUS / "legifrance_cgi-ann2_242nonies_vigueur-2026-09-01.pdf"
    if not path.exists():
        pytest.skip("corpus PDF not present")
    blocks = load_pdf(path)
    text = "\n".join(b.text for b in blocks)
    assert "legifrance.gouv.fr/codes" not in text  # print footer
    assert "EN VIGUEUR DEPUIS LE" in text  # per-article validity line
    assert any(b.section == "Article 242 nonies B" for b in blocks)


def test_manifest_missing_file_raises(tmp_path):
    (tmp_path / "manifest.yaml").write_text(
        "documents:\n  - id: x\n    file: missing.pdf\n", encoding="utf-8"
    )
    with pytest.raises(FileNotFoundError):
        load_manifest(tmp_path / "manifest.yaml")


def _doc(n_blocks: int = 40) -> Document:
    blocks = [
        Block(
            text=f"Paragraphe {i} : " + " ".join(f"mot{i}_{j}" for j in range(25)) + ".",
            page=1 + i // 10,
            section=f"Section {i // 10}",
        )
        for i in range(n_blocks)
    ]
    return Document(doc_id="test", title="Doc de test", source_file="test.txt", blocks=blocks)


def test_chunks_respect_size_and_metadata():
    chunks = chunk_document(_doc(), chunk_size=200, overlap=40)
    assert len(chunks) > 3
    assert all(c.n_tokens <= 200 for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    for c in chunks:
        assert c.page_start is not None and c.page_end is not None
        assert c.page_start <= c.page_end
    assert chunks[0].section == "Section 0"


def test_chunks_overlap_when_requested():
    # Paragraphs (~108 tokens) are longer than the overlap: the tail of the last one is carried.
    with_overlap = chunk_document(_doc(), chunk_size=200, overlap=40)
    for a, b in itertools.pairwise(with_overlap):
        head = b.text.split("\n")[0]
        assert a.text.endswith(head)
        assert 30 <= count_tokens(head) <= 41
    # Overlap large enough for whole paragraphs: the last paragraph is repeated as is.
    whole = chunk_document(_doc(), chunk_size=400, overlap=150)
    for a, b in itertools.pairwise(whole):
        assert b.text.startswith(a.text.split("\n")[-1])
    without = chunk_document(_doc(), chunk_size=200, overlap=0)
    for a, b in itertools.pairwise(without):
        assert not a.text.endswith(b.text.split("\n")[0])


def test_chunk_ids_are_deterministic():
    a = [c.chunk_id for c in chunk_document(_doc(), 200, 40)]
    b = [c.chunk_id for c in chunk_document(_doc(), 200, 40)]
    assert a == b and len(set(a)) == len(a)


def test_long_block_is_split():
    long_text = " ".join(f"token{i}" for i in range(2000))
    doc = Document("long", "Long", "long.txt", blocks=[Block(long_text, 1, None)])
    chunks = chunk_document(doc, chunk_size=300, overlap=0)
    assert len(chunks) >= count_tokens(long_text) // 300
    assert all(c.n_tokens <= 301 for c in chunks)  # decode/encode can shift by one token


def test_invalid_chunk_params():
    with pytest.raises(ValueError):
        chunk_document(_doc(), chunk_size=100, overlap=100)
