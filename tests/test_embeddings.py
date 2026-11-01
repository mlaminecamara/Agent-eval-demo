import math

from agent_eval.embeddings import build_embedder


def _cos(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_mock_embeddings_are_normalised_and_deterministic(mock_settings):
    emb = build_embedder(mock_settings.embeddings)
    v1, v2 = emb.embed(["Factur-X est un format hybride", "Factur-X est un format hybride"])
    assert len(v1) == 256
    assert math.isclose(sum(x * x for x in v1), 1.0, rel_tol=1e-9)
    assert v1 == v2
    assert emb.tokens_used > 0


def test_mock_embeddings_reflect_lexical_overlap(mock_settings):
    emb = build_embedder(mock_settings.embeddings)
    q, close, far = emb.embed(
        [
            "format Factur-X facture électronique",
            "Le format Factur-X combine un PDF et une facture électronique XML",
            "Recette de la tarte aux pommes caramélisées",
        ]
    )
    assert _cos(q, close) > _cos(q, far)


def test_mock_embeddings_handle_empty_text(mock_settings):
    emb = build_embedder(mock_settings.embeddings)
    (v,) = emb.embed([""])
    assert math.isclose(sum(x * x for x in v), 1.0)
