"""CLI: python -m agent_eval.ingest [--chunk-size N --overlap M] [--yes]

Loads the documents listed in data/corpus/manifest.yaml (plus data/adversarial/ when it has a
manifest), chunks them, embeds them and writes them to the configured vector store. With a
real embedding provider, prints the estimated cost and asks for confirmation first.
"""

from __future__ import annotations

import argparse
import sys

from agent_eval.config import load_settings
from agent_eval.embeddings import build_embedder
from agent_eval.ingest.indexer import embedding_token_estimate, index_chunks, plan_chunks
from agent_eval.ingest.loader import load_manifest
from agent_eval.observability.cost import PricingTable
from agent_eval.observability.logging import configure_logging
from agent_eval.retrieval.retriever import build_store


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--overlap", type=int)
    parser.add_argument("--store", choices=["pgvector", "memory"])
    parser.add_argument("--yes", action="store_true", help="skip the cost confirmation")
    parser.add_argument("--allow-unknown-price", action="store_true")
    args = parser.parse_args(argv)

    overrides: dict[str, object] = {}
    if args.chunk_size:
        overrides["retrieval.chunk_size"] = args.chunk_size
    if args.overlap is not None:
        overrides["retrieval.chunk_overlap"] = args.overlap
    if args.store:
        overrides["retrieval.store"] = args.store
    settings = load_settings(overrides=overrides)
    configure_logging(settings.logging.level)

    manifests = [settings.resolve_path(settings.corpus.manifest)]
    adversarial = settings.resolve_path(settings.corpus.adversarial_dir) / "manifest.yaml"
    if adversarial.exists():
        manifests.append(adversarial)
    docs = [d for m in manifests for d in load_manifest(m)]

    r = settings.retrieval
    chunks = plan_chunks(docs, r.chunk_size, r.chunk_overlap)
    est_tokens = embedding_token_estimate(chunks)
    model = settings.embeddings.resolved_model
    print(
        f"{len(docs)} documents -> {len(chunks)} chunks "
        f"(chunk_size={r.chunk_size}, overlap={r.chunk_overlap}), "
        f"~{est_tokens:,} tokens to embed with {model}"
    )

    if settings.embeddings.provider != "mock":
        pricing = PricingTable.load(settings.resolve_path(settings.cost_guard.pricing_file))
        cost = pricing.cost(model, est_tokens, 0)
        if cost is None and not args.allow_unknown_price:
            print(
                f"Refusing to run: no price for {model} in config/pricing.yaml "
                "(or pass --allow-unknown-price)."
            )
            return 2
        shown = "UNKNOWN" if cost is None else f"${cost:.4f}"
        print(f"Estimated embedding cost: {shown}")
        if not args.yes and input("Proceed? [y/N] ").strip().lower() not in {"y", "yes"}:
            print("Aborted.")
            return 1

    store = build_store(settings)
    report = index_chunks(chunks, store, build_embedder(settings.embeddings))
    print(
        f"Indexed {report.chunks} chunks from {report.documents} documents into "
        f"{settings.retrieval.store}:{report.table} ({report.embedding_tokens:,} embedding "
        "tokens)"
    )
    for doc_id, n in sorted(report.per_document.items()):
        print(f"  {doc_id}: {n} chunks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
