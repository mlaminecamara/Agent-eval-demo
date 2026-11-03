"""Integrity checks on the golden dataset and an end-to-end mock eval run."""

import json
from collections import Counter

import pytest

from agent_eval.config import PROJECT_ROOT
from agent_eval.ingest.loader import load_manifest
from evals.dataset import load_dataset
from evals.metrics.answer_quality import evaluate_facts

DATASET = PROJECT_ROOT / "evals" / "datasets" / "golden.jsonl"


@pytest.fixture(scope="module")
def cases():
    return load_dataset(DATASET)


@pytest.fixture(scope="module")
def docs():
    manifests = [
        PROJECT_ROOT / "data" / "corpus" / "manifest.yaml",
        PROJECT_ROOT / "data" / "adversarial" / "manifest.yaml",
    ]
    return {d.doc_id: d for m in manifests for d in load_manifest(m)}


def test_dataset_size_and_categories(cases):
    counts = Counter(c.category for c in cases)
    assert 28 <= len(cases) <= 40
    assert set(counts) == {"factual", "multi_step", "out_of_scope", "injection"}
    assert min(counts.values()) >= 5


def test_expected_sources_exist_in_the_corpus(cases, docs):
    for case in cases:
        for src in case.expected_sources:
            assert src.doc_id in docs, (case.id, src.doc_id)
            if src.page is not None:
                pages = {b.page for b in docs[src.doc_id].blocks}
                assert src.page in pages, (case.id, src.doc_id, src.page)


def test_reference_answers_pass_their_own_fact_checks(cases):
    for case in cases:
        r = evaluate_facts(case.expected_facts, case.expected_answer)
        assert r.all_found in (True, None), (case.id, r.missing)


def test_injection_cases_reference_existing_adversarial_docs(cases, docs):
    for case in cases:
        if case.injection and case.injection.adversarial_doc:
            assert case.injection.adversarial_doc in docs
            assert "adversarial" in case.collections
            assert docs[case.injection.adversarial_doc].collection == "adversarial"


def test_mock_eval_run_end_to_end(tmp_path, monkeypatch):
    from evals import run_eval

    monkeypatch.setattr(run_eval, "RESULTS_DIR", tmp_path)
    code = run_eval.main(
        [
            "--mock",
            "--case",
            "multi-01",
            "--case",
            "oos-02",
            "--case",
            "inj-01",
            "--case",
            "fact-04",
        ]
    )
    assert code == 0
    (json_path,) = (tmp_path / "mock").glob("*.json")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["mock"] is True and payload["complete"] is True
    assert payload["dataset"]["validated"] is True
    by_id = {c["id"]: c for c in payload["cases"]}
    assert by_id["multi-01"]["facts"]["all_found"] is True
    assert by_id["multi-01"]["tools"]["all_matched"] is True
    assert by_id["oos-02"]["safety"]["refused"] is True
    assert by_id["inj-01"]["safety"]["injection_resisted"] is True
    overall = payload["metrics"]["overall"]
    assert overall["n_cases"] == 4
    assert overall["safety"]["correct_refusal"]["n"] == 1
    md = json_path.with_suffix(".md").read_text(encoding="utf-8")
    assert "MOCK RUN" in md and "UNVALIDATED DATASET" not in md


def test_real_run_refuses_draft_dataset(monkeypatch, tmp_path):
    from evals import run_eval

    # A draft copy of one case: the refusal must happen before any API call.
    case = json.loads(DATASET.read_text(encoding="utf-8").splitlines()[0])
    case["status"] = "draft"
    draft = tmp_path / "draft.jsonl"
    draft.write_text(json.dumps(case, ensure_ascii=False) + "\n", encoding="utf-8")
    results = tmp_path / "results"
    results.mkdir()
    monkeypatch.setattr(run_eval, "RESULTS_DIR", results)
    args = ["--provider", "openai", "--dataset", str(draft), "--yes"]
    assert run_eval.main(args) == 2
    assert not any(results.iterdir())
