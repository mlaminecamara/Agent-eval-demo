"""evals.compare (grouping, exclusions, statistics), evals.report and evals.run_matrix (plan and
estimate only: nothing here calls an API)."""

import json

import pytest

from evals import compare, report, run_matrix
from evals.compare import wilson_ci
from evals.metrics.stats import wilson

DATASET_SHA = "d" * 16
JUDGE_SHA = "j" * 16


def _case(cid, ok, must_refuse=False, cost=0.01):
    return {
        "id": cid,
        "must_refuse": must_refuse,
        "judge": {"pass": ok, "cost_usd": 0.001},
        "facts": {"all_found": None if must_refuse else ok},
        "tools": {"all_matched": ok, "expected": 0 if must_refuse else 1},
        "retrieval": None if must_refuse else {"direct": {"hit_at_5": True}},
        "safety": {
            "refusal_correct": ok if must_refuse else None,
            "refused": must_refuse and ok,
            "injection_resisted": None,
            "leaked_system_prompt": False,
        },
        "latency_s": 1.0,
        "cost_usd": cost,
        "steps": 2,
    }


def _run(run_id, agent="react", outcomes=(True, True, False), **over):
    cfg = {
        "agent": agent,
        "llm_provider": "openai",
        "llm_model": "gpt-x",
        "embedding_model": "emb",
        "chunk_size": 500,
        "chunk_overlap": 50,
        "top_k": 5,
        "rerank": False,
        "judge_model": "judge-x",
        "judge_prompt_sha256_16": JUDGE_SHA,
        "git_commit": "abc",
    }
    run = {
        "run_id": run_id,
        "mock": False,
        "complete": True,
        "incomplete_reason": None,
        "config": cfg,
        "dataset": {"sha256_16": DATASET_SHA, "validated": True},
        "prices_verified": {"gpt-x": True},
        "cases": [_case(f"c{i}", ok) for i, ok in enumerate(outcomes)]
        + [_case("r1", True, must_refuse=True)],
    }
    for key, value in over.items():
        if key in cfg:
            cfg[key] = value
        else:
            run[key] = value
    return run


def _write(tmp_path, *runs):
    for run in runs:
        (tmp_path / f"{run['run_id']}.json").write_text(json.dumps(run), encoding="utf-8")


def test_runs_are_grouped_and_non_comparable_runs_excluded(tmp_path):
    _write(
        tmp_path,
        _run("a1"),
        _run("a2", outcomes=(True, False, False)),
        _run("b1", agent="langgraph", outcomes=(True, True, True)),
        _run("old-judge", judge_prompt_sha256_16="other"),
        _run("draft", dataset={"sha256_16": DATASET_SHA, "validated": False}),
        _run("other-ds", dataset={"sha256_16": "x" * 16, "validated": True}),
        _run("broken", complete=False, incomplete_reason="budget"),
    )
    groups, excluded = compare.load_runs(tmp_path, DATASET_SHA, JUDGE_SHA)
    assert sorted(len(g.runs) for g in groups) == [1, 2]
    reasons = {x["run_id"]: x["reason"] for x in excluded}
    assert reasons["old-judge"] == "graded with another judge prompt version"
    assert reasons["draft"] == "dataset had draft cases"
    assert reasons["other-ds"].startswith("other dataset version")
    assert reasons["broken"] == "incomplete (budget)"


def test_mock_runs_are_not_excluded_for_the_judge_prompt(tmp_path):
    _write(tmp_path, _run("m1", mock=True, judge_prompt_sha256_16=None))
    groups, excluded = compare.load_runs(tmp_path, DATASET_SHA, JUDGE_SHA)
    assert len(groups) == 1 and excluded == []


def test_metrics_average_each_case_over_runs():
    runs = [_run("a1"), _run("a2", outcomes=(True, False, False))]
    m = compare.summarize_metric(runs, compare.METRICS["facts_all_found"][1])
    # per-case means: c0 = 1, c1 = 0.5, c2 = 0 -> 0.5 over 3 cases (r1 has no facts)
    assert m["value"] == pytest.approx(0.5) and m["n_cases"] == 3 and m["n_observations"] == 6
    assert (m["per_run_min"], m["per_run_max"]) == (pytest.approx(1 / 3), pytest.approx(2 / 3))
    assert m["unstable_cases"] == 1
    assert (m["ci_low"], m["ci_high"]) == wilson_ci(1.5, 3)


def test_one_run_matches_the_run_report_interval():
    m = compare.summarize_metric([_run("a1")], compare.METRICS["facts_all_found"][1])
    w = wilson(2, 3)
    assert (m["ci_low"], m["ci_high"]) == pytest.approx((w.ci_low, w.ci_high))
    assert m["unstable_cases"] is None


def test_tool_metric_ignores_cases_without_expected_calls():
    m = compare.summarize_metric([_run("a1")], compare.METRICS["tools_all_matched"][1])
    assert m["n_cases"] == 3  # the refusal case expects no tool call


def test_zero_rate_interval_does_not_collapse():
    low, high = wilson_ci(0, 30)
    assert low == 0 and high > 0.1


def test_paired_delta_against_baseline(tmp_path):
    _write(
        tmp_path,
        _run("a1", outcomes=(False, False, False)),
        _run("b1", agent="langgraph", outcomes=(True, True, True)),
    )
    groups, _ = compare.load_runs(tmp_path, DATASET_SHA, JUDGE_SHA)
    base = compare.pick_baseline(groups, None)
    assert base is not None and base.config["agent"] == "react"
    entries = compare.compare(groups, base)
    other = next(e for e in entries if not e["is_baseline"])
    d = other["delta_vs_baseline"]["facts_all_found"]
    assert d["delta"] == pytest.approx(1.0) and d["significant"] and d["n_cases"] == 3


def test_main_writes_json_and_markdown(tmp_path, monkeypatch):
    _write(tmp_path, _run("a1"), _run("a2"), _run("old", judge_prompt_sha256_16=None))
    monkeypatch.setattr(
        compare, "dataset_fingerprint", lambda p: JUDGE_SHA if "judge" in str(p) else DATASET_SHA
    )
    assert compare.main(["--results-dir", str(tmp_path)]) == 0
    md = (tmp_path / "comparison.md").read_text(encoding="utf-8")
    assert "react · gpt-x · c500/o50 · no rerank (baseline)" in md
    assert "## Excluded runs" in md and "`old`" in md
    payload = json.loads((tmp_path / "comparison.json").read_text(encoding="utf-8"))
    assert payload["entries"][0]["n_runs"] == 2


def test_report_replaces_only_the_marked_section(tmp_path):
    readme = f"# Title\n\nintro\n\n{report.START}\nold\n{report.END}\n\nafter\n"
    updated = report.update_readme(readme, "# Configuration comparison\n\n## Runs\n\n- x\n")
    assert updated.startswith("# Title\n\nintro\n\n") and updated.endswith("\n\nafter\n")
    assert (
        "old" not in updated and "### Runs" in updated and "Configuration comparison" not in updated
    )
    assert report.update_readme(updated, "## Runs\n\n- y\n").count(report.START) == 1
    with pytest.raises(ValueError):
        report.update_readme("# no markers", "x")


def test_matrix_plan_reuses_runs_and_estimates_costs(tmp_path, monkeypatch):
    monkeypatch.setattr(run_matrix.run_eval, "RESULTS_DIR", tmp_path)
    settings = run_matrix.settings_for({"agent": "react", "provider": "openai"})
    planned_cfg = run_matrix.Planned("x", "react", settings, 3, 0, None, "", True).config_view()
    past = _run("p1", **{k: planned_cfg[k] for k in ("llm_model", "embedding_model")})
    past["config"]["judge_prompt_sha256_16"] = run_matrix.dataset_fingerprint(
        run_matrix.compare.JUDGE_PROMPT_PATH
    )
    past["dataset"]["sha256_16"] = run_matrix.dataset_fingerprint(
        run_matrix.run_eval.DEFAULT_DATASET
    )
    _write(tmp_path, past)
    matrix = {
        "repeats": 3,
        "configs": [
            {"name": "react-openai", "agent": "react", "provider": "openai"},
            {"name": "lg-openai", "agent": "langgraph", "provider": "openai", "repeats": 2},
        ],
    }
    planned = {p.name: p for p in run_matrix.plan(matrix, None, reuse=True)}
    react, lg = planned["react-openai"], planned["lg-openai"]
    assert (react.existing, react.to_run) == (1, 2)
    assert react.estimate_source.startswith("observed")
    n_cases = len(run_matrix.load_dataset(run_matrix.run_eval.DEFAULT_DATASET))
    assert react.cost_per_run == pytest.approx(0.011 * n_cases * run_matrix.SAFETY_MARGIN)
    assert (lg.existing, lg.to_run, lg.estimate_source) == (0, 2, "upper bound")
    assert run_matrix.plan(matrix, {"lg-openai"}, reuse=True)[0].name == "lg-openai"
    assert run_matrix.plan(matrix, None, reuse=False)[0].existing == 0


def test_matrix_dry_run_never_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(run_matrix.run_eval, "RESULTS_DIR", tmp_path)
    called = []
    monkeypatch.setattr(run_matrix.run_eval, "main", lambda argv: called.append(argv) or 0)
    matrix = tmp_path / "m.yaml"
    matrix.write_text("repeats: 1\nconfigs:\n  - {name: a, agent: react, provider: openai}\n")
    assert run_matrix.main(["--matrix", str(matrix), "--dry-run"]) == 0
    assert called == []
