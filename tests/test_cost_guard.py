import pytest

from agent_eval.config import PROJECT_ROOT
from agent_eval.cost_guard import BudgetExceeded, CostGuard, confirm_run, estimate_cost
from agent_eval.observability.cost import ModelPrice, PricingTable

PRICING = PricingTable(
    {
        "known": ModelPrice("known", 1.0, 2.0, verified=True),
        "unverified": ModelPrice("unverified", 1.0, 2.0, verified=False),
        "unknown": ModelPrice("unknown", None, None, verified=False),
    }
)


def test_token_limit_blocks_next_call():
    guard = CostGuard(max_llm_calls=100, max_total_tokens=1000)
    guard.before_call()
    guard.after_call(900, 200)  # overshoot by the in-flight call is allowed
    with pytest.raises(BudgetExceeded):
        guard.before_call()


def test_cost_computation():
    assert PRICING.cost("known", 1_000_000, 500_000) == pytest.approx(2.0)
    assert PRICING.cost("unknown", 10, 10) is None
    assert PRICING.cost("not-in-table", 10, 10) is None


def test_estimate_and_confirm_known_price():
    est = estimate_cost(
        PRICING,
        "known",
        n_requests=30,
        calls_per_request=4,
        input_tokens_per_call=3000,
        output_tokens_per_call=500,
    )
    assert est.total_calls == 120
    assert est.cost_usd == pytest.approx((360_000 * 1.0 + 60_000 * 2.0) / 1e6)
    out: list[str] = []
    assert confirm_run([est], input_fn=lambda _: "y", print_fn=out.append)
    assert not confirm_run([est], input_fn=lambda _: "", print_fn=out.append)
    assert confirm_run([est], assume_yes=True, print_fn=out.append)


def test_unknown_price_refuses_unless_allowed():
    est = estimate_cost(PRICING, "unknown", 1, 1, 10, 10)
    out: list[str] = []
    assert not confirm_run([est], assume_yes=True, print_fn=out.append)
    assert any("Refusing" in line for line in out)
    assert confirm_run([est], assume_yes=True, allow_unknown_price=True, print_fn=out.append)


def test_unverified_price_warns():
    est = estimate_cost(PRICING, "unverified", 1, 1, 10, 10)
    assert "verified: false" in est.render()


def test_repo_pricing_file_loads():
    table = PricingTable.load(PROJECT_ROOT / "config" / "pricing.yaml")
    assert table.cost("mock-llm-v1", 1000, 1000) == 0
    assert table.cost("gpt-5.4-mini-2026-03-17", 1_000_000, 0) == pytest.approx(0.75)
