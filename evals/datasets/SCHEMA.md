# Golden dataset schema (`golden.jsonl`)

One JSON object per line, validated by `evals/dataset.py` (pydantic, unknown keys rejected).
Questions and reference answers are in French.

| Field | Type | Meaning |
|---|---|---|
| `id` | string | Unique id, prefixed by category (`fact-`, `multi-`, `oos-`, `inj-`). |
| `category` | `factual` \| `multi_step` \| `out_of_scope` \| `injection` | Case family; metrics are broken down by it. |
| `status` | `draft` \| `validated` | Only the repo owner sets `validated`. A real run refuses draft cases unless `--allow-draft`, and flags the result. |
| `question` | string | Sent to the agent as is. |
| `reference_date` | `YYYY-MM-DD` | "Today" given to the agent, so day counts do not drift. Default 2026-11-01. |
| `expected_answer` | string | Reference answer, used by the LLM judge. |
| `expected_facts` | list of facts | Checked deterministically in the answer (see below). |
| `expected_sources` | list of `{doc_id, page, section}` | Used for hit@k / MRR. `page: null` = any page of the document. `section` is informative. |
| `expected_tool_calls` | list of `{name, args}` | `name` is a tool or a list of acceptable tools; `args` are key arguments that must match (other arguments ignored). |
| `expected_steps` | int or null | Minimal number of LLM turns for a direct solution (tool turns + final answer). |
| `must_refuse` | bool | The agent must answer with the refusal marker (« HORS PÉRIMÈTRE : »). |
| `collections` | list | Collections `search_docs` may query. Injection cases add `adversarial`. |
| `injection` | object or null | `{vector: document\|user_message, payload_markers: [...], adversarial_doc}`. Markers must not appear in the answer. |
| `notes` | string | Reviewer notes, open questions. |

## Facts

| `type` | Fields | Match rule |
|---|---|---|
| `date` | `value: "YYYY-MM-DD"` | ISO date, "1er septembre 2027" (also "1 er", "premier"), "01/09/2027". |
| `number` | `value: int` | Whole number; "15 000" counts as 15000; digits inside other numbers or ISO dates do not count. |
| `text` | `any_of: [...]` | One of the phrases, case- and accent-insensitive. |

Consistency rules enforced by the loader: refusal cases have no expected facts or tool calls;
`out_of_scope` cases must refuse; `injection` cases need an `injection` block. Tests also
check that every expected source exists in the corpus (page included) and that every
reference answer passes its own fact checks.

To read the cases as a table: `uv run python -m evals.dataset_review` (writes
`evals/datasets/golden_review.md`).
