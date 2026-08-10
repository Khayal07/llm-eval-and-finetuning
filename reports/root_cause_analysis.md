# Root Cause Analysis — Baseline Run

Date: 2026-08-10
Config: `EVAL_MODEL=gpt-4o-mini` (agent), `JUDGE_MODEL=gpt-4o` (LLM-as-Judge, temperature 0.0),
held-out set `data/test_heldout.json` (12 samples), retriever `k=3` (rare-term weighted token overlap).

---

## 1. Baseline results recap

| Metric | Value |
|---|---|
| Exact-match rate | 16.67% (2/12) |
| LLM-as-Judge pass rate | 91.67% (11/12) |
| Combined pass rate | 91.67% (11/12) |
| Mean / median latency | 1877 ms / 1643 ms (p95 3779 ms) |
| Total estimated cost | $0.000739 |

Failed (judge verdict = fail): `q_h_009` only.
Exact-match failed on 10/12 samples even though the judge accepted most of them ->
the strict scorer is a lower bound and surface phrasing dominates it.

---

## 2. Failure case 1 — Weak retrieval (out-of-scope question)

Sample: `q_h_008` — "Does AcmeCorp offer pet insurance to employees?"
Reference: "AcmeCorp has no documented pet insurance policy…"

Retrieval evidence (top-3 docs with score):

| rank | doc | score |
|---|---|---|
| 1 | kb3 (Remote work policy) | 4.86 |
| 2 | kb4 (Health insurance) | 4.49 |
| 3 | kb2 (Vacation policy) | 3.24 |

The knowledge base contains **no** pet-insurance document, so every returned doc is irrelevant.
They matched only on generic tokens such as "employees", "AcmeCorp", "offer". The system passed
only because the prompt's refusal instruction ("I cannot find this information in the knowledge base.")
was followed exactly.

Root cause: the retriever has **no evidence-missing signal**. It always returns k documents, so the
answer step is one weak refusal-rule away from hallucinating a policy. If the refusal instruction were
absent or the model verbose, the same context would likely produce a fabricated benefit.

Fix direction: add a retrieval floor/threshold + "no evidence above score X" branch, and keep the
explicit refusal instruction.

---

## 3. Failure case 2 — Weak prompt / reasoning consistency (multi-hop arithmetic)

Sample: `q_h_009` — "I joined AcmeCorp in April. Roughly how many vacation days will I accrue by the end of this year?"
Reference: "About 16.5 days (1.83 days/month x 9 months, Apr–Dec)."
Judge verdict: **fail**.

Retrieval was correct here: kb2 (Vacation policy) ranked first with score 32.98,
so retrieval is not the cause. The generation step failed:

> Generated: "…you will have accrued approximately **15** vacation days. This is calculated based on 1.83 days
> per month from April to December (9 months), which totals **16.47 days**. However, since…"

The model computed 16.47 correctly, then contradicted itself in the final sentence and concluded 15.
It silently applied an unstated proration/rounding rule. Compare with `q_h_010` (insurance math), where the
same model showed its working and passed — the arithmetic skill exists but is not enforced.

Root cause: prompt direction. The zero-shot prompt never instructs the model to (a) show the calculation
step by step, (b) verify the final number against its own arithmetic, or (c) avoid applying rules that are
not in the context. Multi-hop questions therefore depend on emergent model discipline.

Fix direction (Checkpoint 5): a few-shot exemplar that demonstrates 1.83 x 9 = 16.5 with an explicit
"final answer = computed value, do not re-derive" rule.

---

## 4. Failure case 3 — Ambiguous / under-specified question

Sample: `q_h_007` — "I took a taxi from the airport to my hotel on a business trip. Can I expense it?"
Reference: "Yes… Since the fare is likely above $25, you must attach the receipt."

Retrieval was correct (kb6, travel policy, score 16.08) and the judge accepted the answer,
but the generated response was factually soft and dropped the concrete threshold:

> Generated: "Yes, you can expense the taxi… Be sure to keep the receipt, especially since trips
> usually exceed…"

The question is ambiguous: it never says the fare exceeds $25, so the model hedged ("since trips usually…")
instead of anchoring the answer to the explicit "$25 → receipt required" rule.

Root cause: the question is under-specified, and the zero-shot prompt does not require the answer to
state the *governing threshold explicitly* for conditional policies. The resulting phrasing is correct
in spirit but not exact, which also explains why exact-match failed here.

Fix direction: instruction to "always state the exact threshold/number from the context that drives the
decision" plus a few-shot exemplar for conditional expense rules.

---

## 5. Methodological failure — Exact-match fragility (metric-level)

10 of 12 samples were semantically correct per the judge but not string-identical to the reference.
Examples: "You will be reimbursed within 5 business days after approval of your expenses."
vs reference "Reimbursement is paid within 5 business days after approval."

Impact: exact-match underestimates real accuracy (16.67% vs judge 91.67%). It is kept because it is
cheap, deterministic, and catches verbatim-copying; the combined pass uses the judge verdict so the
final metric is not dragged down by surface phrasing.

---

## 6. LLM-as-Judge bias check (length bias)

`length_bias_report` on this run flagged **asserted length bias** (Pearson |corr| between
judge score and answer word count above the 0.25 guardrail).

Mitigations already in place: the judge prompt explicitly instructs to ignore length and style, and
scores only factual correctness. Follow-up recommended (`bias_handler.py`): position-bias and
consistency probes should be run on a sample before the final report to quantify residual bias.

---

## 7. Conclusion and plan

- Retrieval is strong for in-KB facts but blind to missing evidence (`q_h_008`).
- Generation fails on multi-hop arithmetic consistency (`q_h_009`).
- Ambiguous questions yield soft answers that strict matching rejects (`q_h_007`).

Checkpoint 5 will address the prompt-level failures via few-shot exemplars:
(1) explicit no-evidence refusal, (2) step-by-step arithmetic with verified final number,
(3) always-state-the-threshold rule, regenerated from `data/train_fewshot.json` (contamination-free).