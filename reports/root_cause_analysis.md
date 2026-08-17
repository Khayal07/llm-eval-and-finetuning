# Root Cause Analysis — Initial Evaluation on the 20-Sample Set

Date: 2026-08-17
Config: `EVAL_MODEL=gpt-4o-mini` (agent), `JUDGE_MODEL=gpt-4o` (LLM-as-Judge, temperature 0.0),
held-out set `data/test_heldout.json` (20 samples), retriever `k=3` (rare-term weighted token
overlap + IDF-weighted cosine no-evidence gate at 0.12).
Artifacts: `reports/runs/20260817-115808__baseline.json`, `reports/runs/20260817-115904__fewshot_optimized.json`.

Every case below is a **genuinely failed record** from a real run with the actual generated
answer, retrieval evidence, and scorer scores attached. No synthetic or successful cases are
presented as failures.

---

## 1. Baseline results recap

| Metric | Value |
|---|---|
| Exact-match rate | 15.00% (3/20) |
| Semantic pass rate (cosine ≥ 0.35) | 90.00% (18/20) |
| LLM-as-Judge pass rate | 90.00% (18/20) |
| Combined pass rate | 90.00% (18/20) |
| Mean / median latency | 1672.53 / 1486.94 ms |
| Total cost | $0.001155 ($5.8e-05 / query) |

**Genuinely failed ids (baseline): `q_h_009`, `q_h_018`** — both multi-hop vacation
proration. All other edge families (ambiguous, arithmetic, contradiction, threshold,
out-of-scope) and all normal questions passed.

---

## 2. Failure case 1 — Weak prompt / arithmetic verification (`q_h_018`)

Sample `q_h_018` — "I joined AcmeCorp in September. Roughly how many vacation days will I
accrue by the end of this year?"
Reference: "About 7.3 days … 1.83 days per month … 4 months (4 x 1.83 is roughly 7.3)."

> Generated: "You will accrue approximately **7.5** vacation days by the end of this year.
> This is calculated as 1.83 days per month for 4 months (September to December)."

Retrieval was correct (kb2, score 32.98 raw / cosine 0.229) — retrieval is **not** the cause.
The model multiplied 1.83 × 4 = **7.5**, but the correct product is **7.32**. It showed the
right structure but never verified the arithmetic, so a small multiplication error shipped as
the final number. (Contrast `q_h_010`, where the model showed its working and passed.)

Root cause: **weak prompt** — the zero-shot prompt does not require showing the calculation
and checking the final number. Multi-hop arithmetic depends on emergent model discipline.
Fixed in the few-shot adapted prompt by the `step_by_step_arithmetic` exemplar
(optimized: 1.83 x 4 = 7.3, combined pass **True**, semantic 0.63).

---

## 3. Failure case 2 — Weak prompt / reasoning consistency (`q_h_009`)

Sample `q_h_009` — "I joined AcmeCorp in April. Roughly how many vacation days will I accrue
by the end of this year?"
Reference: "About 16.5 days … April through December total 9 months (9 x 1.83 is roughly 16.5)."

> Generated: "…you will have accrued approximately **15** vacation days. This is calculated
> based on 1.83 days per month from April to December (9 months), which totals **16.47** days.
> However, since carry-over of unused days to the next year is capped at 5 days, you will have
> 15 days available for use."

Retrieval was correct (kb2, raw 32.98 / cosine 0.229). The model computed 16.47 correctly,
then **contradicted itself** in the final sentence and concluded 15, silently applying the
carry-over cap (which answers a different question: available balance, not accrued days).

Root cause: **weak prompt** — no instruction to (a) verify the final number against its own
arithmetic or (b) avoid applying rules not required by the question. Multi-hop reasoning
consistency is emergent, not enforced. Fixed by the few-shot adapted prompt
(optimized: 1.83 x 9 = 16.47 → "approximately 16.5", combined pass **True**, semantic 0.68).

---

## 4. Failure case 3 — LLM-as-Judge strictness on a correct answer (`q_h_013`, optimized run)

Sample `q_h_013` — "On a 5-hour business flight, which class am I allowed to book?"
Reference: "Economy class, because business class is only allowed on transoceanic flights
longer than 8 hours."

> Generated (optimized): "You are allowed to book economy class on **flights under 6 hours**."

This is the **verbatim economy rule** from kb6 ("economy class on flights under 6 hours");
5 h < 6 h ⇒ economy is the correct decision. The judge failed it:

> Judge: "The AI answer incorrectly states the threshold for business class as flights under
> 6 hours…" — a misreading; the answer never claims business class is allowed under 6 hours.

Evidence that the answer is correct, not the model:
- Semantic-similarity cross-check: **0.523** (above the 0.35 pass threshold).
- The identical fact pattern passed at baseline ("You are allowed to book economy class on a
  5-hour business flight.").
- A 3-round `judge_consistency_check` on this pair returned `[False, False, False]` — the
  judge is **stably strict**, not randomly wrong, on reference-anchored rephrasings.

Root cause: **evaluator limitation** — the LLM judge over-anchors to the reference's phrasing
and penalizes a correct answer that cites a different valid policy rule. This is exactly the
failure mode the deterministic semantic cross-check (Checkpoint 2) is designed to surface, and
it motivates the `--bias-audit` probes.

---

## 5. Failure case 4 — Judge run-to-run variance on a threshold answer (`q_h_019`, optimized run)

Sample `q_h_019` — "…if I stay home tomorrow that will be my fourth day. Will I need a
doctor's note?"
Reference: "Yes, four consecutive days of absence, including Thursday, require a doctor's note."

> Generated: "Yes, you will need a doctor's note because **absences of 4 or more consecutive
> days require a doctor's note**." — the verbatim rule from kb9.

This record failed only in the optimized run; the same answer passed at baseline. A 3-round
`judge_consistency_check` on this exact pair scored `[True, True, True]` (0 flips). The judge
is correct when the answer is graded against the reference's own threshold phrasing, and the
failed verdict in the run is run-to-run sampling variance, not a model error. Semantic score 0.56.

Root cause: **judge nondeterminism** on boundary/threshold answers — one of the reasons the
judge runs at temperature 0.0, `judge_consistency_check` exists, and the objective semantic
cross-check is recorded per sample.

---

## 6. Weak retrieval — out-of-scope safety and the no-evidence threshold

The two out-of-scope questions (`q_h_008` pet insurance, `q_h_020` company cars/relocation)
**passed**, but in the pre-threshold retriever they were a risk: raw token-overlap ranked
irrelevant docs above real evidence (e.g. `q_h_020` raw top score 8.17 > in-scope `q_h_006`
raw score 5.34), so safety depended entirely on the refusal instruction being followed.

Checkpoint 4 adds a **no-evidence cosine gate** (`RETRIEVER_MIN_SIMILARITY`, default 0.12):
when the best-matching document's IDF-weighted cosine similarity falls below the floor, the
context is emptied and the answer step receives the safe fallback ("I cannot find this
information in the knowledge base.") instead of weak context. Measured on the full 20-sample
set:

| Query type | Samples | Cosine evidence | Gate result |
|---|---|---|---|
| Out-of-scope | `q_h_008`, `q_h_020` | 0.065, 0.093 | **no_evidence=True**, context empty |
| In-scope (all 18) | — | 0.140 – 0.355 | no_evidence=False, top-k fed |

The gate separates the classes cleanly and is recorded per sample (`no_evidence`,
`evidence_score` in run artifacts). It hardens out-of-scope safety without touching
in-scope retrieval quality.

---

## 7. Methodological notes

- **Exact match (15%)** is a strict string lower bound; semantic and judge scoring carry the
  reported quality signal.
- **Semantic pass rate (90%)** under-scores the two out-of-scope refusals (terse refusal text
  vs longer reference) even though the judge correctly accepts them; it is a reference-anchored
  floor, not a ceiling.
- **Reconciliation:** the optimized run's two failed records are both **factually correct
  answers** (kb-verified, semantic ≥ 0.52, one stable-3×fail judge misread, one verified
  correct 3×pass under re-grading). Genuine model failures were only `q_h_009`/`q_h_018`
  at baseline, and both are fixed by the few-shot adaptation.

---

## 8. Conclusion

1. **Weak prompt** (multi-hop arithmetic): `q_h_018`, `q_h_009` — genuine failures, fixed by
   few-shot (step-by-step + verified final number).
2. **Evaluator strictness / variance**: `q_h_013`, `q_h_019` — correct answers graded as
   failures; surfaced by the semantic cross-check and consistency probes.
3. **Weak retrieval** (out-of-scope): hardened by the no-evidence cosine gate so weak context
   is never sent to the LLM.