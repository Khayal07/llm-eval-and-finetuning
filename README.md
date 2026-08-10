# AI Output Evaluation + Model Adaptation Framework

An automated framework for evaluating LLM/RAG outputs, tracking accuracy /
latency / cost metrics, running root-cause analysis on failures, and improving
quality via few-shot prompt adaptation (with optional fine-tuning).

The project was built in six checkpoints:
1. Evaluation dataset with edge cases and contamination-free train/test splits.
2. Scoring engine: exact match + LLM-as-Judge with bias handling.
3. Metrics tracking: accuracy, latency, token cost, archived runs.
4. Baseline RAG agent and root-cause analysis.
5. Few-shot prompt adaptation and before/after comparison.
6. Final pipeline runner, report, and this README.

---

## Project structure

```
├── data/
│   ├── dataset_full.json        # master: knowledge base + all 21 samples
│   ├── train_fewshot.json       # 9 exemplars (disjoint topics from test)
│   ├── test_heldout.json        # 12 held-out test samples
│   └── finetune_chat.jsonl      # optional fine-tuning dataset (generated)
├── evals/
│   ├── __init__.py
│   ├── llm_client.py            # OpenAI wrapper (env-based, retry, usage)
│   ├── metrics.py               # latency, token cost, pass-rate aggregation
│   ├── evaluator.py             # exact match + LLM-as-Judge scoring
│   ├── bias_handler.py          # length / position / consistency bias analysis
│   └── run_tracker.py           # run archiving under reports/runs/
├── reports/
│   ├── root_cause_analysis.md   # ≥3 failure cases with evidence
│   ├── final_eval_report.md     # methodology + before/after comparison
│   └── runs/                    # timestamped run artifacts (gitignored)
├── agent_or_rag.py              # evaluated model: RAG policy assistant
├── pipeline.py                  # run orchestration (scoring, aggregation)
├── optimize.py                  # few-shot adaptation + comparison
├── main.py                      # CLI runner (all stages)
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # then edit .env with your OPENAI_API_KEY
```

### Configuration (`.env`)

| Variable | Description | Default |
|---|---|---|
| `OPENAI_API_KEY` | API key (never committed) | — |
| `EVAL_MODEL` | Agent model under evaluation | `gpt-4o-mini` |
| `JUDGE_MODEL` | LLM-as-Judge model | `gpt-4o` |
| `JUDGE_TEMPERATURE` | Judge sampling temperature | `0.0` |
| `FINE_TUNE_MODEL` | Optional fine-tuned model id | *(empty)* |
| `PRICING_INPUT_PER_1M` / `PRICING_OUTPUT_PER_1M` | Cost overrides (USD/1M tokens) | from `evals/metrics.py` |

---

## Usage

```bash
python main.py --mode all          # baseline → optimize → compare (end-to-end)
python main.py --mode baseline     # run baseline only
python main.py --mode optimize     # run few-shot adapted evaluation
python main.py --mode compare      # compare latest baseline vs optimized
python main.py --mode validate     # dataset integrity + contamination check
python main.py --skip-judge ...    # exact-match only (no LLM-as-Judge)
```

`pipeline.run_evaluation()` and `optimize.run_optimized()` expose the same stages
programmatically. Every run is archived to
`reports/runs/<UTC timestamp>__<run_name>.json` and never overwrites history,
so before/after studies stay auditable.

---

## Data and contamination safeguards

- Held-out test set counts for all reported metrics and is never used to build
  few-shot exemplars or fine-tuning data.
- Training exemplars reference knowledge-base documents disjoint from the test
  topics, so no answered topic leaks into adaptation (`main.py --mode validate`
  enforces this).
- The adaptation step (`optimize.py`) addresses failures from the root-cause
  analysis with no-evidence refusal, step-by-step arithmetic, and explicit
  threshold exemplars.

---

## Metrics

- **Exact match** — strict normalized string equality (a cheap lower bound).
- **LLM-as-Judge** — semantic 0/1 grading against a reference, with explicit
  anti-length/anti-style instructions.
- **Combined pass** — judge verdict, falling back to exact match.
- **Latency** — mean / median / p95 (ms). **Cost** — estimated USD from tokens.

Judge bias (length, position, self-consistency) is analyzed in
`evals/bias_handler.py`; see `reports/final_eval_report.md` for this run's bias
findings.

---

## Results

Baseline combined pass **91.67%** (1 multi-hop failure) → optimized **100%**
with no new failures (< ~500 ms faster mean latency; negligible cost increase).
Full numbers: `reports/final_eval_report.md`. Failure analysis:
`reports/root_cause_analysis.md`.