# SignalRank

**Evidence-grounded behavioral intelligence for civic engagement.**

SignalRank is a production-shaped recommendation system that turns fragmented
behavioral histories and unstructured content into ranked, explainable issue
recommendations. It separates prediction from explanation: a measurable hybrid
ranker chooses recommendations, while an evidence layer explains them using only
retrieved source text and model features.

> The bundled demo data is synthetic and exists only for smoke tests. Do not cite
> its metrics as evidence of real-world accuracy. Use the MIND loader and an
> impression-aware temporal split for a credible benchmark.

## What this demonstrates

- Implicit-feedback recommendation with real negative impressions
- Hybrid collaborative and transformer-ready content representations
- Pointwise learning-to-rank with behavioral, semantic, freshness, and popularity features
- Temporal evaluation with AUC, MRR, nDCG@K, Recall@K, coverage, and diversity
- Cold-start fallbacks and cohort-level reporting
- Deterministic, cited recommendation explanations
- FastAPI inference service, health checks, model cards, and regression tests

## Architecture

```text
interactions + articles -> temporal feature pipeline -> candidate set
                                                 |-> hybrid ranker
                                                 |-> diversity reranker
                                                 |-> recommendation API
article evidence + rank features ----------------|-> grounded explanation
```

## Quick start

Requires Python 3.11+.

```bash
python -m signalrank.cli demo --output artifacts
python -m unittest discover -s tests -v
uvicorn signalrank.api:app --reload
```

Then open `http://127.0.0.1:8000/docs` and try:

```bash
curl 'http://127.0.0.1:8000/recommend/u0001?k=5'
curl 'http://127.0.0.1:8000/explain/u0001/i0001'
```

The visual recommendation and cited-chat demo is available at
`http://127.0.0.1:8000/demo/`.

Chat defaults to a deterministic extractive mode. To enable local Ollama:

```bash
ollama pull qwen2.5:3b
export SIGNALRANK_CHAT_PROVIDER=ollama
export OLLAMA_CHAT_MODEL=qwen2.5:3b
uvicorn signalrank.api:app --reload
```

## Benchmark on MIND

Download MINDsmall from the official Microsoft dataset page and extract it so
the directory contains `news.tsv` and `behaviors.tsv`.

```bash
python -m signalrank.cli benchmark-mind \
  --train data/MINDsmall_train \
  --dev data/MINDsmall_dev \
  --output artifacts/mind-benchmark \
  --max-train-impressions 20000 \
  --max-dev-impressions 5000
```

The loader uses pre-impression click histories and displayed-but-not-clicked
articles as observed negatives. Training and development data stay separate,
and the evaluation unit is an impression rather than a random user-item pair.
The report compares four baselines, calculates a paired-bootstrap interval
against the strongest baseline, and measures warm/cold candidate-ranking
latency at P50 and P95.

## Models

The fast reference model combines:

1. TF-IDF article embeddings
2. A user content profile computed from historical clicks
3. Sparse-matrix Truncated-SVD collaborative embeddings
4. Behavioral features such as category affinity, entity overlap, freshness,
   popularity, and user-history length
5. Logistic click ranker with standardized, serving-compatible features
6. Train-only calibration of learned-ranker and category-affinity scores
7. Maximal-marginal-relevance reranking for topical diversity

The interfaces intentionally allow the TF-IDF and SVD retrievers to be replaced
with a transformer two-tower model and ANN index without changing evaluation or
serving contracts.

## Evaluation contract

Primary metrics are `AUC`, `MRR`, `nDCG@5`, `nDCG@10`, `Recall@10`, coverage,
and intra-list diversity. Reports also include cold-start and history-length
slices. All split boundaries and model versions are written into the model card.

Regression gates live in `tests/` and verify:

- no future leakage into a user profile
- clicked candidates outrank at least one displayed negative on a learnable fixture
- rankings are deterministic
- cold-start requests return valid fallbacks
- explanations cite actual article evidence and never invent model scores
- metric implementations match hand-computed examples

## Responsible-use boundary

SignalRank is designed for aggregate issue discovery and evidence review. It
must not be used to infer protected traits, target vulnerable individuals,
optimize political persuasion, or present predictive associations as causal
claims. Production deployments require privacy review, consent and retention
controls, subgroup audits, and human review of high-impact decisions.

## Repository map

```text
signalrank/data.py       MIND loader and synthetic smoke fixture
signalrank/model.py      hybrid ranker, fallbacks, persistence
signalrank/evaluation.py impression-level metrics and slices
signalrank/explain.py    deterministic evidence-grounded explanations
signalrank/api.py        FastAPI service
signalrank/cli.py        training and demo commands
tests/                   unit and end-to-end regression harness
```

