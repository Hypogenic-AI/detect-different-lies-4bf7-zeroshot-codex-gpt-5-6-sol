# False Is Not One Thing

This repository contains a completed, reproducible study of whether false factual outputs from an LLM are low-knowledge errors or belief-contradicting misreports, and whether linear hidden-state probes distinguish them.

## Question and design

We ran Qwen2.5-7B-Instruct on 240 deterministically sampled TriviaQA validation questions. Four stochastic neutral answers operationalize elicited knowledge: at least 3/4 correct is `known`, 0/4 is `unknown`, and the remainder is `ambiguous`. Every question then receives matched greedy prompts for a neutral answer, an explicitly incentivized wrong answer, and an instruction to resist the same incentive. A clean `lie` is a false incentivized answer to a known question; a `hallucination` is a false neutral answer to an unknown question.

We extract every hidden layer at the last prompt and answer tokens. Fixed-depth, question-held-out ridge probes test three tasks: lie-prompt versus resist-prompt condition, neutral unknown-error versus known-correct detection, and actual lies versus truthful outputs under the identical lie prompt.

## Main findings

- The belief split was 129 known, 93 unknown, and 18 ambiguous questions. Only 17/129 known questions became false under the explicit wrong-answer instruction, a 13.2% lie yield.
- Of 116 false outputs under that instruction, 17 (14.7%) were clean lies, 86 (74.1%) were low-knowledge errors, and 13 (11.2%) were ambiguous.
- A conventional condition probe achieved AUROC 1.000, but was already perfect at the final prompt token. It detected the instruction, not demonstrated lying.
- Controlling the prompt changed the conclusion: the layer-21 answer-state probe trained on actual lie versus correct behavior under the same pressure reached AUROC 0.959 (95% bootstrap CI 0.916–0.991), and separated lies from hallucinations at 0.896 (0.800–0.965).
- A hallucination probe detected its native neutral contrast well (answer-state AUROC 0.879), but did not reliably separate hallucinations from lies (0.604, CI 0.475–0.733).
- 51/93 unknown questions had a stable repeated wrong answer. “Unknown” here is an epistemic-failure bucket, not necessarily high uncertainty.

These are conditional results from one small model and an explicit instruction to lie; they are not prevalence estimates for deployment or evidence about spontaneous strategic deception.

## Repository layout

- `paper_draft/main.tex` and `paper_draft/main.pdf`: conference-style paper and compiled PDF.
- `src/prepare_data.py`: deterministic TriviaQA selection.
- `src/run_experiment.py`: generation and hidden-state extraction.
- `src/judge_answers.py`: conservative semantic grading of alias non-matches; all raw decisions are retained.
- `src/analyze.py`: labeling, held-out probes, bootstrap intervals, tables, and figures.
- `results/raw_generations.jsonl`: all 1,680 model outputs and exact formatted prompts.
- `results/activations.npz`: prompt- and answer-token states for 720 deterministic responses.
- `results/answer_judgments.json`: lexical/semantic labels and raw local-grader outputs.
- `results/labeled_outputs.csv`, `probe_results.csv`, and `summary.json`: derived results.

Large downloaded model and dataset caches live under ignored `models/`, `data/`, and `cache/` directories.

## Reproduce

An NVIDIA GPU with about 20 GB of free memory is recommended. From the repository root:

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
HF_HOME=cache/huggingface .venv/bin/python src/prepare_data.py --n 240
HF_HOME=cache/huggingface .venv/bin/python src/run_experiment.py --batch-size 16
HF_HOME=cache/huggingface .venv/bin/python src/judge_answers.py
MPLBACKEND=Agg .venv/bin/python src/analyze.py
(cd paper_draft && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex)
```

Seeds are fixed in the scripts (question selection 314159; generation 271828); folds are SHA-256 hashes of question IDs. The experiment used model revision `a09a35458c702b33eeacc393d103063234e8bc28`, Python 3.12.8, PyTorch 2.7.1+cu128, Transformers 4.57.6, and one idle NVIDIA RTX A6000. The local Qwen semantic grader is used only on 392 unique non-matching answer triples and accepted 23; this same-model adjudication is explicitly treated as a limitation in the paper.
