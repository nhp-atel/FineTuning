# Phase 7 — Capstone

The finale. One end-to-end project that reuses every skill from Phases 1–6 to
fine-tune a small model that classifies sales leads as `hot` / `warm` / `cold`.

## Notebook

| # | Notebook | What you build |
|---|----------|----------------|
| 15 | `15_mini_project_lead_intent.ipynb` | A complete, runnable pipeline: generate data → JSONL + train/val split → prompt/feature design → tokenize → fine-tune → evaluate → inference on new leads → save/serve |

## What it pulls together
- **Data prep** (Phase 4): the lead dataset, JSONL, train/val split.
- **Formatting** (Phase 4): classifier text and the instruction template.
- **Fine-tuning** (Phase 5): a DistilBERT classifier as the runnable path, with the LoRA-LLM path as an optional extension; full hyperparameter walkthrough.
- **Evaluation** (Phase 6): accuracy, precision/recall/F1, confusion matrix, manual review, baseline comparison.
- **Inference** (Phase 6): a `predict(lead)` helper and save/serve.

## After this you'll have…
A fine-tuned, evaluated model and the understanding to swap in your own data,
a bigger model, or QLoRA on a GPU.

## Prerequisites
All earlier phases. The primary DistilBERT path runs on **CPU**; a GPU is
faster and required only for the optional LoRA-LLM extension.

## Next
🎉 You're done with the course. The notebook's "What to learn next" points to
real data, bigger models, GPU QLoRA, deployment, and further topics (RLHF/DPO).
