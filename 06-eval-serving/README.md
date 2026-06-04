# Phase 6 — Evaluation & serving

A trained model is only useful if you can tell whether it's any good and then
actually run it. This phase covers honest evaluation metrics and how to do
inference, save/load, and serve the model behind an API.

## Notebooks (do them in order)

| # | Notebook | What you learn |
|---|----------|----------------|
| 13 | `13_evaluation.ipynb` | Why accuracy can mislead, precision/recall/F1 (in plain words), macro vs weighted, `classification_report`, confusion matrix, manual review, baselines |
| 14 | `14_inference_and_serving.ipynb` | `model.generate` decoding params, classifier inference, loading/merging LoRA adapters, save/load, batched inference, a FastAPI serving template |

## After this phase you can…
- Report accuracy, precision, recall, F1, and a confusion matrix — and interpret them for the lead-intent task.
- Eyeball real predictions and compare against a sensible baseline.
- Run inference (generation and classification), merge a LoRA adapter, and sketch a `/predict` API.

## Prerequisites
Phases 1–5. Runs on **CPU** with the small models used here; the FastAPI server
cell is illustrative (save as `app.py` and run separately).

## Next
➡️ **Phase 7 — Capstone** (`../07-capstone/`): put data prep, fine-tuning, evaluation, and inference together end to end.
