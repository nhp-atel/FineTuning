# Phase 5 — Fine-tuning

The core of the course: actually adapt a pretrained model to the lead-intent
task. You'll learn the two most common practical techniques — **LoRA** and
**QLoRA** — and the hyperparameters that control every training run.

## Notebooks (do them in order)

| # | Notebook | What you learn |
|---|----------|----------------|
| 11 | `11_lora_finetuning.ipynb` | Full fine-tuning vs LoRA, PEFT, `LoraConfig` (`r`, `alpha`, `target_modules`, dropout), a real LoRA run, and key hyperparameters |
| 12 | `12_qlora_finetuning.ipynb` | Quantization (4-bit), `BitsAndBytesConfig`, QLoRA = quantized base + LoRA adapters, fitting bigger models on small GPUs |

## Key hyperparameters covered
Learning rate, epochs, batch size, **gradient accumulation**, **max sequence
length**, and **warmup** — explained in context, with small CPU-friendly values.

## After this phase you can…
- Explain when to use full fine-tuning vs LoRA vs QLoRA.
- Wrap a model with PEFT, count trainable parameters, train it, and save a tiny adapter.
- Set and reason about the hyperparameters that matter most.

## ⚡ GPU vs CPU
- **Notebook 11 (LoRA):** the tiny DistilBERT classifier path runs on **CPU**; a real LLM wants a GPU.
- **Notebook 12 (QLoRA):** 4-bit `bitsandbytes` needs an **NVIDIA GPU** (e.g. free Colab T4). On CPU or Apple Silicon, follow the notebook's LoRA fallback — the GPU-only cells self-skip with a friendly message.

## Prerequisites
Phases 1–4 (especially the dataset from Phase 4).

## Next
➡️ **Phase 6 — Evaluation & serving** (`../06-eval-serving/`): measure how good your model is and put it to use.
