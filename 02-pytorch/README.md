# Phase 2 — PyTorch

PyTorch is the engine under Hugging Face and almost every modern fine-tuning
workflow. Here you learn its building blocks, then write a complete training
loop by hand so the framework magic in later notebooks is never a black box.

## Notebooks (do them in order)

| # | Notebook | What you learn |
|---|----------|----------------|
| 04 | `04_pytorch_fundamentals.ipynb` | Tensors, shapes/dtypes, GPU/CPU devices, autograd, `nn.Module`, loss functions, optimizers |
| 05 | `05_training_loop_from_scratch.ipynb` | `Dataset`/`DataLoader`, batches/epochs, the `zero_grad → forward → loss → backward → step` loop, validation, and the key hyperparameters |

## After this phase you can…
- Create and manipulate tensors and read shape errors instead of fearing them.
- Build a small model as an `nn.Module` and train it with an optimizer.
- Write and explain a full training/validation loop — the same loop Hugging Face's `Trainer` runs for you.
- Feel the effect of learning rate, batch size, and epochs.

## Prerequisites
Phase 1. Runs on **CPU** (tiny models); a GPU only makes it faster.

## Next
➡️ **Phase 3 — Transformers & Hugging Face** (`../03-transformers/`): the architecture and toolkit behind LLMs.
