# Phase 4 — Data preparation

Fine-tuning is only as good as its data. Here you build the **lead-intent**
dataset used through the rest of the course, learn the JSONL format, make a
clean train/validation split, and design the prompt templates that turn raw
records into examples a model can learn from.

## Notebooks (do them in order)

| # | Notebook | What you learn |
|---|----------|----------------|
| 09 | `09_dataset_preparation.ipynb` | What makes a good dataset, generating the lead data, exploring it with pandas, writing/reading **JSONL**, train/validation split, class balance, cleaning |
| 10 | `10_instruction_finetuning_format.ipynb` | Supervised fine-tuning (SFT), prompt-template design, Alpaca/chat formats, system prompts, loss masking |

## The shared dataset
Both notebooks (and every later one) use the same **lead-intent** problem:
classify a lead as `hot` / `warm` / `cold` from `family_size`, `income`,
`rent_or_own`, `cta`, `engagement_month`, and `current_condition`. The exact
data generator is identical across notebooks, so results line up end to end.

## After this phase you can…
- Save and load datasets as JSONL and make a leakage-free train/val split.
- Turn structured records into either classifier text or instruction prompt → response pairs.
- Explain what SFT does and why prompt templates and loss masking matter.

## Prerequisites
Phases 1–3. Runs on **CPU** (mostly pandas + standard library).

## Next
➡️ **Phase 5 — Fine-tuning** (`../05-finetuning/`): actually train a model on this data with LoRA and QLoRA.
