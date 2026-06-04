# Phase 3 — Transformers & Hugging Face

How modern language models actually work, and the library you'll use to run and
fine-tune them. You'll open up self-attention, see how text becomes numbers, and
then drive real pretrained models with Hugging Face.

## Notebooks (do them in order)

| # | Notebook | What you learn |
|---|----------|----------------|
| 06 | `06_transformers_and_attention.ipynb` | Self-attention (query/key/value), scaled dot-product attention, the Transformer block, encoder vs decoder |
| 07 | `07_tokenization_and_embeddings.ipynb` | Tokenizers (subword/BPE/WordPiece), token IDs, special tokens, padding/truncation, `max_length`, embeddings |
| 08 | `08_huggingface_transformers_basics.ipynb` | `pipeline()`, `AutoTokenizer`/`AutoModel`, the `datasets` library, a minimal `Trainer` run, save/load |

## After this phase you can…
- Explain in plain words how attention lets tokens "look at" each other.
- Turn text into `input_ids` + `attention_mask` and back again.
- Load free pretrained models and tokenizers and run inference.
- Recognize the exact tools (`AutoModel`, `Trainer`, `datasets`) used in the fine-tuning phases.

## Prerequisites
Phases 1–2. Runs on **CPU** with the small free models used here; first run
downloads models from the Hugging Face Hub (free, no login).

## Next
➡️ **Phase 4 — Data preparation** (`../04-data-prep/`): build the dataset you'll fine-tune on.
