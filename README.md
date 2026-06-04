# Fine-Tuning LLMs — A Hands-On, Ground-Up Curriculum

A self-study course that takes you from "I know basic Python" to "I can fine-tune
a small language model end-to-end." Everything is in Jupyter notebooks with
runnable code, plain-language explanations, exercises, and debugging tips.

No advanced math required. We use **PyTorch** and **Hugging Face**, small
datasets, and free/open-source models so every notebook runs on a laptop or a
free Google Colab session.

---

## 📚 Learning order

Work through the notebooks **in order** — each one builds on the previous. They
are grouped into seven topic-phase folders, but the numeric prefixes (`01`–`15`)
always tell you the global order.

| #  | Notebook | What you learn |
|----|----------|----------------|
| **`01-foundations/`** | | **Phase 1 — Foundations** |
| 01 | `01-foundations/01_python_refresher.ipynb` | The exact Python you need for ML: lists, dicts, comprehensions, NumPy, functions, classes |
| 02 | `01-foundations/02_ml_fundamentals.ipynb` | What "learning from data" means: features, labels, train/test, overfitting, loss, metrics |
| 03 | `01-foundations/03_neural_network_basics.ipynb` | Neurons, layers, activations, forward pass, gradients, and backpropagation — intuitively |
| **`02-pytorch/`** | | **Phase 2 — PyTorch** |
| 04 | `02-pytorch/04_pytorch_fundamentals.ipynb` | Tensors, autograd, `nn.Module`, optimizers — the building blocks of all training |
| 05 | `02-pytorch/05_training_loop_from_scratch.ipynb` | Write a full training loop yourself so you understand what frameworks hide |
| **`03-transformers/`** | | **Phase 3 — Transformers & Hugging Face** |
| 06 | `03-transformers/06_transformers_and_attention.ipynb` | Self-attention and the Transformer block, explained without heavy math |
| 07 | `03-transformers/07_tokenization_and_embeddings.ipynb` | How text becomes numbers: tokenizers, token IDs, and embeddings |
| 08 | `03-transformers/08_huggingface_transformers_basics.ipynb` | Load models/tokenizers, run inference, use pipelines, the `Trainer` API |
| **`04-data-prep/`** | | **Phase 4 — Data preparation** |
| 09 | `04-data-prep/09_dataset_preparation.ipynb` | Build clean datasets, JSONL format, train/validation split, the lead dataset |
| 10 | `04-data-prep/10_instruction_finetuning_format.ipynb` | Prompt templates, instruction/response format, supervised fine-tuning (SFT) |
| **`05-finetuning/`** | | **Phase 5 — Fine-tuning** |
| 11 | `05-finetuning/11_lora_finetuning.ipynb` | LoRA vs full fine-tuning, adapters, PEFT, and a real LoRA run |
| 12 | `05-finetuning/12_qlora_finetuning.ipynb` | Quantization (4-bit), QLoRA, and fitting bigger models on small GPUs |
| **`06-eval-serving/`** | | **Phase 6 — Evaluation & serving** |
| 13 | `06-eval-serving/13_evaluation.ipynb` | Accuracy, precision, recall, F1, confusion matrix, and manual review |
| 14 | `06-eval-serving/14_inference_and_serving.ipynb` | Generate text, merge adapters, save/load, and serve a model behind an API |
| **`07-capstone/`** | | **Phase 7 — Capstone** |
| 15 | `07-capstone/15_mini_project_lead_intent.ipynb` | **Capstone:** fine-tune a small model to classify lead intent, end to end |

> 💡 Each notebook's "What to learn next" points to the next notebook by name —
> just look for that filename in the next folder. **Every folder also has its own
> `README.md`** describing that phase, its notebooks, and its GPU/CPU needs.

---

## ✅ Prerequisites

- **Basic Python**: variables, `if`/`for`, functions, importing libraries. (Notebook 01 refreshes the rest.)
- **No prior ML, deep learning, or fine-tuning experience needed.**
- **No advanced math.** We explain ideas with words and pictures, not proofs.
- Comfort installing packages with `pip`.

---

## 💻 Recommended environment

You have two easy options.

### Option A — Google Colab (recommended for beginners)
1. Upload the notebook (or open from Google Drive / GitHub).
2. For notebooks 11, 12, and 15, set the runtime to a GPU:
   **Runtime → Change runtime type → Hardware accelerator → T4 GPU**.
3. Run cells top to bottom. The first cell installs dependencies.

### Option B — Local machine
```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install the core stack
pip install --upgrade pip
pip install jupyter torch numpy pandas scikit-learn matplotlib
pip install transformers datasets accelerate evaluate
pip install peft bitsandbytes        # for LoRA / QLoRA (notebooks 11, 12, 15)

# 3. Launch
jupyter notebook
```

Each notebook also has its own install cell at the top, so you can run them
independently — you don't have to pre-install everything.

---

## ⚡ GPU vs CPU notes

| Notebooks | CPU? | GPU? | Notes |
|-----------|------|------|-------|
| 01–10, 13, 14 | ✅ Yes | optional | Concepts, tiny models, and data prep run fine on CPU. |
| 11 (LoRA) | 🐢 Slow but possible | ✅ Recommended | A tiny model trains on CPU in minutes; a real one wants a GPU. |
| 12 (QLoRA) | ❌ Not really | ✅ Required | 4-bit quantization via `bitsandbytes` needs an NVIDIA GPU. On CPU/Mac, read along and run the LoRA fallback. |
| 15 (Capstone) | ✅ Yes (classifier path) | ✅ Better (LLM path) | The DistilBERT classifier path runs on CPU; the LoRA-LLM path wants a GPU. |

**Apple Silicon (M1/M2/M3):** PyTorch runs on the `mps` device, but
`bitsandbytes` (QLoRA) does not. Use the LoRA path instead of QLoRA on Mac.

**Rule of thumb:** if a cell is taking too long on CPU, shrink the dataset,
lower `max_length`, or reduce the number of training steps — every fine-tuning
notebook shows you exactly where to do this.

---

## 🎯 Final project overview

The capstone (`07-capstone/15_mini_project_lead_intent.ipynb`) ties everything together by
fine-tuning a small model to predict **lead intent** for a sales/marketing use
case. Each lead is described by simple fields:

- `family_size` — number of people in the household
- `income` — approximate annual income
- `rent_or_own` — `"rent"` or `"own"`
- `cta` — the call-to-action they engaged with (e.g. `requested_quote`, `booked_demo`)
- `engagement_month` — when they engaged
- `current_condition` — their current situation/condition

The model learns to output a **lead intent** label: `hot`, `warm`, or `cold`.

You'll:
1. Generate and clean a small synthetic dataset.
2. Save it as **JSONL** and make a **train/validation split**.
3. Design a **prompt template** (for the LLM path) or features (for the classifier path).
4. **Fine-tune** a small model (DistilBERT classifier and/or a LoRA LLM).
5. **Evaluate** with accuracy, precision, recall, F1, a confusion matrix, and manual review.
6. **Run inference** on new leads.

By the end you'll have a working, evaluated, fine-tuned model and the
understanding to adapt this recipe to your own data.

---

## 🧭 How to use this course

- Read the markdown, then **run every code cell** — don't just read it.
- Do the **exercises** (look for the ✏️ marker). They're short on purpose.
- When something breaks, check the **"Common mistakes & debugging"** section
  near the end of each notebook — your error is probably listed.
- Each notebook ends with a **Summary** and **What to learn next**.

Happy fine-tuning! 🚀
