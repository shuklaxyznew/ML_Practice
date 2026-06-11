# torch-finetune

Fine-tuning pre-trained language models using PEFT, LoRA, and QLoRA — built on PyTorch and HuggingFace.

---

## What It Does

Three independent, runnable scripts that demonstrate progressively more efficient ways to fine-tune a pre-trained model on your own data, using the same sentiment classification task so results are directly comparable.

---

## Project Structure

```
torch_finetune/
├── torch_finetune/             ← shared utilities (installable package)
│   ├── __init__.py
│   ├── dataset.py              ← SentimentDataset, InstructionDataset, sample data
│   └── utils.py                ← get_device, compute_metrics, print_trainable, run_inference
├── peft_finetune.py            ← Script 1: full fine-tune vs PEFT prompt tuning
├── lora_finetune.py            ← Script 2: LoRA classification + causal LM + merge
├── qlora_finetune.py           ← Script 3: QLoRA with 4-bit quantization
├── pyproject.toml
└── README.md
```

---

## Installation

```bash
git clone https://github.com/yourname/torch-finetune
cd torch_finetune
pip install -e "."

# For QLoRA (GPU only)
pip install -e ".[qlora]"
```

---

## Run the Scripts

Each script is fully independent — run any one without the others:

```bash
python peft_finetune.py     # full fine-tune + PEFT prompt tuning
python lora_finetune.py     # LoRA classification + merge + causal LM
python qlora_finetune.py    # QLoRA classification + causal LM
```

Output models save to `./output/` by default.

---

## The Three Methods

### Script 1 — `peft_finetune.py`

**`full_finetune()`**
Updates every parameter in the model. Trains all ~66 million parameters of DistilBERT.
This is the baseline: best possible accuracy, most memory and compute required.
`learning_rate=2e-5` is kept small to avoid overwriting pre-trained knowledge.

**`peft_prompt_tuning()`**
Freezes the entire base model. Trains only 8 "virtual token" embeddings prepended to each input.
These are not real words — they are learned continuous vectors that steer the frozen model
toward the target task. Only ~8,000 parameters are updated vs 66 million.

---

### Script 2 — `lora_finetune.py`

**`lora_classification()`**
Adds small trainable matrices A and B alongside frozen weight matrices in the attention layers.

```
Forward pass: output = (W_frozen + A × B) × input

W shape:  768 × 768  = 589,824 parameters
A shape:  768 × 8    =   6,144 parameters   (r=8)
B shape:  8   × 768  =   6,144 parameters
LoRA total:              12,288 → 48× fewer than W alone
```

Key hyperparameters:
- `r=8` — rank, the bottleneck size. Higher = more capacity, more parameters.
- `lora_alpha=16` — scaling factor = alpha/r applied to adapter output. Usually 2×r.
- `target_modules=["q_lin", "v_lin"]` — which layers to adapt. Query + Value is the standard choice.
- `bias="none"` — don't train bias terms.

**`lora_merge()`**
After training, bakes the adapter into the base model: `W_final = W_original + (A × B)`.
Result is a standard model with no PEFT overhead — same speed as the original.
Use for production deployment. Swap adapters at runtime if you keep them separate.

**`lora_causal_lm()`**
Same LoRA pattern applied to GPT2 (generative model) using Alpaca-style instruction format.
For production, replace `"gpt2"` with `"meta-llama/Llama-3-8b-hf"` or `"mistralai/Mistral-7B-v0.1"` — the code is identical.

---

### Script 3 — `qlora_finetune.py`

**The problem QLoRA solves:**
A 7B model in float32 = 28 GB GPU memory — needs an A100.
QLoRA loads the base model in 4-bit NF4 format → same model = ~6 GB → fits on a consumer RTX 3080.

**How it works:**
1. Base model quantized to 4-bit NF4 (`BitsAndBytesConfig`) — frozen, not trained
2. LoRA adapters added on top in bfloat16 — only adapters are trained
3. Double quantization: quantization constants themselves are quantized, saving ~0.37 bits/param

**Memory comparison (7B model):**

| Method | GPU RAM | Hardware needed |
|---|---|---|
| Full fine-tuning fp32 | 112 GB | 4× A100 80GB |
| Full fine-tuning bf16 | 56 GB | 2× A100 80GB |
| LoRA bf16 base | 28 GB | 1× A100 80GB |
| QLoRA 4-bit + LoRA | ~6 GB | 1× RTX 3080 10GB |

**`bnb_config()`**
- `load_in_4bit=True` — store weights in 4-bit
- `bnb_4bit_quant_type="nf4"` — NormalFloat4, optimized for normally-distributed neural network weights
- `bnb_4bit_compute_dtype=bfloat16` — upcasts to 16-bit for actual computation (weights stored in 4-bit, computed in 16-bit)
- `bnb_4bit_use_double_quant=True` — quantize the quantization scale constants too

**`prepare_model_for_kbit_training()`**
Required before applying LoRA to a quantized model. Casts LayerNorm to float32, enables gradient checkpointing, freezes base parameters.

**`gradient_accumulation_steps=4`**
Simulates batch_size=8 with only batch_size=2 in memory. Accumulates gradients over 4 forward passes before one weight update.

**`optim="paged_adamw_32bit"`**
Adam optimizer stores 2× extra copies of all parameters (momentum, variance).
PagedAdamW offloads those states to CPU RAM when GPU memory is full.
Essential for fitting QLoRA training on consumer GPUs.

**`lr_scheduler_type="cosine"`** + **`warmup_ratio=0.03`**
Learning rate warms up linearly for the first 3% of steps then decays on a cosine curve.
Prevents large gradient updates at the start from destabilising pre-trained weights.

---

## Comparison Table

| | Full Fine-Tune | PEFT Prompt Tuning | LoRA | QLoRA |
|---|---|---|---|---|
| % params trained | 100% | <0.01% | ~0.3% | ~0.3% |
| Base model precision | fp32 | fp16 | fp16 | 4-bit NF4 |
| Memory (7B model) | 112 GB | 56 GB | 28 GB | ~6 GB |
| Accuracy vs full FT | Baseline | Lower | High | Near-LoRA |
| Saved adapter size | Full model | Few KB | Few MB | Few MB |
| Best for | Max accuracy | Many tasks / one model | 1–5B models | 7B+ on limited GPU |

---

## Decision Tree

```
Need to fine-tune?
    ↓
7B+ model, limited GPU (< 24GB)?
    YES → QLoRA
    NO  ↓
        Many tasks, one shared base model?
            YES → PEFT Prompt Tuning
            NO  ↓
                Max accuracy + large GPU?
                    YES → Full fine-tuning
                    NO  → LoRA  (best default)
```

---

## Shared Utilities

### `get_device()`
Returns best available: CUDA → MPS (Apple Silicon) → CPU.

### `compute_metrics(eval_pred)`
Computes accuracy from HuggingFace Trainer eval predictions.

### `print_trainable(model)`
Prints trainable vs frozen parameter counts and percentage. Key for comparing methods.

### `run_inference(model, tokenizer, device, texts)`
Runs classification inference on a list of strings and prints POSITIVE / NEGATIVE.

---

## Swap to a Larger Model

All three scripts work identically with larger models — just change `MODEL_NAME`:

```python
# LoRA and QLoRA for LLaMA-3
MODEL_NAME = "meta-llama/Llama-3-8b-hf"

# Change target_modules to match LLaMA attention layer names
config = LoraConfig(
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    ...
)
```

For 7B+ models on limited GPU, switch to `qlora_finetune.py` — the code is structurally identical.
