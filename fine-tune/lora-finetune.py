"""
InterviewCoach — Fine-tuning with Unsloth on Modal
Trains Qwen2.5-3B-Instruct for interview question classification using LoRA.

## Objective
The base Qwen2.5-3B model is capable but general-purpose. This fine-tune
teaches it one narrow task: given a raw interview question, output the correct
framework type and ordered coaching steps — nothing else.

Without fine-tuning, a 3B model needs a long system prompt with many few-shot
examples to produce consistent output, and still occasionally adds waffle or
gets the format wrong. After fine-tuning on ~80 domain-specific examples, the
model learns to:
  1. Classify questions reliably into 6 types (System Design, Product Design,
     Data Science, AI Engineering, Behavioural, Estimation)
  2. Return a clean, consistently formatted coaching card every time
  3. Do this faster and with less memory than the 7B model it replaces

The result is a lightweight 3B model that matches 7B quality on this specific
task — suitable for real-time inference on a local Mac during a live interview.

## Usage
    modal run finetune_unsloth_modal.py

## Optional CLI flags
    modal run finetune_unsloth_modal.py --max-steps 300
    modal run finetune_unsloth_modal.py --lora-r 16
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import modal

# ===========================================================================
# Configurations
# ===========================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

# Your Hugging Face username (https://huggingface.co/settings/profile)
<<<<<<< HEAD
HF_USERNAME = "build-small-hackathon"
=======
HF_USERNAME = "vadirajkrishna"
>>>>>>> c0f39ad (initial commit - InterviewCopilotLocal)

# Name for the fine-tuned model repo on Hugging Face
HF_MODEL_NAME = "interview-coach-3b"

# Base model to fine-tune (from Unsloth's HF collection)
BASE_MODEL = "unsloth/Qwen2.5-3B-Instruct"

# Path to training data files (relative to this script)
TRAIN_DATA_FILE = "data/train.json"
VALID_DATA_FILE = "data/valid.json"

# Modal secret name containing HF_TOKEN
# Create at: https://modal.com/secrets  key=HF_TOKEN
MODAL_HF_SECRET = "huggingface-secret"

# Modal GPU — A10G is cost-effective for 3B models (~$1.10/hr, job ~10 min)
# Options: "A10G", "A100", "L40S"
MODAL_GPU = "A10G"

# ===========================================================================
# LoRA hyperparameters — safe defaults for a small classification dataset
# ===========================================================================

LORA_R              = 8       # LoRA rank — increase to 16 for harder tasks
LORA_ALPHA          = 16      # scaling factor, usually 2x rank
LORA_DROPOUT        = 0.05
MAX_SEQ_LENGTH      = 256     # interview questions are short
MAX_STEPS           = 200     # small dataset — 200 is enough, increase if loss plateaus
BATCH_SIZE          = 4
GRAD_ACCUM_STEPS    = 2
LEARNING_RATE       = 2e-4
WARMUP_RATIO        = 0.1
SEED                = 42

# LoRA target modules — all projection layers in Qwen2.5
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# ===========================================================================
# Validation — fail loudly before Modal even starts
# ===========================================================================

def _validate_config():
    errors = []
    if not HF_USERNAME:
        errors.append("  • HF_USERNAME is empty — set your Hugging Face username")
    if not HF_MODEL_NAME:
        errors.append("  • HF_MODEL_NAME is empty — set a repo name")
    if not (SCRIPT_DIR / TRAIN_DATA_FILE).exists():
        errors.append(f"  • TRAIN_DATA_FILE not found: {TRAIN_DATA_FILE}")
    if not (SCRIPT_DIR / VALID_DATA_FILE).exists():
        errors.append(f"  • VALID_DATA_FILE not found: {VALID_DATA_FILE}")
    if errors:
        print("\n❌  Config errors — fix before running:\n")
        for e in errors:
            print(e)
        sys.exit(1)

# ===========================================================================
# Modal infrastructure
# ===========================================================================

app = modal.App("interview-coach-finetune")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "accelerate==1.9.0",
        "datasets==3.6.0",
        "hf-transfer==0.1.9",
        "huggingface_hub==0.34.2",
        "peft==0.16.0",
        "transformers==4.54.0",
        "trl==0.19.1",
        "unsloth[cu128-torch270]==2025.7.8",
        "unsloth_zoo==2025.7.10",
    )
    .env({"HF_HOME": "/model_cache"})
)

with image.imports():
    import unsloth  # noqa: F401 — must be first
    import torch
    from datasets import Dataset
    from transformers import TrainingArguments
    from trl import SFTTrainer
    from unsloth import FastLanguageModel

model_cache_vol = modal.Volume.from_name("ic-model-cache", create_if_missing=True)
checkpoint_vol  = modal.Volume.from_name("ic-checkpoints", create_if_missing=True)

# ===========================================================================
# Training config dataclass — populated from constants above
# ===========================================================================

@dataclass
class TrainingConfig:
    hf_repo:                  str
    model_name:               str
    max_seq_length:           int
    lora_r:                   int
    lora_alpha:               int
    lora_dropout:             float
    learning_rate:            float
    batch_size:               int
    gradient_accumulation_steps: int
    max_steps:                int
    warmup_ratio:             float
    seed:                     int
    train_data:               list
    valid_data:               list


def _build_config(
    max_steps: int = MAX_STEPS,
    lora_r: int    = LORA_R,
) -> TrainingConfig:
    """Load data files and assemble config. Call only after _validate_config()."""
    train_data = json.loads((SCRIPT_DIR / TRAIN_DATA_FILE).read_text())
    valid_data = json.loads((SCRIPT_DIR / VALID_DATA_FILE).read_text())
    return TrainingConfig(
        hf_repo                    = f"{HF_USERNAME}/{HF_MODEL_NAME}",
        model_name                 = BASE_MODEL,
        max_seq_length             = MAX_SEQ_LENGTH,
        lora_r                     = lora_r,
        lora_alpha                 = LORA_ALPHA,
        lora_dropout               = LORA_DROPOUT,
        learning_rate              = LEARNING_RATE,
        batch_size                 = BATCH_SIZE,
        gradient_accumulation_steps= GRAD_ACCUM_STEPS,
        max_steps                  = max_steps,
        warmup_ratio               = WARMUP_RATIO,
        seed                       = SEED,
        train_data                 = train_data,
        valid_data                 = valid_data,
    )

# ===========================================================================
# Fine-tuning function — runs on Modal GPU
# ===========================================================================

@app.function(
    image=image,
    gpu=MODAL_GPU,
    volumes={
        "/model_cache": model_cache_vol,
        "/checkpoints": checkpoint_vol,
    },
    secrets=[modal.Secret.from_name(MODAL_HF_SECRET)],
    timeout=3600,
)
def finetune(config: TrainingConfig):
    import os
    from huggingface_hub import login

    login(token=os.environ["HF_TOKEN"])

    print(f"Model:     {config.model_name}")
    print(f"HF repo:   {config.hf_repo}")
    print(f"Steps:     {config.max_steps}")
    print(f"LoRA rank: {config.lora_r}")
    print(f"Train:     {len(config.train_data)} examples")
    print(f"Valid:     {len(config.valid_data)} examples")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = config.model_name,
        max_seq_length = config.max_seq_length,
        dtype          = None,
        load_in_4bit   = True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r                        = config.lora_r,
        target_modules           = LORA_TARGET_MODULES,
        lora_alpha               = config.lora_alpha,
        lora_dropout             = config.lora_dropout,
        bias                     = "none",
        use_gradient_checkpointing = "unsloth",
        random_state             = config.seed,
        use_rslora               = False,
    )
    model.print_trainable_parameters()

    def format_example(example):
        return {
            "text": (
                f"<|im_start|>user\n{example['prompt']}<|im_end|>\n"
                f"<|im_start|>assistant\n{example['completion']}<|im_end|>"
            )
        }

    train_dataset = Dataset.from_list(config.train_data).map(format_example)
    valid_dataset = Dataset.from_list(config.valid_data).map(format_example)

    training_args = TrainingArguments(
        output_dir                  = "/checkpoints/interview-coach",
        per_device_train_batch_size = config.batch_size,
        gradient_accumulation_steps = config.gradient_accumulation_steps,
        learning_rate               = config.learning_rate,
        max_steps                   = config.max_steps,
        warmup_ratio                = config.warmup_ratio,
        fp16                        = not torch.cuda.is_bf16_supported(),
        bf16                        = torch.cuda.is_bf16_supported(),
        optim                       = "adamw_8bit",
        lr_scheduler_type           = "cosine",
        weight_decay                = 0.01,
        logging_steps               = 10,
        eval_strategy               = "steps",
        eval_steps                  = 50,
        save_strategy               = "steps",
        save_steps                  = 100,
        load_best_model_at_end      = True,
        report_to                   = "none",
        seed                        = config.seed,
    )

    trainer = SFTTrainer(
        model              = model,
        tokenizer          = tokenizer,
        train_dataset      = train_dataset,
        eval_dataset       = valid_dataset,
        dataset_text_field = "text",
        max_seq_length     = config.max_seq_length,
        args               = training_args,
    )

    print("Training...")
    trainer.train()
    print("Training complete.")

    print(f"Pushing to {config.hf_repo}...")
    model.push_to_hub(config.hf_repo)
    tokenizer.push_to_hub(config.hf_repo)
    print(f"Done → https://huggingface.co/{config.hf_repo}")

    return config.hf_repo


# ===========================================================================
# Test function — run after training to verify model output
# ===========================================================================

@app.function(
    image=image,
    gpu=MODAL_GPU,
    volumes={"/model_cache": model_cache_vol},
    secrets=[modal.Secret.from_name(MODAL_HF_SECRET)],
    timeout=600,
)
def test(
    question: str = "How would you design a payment system for a marketplace?",
):
    import os
    from huggingface_hub import login

    login(token=os.environ["HF_TOKEN"])

    hf_repo = f"{HF_USERNAME}/{HF_MODEL_NAME}"
    print(f"Loading {hf_repo}...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = hf_repo,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype          = None,
        load_in_4bit   = True,
    )
    FastLanguageModel.for_inference(model)

    prompt  = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    inputs  = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens = 80,
            temperature    = 0.1,
            do_sample      = False,
        )

    full  = tokenizer.decode(outputs[0], skip_special_tokens=True)
    reply = full.split("<|im_start|>assistant\n")[-1].strip()
    print(f"\nQ: {question}\n\n{reply}")


# ===========================================================================
# Entry point
# ===========================================================================

@app.local_entrypoint()
def main(
    max_steps: int = MAX_STEPS,
    lora_r:    int = LORA_R,
):
    _validate_config()
    config = _build_config(max_steps=max_steps, lora_r=lora_r)

    print("\n🚀  InterviewCoach fine-tuning")
    print(f"   Model:     {config.model_name}")
    print(f"   HF repo:   {config.hf_repo}")
    print(f"   Steps:     {config.max_steps}")
    print(f"   LoRA rank: {config.lora_r}")
    print(f"   Train:     {len(config.train_data)} examples")
    print(f"   Valid:     {len(config.valid_data)} examples\n")

    repo = finetune.remote(config)
    print(f"\n✅  Done → https://huggingface.co/{repo}")
