#!/usr/bin/env python3
"""Production fine-tuning: Gemma-2-27B 16-bit LoRA on 1000-example dataset."""
import gc, json, os, sys, time
os.chdir("/marimo/training")

print("=" * 60)
print("PRODUCTION TRAINING — Gemma-2-27B 16-bit LoRA")
print("=" * 60)
t0 = time.time()

import torch
import unsloth
from unsloth import FastLanguageModel

max_seq_length = 4096
dtype = torch.bfloat16 if unsloth.is_bfloat16_supported() else torch.float16

print(f"\n[1/5] Loading model (dtype={dtype}, seq_len={max_seq_length})...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gemma-2-27b-it",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=False,
)
print(f"  Model loaded in {time.time()-t0:.0f}s")

# ── 2. Apply LoRA ──
print("\n[2/5] Applying LoRA adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)
model.print_trainable_parameters()

# ── 3. Load dataset ──
print("\n[3/5] Loading production dataset...")
from datasets import Dataset

def load_jsonl(path):
    examples = []
    with open(path) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples

def convert_messages_for_gemma(messages):
    """Gemma 2 doesn't support system role. Fold system into user message."""
    converted = []
    system_text = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            system_text = content
        elif role == "user":
            if system_text:
                content = system_text + "\n\n" + content
                system_text = ""
            converted.append({"role": "user", "content": content})
        elif role == "assistant":
            # Gemma uses "model" not "assistant"
            converted.append({"role": "model", "content": content})
        else:
            converted.append({"role": role, "content": content})
    return converted

train_raw = load_jsonl("style_sft_prod/train.jsonl")
eval_raw  = load_jsonl("style_sft_prod/eval.jsonl")

# Convert all messages
for ex in train_raw:
    ex["messages"] = convert_messages_for_gemma(ex["messages"])
for ex in eval_raw:
    ex["messages"] = convert_messages_for_gemma(ex["messages"])

# Format with chat template + EOS
def format_example(example):
    text = tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False
    )
    if not text.endswith(tokenizer.eos_token):
        text += tokenizer.eos_token
    return {"text": text}

train_dataset = Dataset.from_list(train_raw).map(format_example)
eval_dataset  = Dataset.from_list(eval_raw).map(format_example)

print(f"  Train: {len(train_dataset)}  Eval: {len(eval_dataset)}")
print(f"  Sample text length: {len(train_dataset[0]['text'])} chars")

# ── 4. Train ──
print("\n[4/5] Starting training...")
from trl import SFTTrainer, SFTConfig

OUTPUT_DIR = "lawbuddy-prod-27b-16bit"

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=2,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    logging_steps=5,
    eval_strategy="steps",
    eval_steps=25,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    bf16=True,
    max_grad_norm=0.3,
    optim="adamw_8bit",
    report_to="none",
    max_seq_length=max_seq_length,
    dataset_text_field="text",
    seed=42,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
)

print(f"  Effective batch size: {1 * 8} = 8")
print(f"  Epochs: 2")

train_result = trainer.train()
print(f"\n  Training complete!")
print(f"  Final train loss: {train_result.training_loss:.4f}")

# ── 5. Save ──
print("\n[5/5] Saving best model...")
FINAL_DIR = "lawbuddy-prod-27b-16bit-final"
trainer.model.save_pretrained(FINAL_DIR)
tokenizer.save_pretrained(FINAL_DIR)

log_history = trainer.state.log_history
with open("train_prod.log.json", "w") as f:
    json.dump(log_history, f, indent=2)

eval_losses = [e["eval_loss"] for e in log_history if "eval_loss" in e]
print(f"\n{'=' * 60}")
print(f"TRAINING COMPLETE")
print(f"{'=' * 60}")
print(f"  Duration: {(time.time()-t0)/60:.1f} minutes")
print(f"  Final train loss: {train_result.training_loss:.4f}")
if eval_losses:
    print(f"  Best eval loss: {min(eval_losses):.4f}")
    print(f"  Final eval loss: {eval_losses[-1]:.4f}")
print(f"  Adapter saved to: {FINAL_DIR}/")
print(f"  Log saved to: train_prod.log.json")

del model, trainer
gc.collect()
torch.cuda.empty_cache()
print("\nDone. GPU memory freed.")
