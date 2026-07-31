#!/usr/bin/env python3
import gc, json, os, sys, time
import torch
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

print("=" * 60)
print("PRODUCTION TRAINING — Gemma-4-31B 16-bit LoRA")
print("=" * 60)
t0 = time.time()

model_name = "google/gemma-4-31B-it"
max_seq_length = 4096
dtype = torch.bfloat16

print(f"\n[1/5] Loading model {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=dtype,
    device_map="auto",
)
print(f"  Model loaded in {time.time()-t0:.0f}s")

print("\n[2/5] Applying LoRA adapters...")
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj.linear", "k_proj.linear", "v_proj.linear", "o_proj.linear", "gate_proj.linear", "up_proj.linear", "down_proj.linear"],
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
model.gradient_checkpointing_enable()

print("\n[3/5] Loading production dataset...")
def load_jsonl(path):
    examples = []
    with open(path) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples

def convert_messages(messages):
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
            converted.append({"role": "model", "content": content})
        else:
            converted.append({"role": role, "content": content})
    return converted

train_raw = load_jsonl("style_sft_prod/train.jsonl")
eval_raw  = load_jsonl("style_sft_prod/eval.jsonl")

for ex in train_raw: ex["messages"] = convert_messages(ex["messages"])
for ex in eval_raw: ex["messages"] = convert_messages(ex["messages"])

def format_example(example):
    text = tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)
    if not text.endswith(tokenizer.eos_token): text += tokenizer.eos_token
    return {"text": text}

train_dataset = Dataset.from_list(train_raw).map(format_example)
eval_dataset  = Dataset.from_list(eval_raw).map(format_example)

print("\n[4/5] Starting training...")
OUTPUT_DIR = "lawbuddy-prod-31b-16bit"

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
    max_length=max_seq_length,
    dataset_text_field="text",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
)

train_result = trainer.train()

print("\n[5/5] Saving and merging best model...")
FINAL_DIR = "lawbuddy-gemma4-31b-merged"
model = trainer.model.merge_and_unload()
model.save_pretrained(FINAL_DIR)
tokenizer.save_pretrained(FINAL_DIR)

print("\nDone! Merged model saved to", FINAL_DIR)
