#!/usr/bin/env python3
"""Quick standalone test: load Llama-2-13B and generate a few responses.
No dependency on evaluate_v2 or benchmark_eval."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "NousResearch/Llama-2-13b-chat-hf"

print(f"Loading tokenizer: {model_id}")
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
print(f"  pad_token: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")
print(f"  eos_token: {tokenizer.eos_token} (id={tokenizer.eos_token_id})")
print(f"  unk_token: {tokenizer.unk_token} (id={tokenizer.unk_token_id})")

if not tokenizer.pad_token or tokenizer.pad_token == tokenizer.eos_token:
    if tokenizer.unk_token:
        tokenizer.pad_token = tokenizer.unk_token
        print(f"  -> Set pad_token to unk_token: {tokenizer.pad_token}")
tokenizer.padding_side = "left"

print(f"\nLoading model (fp16)...")
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
model.eval()

# Sync pad token
model.config.pad_token_id = tokenizer.pad_token_id
if hasattr(model, 'generation_config'):
    model.generation_config.pad_token_id = tokenizer.pad_token_id

prompts = [
    "Explain photosynthesis in simple terms.",
    "Write a short poem about the ocean.",
    "What is the capital of France and why is it famous?",
]

for p in prompts:
    # Llama-2 chat format (NousResearch version has no chat_template)
    formatted = f"[INST] {p} [/INST]"
    enc = tokenizer(formatted, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=200, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
    response = tokenizer.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"\n{'='*60}")
    print(f"PROMPT: {p}")
    print(f"RESPONSE: {response[:500]}")
