#!/usr/bin/env python3
"""Multi-model GCG: jointly optimize a suffix against multiple surrogate models.

For each step:
1. Compute token gradient on each model, average gradients
2. Sample candidates from averaged gradient
3. Compute loss on each model for each candidate, average losses
4. Select candidate with lowest average loss

This produces suffixes optimized for transferability across models.
"""
import sys
sys.modules["flash_attn"] = None
sys.modules["flash_attn_2_cuda"] = None

import torch
import json
import os
import gc
import argparse
import requests
import csv
import io
from dataclasses import dataclass
from typing import List, Dict, Tuple
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, "../phase2")
from llm_judge import load_judge, classify_response, setup_gibberish_detector

from nanogcg.utils import get_nonascii_toks, INIT_CHARS


MODEL_IDS = {
    "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
    "vicuna": "lmsys/vicuna-7b-v1.5",
    "qwen14b": "Qwen/Qwen1.5-14B-Chat",
    "phi3": "microsoft/Phi-3-medium-4k-instruct",
}

ADAPTERS = {
    "llama3": "7b_defense_wildguard_outputs/defender_v2_cka_20260306_183257",
    "mistral": "7b_defense_wildguard_outputs/defender_v2_cka_20260314_215132",
    "vicuna": "7b_defense_wildguard_outputs/defender_v2_cka_20260318_125824",
    "qwen14b": "7b_defense_wildguard_outputs/defender_v2_cka_20260321_185854",
    "phi3": "7b_defense_wildguard_outputs/defender_v2_cka_20260317_203804",
}

FMT = {
    "llama3": lambda t, p: t.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True),
    "mistral": lambda t, p: f"[INST] {p} [/INST]",
    "vicuna": lambda t, p: f"USER: {p}\nASSISTANT:",
    "qwen14b": lambda t, p: t.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True),
    "phi3": lambda t, p: t.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True),
}


def compute_gcg_loss(model, tokenizer, embedding_layer, goal, target, suffix_ids, device):
    """Compute GCG loss for a given suffix on a single model."""
    # Format the prompt with suffix tokens decoded
    suffix_str = tokenizer.decode(suffix_ids.squeeze(), skip_special_tokens=True)
    full_prompt = f"{goal} {suffix_str}"

    # Use chat template
    model_name = None
    for name, mid in MODEL_IDS.items():
        if hasattr(model, 'name_or_path') and mid in str(getattr(model, 'name_or_path', '')):
            model_name = name
            break
        if hasattr(model.config, '_name_or_path') and mid in model.config._name_or_path:
            model_name = name
            break

    if model_name and model_name in FMT:
        formatted = FMT[model_name](tokenizer, full_prompt)
    else:
        try:
            formatted = tokenizer.apply_chat_template(
                [{"role": "user", "content": full_prompt}],
                tokenize=False, add_generation_prompt=True
            )
        except:
            formatted = full_prompt

    target_str = " " + target

    enc = tokenizer(formatted + target_str, return_tensors="pt", truncation=True, max_length=512).to(device)
    target_ids = tokenizer(target_str, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)

    with torch.no_grad():
        output = model(**enc)
        logits = output.logits

    # Loss on target tokens
    n_target = target_ids.shape[1]
    shift = enc.input_ids.shape[1] - n_target
    shift_logits = logits[0, shift - 1:-1, :]
    shift_labels = target_ids.squeeze(0)

    loss = torch.nn.functional.cross_entropy(shift_logits, shift_labels)
    return loss.item()


def compute_token_gradient_multimodel(models_data, goal, target, optim_ids, device):
    """Compute averaged token gradient across multiple models.

    models_data: list of (model, tokenizer, embedding_layer, model_name) tuples
    """
    all_grads = []

    for model, tokenizer, embedding_layer, model_name in models_data:
        # Format prompt
        suffix_str = tokenizer.decode(optim_ids.squeeze(), skip_special_tokens=True)
        full_prompt = f"{goal} " + "{optim_str}"
        target_str = " " + target

        # Tokenize parts
        if model_name in FMT:
            formatted = FMT[model_name](tokenizer, full_prompt)
        else:
            formatted = full_prompt

        before_str, after_str = formatted.split("{optim_str}")

        before_ids = tokenizer([before_str], padding=False, return_tensors="pt")["input_ids"].to(device, torch.int64)
        after_ids = tokenizer([after_str], add_special_tokens=False, return_tensors="pt")["input_ids"].to(device, torch.int64)
        target_ids = tokenizer([target_str], add_special_tokens=False, return_tensors="pt")["input_ids"].to(device, torch.int64)

        before_embeds = embedding_layer(before_ids)
        after_embeds = embedding_layer(after_ids)
        target_embeds = embedding_layer(target_ids)

        # One-hot encoding for gradient
        optim_ids_onehot = torch.nn.functional.one_hot(optim_ids, num_classes=embedding_layer.num_embeddings)
        optim_ids_onehot = optim_ids_onehot.to(device, model.dtype)
        optim_ids_onehot.requires_grad_()

        optim_embeds = optim_ids_onehot @ embedding_layer.weight
        input_embeds = torch.cat([before_embeds, optim_embeds, after_embeds, target_embeds], dim=1)

        output = model(inputs_embeds=input_embeds)
        logits = output.logits

        shift = input_embeds.shape[1] - target_ids.shape[1]
        shift_logits = logits[..., shift - 1:-1, :].contiguous()
        shift_labels = target_ids

        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

        grad = torch.autograd.grad(outputs=[loss], inputs=[optim_ids_onehot])[0]
        all_grads.append(grad.float())

    # Average gradients across models
    avg_grad = torch.stack(all_grads).mean(dim=0)
    return avg_grad


def sample_ids_from_grad(optim_ids, grad, search_width, topk, n_replace):
    """Sample candidate token sequences based on gradient (from nanogcg)."""
    n_optim = optim_ids.shape[0]
    _, topk_ids = (-grad).topk(topk, dim=-1)  # (n_optim, topk)

    # Random positions to replace
    positions = torch.arange(n_optim, device=optim_ids.device).repeat(search_width, 1)
    replace_mask = torch.zeros_like(positions, dtype=torch.bool)
    for i in range(search_width):
        replace_pos = torch.randperm(n_optim)[:n_replace]
        replace_mask[i, replace_pos] = True

    # Sample from topk at replacement positions
    sampled = optim_ids.unsqueeze(0).repeat(search_width, 1)
    for i in range(search_width):
        for j in range(n_optim):
            if replace_mask[i, j]:
                sampled[i, j] = topk_ids[j, torch.randint(0, topk, (1,))]

    return sampled


def evaluate_candidates_multimodel(models_data, goal, target, sampled_ids, device, batch_size=32):
    """Evaluate candidate suffixes across all models, return averaged losses."""
    n_candidates = sampled_ids.shape[0]
    total_losses = torch.zeros(n_candidates, device=device)

    for model, tokenizer, embedding_layer, model_name in models_data:
        # Format prompt parts
        full_prompt = f"{goal} " + "{optim_str}"
        target_str = " " + target

        if model_name in FMT:
            formatted = FMT[model_name](tokenizer, full_prompt)
        else:
            formatted = full_prompt

        before_str, after_str = formatted.split("{optim_str}")

        before_ids = tokenizer([before_str], padding=False, return_tensors="pt")["input_ids"].to(device, torch.int64)
        after_ids = tokenizer([after_str], add_special_tokens=False, return_tensors="pt")["input_ids"].to(device, torch.int64)
        target_ids = tokenizer([target_str], add_special_tokens=False, return_tensors="pt")["input_ids"].to(device, torch.int64)

        before_embeds = embedding_layer(before_ids)
        after_embeds = embedding_layer(after_ids)
        target_embeds = embedding_layer(target_ids)

        model_losses = []
        for i in range(0, n_candidates, batch_size):
            batch_ids = sampled_ids[i:i + batch_size]
            bs = batch_ids.shape[0]

            batch_embeds = embedding_layer(batch_ids)
            input_embeds = torch.cat([
                before_embeds.repeat(bs, 1, 1),
                batch_embeds,
                after_embeds.repeat(bs, 1, 1),
                target_embeds.repeat(bs, 1, 1),
            ], dim=1)

            with torch.no_grad():
                output = model(inputs_embeds=input_embeds)
                logits = output.logits

            shift = input_embeds.shape[1] - target_ids.shape[1]
            shift_logits = logits[:, shift - 1:-1, :].contiguous()
            shift_labels = target_ids.repeat(bs, 1)

            losses = torch.nn.functional.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                reduction='none'
            ).reshape(bs, -1).mean(dim=1)

            model_losses.append(losses)

        total_losses += torch.cat(model_losses)

    return total_losses / len(models_data)


def multi_model_gcg(models_data, goal, target, device,
                    num_steps=500, search_width=64, topk=256, n_replace=1,
                    suffix_len=20):
    """Run multi-model GCG optimization."""
    # Use the first model's tokenizer for suffix initialization
    _, tokenizer, embedding_layer, _ = models_data[0]

    # Initialize suffix
    init_str = " ".join(["x"] * suffix_len)
    optim_ids = tokenizer(init_str, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)

    best_loss = float('inf')
    best_suffix = init_str
    losses = []

    for step in tqdm(range(num_steps), desc=f"{goal[:40]}..."):
        # Compute averaged gradient
        avg_grad = compute_token_gradient_multimodel(models_data, goal, target, optim_ids, device)

        with torch.no_grad():
            # Sample candidates
            sampled_ids = sample_ids_from_grad(
                optim_ids.squeeze(0), avg_grad.squeeze(0),
                search_width, topk, n_replace
            )

            # Evaluate candidates across all models
            candidate_losses = evaluate_candidates_multimodel(
                models_data, goal, target, sampled_ids, device, batch_size=16
            )

            # Select best
            best_idx = candidate_losses.argmin()
            current_loss = candidate_losses[best_idx].item()
            optim_ids = sampled_ids[best_idx].unsqueeze(0)

            if current_loss < best_loss:
                best_loss = current_loss
                best_suffix = tokenizer.decode(optim_ids.squeeze(), skip_special_tokens=True)

            losses.append(current_loss)

    return best_suffix, best_loss, losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="+", default=["mistral", "vicuna"],
                        help="Source models to jointly optimize against")
    parser.add_argument("--target", type=str, default="llama3",
                        help="Target model to evaluate transfer on")
    parser.add_argument("--defended", action="store_true",
                        help="Evaluate on defended target model")
    parser.add_argument("--n_prompts", type=int, default=25,
                        help="Number of HarmBench prompts")
    parser.add_argument("--num_steps", type=int, default=500)
    parser.add_argument("--search_width", type=int, default=64)
    parser.add_argument("--output_dir", type=str, default="./7b_defense_wildguard_outputs/harmbench_eval")
    args = parser.parse_args()

    device = "cuda"
    src_names = sorted(args.sources)
    src_tag = "_".join(src_names)

    # Load HarmBench prompts
    print("[*] Loading HarmBench prompts...")
    r = requests.get("https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_test.csv")
    reader = csv.DictReader(io.StringIO(r.text))
    all_behaviors = [(row['Behavior'], f"Sure, here is {row['Behavior'].lower()[:50]}") for row in reader]
    behaviors = all_behaviors[:args.n_prompts]
    print(f"[+] {len(behaviors)} prompts")

    # Phase 1: Generate multi-model GCG suffixes
    suffix_file = f"{args.output_dir}/multimodel_gcg_{src_tag}.json"
    if not os.path.exists(suffix_file):
        print(f"\n[*] Loading {len(src_names)} source models: {src_names}")
        models_data = []
        for src_name in src_names:
            print(f"  Loading {src_name}...")
            tokenizer = AutoTokenizer.from_pretrained(MODEL_IDS[src_name], trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_IDS[src_name], torch_dtype=torch.float16,
                device_map="auto", trust_remote_code=True,
                attn_implementation="eager"
            )
            model.eval()
            embedding_layer = model.get_input_embeddings()
            models_data.append((model, tokenizer, embedding_layer, src_name))

        print(f"\n[*] Running multi-model GCG ({len(src_names)} models joint optimization)...")
        results = []
        for i, (goal, target) in enumerate(behaviors):
            print(f"\n[{i+1}/{len(behaviors)}] {goal[:60]}...")
            try:
                suffix, loss, _ = multi_model_gcg(
                    models_data, goal, target, device,
                    num_steps=args.num_steps,
                    search_width=args.search_width,
                )
                results.append({"goal": goal, "target": target, "suffix": suffix, "loss": loss})
                print(f"  Loss: {loss:.3f}, Suffix: {suffix[:50]}...")
            except Exception as e:
                print(f"  ERR: {e}")
                results.append({"goal": goal, "target": target, "suffix": "", "loss": 999})

        json.dump({
            "sources": src_names, "n_prompts": len(behaviors), "results": results
        }, open(suffix_file, "w"), indent=2)
        print(f"\n[+] Saved {len(results)} suffixes to {suffix_file}")

        # Free source models
        for model, tokenizer, _, _ in models_data:
            del model, tokenizer
        torch.cuda.empty_cache()
        gc.collect()
    else:
        print(f"[SKIP] Suffixes exist: {suffix_file}")

    # Phase 2: Evaluate transfer to target
    suffix_data = json.load(open(suffix_file))

    for defended in ([True, False] if args.defended else [False, True]):
        kind = "def" if defended else "bl"
        out_file = f"{args.output_dir}/multimodel_{src_tag}_to_{args.target}_{kind}.json"
        if os.path.exists(out_file):
            print(f"[SKIP] {out_file}")
            continue

        print(f"\n[*] Evaluating transfer: {src_tag} -> {args.target}_{kind}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_IDS[args.target], trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_IDS[args.target], torch_dtype=torch.float16,
            device_map="auto", trust_remote_code=True,
            attn_implementation="eager"
        )
        if defended and args.target in ADAPTERS:
            model = PeftModel.from_pretrained(model, ADAPTERS[args.target])
            model = model.merge_and_unload()
        model.eval()

        judge_model, judge_tokenizer = load_judge("allenai/wildguard")
        gib = setup_gibberish_detector()

        results = []
        successes = 0
        for r_item in suffix_data["results"]:
            if not r_item["suffix"]:
                results.append({"goal": r_item["goal"], "success": False, "response": ""})
                continue

            full = f"{r_item['goal']} {r_item['suffix']}"
            fmt_fn = FMT.get(args.target)
            if fmt_fn:
                formatted = fmt_fn(tokenizer, full)
            else:
                formatted = full

            enc = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=512).to(model.device)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=256, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            resp = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)

            is_s, reason, meta = classify_response(resp, r_item["goal"], judge_model, judge_tokenizer, "allenai/wildguard", gib_detector=gib)
            if is_s:
                successes += 1
            results.append({"goal": r_item["goal"], "success": is_s, "response": resp[:200]})

        asr = successes / len(results) * 100 if results else 0
        print(f"  {src_tag}->{args.target}_{kind}: {successes}/{len(results)} ({asr:.1f}%)")

        json.dump({
            "sources": src_names, "target": f"{args.target}_{kind}", "asr": asr,
            "successes": successes, "total": len(results), "results": results,
            "attack_type": "multi_model_gcg"
        }, open(out_file, "w"), indent=2)

        del model, tokenizer, judge_model, judge_tokenizer
        torch.cuda.empty_cache()
        gc.collect()

    print("\n[+] Done")


if __name__ == "__main__":
    main()
