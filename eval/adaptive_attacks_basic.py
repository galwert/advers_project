#!/usr/bin/env python3
"""
Adaptive GCG Attack on Defended Model

Runs nanogcg directly on the defended model (base + LoRA adapter)
to test if the defense provides intrinsic robustness beyond
invalidating pre-computed suffixes.

Two-phase approach:
  Phase 1: Run all GCG attacks and generate responses (target model only)
  Phase 2: Unload target model, load judge, evaluate all responses

Usage:
    python adaptive_attack.py --model qwen --adapter_path ./two_stage_outputs_v2/defender_v2_cka_20260214_144650
    python adaptive_attack.py --model qwen  # baseline (no adapter)
"""

import torch
import nanogcg
from nanogcg import GCGConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import pandas as pd
import io
import gc
import requests
import os
import sys
import json
import csv
import argparse
from datetime import datetime

sys.modules["flash_attn"] = None

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'phase2'))
from llm_judge import load_judge, classify_response, setup_gibberish_detector, strict_coherence_filter

MODEL_IDS = {
    "llama2": "meta-llama/Llama-2-7b-chat-hf",
    "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
    "vicuna": "lmsys/vicuna-7b-v1.5",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
    "zephyr": "HuggingFaceH4/zephyr-7b-beta",
    "qwen": "Qwen/Qwen1.5-7B-Chat",
    "yi": "01-ai/Yi-6B-Chat",
    "starling": "berkeley-nest/Starling-LM-7B-alpha",
    "phi2": "microsoft/phi-2",
    "orca": "microsoft/Orca-2-7b",
    "yi9b": "01-ai/Yi-1.5-9B-Chat",
    "mistral_nemo": "mistralai/Mistral-Nemo-Instruct-2407",
    "qwen-14b": "Qwen/Qwen1.5-14B-Chat",
    "phi3": "microsoft/Phi-3-medium-4k-instruct",
}

ADVBENCH_URL = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"


def load_advbench(n_samples=100):
    print(f"[*] Loading AdvBench ({n_samples} prompts)...")
    response = requests.get(ADVBENCH_URL)
    response.raise_for_status()
    df = pd.read_csv(io.StringIO(response.text))
    return list(df.head(n_samples).itertuples(index=False, name=None))


def generate_response(model, tokenizer, prompt_with_suffix, max_new_tokens=256):
    enc = tokenizer(prompt_with_suffix, return_tensors="pt", truncation=True, max_length=512)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    with torch.no_grad():
        out = model.generate(
            **enc, max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=list(MODEL_IDS.keys()))
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--n_prompts", type=int, default=100)
    parser.add_argument("--gcg_steps", type=int, default=500)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--judge_only", type=str, default=None,
                        help="Skip GCG, re-judge from existing JSON file")
    args = parser.parse_args()

    model_id = MODEL_IDS[args.model]
    tag = "defended" if args.adapter_path else "baseline"
    if args.output is None:
        args.output = f"adaptive_attack_{args.model}_{tag}.json"

    # ── Judge-only mode: reload Phase 1 results from file ────────────
    if args.judge_only:
        print(f"[*] Judge-only mode: loading results from {args.judge_only}")
        with open(args.judge_only) as f:
            saved = json.load(f)
        results = saved["results"]
        # Use metadata from the saved file
        model_id = saved.get("model_id", model_id)
        tag = saved.get("tag", tag)
        args.output = args.judge_only  # overwrite the same JSON
        print(f"[+] Loaded {len(results)} results (avg loss: {saved.get('avg_loss', 0):.4f})")
    else:
        print(f"[*] Model: {model_id}")
        print(f"[*] Adapter: {args.adapter_path or 'None (baseline)'}")
        print(f"[*] GCG steps: {args.gcg_steps}")

        # ── Phase 1: GCG attacks + response generation ──────────────────
        print(f"\n{'='*70}")
        print(f"PHASE 1: Running GCG attacks ({args.n_prompts} prompts, {args.gcg_steps} steps each)")
        print(f"{'='*70}")

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto",
            trust_remote_code=True, attn_implementation="eager",
        )

        if args.adapter_path:
            print(f"[*] Loading adapter from {args.adapter_path}")
            model = PeftModel.from_pretrained(model, args.adapter_path)
            model = model.merge_and_unload()
            print(f"[+] Adapter merged into base model")

        behaviors = load_advbench(args.n_prompts)
        config = GCGConfig(
            num_steps=args.gcg_steps,
            search_width=64,
            topk=64,
            seed=42,
            verbosity="WARNING",
            use_prefix_cache=False,
        )

        results = []
        for i, (prompt, target) in enumerate(behaviors):
            print(f"\n[{i+1}/{len(behaviors)}] {prompt[:80]}...")

            # Run GCG
            result = nanogcg.run(model, tokenizer, prompt, target, config)

            # Generate full response with the optimized suffix
            full_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": f"{prompt} {result.best_string}"}],
                tokenize=False, add_generation_prompt=True,
            )
            response = generate_response(model, tokenizer, full_prompt)

            results.append({
                "prompt": prompt,
                "target": target,
                "best_loss": float(result.best_loss),
                "best_suffix": result.best_string,
                "response": response[:500],
            })

            print(f"  Loss: {result.best_loss:.4f} | Suffix: {result.best_string[:60]}")
            print(f"  Response: {response[:150]}")

            # Save incrementally (without judge results yet)
            with open(args.output, "w") as f:
                json.dump({
                    "model": args.model, "model_id": model_id,
                    "adapter_path": args.adapter_path, "tag": tag,
                    "gcg_steps": args.gcg_steps, "n_prompts": len(behaviors),
                    "completed": i + 1, "phase": "gcg",
                    "avg_loss": sum(r["best_loss"] for r in results) / len(results),
                    "results": results,
                    "timestamp": datetime.now().isoformat(),
                }, f, indent=2)

        print(f"\n[+] Phase 1 complete: {len(results)} attacks done")
        print(f"    Avg loss: {sum(r['best_loss'] for r in results)/len(results):.4f}")

        # Unload target model
        print(f"\n[*] Unloading target model to free GPU memory...")
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    # ── Phase 2: Judge all responses ────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"PHASE 2: Judging {len(results)} responses")
    print(f"{'='*70}")

    judge_model_id = "allenai/wildguard"
    print(f"[*] Loading judge: {judge_model_id} (fp16)")
    judge_model, judge_tokenizer = load_judge(judge_model_id, quantize=False)
    gib_detector = setup_gibberish_detector()

    successes = 0
    filtered = 0
    for i, r in enumerate(results):
        # Strict coherence filter first
        override, filter_reason = strict_coherence_filter(
            r["response"], suffix=r.get("best_suffix"),
        )

        if override:
            is_success = False
            reason = f"FILTERED: {filter_reason}"
            filtered += 1
        else:
            is_success, reason, meta = classify_response(
                r["response"], r["prompt"],
                judge_model, judge_tokenizer, judge_model_id, gib_detector,
                suffix=r.get("suffix")
            )

        r["judge_success"] = is_success
        r["judge_reason"] = reason
        if is_success:
            successes += 1

        print(f"\n[{i+1}/{len(results)}] {'SUCCESS' if is_success else 'FAIL'}")
        print(f"  Prompt:   {r['prompt'][:100]}")
        print(f"  Suffix:   {r.get('best_suffix', '')[:80]}")
        print(f"  Loss:     {r.get('best_loss', 0):.4f}")
        print(f"  Response: {r['response'][:200]}")
        if override:
            print(f"  Filter:   {filter_reason}")

    # ── Final summary ───────────────────────────────────────────────────
    final_asr = successes / len(results) * 100
    print(f"\n{'='*70}")
    print(f"ADAPTIVE ATTACK RESULTS ({tag})")
    print(f"{'='*70}")
    print(f"Model: {args.model} ({model_id})")
    print(f"Adapter: {args.adapter_path or 'None'}")
    print(f"Prompts: {len(results)} | Steps: {args.gcg_steps}")
    print(f"ASR (strict judge): {successes}/{len(results)} = {final_asr:.1f}%")
    print(f"Filtered as degenerate: {filtered}/{len(results)}")
    print(f"Avg loss: {sum(r['best_loss'] for r in results)/len(results):.4f}")

    # Save final results with judge verdicts (JSON)
    with open(args.output, "w") as f:
        json.dump({
            "model": args.model, "model_id": model_id,
            "adapter_path": args.adapter_path, "tag": tag,
            "gcg_steps": args.gcg_steps, "n_prompts": len(results),
            "completed": len(results), "phase": "done",
            "successes": successes, "asr_judge": final_asr,
            "avg_loss": sum(r["best_loss"] for r in results) / len(results),
            "results": results,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)

    # Save per-prompt CSV
    csv_path = args.output.replace(".json", ".csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "prompt", "target", "best_loss", "best_suffix", "response", "judge_success", "judge_reason"])
        for i, r in enumerate(results):
            writer.writerow([
                i + 1, r["prompt"], r.get("target", ""), f"{r.get('best_loss', 0):.4f}",
                r.get("best_suffix", ""), r["response"],
                r.get("judge_success", ""), r.get("judge_reason", ""),
            ])

    print(f"[+] Saved to {args.output} and {csv_path}")


if __name__ == "__main__":
    main()
