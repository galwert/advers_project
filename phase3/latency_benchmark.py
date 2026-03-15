#!/usr/bin/env python3
"""Lightweight latency & memory benchmark for base vs defended models.
Measures: load time, inference latency (per-token), peak GPU memory, throughput."""

import argparse, json, time, gc, os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def measure_inference(model, tokenizer, prompts, device, n_runs=3, max_new_tokens=128):
    """Measure mean latency and throughput over n_runs."""
    model.eval()
    latencies = []
    total_tokens = 0

    for run_idx in range(n_runs):
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        run_latency = 0.0
        run_tokens = 0

        for prompt in prompts:
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=max_new_tokens,
                                     do_sample=False, pad_token_id=tokenizer.pad_token_id)
            torch.cuda.synchronize()
            t1 = time.perf_counter()

            n_gen = out.shape[1] - enc.input_ids.shape[1]
            run_latency += (t1 - t0)
            run_tokens += n_gen

        latencies.append(run_latency)
        total_tokens += run_tokens

    peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
    mean_latency = sum(latencies) / len(latencies)
    mean_tokens = total_tokens / n_runs
    tokens_per_sec = mean_tokens / mean_latency if mean_latency > 0 else 0

    return {
        "mean_latency_s": round(mean_latency, 3),
        "mean_tokens": int(mean_tokens),
        "tokens_per_sec": round(tokens_per_sec, 1),
        "peak_gpu_gb": round(peak_mem, 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--precision", default="fp16", choices=["fp16", "fp32"])
    parser.add_argument("--merge_lora", action="store_true", help="Merge LoRA into base (zero overhead)")
    parser.add_argument("--n_prompts", type=int, default=20)
    parser.add_argument("--n_runs", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if args.precision == "fp16" else torch.float32

    prompts = [
        "Explain the concept of photosynthesis in simple terms.",
        "What are the main causes of climate change?",
        "Write a short poem about the ocean.",
        "How does a computer processor work?",
        "What is the capital of France and what is it known for?",
        "Describe the water cycle in detail.",
        "What are the benefits of regular exercise?",
        "Explain how vaccines work.",
        "What is machine learning?",
        "Describe the process of making bread from scratch.",
        "What causes earthquakes?",
        "How do airplanes fly?",
        "What is the difference between weather and climate?",
        "Explain the theory of relativity simply.",
        "What are renewable energy sources?",
        "How does the internet work?",
        "What is DNA and why is it important?",
        "Describe the solar system.",
        "What causes rain?",
        "How do plants grow?",
    ][:args.n_prompts]

    results = {"model_id": args.model_id, "precision": args.precision,
               "n_prompts": len(prompts), "n_runs": args.n_runs,
               "max_new_tokens": args.max_new_tokens}

    # --- Base model ---
    print(f"[1/2] Loading base model: {args.model_id}")
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=dtype, device_map=device)
    t_load_base = time.perf_counter() - t0
    results["base_load_time_s"] = round(t_load_base, 2)

    print(f"  Loaded in {t_load_base:.1f}s, measuring inference...")
    base_stats = measure_inference(model, tokenizer, prompts, device, args.n_runs, args.max_new_tokens)
    results["base"] = base_stats
    print(f"  Base: {base_stats}")

    # --- Defended model (if adapter provided) ---
    if args.adapter_path:
        del model
        gc.collect()
        torch.cuda.empty_cache()

        print(f"\n[2/2] Loading defended model (adapter: {args.adapter_path})")
        t0 = time.perf_counter()
        base_model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=dtype, device_map=device)
        model = PeftModel.from_pretrained(base_model, args.adapter_path)

        if args.merge_lora:
            print("  Merging LoRA weights...")
            model = model.merge_and_unload()
            results["lora_mode"] = "merged"
        else:
            results["lora_mode"] = "unmerged"

        t_load_def = time.perf_counter() - t0
        results["defended_load_time_s"] = round(t_load_def, 2)

        print(f"  Loaded in {t_load_def:.1f}s, measuring inference...")
        def_stats = measure_inference(model, tokenizer, prompts, device, args.n_runs, args.max_new_tokens)
        results["defended"] = def_stats
        print(f"  Defended: {def_stats}")

        # Compute overhead
        if base_stats["tokens_per_sec"] > 0:
            overhead = (base_stats["tokens_per_sec"] - def_stats["tokens_per_sec"]) / base_stats["tokens_per_sec"] * 100
            results["throughput_overhead_pct"] = round(overhead, 2)
        results["memory_overhead_gb"] = round(def_stats["peak_gpu_gb"] - base_stats["peak_gpu_gb"], 3)

        # --- Merged mode too if we did unmerged ---
        if not args.merge_lora:
            del model, base_model
            gc.collect()
            torch.cuda.empty_cache()

            print(f"\n[2b] Loading defended model (MERGED)")
            t0 = time.perf_counter()
            base_model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=dtype, device_map=device)
            model = PeftModel.from_pretrained(base_model, args.adapter_path)
            model = model.merge_and_unload()
            t_load_merged = time.perf_counter() - t0
            results["merged_load_time_s"] = round(t_load_merged, 2)

            print(f"  Loaded in {t_load_merged:.1f}s, measuring inference...")
            merged_stats = measure_inference(model, tokenizer, prompts, device, args.n_runs, args.max_new_tokens)
            results["defended_merged"] = merged_stats
            print(f"  Merged: {merged_stats}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Summary
    print("\n=== SUMMARY ===")
    print(f"Base:     {results['base']['tokens_per_sec']} tok/s, {results['base']['peak_gpu_gb']} GB")
    if "defended" in results:
        print(f"Defended: {results['defended']['tokens_per_sec']} tok/s, {results['defended']['peak_gpu_gb']} GB")
        print(f"Overhead: {results.get('throughput_overhead_pct', 'N/A')}% throughput, {results.get('memory_overhead_gb', 'N/A')} GB memory")
    if "defended_merged" in results:
        print(f"Merged:   {results['defended_merged']['tokens_per_sec']} tok/s, {results['defended_merged']['peak_gpu_gb']} GB")

if __name__ == "__main__":
    main()
