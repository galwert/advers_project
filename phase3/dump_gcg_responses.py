#!/usr/bin/env python3
"""Generate and display ALL GCG attack responses with judge verdicts.
Shows both succeeded and refused responses so we can manually verify."""

import sys, os, json, argparse
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'phase2'))
from llm_judge import classify_response as llm_judge_classify, load_judge, setup_gibberish_detector

# Import helpers from evaluate_v2
sys.path.insert(0, os.path.dirname(__file__))
from evaluate_v2 import (
    load_tokenizer, get_load_kwargs, sync_model_tokenizer,
    smart_format, clean_response, MODEL_INDEX_MAP, INDEX_TO_MODEL
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--defender", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--gcg_data_path", type=str, default="../outputs/advbench_suffixes_all_models_fixed.csv")
    parser.add_argument("--anchor", type=str, default="qwen")
    parser.add_argument("--precision", type=str, default="fp16")
    parser.add_argument("--n_eval", type=int, default=100)
    parser.add_argument("--baseline", action="store_true", help="Run on baseline (no adapter)")
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    # Model IDs
    MODEL_IDS = {
        "llama2-13b": "NousResearch/Llama-2-13b-chat-hf",
        "qwen-14b": "Qwen/Qwen1.5-14B-Chat",
        "llama2": "NousResearch/Llama-2-7b-chat-hf",
        "qwen": "Qwen/Qwen1.5-7B-Chat",
    }
    MODEL_TYPES = {
        "llama2-13b": "llama2", "qwen-14b": "qwen",
        "llama2": "llama2", "qwen": "qwen",
    }

    model_id = MODEL_IDS[args.defender]
    model_type = MODEL_TYPES[args.defender]
    defender_idx = MODEL_INDEX_MAP.get(model_type, 0)
    anchor_idx = MODEL_INDEX_MAP.get(args.anchor, 10)

    device = "cuda"

    # Load GCG data
    gcg_data = pd.read_csv(args.gcg_data_path)
    self_attacks = gcg_data[gcg_data['model_index'] == defender_idx]
    anchor_attacks = gcg_data[gcg_data['model_index'] == anchor_idx]
    other_attacks = gcg_data[
        (gcg_data['model_index'] != defender_idx) &
        (gcg_data['model_index'] != anchor_idx)
    ]

    print(f"Attack counts: self={len(self_attacks)}, anchor={len(anchor_attacks)}, other={len(other_attacks)}")

    # Load model
    print(f"\nLoading model: {model_id}")
    tokenizer = load_tokenizer(model_id, model_type)
    model = AutoModelForCausalLM.from_pretrained(model_id, **get_load_kwargs(args.precision))
    sync_model_tokenizer(model, tokenizer)

    if args.adapter_path and not args.baseline:
        print(f"Loading adapter: {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path)

    model.eval()

    # Load judge
    judge_model_id = "allenai/wildguard"
    print(f"Loading judge: {judge_model_id}")
    judge_model, judge_tokenizer = load_judge(judge_model_id, quantize=False)
    gib_detector = setup_gibberish_detector(device=device)

    all_results = []
    batch_size = 8  # Match evaluate_v2.py batch size

    for attack_type, attacks_df in [("self", self_attacks), ("anchor", anchor_attacks), ("other", other_attacks)]:
        samples = attacks_df.head(args.n_eval)
        print(f"\n{'='*80}")
        print(f"ATTACK TYPE: {attack_type} ({len(samples)} prompts)")
        print(f"{'='*80}")

        # Pre-filter valid rows (same as evaluate_v2.py)
        valid_rows = []
        for _, row in samples.iterrows():
            prompt = row.get('prompt', '')
            suffix = row.get('suffix', '')
            if pd.isna(prompt) or pd.isna(suffix) or not suffix:
                continue
            valid_rows.append(row)

        # Process in batches (same as evaluate_v2.py)
        for batch_start in range(0, len(valid_rows), batch_size):
            batch_rows = valid_rows[batch_start:batch_start + batch_size]

            prompts_batch = [r.get('prompt', '') for r in batch_rows]
            suffixes_batch = [r.get('suffix', '') for r in batch_rows]
            full_prompts = [f"{p} {s}" for p, s in zip(prompts_batch, suffixes_batch)]
            formatted_batch = [smart_format(fp, model_type, tokenizer) for fp in full_prompts]

            enc = tokenizer(
                formatted_batch, return_tensors="pt", padding=True,
                truncation=True, max_length=512
            ).to(device)

            with torch.no_grad():
                out = model.generate(
                    **enc, max_new_tokens=150,
                    pad_token_id=tokenizer.pad_token_id,
                    do_sample=False
                )

            for i, (prompt, suffix) in enumerate(zip(prompts_batch, suffixes_batch)):
                response = clean_response(tokenizer.decode(out[i][enc.input_ids.shape[1]:], skip_special_tokens=True))

                is_attack_success, reason, metadata = llm_judge_classify(
                    response, prompt, judge_model, judge_tokenizer, judge_model_id, gib_detector, suffix=suffix
                )

                verdict = "SUCCESS" if is_attack_success else "REFUSED"
                compliance = metadata.get('compliance_score', 'N/A')
                coherence = metadata.get('coherence_score', 'N/A')
                idx = batch_start + i

                print(f"\n[{attack_type.upper()}-{idx+1}] [{verdict}]")
                print(f"  Prompt: {prompt}")
                print(f"  Suffix: {suffix[:80]}...")
                print(f"  Response: {response}")
                print(f"  Reason: {reason}")
                print(f"  Compliance: {compliance}, Coherence: {coherence}")

                all_results.append({
                    "attack_type": attack_type,
                    "index": idx,
                    "prompt": prompt,
                    "suffix": suffix,
                    "response": response,
                    "verdict": verdict,
                    "reason": reason,
                    "compliance": str(compliance),
                    "coherence": str(coherence),
                })

    # Save all results
    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved {len(all_results)} results to {args.output_json}")

    # Summary
    for at in ["self", "anchor", "other"]:
        items = [r for r in all_results if r['attack_type'] == at]
        succ = sum(1 for r in items if r['verdict'] == 'SUCCESS')
        print(f"{at}: {succ}/{len(items)} = {succ/len(items)*100:.1f}% ASR")

if __name__ == "__main__":
    main()
