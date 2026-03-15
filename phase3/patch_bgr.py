#!/usr/bin/env python3
"""Patch existing security_eval.json files with BGR (benign garble rate).

Loads each adapter, generates benign responses, runs is_gibberish check,
and patches the JSON with bgr + garbled_benign fields.
"""
import sys, os, json, gc, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'phase2'))

from transformers import AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm
from evaluate_v2 import (
    MODEL_IDS, load_tokenizer, sync_model_tokenizer, get_load_kwargs,
    load_benign_prompts, smart_format, clean_response, compute_tdr,
)
from llm_judge import is_gibberish

LAYERS = [
    ("0.1935", "defender_v2_cka_20260223_191329"),
    ("0.25",   "defender_v2_cka_20260223_191901"),
    ("0.375",  "defender_v2_cka_20260223_192443"),
    ("0.5",    "defender_v2_cka_20260223_193031"),
    ("0.625",  "defender_v2_cka_20260223_193606"),
    ("0.75",   "defender_v2_cka_20260223_194145"),
]

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "layer_sweep_outputs")
MODEL_TYPE = "mistral"
MODEL_ID = MODEL_IDS[MODEL_TYPE]
PRECISION = "fp32"
DEVICE = "cuda"


def run_benign_garble_check(model, tokenizer, model_type, device, n_samples=100, batch_size=8):
    """Generate benign responses and check for garble. Returns (bgr, tdr, responses, garbled_list)."""
    model.eval()
    garbled_count = 0
    refusal_count = 0
    garbled_examples = []
    refused_examples = []
    responses = []
    prompts = load_benign_prompts(n_samples)

    from llm_judge import check_refusal_keywords

    for batch_start in tqdm(range(0, len(prompts), batch_size), desc="Benign+garble"):
        batch_prompts = prompts[batch_start:batch_start + batch_size]
        formatted_batch = [smart_format(p, model_type, tokenizer) for p in batch_prompts]

        enc = tokenizer(
            formatted_batch, return_tensors="pt", padding=True,
            truncation=True, max_length=128
        ).to(device)

        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=80,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False
            )

        for i, prompt in enumerate(batch_prompts):
            gen_ids = out[i][enc.input_ids.shape[1]:]
            raw_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            response = clean_response(raw_text)
            responses.append(response)

            # Check refusal first
            is_refusal, reason = check_refusal_keywords(response)
            if is_refusal:
                refusal_count += 1
                refused_examples.append({
                    "prompt": prompt, "response": response[:300], "reason": reason,
                })
                continue

            # Check garble
            is_garble, garble_conf, garble_reason = is_gibberish(response, gib_detector=None)
            if is_garble:
                garbled_count += 1
                garbled_examples.append({
                    "prompt": prompt, "response": response[:300], "reason": garble_reason,
                })

    bgr = garbled_count / len(prompts)
    brr = refusal_count / len(prompts)
    tdr = compute_tdr(responses)

    print(f"  BRR={brr*100:.1f}%, BGR={bgr*100:.1f}%, TDR={tdr:.4f}")
    return brr, bgr, tdr, responses, refused_examples, garbled_examples


def main():
    print(f"Loading base model: {MODEL_ID}")
    tokenizer = load_tokenizer(MODEL_ID, MODEL_TYPE)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **get_load_kwargs(PRECISION))
    sync_model_tokenizer(base_model, tokenizer)

    for layer_pct, adapter_name in LAYERS:
        adapter_dir = os.path.join(BASE_DIR, f"mistral_layer{layer_pct}")
        adapter_path = os.path.join(adapter_dir, adapter_name)
        json_path = os.path.join(adapter_dir, "security_eval.json")

        if not os.path.exists(adapter_path):
            print(f"\n[!] Skipping layer {layer_pct} - adapter not found: {adapter_path}")
            continue

        print(f"\n{'='*60}")
        print(f"Layer {layer_pct}: {adapter_path}")
        print(f"{'='*60}")

        # Load adapter on top of base model
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model.eval()

        brr, bgr, tdr, responses, refused_examples, garbled_examples = run_benign_garble_check(
            model, tokenizer, MODEL_TYPE, DEVICE
        )

        # Print garbled examples
        if garbled_examples:
            print(f"\n  Garbled responses ({len(garbled_examples)}):")
            for ex in garbled_examples[:5]:
                print(f"    Q: {ex['prompt']}")
                print(f"    A: {ex['response'][:150]}")
                print(f"    Reason: {ex['reason']}")
                print()

        # Patch security_eval.json
        if os.path.exists(json_path):
            with open(json_path) as f:
                data = json.load(f)

            data["defended"]["bgr"] = bgr
            data["defended"]["tdr"] = tdr  # update TDR too
            data["defended"]["brr"] = brr  # update BRR with consistent checker
            data["defended_examples"]["garbled_benign"] = garbled_examples
            data["defended_examples"]["refused_benign"] = refused_examples

            with open(json_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            print(f"  [+] Patched {json_path}")
        else:
            print(f"  [!] No security_eval.json to patch")

        # Unload adapter
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Free base model
    del base_model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print("\nDone!")


if __name__ == "__main__":
    main()
