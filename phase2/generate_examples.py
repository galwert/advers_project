import torch
import nanogcg
from nanogcg import GCGConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import io
import requests
import os
import argparse
import sys
import fcntl
import csv
import sys
# Manually block flash_attn from being imported
sys.modules["flash_attn"] = None
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ADVBENCH_URL = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"

# Reduced model list
MODELS = [
    # ("Llama2-7b", "meta-llama/Llama-2-7b-chat-hf"),
    # ("Llama3-8b", "meta-llama/Meta-Llama-3-8B-Instruct"),
    # ("Vicuna-7b", "lmsys/vicuna-7b-v1.5"),
    # ("Mistral-7b", "mistralai/Mistral-7B-Instruct-v0.2"),
    # ("Zephyr-7b", "HuggingFaceH4/zephyr-7b-beta"),
    # ("Hermes-2", "NousResearch/Nous-Hermes-2-Mistral-7B-DPO"),
    ("Starling-7b", "berkeley-nest/Starling-LM-7B-alpha"),
    ("OpenChat-3.5", "openchat/openchat_3.5"),
    # ("Gemma-7b", "google/gemma-7b-it"),
    # ("Phi-2", "microsoft/phi-2"),
    # ("Qwen1.5-7b", "Qwen/Qwen1.5-7B-Chat"),
    # ("Yi-6b", "01-ai/Yi-6B-Chat"),
    # ("Baichuan2-7b", "baichuan-inc/Baichuan2-7B-Chat"),
    # ("DeepSeek-7b", "deepseek-ai/deepseek-llm-7b-chat"),
    # ("InternLM2-7b", "internlm/internlm2-chat-7b"),
    # ("Falcon-7b", "tiiuae/falcon-7b-instruct"),
    # ("Solar-10.7b", "upstage/SOLAR-10.7B-Instruct-v1.0"),
    # ("Orca-2-7b", "microsoft/Orca-2-7b"),
    # ("NeuralChat-7b", "Intel/neural-chat-7b-v3-1"),
    # ("StableZephyr-3b", "stabilityai/stablelm-zephyr-3b"),
]


def load_advbench(n_samples=10):
    print(f"[*] Downloading AdvBench from {ADVBENCH_URL}...")
    try:
        response = requests.get(ADVBENCH_URL)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text))
        df = df.rename(columns={"goal": "prompt"})
        subset = df.head(n_samples).to_dict(orient="records")
        print(f"[+] Loaded {len(subset)} behaviors from AdvBench.")
        return subset
    except Exception as e:
        print(f"[-] Failed to download AdvBench: {e}")
        return []


def load_model(model_id):
    print(f"\n[+] Loading Model: {model_id}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager"
        )
        return model, tokenizer
    except Exception as e:
        print(f"[-] Error loading {model_id}: {e}")
        return None, None


def generate_attack(model, tokenizer, prompt, target):
    config = GCGConfig(
        num_steps=500,  # Kept at 50 to see the "slope"
        search_width=64,  # Kept low to avoid saturation
        topk=64,
        seed=42,
        verbosity="WARNING"
    )

    # We return the whole result object now to access history
    result = nanogcg.run(model, tokenizer, prompt, target, config)
    return result


def save_results_to_csv(results, output_path, source_name):
    """Save results to CSV with file locking."""
    results_df = pd.DataFrame(results)

    with open(output_path, 'a') as f:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            file_is_empty = f.tell() == 0

            # Ensure columns are ordered nicely if creating new file
            if file_is_empty:
                cols = list(results_df.columns)
                # Move 'loss' and steps to end for readability
                base_cols = ['model_index', 'model', 'example_index', 'prompt', 'target', 'suffix']
                other_cols = [c for c in cols if c not in base_cols]
                results_df = results_df[base_cols + other_cols]

            results_df.to_csv(f, mode='a', header=file_is_empty, index=False,
                              quoting=csv.QUOTE_ALL, escapechar='\\')

            print(f"\n{'=' * 60}")
            print(f"[+] Done! Processed {len(results)} examples for {source_name}")
            print(f"[+] Results appended to: {output_path}")
            print(f"{'=' * 60}\n")

        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def process_model(model_index, all_behaviors, examples_per_model, output_path):
    dataset_size = len(all_behaviors)
    source_name, source_id = MODELS[model_index]

    model_behaviors = []
    for i in range(examples_per_model):
        behavior_idx = i % dataset_size
        model_behaviors.append(all_behaviors[behavior_idx])

    print(f"\n{'=' * 60}")
    print(f"Model {model_index + 1}/{len(MODELS)}")
    print(f"Model Name: {source_name}")
    print(f"Examples: {len(model_behaviors)}")
    print(f"{'=' * 60}\n")

    model, tokenizer = load_model(source_id)
    if not model:
        print(f"[-] Failed to load model {source_name}, skipping...")
        return False

    results = []

    for i, b in enumerate(model_behaviors):
        print(f"\n--- [{model_index}] {source_name} | Example {i + 1}/{len(model_behaviors)} ---")
        print(f"Goal: {b['prompt']}")

        try:
            # Run attack
            result = generate_attack(model, tokenizer, b['prompt'], b['target'])

            best_suffix = result.best_string
            final_loss = result.best_loss

            # --- EXTRACT LOSS HISTORY ---
            # nanogcg typically stores history in result.losses (list of float)
            # We want steps 10, 20, 30, 40, 50


            print(f"[+] Success! Final Loss: {final_loss:.4f}")

            # Build row
            row = {
                "model_index": model_index+6,
                "model": source_name,
                "example_index": i,
                "prompt": b['prompt'],
                "target": b['target'],
                "suffix": best_suffix,
                "final_loss": final_loss
            }
            # Merge history into row
            results.append(row)

        except Exception as e:
            print(f"[-] Attack failed: {e}")
            # Add failed row with Nones
            row = {
                "model_index": model_index,
                "model": source_name,
                "example_index": i,
                "prompt": b['prompt'],
                "target": b['target'],
                "suffix": None,
                "final_loss": None
            }
            # Add empty history keys
            for step in [10, 20, 30, 40, 50]:
                row[f"loss_step_{step}"] = None
            results.append(row)

    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    save_results_to_csv(results, output_path, source_name)
    return True


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate adversarial examples via GCG.")
    parser.add_argument("--model-indices", type=str, default=None)
    parser.add_argument("--examples-per-model", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default="../outputs")
    parser.add_argument("--list-models", action="store_true")

    args = parser.parse_args()

    if args.list_models:
        print("\n=== Available Models ===")
        for idx, (name, model_id) in enumerate(MODELS):
            print(f"  [{idx}] {name:20s} ({model_id})")
        sys.exit(0)

    if args.model_indices is not None:
        try:
            model_indices = [int(idx.strip()) for idx in args.model_indices.split(',')]
        except ValueError:
            print("[-] Invalid indices format.")
            sys.exit(1)
    else:
        model_indices = list(range(len(MODELS)))

    os.makedirs(args.output_dir, exist_ok=True)
    all_behaviors = load_advbench(n_samples=10000)

    if not all_behaviors:
        sys.exit(1)

    # Modified filename to indicate history logging
    output_filename = "advbench_suffixes_all_models.csv"
    output_path = os.path.join(args.output_dir, output_filename)

    print(f"\n{'=' * 60}")
    print(f"Starting Run: 500 Steps")
    print(f"Output: {output_path}")
    print(f"{'=' * 60}\n")

    successful = 0
    failed = 0
    for model_index in model_indices:
        if process_model(model_index, all_behaviors, args.examples_per_model, output_path):
            successful += 1
        else:
            failed += 1

    # Final summary
    print(f"\n{'='*60}")
    print(f"=== FINAL SUMMARY ===")
    print(f"Models processed successfully: {successful}/{len(model_indices)}")
    if failed > 0:
        print(f"Models failed: {failed}")
    print(f"Total examples generated: {successful * args.examples_per_model}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}\n")

