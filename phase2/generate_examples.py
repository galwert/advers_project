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
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Raw URL for the standard AdvBench "Harmful Behaviors" dataset
ADVBENCH_URL = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"

# Reduced model list for initial testing
MODELS = [
    # --- The Llama Family (Baseline) ---
    ("Llama2-7b", "meta-llama/Llama-2-7b-chat-hf"),
    ("Llama3-8b", "meta-llama/Meta-Llama-3-8B-Instruct"),
    ("Vicuna-7b", "lmsys/vicuna-7b-v1.5"),

    # --- The Mistral Family (Strong/Sparse) ---
    ("Mistral-7b", "mistralai/Mistral-7B-Instruct-v0.2"),
    ("Zephyr-7b", "HuggingFaceH4/zephyr-7b-beta"),
    ("Hermes-2", "NousResearch/Nous-Hermes-2-Mistral-7B-DPO"),
    ("Starling-7b", "berkeley-nest/Starling-LM-7B-alpha"),
    ("OpenChat-3.5", "openchat/openchat_3.5"),

    # --- Google & Microsoft (Different Architectures) ---
    ("Gemma-7b", "google/gemma-7b-it"),
    ("Phi-2", "microsoft/phi-2"),  # Small but distinct geometry

    # --- The "Eastern" Models (Different Training Data) ---
    ("Qwen1.5-7b", "Qwen/Qwen1.5-7B-Chat"),
    ("Yi-6b", "01-ai/Yi-6B-Chat"),
    ("Baichuan2-7b", "baichuan-inc/Baichuan2-7B-Chat"),
    ("DeepSeek-7b", "deepseek-ai/deepseek-llm-7b-chat"),
    ("InternLM2-7b", "internlm/internlm2-chat-7b"),

    # --- Distinct Architectures (For Robustness) ---
    ("Falcon-7b", "tiiuae/falcon-7b-instruct"),
    ("Solar-10.7b", "upstage/SOLAR-10.7B-Instruct-v1.0"), # Slightly larger, good test
    ("Orca-2-7b", "microsoft/Orca-2-7b"),
    ("NeuralChat-7b", "Intel/neural-chat-7b-v3-1"),
    ("StableZephyr-3b", "stabilityai/stablelm-zephyr-3b"),
]


def load_advbench(n_samples=10):
    """
    Downloads AdvBench and returns the first n_samples as a list of dicts.
    Returns: [{'prompt': '...', 'target': '...'}, ...]
    """
    print(f"[*] Downloading AdvBench from {ADVBENCH_URL}...")
    try:
        response = requests.get(ADVBENCH_URL)
        response.raise_for_status()

        # The file is a CSV with header: goal, target
        df = pd.read_csv(io.StringIO(response.text))

        # Rename 'goal' to 'prompt' to match our logic
        df = df.rename(columns={"goal": "prompt"})

        # Take the first N samples (standard practice for "dev" runs)
        subset = df.head(n_samples).to_dict(orient="records")
        print(f"[+] Loaded {len(subset)} behaviors from AdvBench.")
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
            trust_remote_code=True
        )
        return model, tokenizer
    except Exception as e:
        print(f"[-] Error loading {model_id}: {e}")
        return None, None


def generate_attack(model, tokenizer, prompt, target):
    config = GCGConfig(
        num_steps=500,  # Increase to 500 for final paper run
        search_width=64,
        topk=64,
        seed=42,
        verbosity="WARNING"
    )

    result = nanogcg.run(model, tokenizer, prompt, target, config)
    return result.best_string, result.best_loss


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Parse CLI arguments
    parser = argparse.ArgumentParser(
        description="Generate adversarial examples via GCG for a specific model by index."
    )
    parser.add_argument(
        "--model-index",
        type=int,
        required=True,
        help=f"Index of the model to run (0-{len(MODELS)-1}). Use --list-models to see available models."
    )
    parser.add_argument(
        "--examples-per-model",
        type=int,
        default=100,
        help="Number of examples to process for this model (default: 100)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save results (default: outputs)"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all available models with their indices and exit"
    )

    args = parser.parse_args()

    # Handle --list-models
    if args.list_models:
        print("\n=== Available Models ===")
        for idx, (name, model_id) in enumerate(MODELS):
            print(f"  [{idx}] {name:20s} ({model_id})")
        print(f"\nTotal: {len(MODELS)} models")
        print(f"\nUsage: python generate_examples.py --model-index <INDEX>")
        sys.exit(0)

    # Validate model index
    if args.model_index < 0 or args.model_index >= len(MODELS):
        print(f"[-] Error: model-index must be between 0 and {len(MODELS)-1}")
        print(f"    Use --list-models to see available models")
        sys.exit(1)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load all available samples from AdvBench
    # We'll cycle through them if we need more than available
    print(f"[*] Loading AdvBench dataset...")
    all_behaviors = load_advbench(n_samples=10000)  # Load all available samples

    if not all_behaviors:
        print("[-] Failed to load dataset")
        sys.exit(1)

    dataset_size = len(all_behaviors)
    print(f"[+] Dataset contains {dataset_size} total samples")

    # Calculate which examples this model should process
    # Each model processes examples_per_model examples, cycling through the dataset if needed
    model_behaviors = []
    for i in range(args.examples_per_model):
        # Use modulo to cycle through dataset if we need more samples than available
        behavior_idx = (args.model_index * args.examples_per_model + i) % dataset_size
        model_behaviors.append(all_behaviors[behavior_idx])

    print(f"[+] Model will process {len(model_behaviors)} examples")

    # Get the model to process
    source_name, source_id = MODELS[args.model_index]

    print(f"\n{'='*60}")
    print(f"Model Index: {args.model_index}")
    print(f"Model Name: {source_name}")
    print(f"Model ID: {source_id}")
    print(f"Examples to process: {len(model_behaviors)}")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*60}\n")

    # Load the model
    model, tokenizer = load_model(source_id)
    if not model:
        print(f"[-] Failed to load model {source_name}")
        sys.exit(1)

    results = []

    # Process examples for this model
    for i, b in enumerate(model_behaviors):
        global_idx = (args.model_index * args.examples_per_model + i) % dataset_size
        print(f"\n--- [{args.model_index}] {source_name} | Example {i+1}/{len(model_behaviors)} (Dataset idx: {global_idx}) ---")
        print(f"Goal: {b['prompt']}")

        try:
            suffix, loss = generate_attack(model, tokenizer, b['prompt'], b['target'])
            print(f"[+] Success! Suffix: {suffix} (Loss: {loss:.4f})")

            results.append({
                "model_index": args.model_index,
                "model": source_name,
                "example_index": global_idx,
                "prompt": b['prompt'],
                "target": b['target'],
                "suffix": suffix,
                "loss": loss
            })
        except Exception as e:
            print(f"[-] Attack failed: {e}")
            results.append({
                "model_index": args.model_index,
                "model": source_name,
                "example_index": global_idx,
                "prompt": b['prompt'],
                "target": b['target'],
                "suffix": None,
                "loss": None
            })

    # Clean up
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Save results to shared CSV file with file locking to prevent conflicts
    output_filename = "advbench_suffixes_all_models.csv"
    output_path = os.path.join(args.output_dir, output_filename)

    # Use file locking to safely append from multiple processes
    results_df = pd.DataFrame(results)

    # Acquire exclusive lock on the file
    with open(output_path, 'a') as f:
        try:
            # Lock the file (blocks until lock is available)
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)

            # Check if file is empty (needs header)
            file_is_empty = f.tell() == 0

            # Write results with proper CSV quoting to handle special characters
            # Use QUOTE_ALL to ensure all fields are quoted, preventing parsing issues
            results_df.to_csv(f, mode='a', header=file_is_empty, index=False,
                            quoting=csv.QUOTE_ALL, escapechar='\\')

            print(f"\n{'='*60}")
            print(f"[+] Done! Processed {len(results)} examples for {source_name}")
            print(f"[+] Results appended to: {output_path}")
            print(f"{'='*60}\n")

        finally:
            # Release the lock
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
