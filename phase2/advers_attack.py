import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import csv

# --- CONFIGURATION ---
INPUT_SUFFIXES = "outputs/advbench_suffixes_all_models.csv"  # The file with the suffixes (from Phase 2A)
OUTPUT_RESPONSES = "outputs/phase2_transfer_results.csv"  # The new file with the ACTUAL text responses
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# List of models to test.
# Make sure to include both your Source models (to verify jailbreak) and Target models (to test transfer).
TARGET_MODELS = [
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


def load_model(model_id):
    print(f"[*] Loading Target Model: {model_id}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        # Ensure pad_token is set (crucial for some models like Llama-2)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer
    except Exception as e:
        print(f"[-] Error loading {model_id}: {e}")
        return None, None


if __name__ == "__main__":
    # 1. Load Suffixes
    if not os.path.exists(INPUT_SUFFIXES):
        print(f"[-] Input file {INPUT_SUFFIXES} not found! Run the attack script first.")
        exit()

    # Read CSV with proper handling of escaped characters
    df_suffixes = pd.read_csv(INPUT_SUFFIXES, escapechar='\\', quoting=csv.QUOTE_ALL)
    print(f"[+] Loaded {len(df_suffixes)} adversarial suffixes to test.")

    # 2. Check for existing progress (Resume Capability)
    completed_pairs = set()
    if os.path.exists(OUTPUT_RESPONSES):
        try:
            existing_df = pd.read_csv(OUTPUT_RESPONSES, escapechar='\\', quoting=csv.QUOTE_ALL)
            # We track progress by (Target Model + Prompt) to avoid duplicates
            for _, row in existing_df.iterrows():
                # Signature: Which target model evaluated which prompt?
                sig = (row['target_model'], row['prompt'])
                completed_pairs.add(sig)
            print(f"[+] Found {len(existing_df)} existing results. Resuming...")
        except Exception:
            print("[!] Output file exists but is empty or unreadable. Starting fresh.")

    # 3. Loop through each TARGET model (The model checking the key)
    for target_name, target_id in TARGET_MODELS:

        # Load the model once
        model, tokenizer = load_model(target_id)
        if not model: continue

        print(f"\n--- Testing on {target_name} ---")

        for idx, row in df_suffixes.iterrows():
            source_model = row['model']  # <--- Defined here as 'source_model'
            prompt = row['prompt']
            suffix = row['suffix']

            # Skip if already done
            if (target_name, prompt) in completed_pairs:
                print(f"[Skipping] {target_name} -> {prompt[:30]}... (Already done)")
                continue

            # Construct input
            full_input = f"{prompt} {suffix}"

            try:
                # Tokenize
                inputs = tokenizer(full_input, return_tensors="pt").to(DEVICE)

                # Generate Response
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )

                # Decode
                full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                response_only = full_text.replace(full_input, "").strip()

                # --- FIX IS HERE: Use 'source_model', not 'source_name' ---
                print(f"[Generated] {source_model} -> {target_name}: {response_only[:50]}...")

                # Prepare Result Row
                result_row = {
                    "source_model": source_model,  # Correct variable
                    "target_model": target_name,
                    "prompt": prompt,
                    "suffix": suffix,
                    "full_response": response_only,
                    "transfer_type": "Self-Attack" if source_model == target_name else "Transfer"
                }

                # Save immediately with proper quoting
                df_row = pd.DataFrame([result_row])
                header_mode = not os.path.exists(OUTPUT_RESPONSES)
                df_row.to_csv(OUTPUT_RESPONSES, mode='a', header=header_mode, index=False,
                            quoting=csv.QUOTE_ALL, escapechar='\\')

            except Exception as e:
                print(f"[-] Generation failed for {target_name} on prompt {idx}: {e}")

        # Free memory before loading the next target model
        del model, tokenizer
        torch.cuda.empty_cache()

    print(f"\n[+] All Done. Full responses saved to {OUTPUT_RESPONSES}")