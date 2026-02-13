"""Pre-download all models to HuggingFace cache so offline mode works."""

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

CAUSAL_MODELS = {
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
    "judge": "meta-llama/Llama-Guard-3-8B",
}

CLASSIFICATION_MODELS = {
    "gibberish": "madhurjindal/autonlp-Gibberish-Detector-492513457",
}

for name, model_id in CAUSAL_MODELS.items():
    print(f"\n{'='*60}")
    print(f"Downloading: {name} ({model_id})")
    print(f"{'='*60}")
    try:
        AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
        print(f"[+] {name} cached successfully")
    except Exception as e:
        print(f"[-] {name} failed: {e}")

for name, model_id in CLASSIFICATION_MODELS.items():
    print(f"\n{'='*60}")
    print(f"Downloading: {name} ({model_id})")
    print(f"{'='*60}")
    try:
        AutoTokenizer.from_pretrained(model_id)
        AutoModelForSequenceClassification.from_pretrained(model_id)
        print(f"[+] {name} cached successfully")
    except Exception as e:
        print(f"[-] {name} failed: {e}")

# Pre-cache datasets
print(f"\n{'='*60}")
print("Downloading: WikiText dataset")
print(f"{'='*60}")
try:
    from datasets import load_dataset
    load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    print("[+] WikiText cached successfully")
except Exception as e:
    print(f"[-] WikiText failed: {e}")

print("\nDone. You can now run with TRANSFORMERS_OFFLINE=1.")
