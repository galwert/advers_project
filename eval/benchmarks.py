#!/usr/bin/env python3
"""
Benchmark Evaluation (Phase 3)

Evaluates baseline vs defended (LoRA-adapted) models on standard safety and
capability benchmarks used by competing defenses (Circuit Breakers, RepBend,
ReFAT, LAT, etc.). Reports deltas to quantify the defense's side effects.

Benchmarks:
  - OR-Bench Hard (1,320 prompts) - over-refusal rate on safe-but-edgy prompts
  - XSTest Safe (250 prompts) - exaggerated safety refusal rate
  - MMLU (14,042 questions) - knowledge accuracy (5-shot, via lm-eval-harness)
  - MT-Bench (80 multi-turn questions) - conversation quality (1-10, local judge)

Pipeline:
  1. Loads adapter from CSV (best pair) or direct path
  2. Runs baseline model on all benchmarks
  3. Runs defended model (base + LoRA adapter) on all benchmarks
  4. Reports per-benchmark deltas (lower OR-Bench/XSTest = better)

Usage:
    python benchmark_eval.py --defender vicuna --csv_path all_pairs_results.csv
    python benchmark_eval.py --defender vicuna --csv_path all_pairs_results.csv --skip_mmlu --skip_mtbench
    python benchmark_eval.py --defender vicuna --baseline_only
"""

import torch
import os
import sys
import gc
import re
import csv
import argparse
import json
import tempfile
import subprocess
import urllib.request
import pandas as pd
from tqdm.auto import tqdm
from functools import partialmethod
if os.environ.get("TQDM_DISABLE"):
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset

import warnings
warnings.filterwarnings("ignore")

sys.modules["flash_attn"] = None

# Import refusal checker from phase2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'phase2'))
from llm_judge import check_refusal_keywords, is_gibberish, setup_gibberish_detector


# ==========================================
# MODEL MAPPINGS
# ==========================================
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
    # 13B+ models
    "llama2-13b": "NousResearch/Llama-2-13b-chat-hf",
    "qwen-14b": "Qwen/Qwen1.5-14B-Chat",
    "phi3": "microsoft/Phi-3-medium-4k-instruct",
    "yi9b": "01-ai/Yi-1.5-9B-Chat",
    "mistral_nemo": "mistralai/Mistral-Nemo-Instruct-2407",
}

# Map full HF IDs back to model_type for prompt formatting
_HF_ID_TO_TYPE = {v.lower(): k for k, v in MODEL_IDS.items()}
_HF_ID_TO_TYPE["nousresearch/llama-2-13b-chat-hf"] = "llama2"
_HF_ID_TO_TYPE["meta-llama/llama-2-13b-chat-hf"] = "llama2"
_HF_ID_TO_TYPE["qwen/qwen1.5-14b-chat"] = "qwen"

SLOW_TOKENIZER_MODELS = {"llama2", "llama2-13b", "vicuna", "orca", "mistral", "zephyr", "starling", "yi", "phi2"}


def load_tokenizer(model_id, model_type=None):
    """Load tokenizer with correct settings per model."""
    # Use LlamaTokenizer for Llama-2 (AutoTokenizer in transformers 5.x
    # loads a generic backend that doesn't decode SentencePiece ▁ properly)
    if model_type and ("llama2" in model_type.lower()):
        from transformers import LlamaTokenizer
        tokenizer = LlamaTokenizer.from_pretrained(model_id)
    else:
        use_fast = model_type not in SLOW_TOKENIZER_MODELS if model_type else True
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=use_fast)
    if not tokenizer.pad_token:
        if tokenizer.unk_token:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def sync_model_tokenizer(model, tokenizer):
    """Sync model config and generation_config with tokenizer token IDs."""
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    if hasattr(model, 'generation_config'):
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id


MT_BENCH_URL = "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl"


# ==========================================
# UTILITIES
# ==========================================
def get_bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )


def get_load_kwargs(precision="4bit"):
    """Build kwargs for AutoModelForCausalLM.from_pretrained().

    precision: "4bit" (default, NF4 quantization), "fp16", or "fp32".

    For fp16/fp32, uses device_map={"": 0} to force single-GPU placement.
    device_map="auto" can silently offload layers to CPU, which breaks
    PeftModel adapter loading/inference.
    """
    kwargs = dict(trust_remote_code=True, attn_implementation="eager")
    if precision == "fp16":
        kwargs["torch_dtype"] = torch.float16
        kwargs["device_map"] = {"": 0}
    elif precision == "fp32":
        kwargs["torch_dtype"] = torch.float32
        kwargs["device_map"] = {"": 0}
    else:  # 4bit
        kwargs["quantization_config"] = get_bnb_config()
        kwargs["device_map"] = "auto"
    return kwargs


def format_prompt(text: str, model_type: str) -> str:
    model_type = model_type.lower()
    if model_type == "llama2":
        return f"[INST] {text} [/INST]"
    elif model_type == "llama3":
        return f"<|start_header_id|>user<|end_header_id|>\n\n{text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    elif model_type == "vicuna":
        return f"USER: {text} ASSISTANT:"
    elif model_type == "mistral":
        return f"<s>[INST] {text} [/INST]"
    elif model_type == "zephyr":
        return f"<|user|>\n{text}</s>\n<|assistant|>\n"
    elif model_type == "qwen":
        return f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
    elif model_type == "starling":
        return f"GPT4 Correct User: {text}<|end_of_turn|>GPT4 Correct Assistant:"
    elif model_type in ("yi", "orca"):
        return f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
    elif model_type == "phi2":
        return f"Instruct: {text}\nOutput:"
    return text


def format_prompt_native(text: str, tokenizer) -> str:
    """Format prompt using the tokenizer's built-in chat template (matches nanogcg)."""
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return None


def smart_format(text: str, model_type: str, tokenizer) -> str:
    """Format prompt using native chat template, falling back to manual format."""
    native = format_prompt_native(text, tokenizer)
    return native if native else format_prompt(text, model_type)


def format_multi_turn(q1: str, r1: str, q2: str, model_type: str, tokenizer=None) -> str:
    """Format a 2-turn conversation for generating the turn-2 response."""
    if tokenizer is not None:
        try:
            return tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": q1},
                    {"role": "assistant", "content": r1},
                    {"role": "user", "content": q2},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    model_type = model_type.lower()
    if model_type == "llama2":
        return f"[INST] {q1} [/INST] {r1} </s><s>[INST] {q2} [/INST]"
    elif model_type == "llama3":
        return (
            f"<|start_header_id|>user<|end_header_id|>\n\n{q1}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n{r1}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n{q2}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    elif model_type == "vicuna":
        return f"USER: {q1} ASSISTANT: {r1}</s>USER: {q2} ASSISTANT:"
    elif model_type == "mistral":
        return f"<s>[INST] {q1} [/INST] {r1}</s>[INST] {q2} [/INST]"
    elif model_type == "zephyr":
        return (
            f"<|user|>\n{q1}</s>\n<|assistant|>\n{r1}</s>\n"
            f"<|user|>\n{q2}</s>\n<|assistant|>\n"
        )
    elif model_type in ("qwen", "yi", "orca"):
        return (
            f"<|im_start|>user\n{q1}<|im_end|>\n"
            f"<|im_start|>assistant\n{r1}<|im_end|>\n"
            f"<|im_start|>user\n{q2}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    elif model_type == "starling":
        return (
            f"GPT4 Correct User: {q1}<|end_of_turn|>"
            f"GPT4 Correct Assistant: {r1}<|end_of_turn|>"
            f"GPT4 Correct User: {q2}<|end_of_turn|>"
            f"GPT4 Correct Assistant:"
        )
    elif model_type == "phi2":
        return f"Instruct: {q1}\nOutput: {r1}\nInstruct: {q2}\nOutput:"
    return f"{q1}\n{r1}\n{q2}\n"


def detect_model_type(model_id: str) -> str:
    """Auto-detect model type from HuggingFace model ID."""
    m = model_id.lower()
    if "llama-2" in m:
        return "llama2"
    elif "llama-3" in m or "llama3" in m:
        return "llama3"
    elif "vicuna" in m:
        return "vicuna"
    elif "mistral" in m:
        return "mistral"
    elif "zephyr" in m:
        return "zephyr"
    elif "qwen" in m:
        return "qwen"
    elif "yi-" in m:
        return "yi"
    elif "starling" in m:
        return "starling"
    elif "phi" in m:
        return "phi2"
    elif "orca" in m:
        return "orca"
    return "llama3"


# ==========================================
# LOAD BENCHMARKS
# ==========================================
def load_orbench_hard(exclude_prompts: set = None) -> tuple:
    """Load OR-Bench Hard-1K from HuggingFace.

    Args:
        exclude_prompts: Set of prompts to exclude (train split).

    Returns:
        (full_prompts, held_out_prompts) where held_out_prompts excludes
        the training set. If exclude_prompts is None, held_out == full.
        Returns ([], []) and prints a warning if the dataset is unreachable
        (offline run, dataset removed) so callers can skip the OR-Bench
        evaluation gracefully instead of crashing.
    """
    print("[*] Loading OR-Bench Hard-1K...")
    try:
        ds = load_dataset("bench-llm/or-bench", "or-bench-hard-1k", split="train")
    except Exception as e:
        print(f"[!] OR-Bench could not be loaded ({type(e).__name__}: {e}); "
              "skipping OR-Bench refusal-rate / BGR evaluation.")
        return [], []
    full_prompts = [row["prompt"] for row in ds]
    print(f"    Loaded {len(full_prompts)} prompts")

    if exclude_prompts:
        held_out = [p for p in full_prompts if p not in exclude_prompts]
        print(f"    After excluding {len(exclude_prompts)} train prompts: {len(held_out)} held-out")
    else:
        held_out = full_prompts

    return full_prompts, held_out


def load_xstest_safe() -> list:
    """Load XSTest safe subset from HuggingFace."""
    print("[*] Loading XSTest (safe subset)...")
    ds = load_dataset("Paul/XSTest", split="train")
    safe_prompts = [row["prompt"] for row in ds if row.get("label") == "safe"]
    print(f"    Loaded {len(safe_prompts)} safe prompts")
    return safe_prompts



def load_mtbench_questions() -> list:
    """Load MT-Bench questions (80 multi-turn). Downloads and caches from FastChat."""
    cache_path = os.path.join(os.path.dirname(__file__), ".mtbench_questions.jsonl")

    if not os.path.exists(cache_path):
        print("[*] Downloading MT-Bench questions from FastChat...")
        urllib.request.urlretrieve(MT_BENCH_URL, cache_path)

    questions = []
    with open(cache_path) as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))

    print(f"[*] Loaded {len(questions)} MT-Bench questions")
    return questions


# ==========================================
# ADAPTER LOOKUP
# ==========================================
def find_adapter_from_csv(csv_path: str, defender: str, anchor: str = None):
    """Find adapter path from all-pairs CSV.

    If anchor is specified, returns that pair's adapter.
    Otherwise picks the pair with lowest defended ASR.
    """
    rows = []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    df = pd.DataFrame(rows)
    print(f"[CSV] Total rows: {len(df)}")
    print(f"[CSV] Statuses: {df['status'].value_counts().to_dict()}")
    print(f"[CSV] Defenders: {sorted(df['defender'].unique())}")
    if anchor:
        matches = df[(df["defender"] == defender) & (df["anchor"] == anchor)]
        print(f"[CSV] Rows for anchor={anchor}, defender={defender}: {len(matches)}")
        if len(matches) > 0:
            print(f"[CSV]   Statuses: {matches['status'].value_counts().to_dict()}")
    # Accept OK rows, or rows with a non-empty adapter_path (completed but status not set)
    df = df[(df["status"] == "OK") | (df["adapter_path"].fillna("").str.len() > 0)]
    ok_for_defender = df[df["defender"] == defender]
    if len(ok_for_defender) > 0:
        print(f"[CSV] OK pairs for defender={defender}: {sorted(ok_for_defender['anchor'].tolist())}")
    else:
        print(f"[CSV] No OK pairs for defender={defender}!")
        all_ok = df[["anchor", "defender"]].values.tolist()
        print(f"[CSV] All OK pairs: {[(a,d) for a,d in all_ok]}")
    df = ok_for_defender

    if len(df) == 0:
        raise ValueError(f"No OK results for defender={defender} in {csv_path}")

    if anchor:
        row = df[df["anchor"] == anchor]
        if len(row) == 0:
            raise ValueError(f"No OK result for anchor={anchor}, defender={defender}")
        row = row.iloc[0]
    else:
        asr_cols = ["defended_asr_self", "defended_asr_anchor", "defended_asr_other"]
        for col in asr_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["avg_asr"] = df[asr_cols].mean(axis=1)
        row = df.loc[df["avg_asr"].idxmin()]
        print(f"[*] Best anchor for {defender}: {row['anchor']} (avg ASR={row['avg_asr']:.4f})")

    adapter_path = row["adapter_path"]
    if not os.path.exists(adapter_path):
        alt = os.path.join(os.path.dirname(__file__), adapter_path)
        if os.path.exists(alt):
            adapter_path = alt
        else:
            raise FileNotFoundError(f"Adapter not found: {adapter_path}")

    return adapter_path, row["anchor"]


# ==========================================
# EVALUATION: REFUSAL RATE (OR-Bench, XSTest)
# ==========================================
_GIB_DETECTOR = None  # Lazily initialized by init_gibberish_detector()


def init_gibberish_detector(device="cuda"):
    """Initialize the shared gibberish detector (same as evaluate_v2.py)."""
    global _GIB_DETECTOR
    if _GIB_DETECTOR is None:
        _GIB_DETECTOR = setup_gibberish_detector(device=device)
    return _GIB_DETECTOR


def _is_garbled(text: str) -> bool:
    """Garble detection using the same is_gibberish() pipeline as evaluate_v2.py."""
    result, confidence, reason = is_gibberish(text, _GIB_DETECTOR)
    return result


def evaluate_refusal_rate(
    model, tokenizer, prompts: list, model_type: str,
    device: str, batch_size: int = 8, max_new_tokens: int = 100,
    label: str = "eval", verbose: bool = False,
) -> dict:
    """Generate responses and compute refusal rate + BGR."""
    model.eval()
    refusals = 0
    garbled = 0
    total = 0
    refused_examples = []
    all_responses = []
    verbose_samples = []

    for batch_start in tqdm(range(0, len(prompts), batch_size), desc=label):
        batch_prompts = prompts[batch_start:batch_start + batch_size]
        formatted = [smart_format(p, model_type, tokenizer) for p in batch_prompts]

        enc = tokenizer(
            formatted, return_tensors="pt", padding=True,
            truncation=True, max_length=256
        ).to(device)

        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False
            )

        for i, prompt in enumerate(batch_prompts):
            response = tokenizer.decode(
                out[i][enc.input_ids.shape[1]:], skip_special_tokens=True
            ).strip()

            is_refusal, reason = check_refusal_keywords(response)
            is_garble = _is_garbled(response)
            total += 1

            # Garbled takes priority: if both garbled AND refused, count as garbled only
            if is_garble:
                garbled += 1
            elif is_refusal:
                refusals += 1
                if len(refused_examples) < 20:
                    refused_examples.append({
                        "prompt": prompt, "response": response[:300], "reason": reason
                    })

            all_responses.append({
                "prompt": prompt, "response": response,
                "is_refusal": is_refusal, "refusal_reason": reason if is_refusal else None,
                "is_garbled": is_garble,
            })

            if verbose and len(verbose_samples) < 5:
                verdict = "GARBLED" if is_garble else (f"REFUSED ({reason})" if is_refusal else "OK")
                verbose_samples.append((prompt, response, verdict))

    if verbose and verbose_samples:
        print(f"\n  [{label}] Sample generations:")
        for idx, (p, r, v) in enumerate(verbose_samples, 1):
            print(f"  [{idx}] Prompt:   {p[:80]}")
            print(f"      Response: {r[:150]}")
            print(f"      Verdict:  {v}")

    refusal_rate = refusals / total if total > 0 else 0.0
    bgr = garbled / total if total > 0 else 0.0
    print(f"{label} BGR: {bgr*100:.1f}% ({garbled}/{total})")
    return {
        "refusal_rate": refusal_rate,
        "refusals": refusals,
        "bgr": bgr,
        "garbled": garbled,
        "total": total,
        "refused_examples": refused_examples,
        "all_responses": all_responses,
    }


# ==========================================
# EVALUATION: MMLU (via lm-evaluation-harness)
# ==========================================
def evaluate_mmlu_lmeval(
    model_id: str, adapter_path: str = None,
    batch_size: int = 8, n_shot: int = 5, limit: int = 0,
    precision: str = "4bit",
) -> float:
    """Evaluate MMLU accuracy using lm-evaluation-harness (proper implementation).

    Shells out to lm_eval CLI which handles prompt formatting, tokenization,
    and scoring correctly for each model architecture.
    """
    # Build model_args
    # Use fp16 for MMLU — multiple-choice accuracy is precision-insensitive,
    # and avoids load_in_4bit kwarg incompatibilities with newer transformers.
    # Detect full-model checkpoint (e.g., RMU): has config.json but no adapter_config.json.
    is_full_model_ckpt = (
        adapter_path
        and os.path.exists(adapter_path)
        and os.path.exists(os.path.join(adapter_path, "config.json"))
        and not os.path.exists(os.path.join(adapter_path, "adapter_config.json"))
    )
    pretrained_arg = adapter_path if is_full_model_ckpt else model_id
    model_args = [
        f"pretrained={pretrained_arg}",
        "dtype=float16",
        "trust_remote_code=True",
    ]
    if adapter_path and os.path.exists(adapter_path) and not is_full_model_ckpt:
        model_args.append(f"peft={adapter_path}")

    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            sys.executable, "-m", "lm_eval",
            "--model", "hf",
            "--model_args", ",".join(model_args),
            "--tasks", "mmlu",
            "--num_fewshot", str(n_shot),
            "--batch_size", "auto:4",
            "--output_path", tmpdir,
        ]
        if limit > 0:
            cmd.extend(["--limit", str(limit)])

        tag = "defended" if adapter_path else "baseline"
        print(f"\n--- MMLU 5-shot via lm-eval-harness ({tag}) ---")
        print(f"[*] Running: {' '.join(cmd)}")

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=7200,
        )

        if result.returncode != 0:
            print(f"[!] lm-eval failed (rc={result.returncode})")
            print(f"    stderr: {result.stderr[-500:]}")
            return None

        # Parse results JSON — lm-eval writes to {output_path}/{model_name}/results_*.json
        results_json = None
        for root, dirs, files in os.walk(tmpdir):
            for f in files:
                if f.startswith("results") and f.endswith(".json"):
                    results_json = os.path.join(root, f)
                    break
            if results_json:
                break

        if not results_json:
            print("[!] Could not find lm-eval results JSON")
            # Try to parse accuracy from stdout
            for line in result.stdout.split("\n"):
                if "mmlu" in line.lower() and "acc" in line.lower():
                    print(f"    stdout: {line}")
            return None

        with open(results_json, 'r') as f:
            lm_results = json.load(f)

        # Extract accuracy — try group-level first, then individual
        results_dict = lm_results.get("results", {})
        if "mmlu" in results_dict:
            acc = results_dict["mmlu"].get("acc,none", results_dict["mmlu"].get("acc"))
        else:
            # Average across subtasks
            accs = []
            for task_name, task_results in results_dict.items():
                if "mmlu" in task_name:
                    a = task_results.get("acc,none", task_results.get("acc"))
                    if a is not None:
                        accs.append(a)
            acc = sum(accs) / len(accs) if accs else None

        if acc is not None:
            print(f"\nMMLU Accuracy: {acc*100:.1f}%")
        else:
            print("[!] Could not parse MMLU accuracy from results")

        return acc


# ==========================================
# EVALUATION: MT-Bench (generate responses)
# ==========================================
def generate_mtbench_responses(
    model, tokenizer, questions: list, model_type: str,
    device: str, max_new_tokens: int = 512, verbose: bool = False,
) -> list:
    """Generate multi-turn responses for all MT-Bench questions."""
    model.eval()
    results = []
    verbose_count = 0

    for q in tqdm(questions, desc="MT-Bench generation"):
        q1 = q["turns"][0]
        q2 = q["turns"][1] if len(q["turns"]) > 1 else None

        # Turn 1
        fmt1 = smart_format(q1, model_type, tokenizer)
        enc1 = tokenizer(fmt1, return_tensors="pt", truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            out1 = model.generate(
                **enc1, max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id, do_sample=False
            )
        r1 = tokenizer.decode(out1[0][enc1.input_ids.shape[1]:], skip_special_tokens=True).strip()

        # Turn 2
        r2 = None
        if q2:
            fmt2 = format_multi_turn(q1, r1, q2, model_type, tokenizer)
            enc2 = tokenizer(fmt2, return_tensors="pt", truncation=True, max_length=2048).to(device)
            with torch.no_grad():
                out2 = model.generate(
                    **enc2, max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id, do_sample=False
                )
            r2 = tokenizer.decode(out2[0][enc2.input_ids.shape[1]:], skip_special_tokens=True).strip()

        if verbose and verbose_count < 5:
            print(f"\n  [MT-Bench {verbose_count+1}] Q: {q1[:80]}")
            print(f"      Turn 1: {r1[:150]}")
            if r2:
                print(f"      Turn 2: {r2[:150]}")
            verbose_count += 1

        results.append({
            "question_id": q["question_id"],
            "category": q["category"],
            "turns": q["turns"],
            "response_1": r1,
            "response_2": r2,
        })

    return results


# ==========================================
# MT-BENCH JUDGING (local model)
# ==========================================
JUDGE_PROMPT_T1 = """Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".

[Question]
{question}

[The Start of Assistant's Answer]
{answer}
[The End of Assistant's Answer]"""

JUDGE_PROMPT_T2 = """Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".

<|The Start of Assistant Conversation with User|>

### User:
{question_1}

### Assistant:
{answer_1}

### User:
{question_2}

### Assistant:
{answer_2}

<|The End of Assistant Conversation with User|>"""


def parse_score(text: str) -> float:
    """Parse [[rating]] from judge output."""
    match = re.search(r'\[\[(\d+(?:\.\d+)?)\]\]', text)
    if match:
        return min(float(match.group(1)), 10.0)
    match = re.search(r'[Rr]ating[:\s]*(\d+(?:\.\d+)?)', text)
    if match:
        return min(float(match.group(1)), 10.0)
    match = re.search(r'\b(\d+(?:\.\d+)?)\s*/\s*10\b', text)
    if match:
        return min(float(match.group(1)), 10.0)
    return 5.0  # default if parsing fails


def judge_mtbench(
    responses: list,
    judge_model, judge_tokenizer, judge_type: str,
    device: str, max_new_tokens: int = 256,
) -> list:
    """Judge MT-Bench responses with a local model. Returns per-question scores."""
    judge_model.eval()
    scored = []

    for resp in tqdm(responses, desc="Judging MT-Bench"):
        scores = []

        # Judge turn 1
        prompt_t1 = JUDGE_PROMPT_T1.format(
            question=resp["turns"][0],
            answer=resp["response_1"],
        )
        fmt = smart_format(prompt_t1, judge_type, judge_tokenizer)
        enc = judge_tokenizer(fmt, return_tensors="pt", truncation=True, max_length=2048).to(device)
        with torch.no_grad():
            out = judge_model.generate(
                **enc, max_new_tokens=max_new_tokens,
                pad_token_id=judge_tokenizer.pad_token_id, do_sample=False,
            )
        judgment = judge_tokenizer.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True)
        scores.append(parse_score(judgment))

        # Judge turn 2
        if resp["response_2"] and len(resp["turns"]) > 1:
            prompt_t2 = JUDGE_PROMPT_T2.format(
                question_1=resp["turns"][0],
                answer_1=resp["response_1"],
                question_2=resp["turns"][1],
                answer_2=resp["response_2"],
            )
            fmt2 = smart_format(prompt_t2, judge_type, judge_tokenizer)
            enc2 = judge_tokenizer(fmt2, return_tensors="pt", truncation=True, max_length=3072).to(device)
            with torch.no_grad():
                out2 = judge_model.generate(
                    **enc2, max_new_tokens=max_new_tokens,
                    pad_token_id=judge_tokenizer.pad_token_id, do_sample=False,
                )
            judgment2 = judge_tokenizer.decode(out2[0][enc2.input_ids.shape[1]:], skip_special_tokens=True)
            scores.append(parse_score(judgment2))

        avg = sum(scores) / len(scores) if scores else 0
        scored.append({
            "question_id": resp["question_id"],
            "category": resp["category"],
            "scores": scores,
            "avg_score": avg,
        })

    return scored


# ==========================================
# MAIN BENCHMARK RUNNER
# ==========================================
def run_benchmark(
    model_id: str, model_type: str, adapter_path: str = None,
    device: str = "cuda", batch_size: int = 8,
    run_mtbench: bool = False, mtbench_questions: list = None,
    verbose: bool = False, precision: str = "4bit",
    orbench_full: list = None, orbench_held_out: list = None,
) -> dict:
    """Load model, evaluate all requested benchmarks, unload."""

    tag = "DEFENDED" if adapter_path else "BASELINE"
    print(f"\n{'='*60}")
    print(f"  {tag}: {model_type} ({model_id})")
    if adapter_path:
        print(f"  Adapter: {adapter_path}")
    print(f"{'='*60}")

    tokenizer = load_tokenizer(model_id, model_type)

    # Detect whether adapter_path is a LoRA adapter or a full-model checkpoint
    is_full_model_ckpt = (
        adapter_path
        and os.path.exists(adapter_path)
        and os.path.exists(os.path.join(adapter_path, "config.json"))
        and not os.path.exists(os.path.join(adapter_path, "adapter_config.json"))
    )

    if is_full_model_ckpt:
        print(f"[*] Loading full-model checkpoint from {adapter_path}...")
        model = AutoModelForCausalLM.from_pretrained(
            adapter_path, **get_load_kwargs(precision)
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, **get_load_kwargs(precision)
        )

    sync_model_tokenizer(model, tokenizer)

    if adapter_path and os.path.exists(adapter_path) and not is_full_model_ckpt:
        print(f"[*] Loading adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)
        model.print_trainable_parameters()

        # --- Diagnostic 1: LoRA weight norms ---
        lora_a_norms = []
        lora_b_norms = []
        for name, param in model.named_parameters():
            if 'lora_A' in name:
                lora_a_norms.append(param.data.float().abs().mean().item())
            elif 'lora_B' in name:
                lora_b_norms.append(param.data.float().abs().mean().item())
        if lora_b_norms:
            avg_b = sum(lora_b_norms) / len(lora_b_norms)
            max_b = max(lora_b_norms)
            print(f"[*] LoRA weight analysis:")
            print(f"    lora_A layers: {len(lora_a_norms)}, avg |w|: {sum(lora_a_norms)/len(lora_a_norms):.6f}")
            print(f"    lora_B layers: {len(lora_b_norms)}, avg |w|: {avg_b:.8f}, max |w|: {max_b:.8f}")
            if max_b < 1e-7:
                print(f"    [!] CRITICAL: lora_B weights are ~0! Training did NOT update the adapter.")
            elif max_b < 1e-4:
                print(f"    [!] WARNING: lora_B weights are very small. Adapter effect may be negligible.")
        else:
            print(f"[!] CRITICAL: No LoRA parameters found in model! Adapter not injected.")

        # --- Diagnostic 2: Full logit comparison ---
        _test_prompt = smart_format("Hello, how are you?", model_type, tokenizer)
        _test_enc = tokenizer(_test_prompt, return_tensors="pt", truncation=True, max_length=64).to(device)
        with torch.no_grad():
            _logits_adapted = model(**_test_enc).logits[:, -1, :].float()
            with model.disable_adapter():
                _logits_base = model(**_test_enc).logits[:, -1, :].float()
        _diff = (_logits_adapted - _logits_base).abs()
        _top1_base = _logits_base.argmax(dim=-1).item()
        _top1_adapted = _logits_adapted.argmax(dim=-1).item()
        print(f"[*] Logit diff: max={_diff.max().item():.6f}, mean={_diff.mean().item():.8f}")
        print(f"[*] Top-1 token: base={_top1_base} ({tokenizer.decode([_top1_base])}), "
              f"adapted={_top1_adapted} ({tokenizer.decode([_top1_adapted])})")
        if _top1_base == _top1_adapted and _diff.max().item() < 0.01:
            print(f"[!] WARNING: Adapter barely changes logits. Generated text will be IDENTICAL.")

        # --- Diagnostic 3: Short generation comparison ---
        with torch.no_grad():
            _out_adapted = model.generate(
                **_test_enc, max_new_tokens=30, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            _resp_adapted = tokenizer.decode(_out_adapted[0][_test_enc.input_ids.shape[1]:], skip_special_tokens=True)
            with model.disable_adapter():
                _out_base = model.generate(
                    **_test_enc, max_new_tokens=30, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
                _resp_base = tokenizer.decode(_out_base[0][_test_enc.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"[*] Base response:    {_resp_base[:100]}")
        print(f"[*] Adapted response: {_resp_adapted[:100]}")
        if _resp_base == _resp_adapted:
            print(f"[!] WARNING: Responses are IDENTICAL. Adapter has no effect on generation!")
        else:
            print(f"[+] Adapter is active and modifying generation.")
        del _test_enc, _logits_adapted, _logits_base, _out_adapted, _out_base
    elif adapter_path and not is_full_model_ckpt:
        raise FileNotFoundError(
            f"Adapter path does not exist: {adapter_path}\n"
            f"Cannot run defended evaluation without a valid adapter."
        )

    model.eval()
    results = {}

    # --- OR-Bench (skipped gracefully if the dataset cannot be loaded) ---
    if orbench_full is None:
        orbench_full, _ = load_orbench_hard()
    orbench_prompts = orbench_full
    if not orbench_prompts:
        print("\n[!] Skipping OR-Bench evaluation (no prompts loaded).")
        results["orbench"] = {"refusal_rate": None, "bgr": None, "skipped": True}
    else:
        print(f"\n--- OR-Bench Hard ({len(orbench_prompts)} prompts) ---")
        results["orbench"] = evaluate_refusal_rate(
            model, tokenizer, orbench_prompts, model_type, device,
            batch_size=batch_size, max_new_tokens=100, label="OR-Bench",
            verbose=verbose,
        )
        print(f"OR-Bench Refusal Rate: {results['orbench']['refusal_rate']*100:.1f}%")
        print(f"OR-Bench BGR: {results['orbench'].get('bgr', 0)*100:.1f}%")

    # --- OR-Bench Held-Out (test split, excluding training prompts) ---
    if orbench_prompts and orbench_held_out is not None and len(orbench_held_out) < len(orbench_prompts):
        print(f"\n--- OR-Bench Held-Out ({len(orbench_held_out)} prompts) ---")
        results["orbench_held_out"] = evaluate_refusal_rate(
            model, tokenizer, orbench_held_out, model_type, device,
            batch_size=batch_size, max_new_tokens=100, label="OR-Bench Held-Out",
            verbose=verbose,
        )
        print(f"OR-Bench Held-Out Refusal Rate: {results['orbench_held_out']['refusal_rate']*100:.1f}%")

    # --- XSTest ---
    xstest_prompts = load_xstest_safe()
    print(f"\n--- XSTest Safe ({len(xstest_prompts)} prompts) ---")
    results["xstest"] = evaluate_refusal_rate(
        model, tokenizer, xstest_prompts, model_type, device,
        batch_size=batch_size, max_new_tokens=100, label="XSTest",
        verbose=verbose,
    )
    print(f"XSTest Refusal Rate: {results['xstest']['refusal_rate']*100:.1f}%")

    # NOTE: MMLU now runs separately via lm-eval-harness (outside run_benchmark)

    # --- MT-Bench (generate only, judge later) ---
    if run_mtbench and mtbench_questions:
        print(f"\n--- MT-Bench ({len(mtbench_questions)} questions, 2 turns) ---")
        results["mtbench_responses"] = generate_mtbench_responses(
            model, tokenizer, mtbench_questions, model_type, device,
            verbose=verbose,
        )

    # Cleanup
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return results


# ==========================================
# MAIN
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Benchmark Evaluation")

    parser.add_argument("--defender", type=str, required=True,
                        help="Defender model name (e.g., vicuna, llama3)")
    parser.add_argument("--anchor", type=str, default=None,
                        help="Specific anchor (default: best from CSV)")
    parser.add_argument("--csv_path", type=str, default="all_pairs_results.csv")
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="Direct adapter path (overrides CSV)")
    parser.add_argument("--baseline_only", action="store_true")
    parser.add_argument("--no_baseline", action="store_true",
                        help="Skip baseline evaluation, only run defended model")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--low_memory", action="store_true",
                        help="Smaller batch sizes for low VRAM")

    # Benchmark toggles
    parser.add_argument("--skip_mmlu", action="store_true", help="Skip MMLU evaluation")
    parser.add_argument("--skip_mtbench", action="store_true", help="Skip MT-Bench evaluation")
    parser.add_argument("--mmlu_limit", type=int, default=0,
                        help="Limit MMLU questions (0 = all ~14K)")

    # MT-Bench judge
    parser.add_argument("--judge_model", type=str,
                        default="meta-llama/Meta-Llama-3-8B-Instruct",
                        help="Local model for MT-Bench judging")
    parser.add_argument("--baseline_cache_dir", type=str, default=None,
                        help="Directory to cache baseline results. If set, looks for cached "
                             "baseline before running; saves after running if not found.")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Print 5 sample generations from each benchmark")
    parser.add_argument("--precision", type=str, default="4bit",
                        choices=["4bit", "fp16", "fp32"],
                        help="Model precision: 4bit (NF4 quantization), fp16, or fp32 (default: 4bit)")
    parser.add_argument("--exclude_train_prompts", action="store_true",
                        help="Exclude borderline prompts used during training from OR-Bench eval")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_TYPE_ALIASES = {"llama2-13b": "llama2", "qwen-14b": "qwen"}
    defender_arg = args.defender.lower()
    model_id = MODEL_IDS.get(defender_arg)
    if model_id:
        model_type = MODEL_TYPE_ALIASES.get(defender_arg, defender_arg)
    else:
        # Treat as full HuggingFace model ID
        model_id = args.defender
        model_type = _HF_ID_TO_TYPE.get(defender_arg)
        if not model_type:
            # Infer model_type from name
            name_lower = defender_arg
            if "llama-2" in name_lower or "llama2" in name_lower:
                model_type = "llama2"
            elif "llama-3" in name_lower or "llama3" in name_lower:
                model_type = "llama3"
            elif "qwen" in name_lower:
                model_type = "qwen"
            elif "mistral" in name_lower:
                model_type = "mistral"
            elif "vicuna" in name_lower:
                model_type = "vicuna"
            elif "zephyr" in name_lower:
                model_type = "zephyr"
            elif "phi" in name_lower:
                model_type = "phi2"
            elif "yi" in name_lower:
                model_type = "yi"
            else:
                print(f"[!] Cannot infer model type from: {args.defender}. Use --defender with a known name or HF ID.")
                sys.exit(1)
        print(f"[+] Using full HF model ID: {model_id} (type: {model_type})")

    batch_size = 4 if args.low_memory else args.batch_size
    run_mmlu = not args.skip_mmlu
    run_mtbench = not args.skip_mtbench

    # Find adapter
    adapter_path = None
    anchor_used = args.anchor
    if not args.baseline_only:
        if args.adapter_path:
            adapter_path = args.adapter_path
        else:
            csv_path = args.csv_path
            if not os.path.isabs(csv_path):
                csv_path = os.path.join(os.path.dirname(__file__), csv_path)
            adapter_path, anchor_used = find_adapter_from_csv(csv_path, model_type, args.anchor)
        print(f"[+] Using adapter: {adapter_path}")
        print(f"[+] Anchor: {anchor_used}")

    mtbench_questions = None
    if run_mtbench:
        mtbench_questions = load_mtbench_questions()

    all_results = {"defender": model_type, "model_id": model_id, "anchor": anchor_used}

    # Initialize gibberish detector (same pipeline as evaluate_v2.py)
    init_gibberish_detector(device=device)

    # Load OR-Bench prompts and handle train/test split
    exclude_prompts = None
    if args.exclude_train_prompts and adapter_path:
        bl_json_path = os.path.join(adapter_path, "borderline_train_prompts.json")
        if os.path.exists(bl_json_path):
            with open(bl_json_path, 'r') as f:
                train_prompts = json.load(f)
            exclude_prompts = set(train_prompts)
            print(f"[+] Loaded {len(exclude_prompts)} train prompts to exclude from OR-Bench")
        else:
            print(f"[!] No borderline_train_prompts.json found in {adapter_path}")

    orbench_full, orbench_held_out = load_orbench_hard(exclude_prompts=exclude_prompts)

    # =========== BASELINE ===========
    baseline = None
    if not args.no_baseline:
        print("\n" + "#" * 70)
        print("# BASELINE EVALUATION")
        print("#" * 70)

        # Check baseline cache
        baseline_cache_path = None
        if args.baseline_cache_dir:
            os.makedirs(args.baseline_cache_dir, exist_ok=True)
            baseline_cache_path = os.path.join(args.baseline_cache_dir, f"baseline_{model_type}.json")

        if baseline_cache_path and os.path.exists(baseline_cache_path):
            print(f"[*] Loading cached baseline from {baseline_cache_path}")
            with open(baseline_cache_path, 'r') as f:
                baseline = json.load(f)
            print(f"[+] Cached baseline loaded (keys: {list(baseline.keys())})")
        else:
            baseline = run_benchmark(
                model_id, model_type, adapter_path=None, device=device,
                batch_size=batch_size, run_mtbench=run_mtbench,
                mtbench_questions=mtbench_questions, verbose=args.verbose,
                precision=args.precision,
                orbench_full=orbench_full, orbench_held_out=orbench_held_out,
            )
            if baseline_cache_path:
                with open(baseline_cache_path, 'w') as f:
                    json.dump(baseline, f, indent=2, default=str)
                print(f"[+] Baseline cached to {baseline_cache_path}")

        all_results["baseline"] = {
            "orbench_refusal_rate": baseline["orbench"]["refusal_rate"],
            "orbench_bgr": baseline["orbench"].get("bgr", 0.0),
            "xstest_refusal_rate": baseline["xstest"]["refusal_rate"],
            "xstest_bgr": baseline["xstest"].get("bgr", 0.0),
        }
        if "orbench_held_out" in baseline:
            all_results["baseline"]["orbench_held_out_refusal_rate"] = baseline["orbench_held_out"]["refusal_rate"]
        if "mmlu_accuracy" in baseline:
            all_results["baseline"]["mmlu_accuracy"] = baseline["mmlu_accuracy"]
        all_results["baseline_orbench_all_responses"] = baseline["orbench"].get("all_responses", [])
        all_results["baseline_orbench_refused_examples"] = baseline["orbench"].get("refused_examples", [])
        all_results["baseline_xstest_all_responses"] = baseline["xstest"].get("all_responses", [])
        all_results["baseline_xstest_refused_examples"] = baseline["xstest"].get("refused_examples", [])
    else:
        print("\n[*] Skipping baseline evaluation (--no_baseline)")

    # =========== DEFENDED ===========
    defended = None
    if adapter_path:
        print("\n" + "#" * 70)
        print("# DEFENDED EVALUATION")
        print("#" * 70)
        defended = run_benchmark(
            model_id, model_type, adapter_path=adapter_path, device=device,
            batch_size=batch_size, run_mtbench=run_mtbench,
            mtbench_questions=mtbench_questions, verbose=args.verbose,
            precision=args.precision,
            orbench_full=orbench_full, orbench_held_out=orbench_held_out,
        )

        all_results["defended"] = {
            "orbench_refusal_rate": defended["orbench"]["refusal_rate"],
            "orbench_bgr": defended["orbench"].get("bgr", 0.0),
            "xstest_refusal_rate": defended["xstest"]["refusal_rate"],
            "xstest_bgr": defended["xstest"].get("bgr", 0.0),
        }
        if "orbench_held_out" in defended:
            all_results["defended"]["orbench_held_out_refusal_rate"] = defended["orbench_held_out"]["refusal_rate"]
        if "mmlu_accuracy" in defended:
            all_results["defended"]["mmlu_accuracy"] = defended["mmlu_accuracy"]

        all_results["defended_orbench_refused_examples"] = defended["orbench"]["refused_examples"]
        all_results["defended_orbench_all_responses"] = defended["orbench"].get("all_responses", [])
        all_results["defended_xstest_refused_examples"] = defended["xstest"]["refused_examples"]
        all_results["defended_xstest_all_responses"] = defended["xstest"].get("all_responses", [])

    # =========== MMLU (via lm-eval-harness) ===========
    if run_mmlu:
        # Force full GPU cleanup before shelling out to lm-eval subprocess
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        if not args.no_baseline and baseline is not None:
            bl_mmlu = evaluate_mmlu_lmeval(
                model_id, adapter_path=None,
                batch_size=batch_size, limit=args.mmlu_limit,
                precision=args.precision,
            )
            if bl_mmlu is not None:
                baseline["mmlu_accuracy"] = bl_mmlu
                all_results.setdefault("baseline", {})["mmlu_accuracy"] = bl_mmlu

        if adapter_path and defended is not None:
            df_mmlu = evaluate_mmlu_lmeval(
                model_id, adapter_path=adapter_path,
                batch_size=batch_size, limit=args.mmlu_limit,
                precision=args.precision,
            )
            if df_mmlu is not None:
                defended["mmlu_accuracy"] = df_mmlu
                all_results.setdefault("defended", {})["mmlu_accuracy"] = df_mmlu

    # =========== MT-BENCH JUDGING ===========
    has_bl_mtbench = baseline and "mtbench_responses" in baseline
    has_df_mtbench = defended and "mtbench_responses" in defended
    if run_mtbench and (has_bl_mtbench or has_df_mtbench):
        print("\n" + "#" * 70)
        print("# MT-BENCH JUDGING")
        print("#" * 70)

        judge_type = detect_model_type(args.judge_model)
        print(f"[*] Loading judge: {args.judge_model} (type: {judge_type})")

        judge_tokenizer = AutoTokenizer.from_pretrained(args.judge_model, trust_remote_code=True)
        if not judge_tokenizer.pad_token:
            judge_tokenizer.pad_token = judge_tokenizer.eos_token
        judge_tokenizer.padding_side = "left"

        judge_model = AutoModelForCausalLM.from_pretrained(
            args.judge_model, **get_load_kwargs(args.precision)
        )

        # Judge baseline
        if has_bl_mtbench:
            print("\n--- Judging baseline responses ---")
            bl_scores = judge_mtbench(
                baseline["mtbench_responses"],
                judge_model, judge_tokenizer, judge_type, device,
            )
            bl_avg = sum(s["avg_score"] for s in bl_scores) / len(bl_scores)
            all_results["baseline"]["mtbench_score"] = bl_avg
            all_results["baseline"]["mtbench_per_category"] = _category_scores(bl_scores)
            print(f"Baseline MT-Bench: {bl_avg:.2f}/10")

        # Judge defended
        if has_df_mtbench:
            print("\n--- Judging defended responses ---")
            df_scores = judge_mtbench(
                defended["mtbench_responses"],
                judge_model, judge_tokenizer, judge_type, device,
            )
            df_avg = sum(s["avg_score"] for s in df_scores) / len(df_scores)
            all_results["defended"]["mtbench_score"] = df_avg
            all_results["defended"]["mtbench_per_category"] = _category_scores(df_scores)
            print(f"Defended MT-Bench: {df_avg:.2f}/10")

        del judge_model, judge_tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    # =========== SUMMARY ===========
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 70)
    print(f"Model: {model_type} ({model_id})")
    if adapter_path:
        print(f"Anchor: {anchor_used}")
        print(f"Adapter: {adapter_path}")
    if run_mtbench:
        print(f"MT-Bench Judge: {args.judge_model}")
    print()

    has_baseline = baseline is not None
    has_defended = adapter_path and defended is not None

    # Header
    if has_baseline and has_defended:
        print(f"{'Benchmark':<20} {'Baseline':>12} {'Defended':>12} {'Delta':>10}")
        print("-" * 56)
    elif has_defended:
        print(f"{'Benchmark':<20} {'Defended':>12}")
        print("-" * 34)
    else:
        print(f"{'Benchmark':<20} {'Baseline':>12}")
        print("-" * 34)

    def _pct(d, key):
        """Return d[key]['refusal_rate'] * 100 if present and not None, else None."""
        if d is None or key not in d:
            return None
        rate = d[key].get("refusal_rate")
        return rate * 100 if rate is not None else None

    # OR-Bench
    bl_or = _pct(baseline if has_baseline else None, "orbench")
    df_or = _pct(defended if has_defended else None, "orbench")
    if bl_or is not None and df_or is not None:
        print(f"{'OR-Bench Hard':<20} {bl_or:>11.1f}% {df_or:>11.1f}% {df_or - bl_or:>+9.1f}%")
    elif df_or is not None:
        print(f"{'OR-Bench Hard':<20} {df_or:>11.1f}%")
    elif bl_or is not None:
        print(f"{'OR-Bench Hard':<20} {bl_or:>11.1f}%")
    else:
        print(f"{'OR-Bench Hard':<20} (skipped - dataset unavailable)")

    # OR-Bench Held-Out (test split)
    bl_or_ho = _pct(baseline if has_baseline else None, "orbench_held_out")
    df_or_ho = _pct(defended if has_defended else None, "orbench_held_out")
    if bl_or_ho is not None and df_or_ho is not None:
        print(f"{'OR-Bench Held-Out':<20} {bl_or_ho:>11.1f}% {df_or_ho:>11.1f}% {df_or_ho - bl_or_ho:>+9.1f}%")
    elif df_or_ho is not None:
        print(f"{'OR-Bench Held-Out':<20} {df_or_ho:>11.1f}%")
    elif bl_or_ho is not None:
        print(f"{'OR-Bench Held-Out':<20} {bl_or_ho:>11.1f}%")

    # XSTest
    bl_xs = baseline["xstest"]["refusal_rate"] * 100 if has_baseline else None
    df_xs = defended["xstest"]["refusal_rate"] * 100 if has_defended else None
    if bl_xs is not None and df_xs is not None:
        print(f"{'XSTest Safe':<20} {bl_xs:>11.1f}% {df_xs:>11.1f}% {df_xs - bl_xs:>+9.1f}%")
    elif df_xs is not None:
        print(f"{'XSTest Safe':<20} {df_xs:>11.1f}%")
    elif bl_xs is not None:
        print(f"{'XSTest Safe':<20} {bl_xs:>11.1f}%")

    # MMLU
    bl_mm = all_results.get("baseline", {}).get("mmlu_accuracy")
    df_mm = all_results.get("defended", {}).get("mmlu_accuracy")
    if bl_mm is not None and df_mm is not None:
        print(f"{'MMLU (5-shot)':<20} {bl_mm*100:>11.1f}% {df_mm*100:>11.1f}% {(df_mm-bl_mm)*100:>+9.1f}%")
    elif df_mm is not None:
        print(f"{'MMLU (5-shot)':<20} {df_mm*100:>11.1f}%")
    elif bl_mm is not None:
        print(f"{'MMLU (5-shot)':<20} {bl_mm*100:>11.1f}%")

    # MT-Bench
    bl_mt = all_results.get("baseline", {}).get("mtbench_score")
    df_mt = all_results.get("defended", {}).get("mtbench_score")
    if bl_mt is not None and df_mt is not None:
        print(f"{'MT-Bench':<20} {bl_mt:>10.2f}/10 {df_mt:>10.2f}/10 {df_mt - bl_mt:>+8.2f}")
    elif df_mt is not None:
        print(f"{'MT-Bench':<20} {df_mt:>10.2f}/10")
    elif bl_mt is not None:
        print(f"{'MT-Bench':<20} {bl_mt:>10.2f}/10")

    print()
    print("(OR-Bench/XSTest: lower = better.  MMLU/MT-Bench: higher = better.)")
    print("(Circuit Breakers: 38.5% OR-Bench | ReFAT: 65.2% MMLU, 6.98 MT-Bench)")
    print("=" * 70)

    # Per-category MT-Bench breakdown
    bl_cats = all_results.get("baseline", {}).get("mtbench_per_category", {})
    df_cats = all_results.get("defended", {}).get("mtbench_per_category", {})
    if bl_cats or df_cats:
        print("\nMT-Bench Per-Category Breakdown:")
        if bl_cats and df_cats:
            print(f"  {'Category':<20} {'Baseline':>10} {'Defended':>10} {'Delta':>8}")
            print(f"  {'-'*50}")
            for cat in sorted(bl_cats.keys()):
                b = bl_cats[cat]
                d = df_cats.get(cat, 0)
                print(f"  {cat:<20} {b:>9.2f} {d:>9.2f} {d - b:>+7.2f}")
        elif df_cats:
            print(f"  {'Category':<20} {'Defended':>10}")
            print(f"  {'-'*32}")
            for cat in sorted(df_cats.keys()):
                print(f"  {cat:<20} {df_cats[cat]:>9.2f}")
        else:
            print(f"  {'Category':<20} {'Baseline':>10}")
            print(f"  {'-'*32}")
            for cat in sorted(bl_cats.keys()):
                print(f"  {cat:<20} {bl_cats[cat]:>9.2f}")

    # Save JSON
    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        # Strip non-serializable data (mt_bench raw responses are large)
        save_data = {k: v for k, v in all_results.items()}
        with open(args.output_json, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        print(f"\n[+] Results saved to {args.output_json}")


def _category_scores(scored_list: list) -> dict:
    """Aggregate MT-Bench scores by category."""
    by_cat = {}
    for s in scored_list:
        cat = s["category"]
        if cat not in by_cat:
            by_cat[cat] = []
        by_cat[cat].append(s["avg_score"])
    return {cat: sum(v) / len(v) for cat, v in by_cat.items()}


if __name__ == "__main__":
    main()
