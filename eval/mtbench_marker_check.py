"""MT-Bench turn-marker hallucination check.

Loads a Llama-3 base + LoRA adapter (or just the base), generates greedy
responses for all 80 MT-Bench questions (Q1 and Q2), and counts how many
contain the decoded `assistant\\n\\n` role marker that signals turn-boundary
overshoot. Used to confirm that retrained CB does not exhibit the
public-checkpoint anomaly documented in app:cb_mtbench.

Usage:
    python mtbench_marker_check.py \
        --base meta-llama/Meta-Llama-3-8B-Instruct \
        --adapter <lora-dir|none> \
        --label retrained_cb \
        --output /path/to/out.json
"""
import argparse
import json
import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

QUESTIONS_PATH = Path(__file__).parent / ".mtbench_questions.jsonl"


def llama3_format_turn1(tok, q1: str) -> str:
    msgs = [{"role": "user", "content": q1}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def llama3_format_turn2(tok, q1: str, r1: str, q2: str) -> str:
    msgs = [
        {"role": "user", "content": q1},
        {"role": "assistant", "content": r1},
        {"role": "user", "content": q2},
    ]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def load_model(base: str, adapter: str | None):
    is_full = (
        adapter
        and adapter != "none"
        and os.path.exists(adapter)
        and os.path.exists(os.path.join(adapter, "config.json"))
        and not os.path.exists(os.path.join(adapter, "adapter_config.json"))
    )

    if is_full:
        print(f"[*] full-model checkpoint: {adapter}")
        tok = AutoTokenizer.from_pretrained(adapter)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            adapter, torch_dtype=torch.float16, device_map={"": 0},
        )
    else:
        print(f"[*] tokenizer: {base}")
        tok = AutoTokenizer.from_pretrained(base)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        print(f"[*] base model: {base}")
        model = AutoModelForCausalLM.from_pretrained(
            base, torch_dtype=torch.float16, device_map={"": 0},
        )
        if adapter and adapter != "none" and os.path.exists(adapter):
            print(f"[*] LoRA adapter: {adapter}")
            model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tok


@torch.no_grad()
def generate(model, tok, prompt: str, max_new_tokens: int = 512) -> tuple[str, bool]:
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        pad_token_id=tok.pad_token_id,
        do_sample=False,
    )
    n_in = enc.input_ids.shape[1]
    new_tokens = out[0][n_in:]
    text = tok.decode(new_tokens, skip_special_tokens=True).strip()
    hit_limit = (new_tokens.shape[0] >= max_new_tokens)
    return text, hit_limit


def count_markers(text: str) -> int:
    """Count `assistant\\n\\n` decoded role-marker occurrences."""
    return text.count("assistant\n\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--adapter", default="none")
    ap.add_argument("--label", required=True, help="e.g. retrained_cb / baseline")
    ap.add_argument("--output", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    args = ap.parse_args()

    questions = []
    with open(QUESTIONS_PATH) as f:
        for line in f:
            questions.append(json.loads(line))
    print(f"[*] loaded {len(questions)} MT-Bench questions")

    model, tok = load_model(args.base, args.adapter)
    print(f"[*] tokenizer.eos_token = {tok.eos_token!r} (id={tok.eos_token_id})")

    responses = []
    for q in tqdm(questions, desc="MT-Bench gen"):
        q1 = q["turns"][0]
        q2 = q["turns"][1] if len(q["turns"]) > 1 else None

        p1 = llama3_format_turn1(tok, q1)
        r1, hit1 = generate(model, tok, p1, args.max_new_tokens)
        m1 = count_markers(r1)

        r2, m2, hit2 = None, 0, False
        if q2:
            p2 = llama3_format_turn2(tok, q1, r1, q2)
            r2, hit2 = generate(model, tok, p2, args.max_new_tokens)
            m2 = count_markers(r2)

        responses.append({
            "question_id": q["question_id"],
            "category": q["category"],
            "turns": q["turns"],
            "response_1": r1,
            "response_2": r2,
            "markers_1": m1,
            "markers_2": m2,
            "hit_token_limit_1": hit1,
            "hit_token_limit_2": hit2,
            "len_chars_1": len(r1),
            "len_chars_2": len(r2) if r2 else 0,
        })

    n = len(responses)
    n_resps = 2 * n  # Q1 + Q2
    total_markers = sum(r["markers_1"] + r["markers_2"] for r in responses)
    n_with_marker = sum((r["markers_1"] > 0) + (r["markers_2"] > 0) for r in responses)
    n_token_limit = sum(int(r["hit_token_limit_1"]) + int(r["hit_token_limit_2"]) for r in responses)
    avg_len_1 = sum(r["len_chars_1"] for r in responses) / n
    avg_len_2 = sum(r["len_chars_2"] for r in responses) / n

    summary = {
        "label": args.label,
        "base": args.base,
        "adapter": args.adapter,
        "eos_token": tok.eos_token,
        "eos_token_id": tok.eos_token_id,
        "n_questions": n,
        "n_responses": n_resps,
        "responses_with_hallucinated_marker": n_with_marker,
        "responses_with_marker_pct": 100.0 * n_with_marker / n_resps,
        "total_markers": total_markers,
        "responses_hit_token_limit": n_token_limit,
        "avg_len_chars_turn1": avg_len_1,
        "avg_len_chars_turn2": avg_len_2,
    }
    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    out = {"summary": summary, "responses": responses}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[+] saved {args.output}")


if __name__ == "__main__":
    main()
