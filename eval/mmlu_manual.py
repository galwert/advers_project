"""Manual MMLU 5-shot evaluator.

Standalone, no lm-eval dependency. Loads the cached `hails/mmlu_no_train`
subjects directly, builds 5-shot prompts in the standard CAIS format, scores
the next-token logprobs over " A"/" B"/" C"/" D", and reports per-subject and
overall accuracy.

Usage:
    python mmlu_manual.py --base meta-llama/Meta-Llama-3-8B-Instruct \
                          --adapter <lora-or-fullmodel-dir|none> \
                          --output <out.json>
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

MMLU_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology", "high_school_statistics",
    "high_school_us_history", "high_school_world_history", "human_aging",
    "human_sexuality", "international_law", "jurisprudence", "logical_fallacies",
    "machine_learning", "management", "marketing", "medical_genetics",
    "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition",
    "philosophy", "prehistory", "professional_accounting", "professional_law",
    "professional_medicine", "professional_psychology", "public_relations",
    "security_studies", "sociology", "us_foreign_policy", "virology",
    "world_religions",
]


def format_subject(subject: str) -> str:
    return subject.replace("_", " ")


def format_example(question: str, choices, answer_letter: str | None) -> str:
    s = question.strip()
    s += f"\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer:"
    if answer_letter is not None:
        s += f" {answer_letter}"
    return s


def build_prompt(subject: str, dev_examples, test_question, test_choices) -> str:
    """5-shot prompt in the canonical CAIS format."""
    header = f"The following are multiple choice questions (with answers) about {format_subject(subject)}.\n\n"
    shots = []
    for ex in dev_examples[:5]:
        letter = "ABCD"[ex["answer"]]
        shots.append(format_example(ex["question"], ex["choices"], letter))
    shots.append(format_example(test_question, test_choices, None))
    return header + "\n\n".join(shots)


def load_model(base: str, adapter: str | None):
    print(f"[*] tokenizer: {base}")
    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    is_full = (
        adapter
        and adapter != "none"
        and os.path.exists(adapter)
        and os.path.exists(os.path.join(adapter, "config.json"))
        and not os.path.exists(os.path.join(adapter, "adapter_config.json"))
    )

    if is_full:
        print(f"[*] loading full-model checkpoint: {adapter}")
        model = AutoModelForCausalLM.from_pretrained(
            adapter, torch_dtype=torch.float16, device_map={"": 0},
        )
    else:
        print(f"[*] loading base: {base}")
        model = AutoModelForCausalLM.from_pretrained(
            base, torch_dtype=torch.float16, device_map={"": 0},
        )
        if adapter and adapter != "none" and os.path.exists(adapter):
            print(f"[*] loading LoRA adapter: {adapter}")
            model = PeftModel.from_pretrained(model, adapter)

    model.eval()
    return model, tok


@torch.no_grad()
def score_question(model, tok, prompt: str, choice_token_ids):
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    out = model(**enc)
    last_logits = out.logits[0, -1, :]
    log_probs = F.log_softmax(last_logits.float(), dim=-1)
    scores = [log_probs[t].item() for t in choice_token_ids]
    return int(max(range(4), key=lambda i: scores[i])), scores


def get_choice_token_ids(tok):
    """Token id for the FIRST sub-token of each ' A'/' B'/' C'/' D' continuation."""
    ids = []
    for letter in "ABCD":
        # Try " A" first (with leading space), as the prompt ends in "Answer:"
        toks = tok.encode(" " + letter, add_special_tokens=False)
        ids.append(toks[0])
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--adapter", default=None,
                    help="LoRA adapter dir, full-model checkpoint dir, or 'none'")
    ap.add_argument("--output", required=True, help="JSON output path")
    ap.add_argument("--subject_limit", type=int, default=0,
                    help="limit per-subject test questions (0 = all)")
    args = ap.parse_args()

    model, tok = load_model(args.base, args.adapter)
    choice_ids = get_choice_token_ids(tok)
    print(f"[*] choice token ids (' A',' B',' C',' D'): {choice_ids}")
    print(f"[*] decoded back: " + repr([tok.decode([t]) for t in choice_ids]))

    per_subject = {}
    total_correct, total_n = 0, 0

    for subject in tqdm(MMLU_SUBJECTS, desc="subjects"):
        ds = load_dataset("hails/mmlu_no_train", subject)
        dev = list(ds["dev"])
        test = list(ds["test"])
        if args.subject_limit:
            test = test[: args.subject_limit]

        correct, n = 0, 0
        for ex in test:
            prompt = build_prompt(subject, dev, ex["question"], ex["choices"])
            pred, _ = score_question(model, tok, prompt, choice_ids)
            if pred == ex["answer"]:
                correct += 1
            n += 1
        acc = correct / n if n else 0.0
        per_subject[subject] = {"acc": acc, "n": n, "correct": correct}
        total_correct += correct
        total_n += n
        tqdm.write(f"  {subject}: {acc:.3f} ({correct}/{n})  [running overall: {total_correct/total_n:.3f}]")

    overall = total_correct / total_n if total_n else 0.0
    out = {
        "base": args.base,
        "adapter": args.adapter,
        "overall_accuracy": overall,
        "total_correct": total_correct,
        "total_n": total_n,
        "per_subject": per_subject,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"\n[+] OVERALL MMLU accuracy: {overall*100:.2f}% ({total_correct}/{total_n})")
    print(f"[+] saved: {args.output}")


if __name__ == "__main__":
    main()
