#!/bin/bash
#SBATCH --job-name=cb_mt_inv
#SBATCH --partition=public
#SBATCH --account=cs
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=7b_defense_wildguard_outputs/slurm_cb_mt_inv_%j.out
#SBATCH --error=7b_defense_wildguard_outputs/slurm_cb_mt_inv_%j.err

source /home/wertheizer/miniforge3/etc/profile.d/conda.sh
conda activate defense_env
cd /home/wertheizer/advers_project/phase3

export HF_TOKEN="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"
export TQDM_DISABLE=1

python3 << 'PYEOF'
import json, torch, sys, os, re, gc
sys.modules["flash_attn"] = None
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
device = "cuda"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
if not tokenizer.pad_token:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# Load MT-Bench questions
questions = []
with open(".mtbench_questions.jsonl") as f:
    for line in f:
        if line.strip():
            questions.append(json.loads(line))
q_by_id = {q["question_id"]: q for q in questions}

# Load CB Llama3 data
cb_resp = json.load(open("7b_defense_wildguard_outputs/bench_competing/responses_cb_llama3.json"))
cb_mt = {r["question_id"]: r for r in cb_resp["mtbench"]}
cb_scores = json.load(open("7b_defense_wildguard_outputs/bench_competing/bench_cb_llama3.json"))
cb_per_q = {s["question_id"]: s for s in cb_scores["mtbench_per_question"]}

# ── Phase 1: Generate ALL 80 baseline responses ──
print("[*] Loading baseline Llama-3-8B-Instruct (fp32)...")
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.float32,
    device_map="auto", trust_remote_code=True, attn_implementation="eager",
)
model.eval()

print("[*] Generating baseline responses for all 80 questions...")
bl_responses = []
for q in questions:
    q1 = q["turns"][0]
    q2 = q["turns"][1] if len(q["turns"]) > 1 else None

    fmt1 = tokenizer.apply_chat_template(
        [{"role": "user", "content": q1}],
        tokenize=False, add_generation_prompt=True,
    )
    enc1 = tokenizer(fmt1, return_tensors="pt", truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        out1 = model.generate(**enc1, max_new_tokens=512, pad_token_id=tokenizer.pad_token_id, do_sample=False)
    r1 = tokenizer.decode(out1[0][enc1.input_ids.shape[1]:], skip_special_tokens=True).strip()

    r2 = None
    if q2:
        try:
            fmt2 = tokenizer.apply_chat_template(
                [{"role": "user", "content": q1},
                 {"role": "assistant", "content": r1},
                 {"role": "user", "content": q2}],
                tokenize=False, add_generation_prompt=True,
            )
        except:
            fmt2 = (
                f"<|start_header_id|>user<|end_header_id|>\n\n{q1}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n{r1}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n{q2}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        enc2 = tokenizer(fmt2, return_tensors="pt", truncation=True, max_length=2048).to(device)
        with torch.no_grad():
            out2 = model.generate(**enc2, max_new_tokens=512, pad_token_id=tokenizer.pad_token_id, do_sample=False)
        r2 = tokenizer.decode(out2[0][enc2.input_ids.shape[1]:], skip_special_tokens=True).strip()

    bl_responses.append({
        "question_id": q["question_id"],
        "category": q["category"],
        "turns": q["turns"],
        "response_1": r1,
        "response_2": r2,
    })
    if (len(bl_responses)) % 10 == 0:
        print(f"  {len(bl_responses)}/80")

# Save baseline responses (for future GPT-4 judging too)
with open("7b_defense_wildguard_outputs/mtbench_baseline_llama3_fp32_responses.json", "w") as f:
    json.dump({"model_id": model_id, "precision": "fp32", "responses": bl_responses}, f, indent=2)
print("[+] Saved baseline responses")

del model; gc.collect(); torch.cuda.empty_cache()

# ── Phase 2: Judge both baseline and CB with same judge ──
print("\n[*] Loading judge (same Llama-3-8B-Instruct fp32)...")
judge_model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.float32,
    device_map="auto", trust_remote_code=True, attn_implementation="eager",
)
judge_model.eval()

JUDGE_T1 = """Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".

[Question]
{question}

[The Start of Assistant's Answer]
{answer}
[The End of Assistant's Answer]"""

JUDGE_T2 = """Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".

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

def parse_score(text):
    m = re.search(r'\[\[(\d+(?:\.\d+)?)\]\]', text)
    if m: return min(float(m.group(1)), 10.0)
    m = re.search(r'[Rr]ating[:\s]*(\d+(?:\.\d+)?)', text)
    if m: return min(float(m.group(1)), 10.0)
    m = re.search(r'\b(\d+(?:\.\d+)?)\s*/\s*10\b', text)
    if m: return min(float(m.group(1)), 10.0)
    return 5.0

def judge_turn1(question, answer):
    prompt = JUDGE_T1.format(question=question, answer=answer)
    fmt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True,
    )
    enc = tokenizer(fmt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    with torch.no_grad():
        out = judge_model.generate(**enc, max_new_tokens=256, pad_token_id=tokenizer.pad_token_id, do_sample=False)
    judgment = tokenizer.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True)
    return parse_score(judgment)

def judge_turn2(q1, a1, q2, a2):
    prompt = JUDGE_T2.format(question_1=q1, answer_1=a1, question_2=q2, answer_2=a2)
    fmt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True,
    )
    enc = tokenizer(fmt, return_tensors="pt", truncation=True, max_length=3072).to(device)
    with torch.no_grad():
        out = judge_model.generate(**enc, max_new_tokens=256, pad_token_id=tokenizer.pad_token_id, do_sample=False)
    judgment = tokenizer.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True)
    return parse_score(judgment)

print("[*] Judging all 80 questions for BOTH baseline and CB...")
bl_by_id = {r["question_id"]: r for r in bl_responses}

comparison = []
bl_scores_all = []
cb_scores_all = []

for qi, q in enumerate(questions):
    qid = q["question_id"]
    bl_r = bl_by_id[qid]
    cb_r = cb_mt[qid]

    # Judge baseline
    bl_s1 = judge_turn1(q["turns"][0], bl_r["response_1"])
    bl_scores = [bl_s1]
    if bl_r["response_2"] and len(q["turns"]) > 1:
        bl_s2 = judge_turn2(q["turns"][0], bl_r["response_1"], q["turns"][1], bl_r["response_2"])
        bl_scores.append(bl_s2)

    # Judge CB
    cb_s1 = judge_turn1(q["turns"][0], cb_r["response_1"])
    cb_scores_q = [cb_s1]
    if cb_r["response_2"] and len(q["turns"]) > 1:
        cb_s2 = judge_turn2(q["turns"][0], cb_r["response_1"], q["turns"][1], cb_r["response_2"])
        cb_scores_q.append(cb_s2)

    bl_avg = sum(bl_scores) / len(bl_scores)
    cb_avg = sum(cb_scores_q) / len(cb_scores_q)
    bl_scores_all.append(bl_avg)
    cb_scores_all.append(cb_avg)

    delta = cb_avg - bl_avg
    comparison.append({
        "qid": qid, "category": q["category"],
        "bl_scores": bl_scores, "cb_scores": cb_scores_q,
        "bl_avg": bl_avg, "cb_avg": cb_avg, "delta": delta,
        "bl_r1_len": len(bl_r["response_1"]),
        "cb_r1_len": len(cb_r["response_1"]),
    })

    if (qi + 1) % 10 == 0:
        print(f"  Judged {qi+1}/80")

# ── Phase 3: Analysis ──
print("\n" + "=" * 80)
print("RESULTS: Baseline vs CB Llama3 (same judge, same session)")
print("=" * 80)

bl_overall = sum(bl_scores_all) / len(bl_scores_all)
cb_overall = sum(cb_scores_all) / len(cb_scores_all)
print(f"\nOVERALL: Baseline={bl_overall:.2f}  CB={cb_overall:.2f}  Delta={cb_overall-bl_overall:+.2f}")

# Per category
cats = {}
for c in comparison:
    cat = c["category"]
    cats.setdefault(cat, {"bl": [], "cb": []})
    cats[cat]["bl"].append(c["bl_avg"])
    cats[cat]["cb"].append(c["cb_avg"])

print(f"\n{'Category':<15} {'BL':>6} {'CB':>6} {'Delta':>7} {'N':>4}")
print("-" * 42)
for cat in sorted(cats.keys()):
    bl_cat = sum(cats[cat]["bl"]) / len(cats[cat]["bl"])
    cb_cat = sum(cats[cat]["cb"]) / len(cats[cat]["cb"])
    n = len(cats[cat]["bl"])
    print(f"{cat:<15} {bl_cat:>6.2f} {cb_cat:>6.2f} {cb_cat-bl_cat:>+7.2f} {n:>4}")

# Length comparison
bl_lens = [c["bl_r1_len"] for c in comparison]
cb_lens = [c["cb_r1_len"] for c in comparison]
print(f"\nResponse length (T1): BL avg={sum(bl_lens)/len(bl_lens):.0f}  CB avg={sum(cb_lens)/len(cb_lens):.0f}")

# Biggest deltas
print("\nBiggest CB advantages (CB >> BL):")
for c in sorted(comparison, key=lambda x: x["delta"], reverse=True)[:5]:
    q = q_by_id[c["qid"]]
    print(f"  Q{c['qid']} [{c['category']}] BL={c['bl_avg']:.0f} CB={c['cb_avg']:.0f} ({c['delta']:+.0f}): {q['turns'][0][:60]}")

print("\nBiggest BL advantages (BL >> CB):")
for c in sorted(comparison, key=lambda x: x["delta"])[:5]:
    q = q_by_id[c["qid"]]
    print(f"  Q{c['qid']} [{c['category']}] BL={c['bl_avg']:.0f} CB={c['cb_avg']:.0f} ({c['delta']:+.0f}): {q['turns'][0][:60]}")

# Save full comparison
with open("7b_defense_wildguard_outputs/cb_llama3_mtbench_investigation.json", "w") as f:
    json.dump({
        "baseline_overall": bl_overall,
        "cb_overall": cb_overall,
        "per_question": comparison,
    }, f, indent=2)
print("\n[+] Full comparison saved to cb_llama3_mtbench_investigation.json")
PYEOF

echo ""
echo "[+] Done"
