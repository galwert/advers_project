"""Split the 181 new items into batches for Rater-2 subagents.

Emits per-batch JSON blob files (prompt+response only, no R1 verdict, no config).
Each subagent will read one batch file and write a verdicts file.
"""
import json, os

SAMPLE = "/home/wertheizer/advers_project/rebuttal_data/v5_evals_v2/stratified_sample.json"
BATCH_DIR = "/home/wertheizer/advers_project/rebuttal_data/v5_evals_v2/rater2_batches"
BATCH_SIZE = 20

def main():
    os.makedirs(BATCH_DIR, exist_ok=True)
    d = json.load(open(SAMPLE))
    to_rate = [it for it in d["items"] if not it["reused"]]
    print(f"items to rate: {len(to_rate)}")
    batches = []
    for i in range(0, len(to_rate), BATCH_SIZE):
        chunk = to_rate[i:i+BATCH_SIZE]
        batch_id = f"batch_{i//BATCH_SIZE:03d}"
        # Only blind fields: sample_idx (as opaque id), prompt, response
        blind_chunk = [{
            "id": it["sample_idx"],  # opaque id for round-trip
            "prompt": it["prompt"],
            "response": it["response"],
        } for it in chunk]
        path = os.path.join(BATCH_DIR, f"{batch_id}.json")
        with open(path, "w") as f:
            json.dump(blind_chunk, f, indent=2)
        batches.append((batch_id, path, len(chunk)))
    print(f"wrote {len(batches)} batches:")
    for bid, p, n in batches:
        print(f"  {bid}: n={n}  {p}")

if __name__ == "__main__":
    main()
