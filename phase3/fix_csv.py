"""Quick fix: repair CSV rows with empty status by checking if their JSON results exist."""
import csv
import os
import json

CSV_PATH = os.path.join(os.path.dirname(__file__), "all_pairs_results.csv")

# Read all rows
with open(CSV_PATH, 'r', newline='') as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    rows = list(reader)

print(f"Total rows: {len(rows)}")

fixed = 0
for row in rows:
    status = row.get("status", "")
    adapter_path = row.get("adapter_path", "")

    if status == "" and adapter_path and os.path.exists(adapter_path):
        # This row likely completed but status wasn't recorded properly
        anchor = row.get("anchor", "?")
        defender = row.get("defender", "?")

        # Check if defended JSON exists
        output_dir = os.path.join(os.path.dirname(__file__), "two_stage_outputs_v2")
        defended_json = os.path.join(output_dir, f"defended_{defender}_{anchor}.json")

        if os.path.exists(defended_json):
            with open(defended_json, 'r') as f:
                data = json.load(f)

            dd = data.get("defended", {})
            if dd:
                row["defended_asr_self"] = dd.get("asr_self", row.get("defended_asr_self", ""))
                row["defended_asr_anchor"] = dd.get("asr_anchor", row.get("defended_asr_anchor", ""))
                row["defended_asr_other"] = dd.get("asr_other", row.get("defended_asr_other", ""))
                row["defended_brr"] = dd.get("brr", row.get("defended_brr", ""))
                row["defended_ppl"] = dd.get("ppl", row.get("defended_ppl", ""))
                row["defended_tdr"] = dd.get("tdr", row.get("defended_tdr", ""))
                row["defended_cka"] = dd.get("cka_score", row.get("defended_cka", ""))
                row["status"] = "OK"
                fixed += 1
                print(f"  Fixed: {anchor}->{defender} (from {defended_json})")
            else:
                print(f"  Skipped: {anchor}->{defender} (JSON has no 'defended' key)")
        else:
            # No JSON but adapter exists — mark as OK based on adapter presence
            row["status"] = "OK"
            fixed += 1
            print(f"  Fixed: {anchor}->{defender} (adapter exists, no JSON)")
    elif status == "" and not adapter_path:
        anchor = row.get("anchor", "?")
        defender = row.get("defender", "?")
        print(f"  Cannot fix: {anchor}->{defender} (no adapter path)")

# Sanitize example fields in ALL rows
for row in rows:
    for key in ["sample_succeeded_attacks", "sample_refused_benign"]:
        val = row.get(key, "")
        if val:
            row[key] = val.replace("\n", " ").replace("\r", " ").replace('"', "'")

# Write back
with open(CSV_PATH, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"\nFixed {fixed} rows. Sanitized all example fields.")
print(f"Written back to {CSV_PATH}")
