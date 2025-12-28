import requests
import pandas as pd
import io

ADVBENCH_URL = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"
response = requests.get(ADVBENCH_URL)
response.raise_for_status()

# The file is a CSV with header: goal, target
df = pd.read_csv(io.StringIO(response.text))

# Rename 'goal' to 'prompt' to match our logic
df = df.rename(columns={"goal": "prompt"})

n_samples = 50
# Take the first N samples (standard practice for "dev" runs)
subset = df.head(n_samples).to_dict(orient="records")
print(len(df))
print(f"[+] Loaded {len(subset)} behaviors from AdvBench.")
print(f"[+] Loaded {len(subset)} behaviors from AdvBench.")