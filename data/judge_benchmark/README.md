# Judge Benchmark Data

Calibration set used to derive `tab:judge_standalone` and `tab:judge_pipeline`
in the paper appendix (App. `app:judge_benchmark`). 500 prompt-response pairs
sampled from WildGuardTest + GCG-garbled outputs:

| Bucket    | Count | Source                                                |
|-----------|-------|-------------------------------------------------------|
| Unsafe    | 200   | `allenai/wildguardmix` `wildguardtest`, `response_harm_label == "harmful"` |
| Refusal   | 200   | `allenai/wildguardmix` `wildguardtest`, `response_harm_label != "harmful"` |
| Garbled   | 100   | GCG outputs flagged as gibberish/degenerate by upstream filter |

Sampling is deterministic: `SEED = 42` for the WildGuardTest pool,
`SEED + 1 = 43` for garbled, `SEED + 2 = 44` for the final shuffle.
The full sampling code lives in `phase2/judge_benchmark_full.py:81-138` of
the development tree (not part of this repo because it depends on
unreleased intermediate scoring artifacts).

## Files

- `full_benchmark_results.csv` (500 rows) — per-sample predictions from every
  evaluated system. Columns: `idx`, `source`, `ground_truth`, `prompt_preview`,
  `response_preview`, then one column per system:
    - `heur_v3_pred` — heuristic-only baseline (Stages 0-4 of the pipeline)
    - `llamaguard_pred`, `wildguard_pred`, `harmbench_7b_pred`, `harmbench_13b_pred`
      — standalone neural judges
    - `pipe_v3_lg_pred`, `pipe_v3_wg_pred`, `pipe_v3_hb_pred`, `pipe_v3_hb13_pred`
      — pipeline (Stages 0-4) with each neural judge as the Stage 5 tiebreaker
- `full_benchmark_summary.json` — aggregated TP/FP/TN/FN/precision/recall/F1/accuracy
  for each of the 9 systems above. The values reported in `tab:judge_standalone`
  and `tab:judge_pipeline` come directly from this file. The paper-pick
  configuration is `pipe_v3_wg` (Pipeline + WildGuard) at TP=144, FP=26, TN=274,
  FN=56, precision=0.847, recall=0.720, F1=0.778, accuracy=0.836.

## Reproducing the tables

```python
import pandas as pd, json
df = pd.read_csv("data/judge_benchmark/full_benchmark_results.csv")
df["gt_unsafe"] = (df["ground_truth"] == "unsafe").astype(int)

def metrics(pred_col):
    p = (df[pred_col] == "unsafe").astype(int)
    tp = ((p==1) & (df.gt_unsafe==1)).sum()
    fp = ((p==1) & (df.gt_unsafe==0)).sum()
    tn = ((p==0) & (df.gt_unsafe==0)).sum()
    fn = ((p==0) & (df.gt_unsafe==1)).sum()
    return tp, fp, tn, fn

for col in ["pipe_v3_wg_pred", "pipe_v3_hb_pred", "pipe_v3_hb13_pred", "pipe_v3_lg_pred"]:
    print(col, metrics(col))
```

## License

CC BY 4.0 (see `LICENSE-DATA` at repo root). Note that the response previews
embed text from `allenai/wildguardmix`, which itself is released under ODC-BY
1.0 in upstream form.
