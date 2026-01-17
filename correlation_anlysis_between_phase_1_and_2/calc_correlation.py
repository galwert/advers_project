import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


def calc_correlation(csv_folder_path):
    folder = Path(csv_folder_path)
    csv_files = list(folder.glob("*_matrix.csv"))

    # Extract and split names
    samples = []
    for f in csv_files:
        # Remove "_matrix.csv" suffix
        name = f.stem.replace("_matrix", "")
        # Split by "-"
        name1, name2 = name.split("_")

        samples.append((name1, name2, f))

    # Load into DataFrames with metadata
    dfs = []
    for name1, name2, filepath in samples:
        df = pd.read_csv(filepath, header=None)
        # df['name1'] = name1
        # df['name2'] = name2
        dfs.append([name1, name2, df])

    # combined_df = pd.concat(dfs, ignore_index=True)
    # print(combined_df.head())
    # print("here")
    return dfs


def plot_correlation(models_correlations_dfs, success_attack_rate_df):
    max_values = []
    attack_rates = []
    labels = []

    for name1, name2, df in models_correlations_dfs:
        # max_row_idx = df.shape[0] - 1
        # max_col_idx = df.shape[1] - 1
        # similarity_score = df.iloc[max_row_idx, max_col_idx]

        mid_row = df.shape[0] // 2
        mid_col = df.shape[1] // 2
        similarity_score = df.iloc[mid_row, mid_col]
        similarity_score = df.iloc[12, 12]


        # Get attack success rate for this pair
        mask = ((success_attack_rate_df['source_model'] == name1) &
                (success_attack_rate_df[name2].notna()))
        if mask.any():
            attack_rate = success_attack_rate_df.loc[mask, name2].values[0]
            max_values.append(similarity_score)
            attack_rates.append(attack_rate)
            labels.append(f"{name1}->{name2}")

    # Single correlation for all model pairs
    corr, pval = pearsonr(max_values, attack_rates)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(max_values, attack_rates, alpha=0.6, s=50)
    plt.xlabel('Max Similarity Score')
    plt.ylabel('Attack Success Rate (%)')
    plt.title(f'Similarity vs Attack Transferability\nCorrelation: {corr:.3f} (p={pval:.3e})')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print(f"Number of model pairs: {len(max_values)}")
    print(f"Correlation: {corr:.3f}, p-value: {pval:.3e}")

    return corr, pval

def main():
    csv_folder_path = r"C:\Users\perez\OneDrive - Technion\masters\semester_A_2025 (last one!!)\adversarial attacks on LLMs\alignment_results"
    models_correlations_dfs = calc_correlation(csv_folder_path)

    success_attack_rate_path = r"C:\Users\perez\OneDrive - Technion\masters\semester_A_2025 (last one!!)\adversarial attacks on LLMs\phase2_final_matrix.csv"
    success_attack_rate_df = pd.read_csv(success_attack_rate_path)

    corr, pval = plot_correlation(models_correlations_dfs, success_attack_rate_df)

    print("here")
if __name__ == '__main__':
    main()
