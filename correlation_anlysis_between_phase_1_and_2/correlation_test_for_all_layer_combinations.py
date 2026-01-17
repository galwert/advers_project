import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np


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
        dfs.append([name1, name2, df])

    return dfs


def find_best_correlation(models_correlations_dfs, success_attack_rate_df):
    # Find the minimum dimensions across all matrices
    min_rows = min(df.shape[0] for _, _, df in models_correlations_dfs)
    min_cols = min(df.shape[1] for _, _, df in models_correlations_dfs)

    print(f"Searching over {min_rows} rows and {min_cols} columns...")

    best_corr = 0
    best_position = (0, 0)
    best_pval = 1.0

    correlations_grid = np.zeros((min_rows, min_cols))

    for row_idx in range(min_rows):
        for col_idx in range(min_cols):
            similarity_scores = []
            attack_rates = []

            for name1, name2, df in models_correlations_dfs:
                similarity_score = df.iloc[row_idx, col_idx]

                # Get attack success rate for this pair
                mask = ((success_attack_rate_df['source_model'] == name1) &
                        (success_attack_rate_df[name2].notna()))
                if mask.any():
                    attack_rate = success_attack_rate_df.loc[mask, name2].values[0]
                    similarity_scores.append(similarity_score)
                    attack_rates.append(attack_rate)

            # Calculate correlation
            if len(similarity_scores) > 2:
                corr, pval = pearsonr(similarity_scores, attack_rates)
                correlations_grid[row_idx, col_idx] = abs(corr)

                if abs(corr) > abs(best_corr):
                    best_corr = corr
                    best_position = (row_idx, col_idx)
                    best_pval = pval

        if (row_idx + 1) % 5 == 0:
            print(f"Processed {row_idx + 1}/{min_rows} rows...")

    print(f"\nBest correlation: {best_corr:.4f} (p={best_pval:.3e})")
    print(f"Position: row={best_position[0]}, col={best_position[1]}")

    # Plot heatmap of correlations
    plt.figure(figsize=(12, 8))
    plt.imshow(correlations_grid, aspect='auto', cmap='viridis')
    plt.colorbar(label='|Correlation|')
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')
    plt.title('Absolute Correlation at Each Matrix Position')
    plt.plot(best_position[1], best_position[0], 'r*', markersize=20, label='Best')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return best_corr, best_position, best_pval


def plot_correlation_at_position(models_correlations_dfs, success_attack_rate_df, row_idx, col_idx):
    similarity_scores = []
    attack_rates = []
    labels = []

    for name1, name2, df in models_correlations_dfs:
        similarity_score = df.iloc[row_idx, col_idx]

        # Get attack success rate for this pair
        mask = ((success_attack_rate_df['source_model'] == name1) &
                (success_attack_rate_df[name2].notna()))
        if mask.any():
            attack_rate = success_attack_rate_df.loc[mask, name2].values[0]
            similarity_scores.append(similarity_score)
            attack_rates.append(attack_rate)
            labels.append(f"{name1}->{name2}")

    # Calculate correlation
    corr, pval = pearsonr(similarity_scores, attack_rates)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(similarity_scores, attack_rates, alpha=0.6, s=50)
    plt.xlabel(f'Similarity Score at (row={row_idx}, col={col_idx})')
    plt.ylabel('Attack Success Rate (%)')
    plt.title(f'Similarity vs Attack Transferability\nCorrelation: {corr:.3f} (p={pval:.3e})')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print(f"Number of model pairs: {len(similarity_scores)}")
    print(f"Correlation: {corr:.3f}, p-value: {pval:.3e}")

    return corr, pval


def main():
    csv_folder_path = r"C:\Users\perez\OneDrive - Technion\masters\semester_A_2025 (last one!!)\adversarial attacks on LLMs\alignment_results"
    models_correlations_dfs = calc_correlation(csv_folder_path)

    success_attack_rate_path = r"C:\Users\perez\OneDrive - Technion\masters\semester_A_2025 (last one!!)\adversarial attacks on LLMs\phase2_final_matrix.csv"
    success_attack_rate_df = pd.read_csv(success_attack_rate_path)

    # Find best position
    best_corr, best_position, best_pval = find_best_correlation(models_correlations_dfs, success_attack_rate_df)

    # Plot the best correlation
    print("\nPlotting best correlation...")
    plot_correlation_at_position(models_correlations_dfs, success_attack_rate_df,
                                 best_position[0], best_position[1])


if __name__ == '__main__':
    main()