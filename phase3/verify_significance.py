import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from io import StringIO
import math

# ================= DATA LOADING =================
asr_csv_data = """source_model,Baichuan2-7b,DeepSeek-7b,Falcon-7b,Gemma-7b,Hermes-2,InternLM2-7b,Llama2-7b,Llama3-8b,Mistral-7b,NeuralChat-7b,OpenChat-3.5,Orca-2-7b,Phi-2,Qwen1.5-7b,Solar-10.7b,StableZephyr-3b,Starling-7b,Vicuna-7b,Yi-6b,Zephyr-7b
Baichuan2-7b,83.0,87.0,95.0,17.0,90.0,90.0,100.0,83.0,93.0,93.0,93.0,95.0,89.0,57.0,92.0,84.0,95.0,90.6,69.0,96.0
DeepSeek-7b,72.0,79.0,89.0,44.0,86.0,84.0,66.7,90.0,94.0,82.0,90.0,85.0,85.0,67.0,82.0,78.0,93.0,82.0,63.0,78.0
Falcon-7b,73.0,77.0,71.0,34.0,86.0,70.0,41.2,77.0,91.0,78.0,86.0,84.0,81.0,68.0,88.0,70.0,90.0,75.0,71.0,78.0
Gemma-7b,81.0,82.0,90.0,67.0,88.0,87.0,45.0,94.0,86.0,87.0,91.0,91.0,88.0,76.0,87.0,77.0,91.0,84.0,73.0,81.0
Hermes-2,75.0,82.5,89.2,34.2,86.7,89.2,51.0,93.3,83.3,79.2,85.8,94.2,87.5,72.5,87.5,77.5,84.2,90.0,75.0,77.5
InternLM2-7b,79.0,78.0,82.0,51.0,86.0,87.0,58.0,81.0,88.0,90.0,87.0,89.0,84.0,77.0,79.0,75.0,87.0,85.0,79.0,79.0
Llama2-7b,70.0,86.0,87.0,36.0,88.0,89.0,64.0,87.0,89.0,86.0,90.0,86.0,87.0,76.0,87.0,77.0,86.0,82.0,69.0,74.0
Llama3-8b,72.0,74.0,76.0,31.0,83.0,79.0,35.0,83.0,82.0,83.0,81.0,80.0,83.0,71.0,78.0,68.0,89.0,81.0,61.0,84.0
Mistral-7b,82.0,84.0,85.0,54.0,95.0,82.0,43.0,86.0,92.0,91.0,95.0,94.0,90.0,80.0,90.0,85.0,89.0,83.0,68.0,78.0
NeuralChat-7b,77.0,75.0,73.0,40.0,86.0,81.0,50.0,81.0,90.0,93.0,85.0,89.0,89.0,83.0,80.0,75.0,91.0,81.0,60.0,81.0
Orca-2-7b,79.0,93.0,90.0,26.0,97.0,93.0,100.0,93.0,90.0,93.0,98.0,96.0,91.0,61.0,95.0,82.0,94.0,93.8,69.0,94.0
Phi-2,55.0,60.0,72.0,38.0,80.0,49.0,23.0,63.0,66.0,89.0,82.0,88.0,93.0,48.0,70.0,73.0,85.0,73.0,37.0,80.0
Qwen1.5-7b,72.0,71.0,83.0,39.0,79.0,77.0,53.8,74.0,85.0,81.0,89.0,78.0,75.0,86.0,77.0,71.0,83.0,77.5,68.0,77.0
Solar-10.7b,79.0,83.0,94.0,38.0,96.0,93.0,63.3,89.0,93.0,92.0,96.0,91.0,87.0,63.0,94.0,80.0,93.0,85.0,78.0,82.0
StableZephyr-3b,87.0,79.0,87.0,41.0,85.0,85.0,52.0,83.0,89.0,82.0,92.0,89.0,84.0,57.0,80.0,83.0,90.0,88.0,62.0,86.0
Vicuna-7b,84.0,85.0,94.0,27.0,95.0,90.0,62.0,82.0,86.0,92.0,94.0,92.0,90.0,54.0,89.0,80.0,94.0,98.0,77.0,97.0
Yi-6b,80.0,77.0,86.0,33.0,82.0,78.0,47.5,86.0,87.0,80.0,78.0,88.0,80.0,65.0,85.0,74.0,79.0,75.0,68.0,70.0
Zephyr-7b,73.0,85.0,87.0,36.0,85.0,88.0,47.0,88.0,86.0,80.0,87.0,89.0,89.0,64.0,86.0,78.0,83.0,81.0,69.0,78.0"""

SIM_MATRIX_PATH = "../phase1/grand_geometry_matrix.csv"


def analyze_individual_models():
    # Load DataFrames
    asr_df = pd.read_csv(StringIO(asr_csv_data), index_col="source_model")
    try:
        sim_df = pd.read_csv(SIM_MATRIX_PATH, index_col=0)
    except FileNotFoundError:
        print(f"[-] Error: Could not find {SIM_MATRIX_PATH}")
        return

    # Helper
    def normalize(name):
        return name.lower().replace("-", "").replace("_", "").replace(".", "")

    sim_map = {normalize(k): k for k in sim_df.index}
    models = asr_df.index.tolist()

    # Setup Grid Plot (5 rows x 4 columns = 20 models)
    num_models = len(models)
    cols = 4
    rows = math.ceil(num_models / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(20, 24))
    axes = axes.flatten()

    print("[*] Generating individual model plots...")

    for i, current_model in enumerate(models):
        ax = axes[i]

        # 1. OUTGOING: Current Model -> Attacks Others (Blue)
        outgoing_x = []
        outgoing_y = []

        # 2. INCOMING: Others -> Attack Current Model (Red)
        incoming_x = []
        incoming_y = []

        # Collect Data
        for other_model in models:
            if current_model == other_model: continue

            # Key matching
            cm_key = normalize(current_model)
            om_key = normalize(other_model)

            if cm_key not in sim_map or om_key not in sim_map: continue

            real_cm = sim_map[cm_key]
            real_om = sim_map[om_key]
            sim_val = sim_df.loc[real_cm, real_om]

            # Outgoing Data (X = Sim, Y = ASR where Source is Current)
            try:
                out_asr = asr_df.loc[current_model, other_model]
                outgoing_x.append(sim_val)
                outgoing_y.append(out_asr)
            except:
                pass

            # Incoming Data (X = Sim, Y = ASR where Target is Current)
            try:
                in_asr = asr_df.loc[other_model, current_model]
                incoming_x.append(sim_val)
                incoming_y.append(in_asr)
            except:
                pass

        # Plot Scatter
        ax.scatter(outgoing_x, outgoing_y, color='#3498db', label='Outgoing (Source)', alpha=0.7, s=40)
        ax.scatter(incoming_x, incoming_y, color='#e74c3c', label='Incoming (Target)', alpha=0.7, s=40)

        # Optional: Add regression lines if enough points
        if len(outgoing_x) > 2:
            m, b = np.polyfit(outgoing_x, outgoing_y, 1)
            ax.plot(outgoing_x, m * np.array(outgoing_x) + b, color='#3498db', alpha=0.3, linestyle='--')

        if len(incoming_x) > 2:
            m, b = np.polyfit(incoming_x, incoming_y, 1)
            ax.plot(incoming_x, m * np.array(incoming_x) + b, color='#e74c3c', alpha=0.3, linestyle='--')

        ax.set_title(current_model, fontsize=10, fontweight='bold')
        ax.set_ylim(0, 105)
        ax.set_xlim(0.5, 1.0)
        ax.grid(True, alpha=0.3)

        # Only show labels on edges to clean up
        if i % cols == 0: ax.set_ylabel("ASR (%)")
        if i >= (rows - 1) * cols: ax.set_xlabel("Similarity")

    # Legend on the first plot or outside
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=14)

    plt.tight_layout(rect=[0, 0.0, 1, 0.96])  # Adjust for legend space
    plt.savefig("individual_model_analysis.png")
    print("[+] Saved 'individual_model_analysis.png'")
    plt.show()


if __name__ == "__main__":
    analyze_individual_models()