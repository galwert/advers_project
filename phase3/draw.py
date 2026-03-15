import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the data
# Assuming you save the data you provided into 'comprehensive_results.csv'
# For now, I'll create a dataframe from the raw data string for this script to run standalone
from io import StringIO

csv_data = """Model,Model_ID,PPL_Before,ASR_Self_Before,ASR_Cross_Before,PPL_After,Geo_Cosine_Sim,ASR_Self,ASR_Cross,Status,Error
Llama2-7b,meta-llama/Llama-2-7b-chat-hf,5.53,70.0,65.0,9.32,0.7979,28.0,32.0,,
Llama3-8b,meta-llama/Meta-Llama-3-8B-Instruct,7.16,74.0,69.0,12.32,0.3713,72.0,82.0,,
Vicuna-7b,lmsys/vicuna-7b-v1.5,5.59,88.0,82.0,7.27,0.6719,69.0,76.0,,
Mistral-7b,mistralai/Mistral-7B-Instruct-v0.2,4.93,86.0,80.0,9.21,0.0,28.0,26.0,,
Zephyr-7b,HuggingFaceH4/zephyr-7b-beta,5.06,67.0,74.0,8.25,0.0,39.0,38.0,,
Hermes-2,NousResearch/Nous-Hermes-2-Mistral-7B-DPO,4.89,86.0,85.0,9.22,0.0,21.0,18.0,,
Gemma-7b,google/gemma-7b-it,21.79,50.0,35.0,22.75,0.0,54.0,56.0,,
Phi-2,microsoft/phi-2,8.22,87.0,79.0,17.6,0.8633,42.0,42.0,,
Qwen1.5-7b,Qwen/Qwen1.5-7B-Chat,9.11,74.0,65.0,15.39,0.2854,83.0,86.0,,
Yi-6b,01-ai/Yi-6B-Chat,5.31,81.0,79.0,8.75,0.2179,92.0,91.0,,
Baichuan2-7b,,,,,,,,,Failed,Shape Error
DeepSeek-7b,deepseek-ai/deepseek-llm-7b-chat,6.63,87.0,81.0,8.67,0.1365,51.0,49.0,,
InternLM2-7b,,,,,,,,,Failed,Tensor Size Error
Falcon-7b,,,,,,,,,Failed,Shape Error
Solar-10.7b,upstage/SOLAR-10.7B-Instruct-v1.0,4.31,80.0,71.0,6.9,0.0,16.0,22.0,,
Orca-2-7b,microsoft/Orca-2-7b,6.99,88.0,81.0,8.82,0.3416,94.0,88.0,,
NeuralChat-7b,Intel/neural-chat-7b-v3-1,5.03,80.0,82.0,10.47,0.0,24.0,19.0,,
StableZephyr-3b,stabilityai/stablelm-zephyr-3b,35.49,87.0,81.0,39.9,0.7881,46.0,35.0,,"""

df = pd.read_csv(StringIO(csv_data))

# Filter out failed models
df_clean = df[df['Status'] != 'Failed'].copy()

# Calculate Delta Metrics
df_clean['ASR_Reduction'] = df_clean['ASR_Self_Before'] - df_clean['ASR_Self']
df_clean['PPL_Increase_Pct'] = ((df_clean['PPL_After'] - df_clean['PPL_Before']) / df_clean['PPL_Before']) * 100

# Set plot style
sns.set_theme(style="whitegrid")
plt.figure(figsize=(20, 15))

# --- PLOT 1: ASR Comparison (Before vs After) ---
plt.subplot(2, 2, 1)
# Melt for grouped bar chart
asr_melt = df_clean.melt(id_vars=['Model'], value_vars=['ASR_Self_Before', 'ASR_Self'],
                         var_name='Condition', value_name='ASR (%)')
sns.barplot(data=asr_melt, x='Model', y='ASR (%)', hue='Condition', palette=['#e74c3c', '#2ecc71'])
plt.title('Defense Effectiveness: ASR Reduction (Self)', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 100)
plt.legend(title='Condition', labels=['Before Defense', 'After Defense'])

# --- PLOT 2: Latent Geometry Shift (Cosine Sim) ---
plt.subplot(2, 2, 2)
# Color bars by whether the defense "Worked" (Reduction > 40%)
colors = ['#2ecc71' if x > 40 else '#e74c3c' for x in df_clean['ASR_Reduction']]
sns.barplot(data=df_clean, x='Model', y='Geo_Cosine_Sim', palette=colors)
plt.title('Latent Geometry Shift (Cosine Similarity)\nGreen = Good Defense, Red = Failed Defense', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Cosine Similarity (Pre vs Post Refusal Vector)')

# --- PLOT 3: Safety vs Utility Trade-off ---
plt.subplot(2, 2, 3)
sns.scatterplot(data=df_clean, x='PPL_Increase_Pct', y='ASR_Reduction', s=150, hue='Model', legend=False)

# Annotate points
for i in range(df_clean.shape[0]):
    plt.text(df_clean.PPL_Increase_Pct.iloc[i]+0.5, df_clean.ASR_Reduction.iloc[i],
             df_clean.Model.iloc[i], fontsize=9)

plt.title('Safety vs. Utility Trade-off', fontsize=14, fontweight='bold')
plt.xlabel('Perplexity Increase (%)')
plt.ylabel('ASR Reduction (Percentage Points)')
plt.axhline(y=0, color='black', linestyle='--')
plt.axvline(x=0, color='black', linestyle='--')

# --- PLOT 4: Cross-Model Transferability ---
plt.subplot(2, 2, 4)
cross_melt = df_clean.melt(id_vars=['Model'], value_vars=['ASR_Cross_Before', 'ASR_Cross'],
                           var_name='Condition', value_name='ASR (%)')
sns.barplot(data=cross_melt, x='Model', y='ASR (%)', hue='Condition', palette=['#3498db', '#9b59b6'])
plt.title('Generalization: Cross-Attack Robustness', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 100)

plt.tight_layout()
plt.savefig('defense_analysis_summary.png')
plt.show()