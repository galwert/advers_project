#!/usr/bin/env python3
"""
Geometric Similarity Matrix for Cross-Model GCG Transferability Analysis (Phase 1)

Extracts hidden-state embeddings from 20 LLMs on clean (WikiText) and harmful
(AdvBench) prompts, then computes pairwise geometric similarity matrices.
These matrices are later correlated with cross-model GCG attack success rates
to test the hypothesis that representational similarity predicts adversarial
transferability.

Metrics computed (same-dimension models only):
  - Direct cosine similarity of refusal directions (no alignment)
  - CKA (Centered Kernel Alignment) - rotation-invariant structural similarity
  - Procrustes-aligned cosine similarity - optimal orthogonal alignment

Pipeline:
    # Step 1: Extract embeddings for all models (GPU, ~2h for 20 models)
    python geometric_similarity_matrix.py --extract-all

    # Step 2: Compute 20x20 similarity matrices from cached embeddings
    python geometric_similarity_matrix.py --compute-matrix --layer 16

    # Step 3: Correlate with ASR matrix
    python geometric_similarity_matrix.py --correlate --asr-path ../outputs/asr_matrix.csv

    # Full pipeline
    python geometric_similarity_matrix.py --extract-all --compute-matrix --correlate --asr-path ../outputs/asr_matrix.csv

Note:
    This is the v1 script that requires same hidden dimensions for Procrustes
    and direct cosine metrics. See robust_geometry_matrix_v2.py for the
    dimension-agnostic v2 that handles all model pairs.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from tqdm import tqdm
import argparse
import io
from scipy.linalg import orthogonal_procrustes
from scipy.stats import pearsonr, spearmanr
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer
import requests
import time
import gc

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

ADVBENCH_URL = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"

# Full model list - index matches GCG CSV model_index
MODELS_LIST = [
    ("llama2", "meta-llama/Llama-2-7b-chat-hf"),           # 0
    ("llama3", "meta-llama/Meta-Llama-3-8B-Instruct"),     # 1
    ("vicuna", "lmsys/vicuna-7b-v1.5"),                    # 2
    ("mistral", "mistralai/Mistral-7B-Instruct-v0.2"),     # 3
    ("zephyr", "HuggingFaceH4/zephyr-7b-beta"),            # 4
    ("hermes2", "NousResearch/Nous-Hermes-2-Mistral-7B-DPO"),  # 5
    ("starling", "berkeley-nest/Starling-LM-7B-alpha"),    # 6
    ("openchat", "openchat/openchat_3.5"),                 # 7
    ("gemma", "google/gemma-7b-it"),                       # 8
    ("phi2", "microsoft/phi-2"),                           # 9
    ("qwen", "Qwen/Qwen1.5-7B-Chat"),                      # 10
    ("yi", "01-ai/Yi-6B-Chat"),                            # 11
    ("baichuan2", "baichuan-inc/Baichuan2-7B-Chat"),       # 12
    ("deepseek", "deepseek-ai/deepseek-llm-7b-chat"),      # 13
    ("internlm2", "internlm/internlm2-chat-7b"),           # 14
    ("falcon", "tiiuae/falcon-7b-instruct"),               # 15
    ("solar", "upstage/SOLAR-10.7B-Instruct-v1.0"),        # 16
    ("orca2", "microsoft/Orca-2-7b"),                      # 17
    ("neuralchat", "Intel/neural-chat-7b-v3-1"),           # 18
    ("stablelm", "stabilityai/stablelm-zephyr-3b"),        # 19
]

MODEL_NAMES = [name for name, _ in MODELS_LIST]
MODEL_PATHS = {name: path for name, path in MODELS_LIST}


@dataclass
class Config:
    num_wikitext: int = 520
    num_harmbench: int = 520
    batch_size: int = 8
    cache_dir: str = "./embeddings_cache"
    output_dir: str = "./geometry_output"
    # Which layer to use for similarity (None = use best layer per pair, or specific int)
    target_layer: Optional[int] = None
    # Layers to analyze (relative to model depth)
    layer_percentiles: List[float] = field(default_factory=lambda: [0.25, 0.5, 0.75])


def load_wikitext(num_samples: int = 520, min_length: int = 50) -> List[str]:
    """Load WikiText examples"""
    print(f"Loading {num_samples} WikiText examples...")
    dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")
    texts = []
    for item in dataset:
        text = item['text'].strip()
        if len(text) >= min_length and not text.startswith('='):
            texts.append(text)
        if len(texts) >= num_samples:
            break
    return texts[:num_samples]


def load_harmbench(num_samples: int = 520) -> List[str]:
    """Load HarmBench/AdvBench prompts"""
    print(f"Loading {num_samples} HarmBench examples...")
    try:
        response = requests.get(ADVBENCH_URL)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to download: HTTP {response.status_code}")
        df = pd.read_csv(io.StringIO(response.text))
        df = df.rename(columns={"goal": "prompt"})
        return df.head(num_samples)['prompt'].astype(str).tolist()
    except Exception as e:
        print(f"Failed to download AdvBench: {repr(e)}")
        return []


class EmbeddingExtractor:
    """Extracts hidden state embeddings from transformer models"""

    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

        if hasattr(model.config, 'num_hidden_layers'):
            self.num_layers = model.config.num_hidden_layers
        else:
            self.num_layers = len(model.model.layers)

        if hasattr(model.config, 'hidden_size'):
            self.hidden_dim = model.config.hidden_size
        else:
            self.hidden_dim = None

    @torch.no_grad()
    def extract_embeddings(
            self,
            texts: List[str],
            layers: Optional[List[int]] = None,
            batch_size: int = 8,
            position: str = "last"
    ) -> Dict[int, torch.Tensor]:
        """Extract embeddings from specified layers"""
        texts = [str(t) if not isinstance(t, str) else t for t in texts]

        if layers is None:
            layers = list(range(self.num_layers))

        embeddings = {layer: [] for layer in layers}

        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting", leave=False):
            batch_texts = texts[i:i + batch_size]

            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)

            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )

            for layer_idx in layers:
                hidden_state = outputs.hidden_states[layer_idx + 1]

                if position == "last":
                    attention_mask = inputs['attention_mask']
                    seq_lengths = attention_mask.sum(dim=1) - 1
                    batch_embeds = hidden_state[torch.arange(hidden_state.size(0)), seq_lengths]
                elif position == "first":
                    batch_embeds = hidden_state[:, 0, :]
                else:  # mean
                    attention_mask = inputs['attention_mask'].unsqueeze(-1)
                    batch_embeds = (hidden_state * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)

                embeddings[layer_idx].append(batch_embeds.cpu())

        for layer_idx in layers:
            embeddings[layer_idx] = torch.cat(embeddings[layer_idx], dim=0)

        return embeddings


def get_cache_path(model_name: str, dataset_name: str, cache_dir: str) -> Path:
    """Get standardized cache path"""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path / f"{model_name}_{dataset_name}_embeddings.pt"


def extract_all_models(config: Config, models_to_extract: Optional[List[str]] = None):
    """Extract embeddings for all (or specified) models"""
    cache_dir = Path(config.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets once
    wikitext = load_wikitext(config.num_wikitext)
    harmbench = load_harmbench(config.num_harmbench)
    datasets = {'clean': wikitext, 'harm': harmbench}

    if models_to_extract is None:
        models_to_extract = MODEL_NAMES

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    for model_name in models_to_extract:
        # Check if already cached
        all_cached = all(
            get_cache_path(model_name, ds, config.cache_dir).exists()
            for ds in datasets.keys()
        )
        if all_cached:
            print(f"[{model_name}] Already cached, skipping...")
            continue

        print(f"\n{'='*60}")
        print(f"Extracting embeddings for: {model_name}")
        print(f"{'='*60}")

        model_path = MODEL_PATHS[model_name]

        try:
            # Load model with retries
            max_retries = 3
            model = None
            tokenizer = None

            for attempt in range(max_retries):
                try:
                    print(f"  Loading model (attempt {attempt + 1}/{max_retries})...")
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        trust_remote_code=True
                    )
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_path,
                        trust_remote_code=True,
                        timeout=60  # Increase timeout
                    )
                    break
                except Exception as load_error:
                    print(f"  Attempt {attempt + 1} failed: {load_error}")
                    if attempt < max_retries - 1:
                        time.sleep(5)  # Wait before retry
                    else:
                        raise load_error

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            extractor = EmbeddingExtractor(model, tokenizer, device)
            print(f"  Layers: {extractor.num_layers}, Hidden dim: {extractor.hidden_dim}")

            for ds_name, texts in datasets.items():
                print(f"  Extracting {ds_name}...")
                embeddings = extractor.extract_embeddings(
                    texts,
                    layers=None,  # All layers
                    batch_size=config.batch_size
                )
                cache_path = get_cache_path(model_name, ds_name, config.cache_dir)
                torch.save(embeddings, cache_path)
                print(f"  Saved to {cache_path}")

            # Free memory
            del model, tokenizer, extractor
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"  ERROR: Failed to process {model_name}: {e}")
            # Clean up partial downloads
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

    print("\nExtraction complete!")


def compute_refusal_direction(clean_embeds: torch.Tensor, harm_embeds: torch.Tensor) -> torch.Tensor:
    """Compute normalized refusal direction"""
    direction = harm_embeds.mean(dim=0) - clean_embeds.mean(dim=0)
    return direction / (torch.norm(direction) + 1e-8)


def compute_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Compute CKA similarity (rotation-invariant)"""
    X = X.float()
    Y = Y.float()

    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    K = X @ X.T
    L = Y @ Y.T

    n = K.shape[0]
    H = torch.eye(n) - torch.ones(n, n) / n
    K_centered = H @ K @ H
    L_centered = H @ L @ H

    hsic_kl = (K_centered * L_centered).sum()
    hsic_kk = (K_centered * K_centered).sum()
    hsic_ll = (L_centered * L_centered).sum()

    cka = hsic_kl / (torch.sqrt(hsic_kk * hsic_ll) + 1e-10)
    return cka.item()


def compute_procrustes_similarity(
    source_embeds: torch.Tensor,
    target_embeds: torch.Tensor,
    source_direction: torch.Tensor,
    target_direction: torch.Tensor
) -> float:
    """Compute Procrustes-aligned cosine similarity"""
    X = source_embeds.cpu().numpy().astype(np.float64)
    Y = target_embeds.cpu().numpy().astype(np.float64)

    # Subsample for stability
    if X.shape[0] > 256:
        np.random.seed(42)
        idx = np.random.choice(X.shape[0], 256, replace=False)
        X, Y = X[idx], Y[idx]

    # Center and normalize
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    X = X / (np.linalg.norm(X, 'fro') + 1e-10)
    Y = Y / (np.linalg.norm(Y, 'fro') + 1e-10)

    try:
        R, _ = orthogonal_procrustes(X, Y)
        R_tensor = torch.tensor(R, dtype=source_direction.dtype)
        transferred = R_tensor @ source_direction
        similarity = torch.dot(transferred, target_direction).item()
    except (np.linalg.LinAlgError, ValueError):
        similarity = 0.0

    return similarity


def compute_pairwise_similarity(
    model_a: str,
    model_b: str,
    cache_dir: str,
    layer_idx: int
) -> Dict[str, float]:
    """Compute all similarity metrics between two models at a specific layer"""
    # Load embeddings
    emb_a_clean = torch.load(get_cache_path(model_a, 'clean', cache_dir))
    emb_a_harm = torch.load(get_cache_path(model_a, 'harm', cache_dir))
    emb_b_clean = torch.load(get_cache_path(model_b, 'clean', cache_dir))
    emb_b_harm = torch.load(get_cache_path(model_b, 'harm', cache_dir))

    # Get embeddings at specified layer
    # Handle different layer counts by using relative position
    layers_a = sorted(emb_a_clean.keys())
    layers_b = sorted(emb_b_clean.keys())

    # Use the requested layer if available, otherwise map proportionally
    if layer_idx in layers_a and layer_idx in layers_b:
        layer_a, layer_b = layer_idx, layer_idx
    else:
        # Map to equivalent relative position
        rel_pos = layer_idx / max(layers_a) if layers_a else 0.5
        layer_a = layers_a[int(rel_pos * (len(layers_a) - 1))]
        layer_b = layers_b[int(rel_pos * (len(layers_b) - 1))]

    clean_a = emb_a_clean[layer_a]
    harm_a = emb_a_harm[layer_a]
    clean_b = emb_b_clean[layer_b]
    harm_b = emb_b_harm[layer_b]

    # Compute refusal directions
    dir_a = compute_refusal_direction(clean_a, harm_a)
    dir_b = compute_refusal_direction(clean_b, harm_b)

    # Handle dimension mismatch (different model sizes)
    if dir_a.shape[0] != dir_b.shape[0]:
        # Can't directly compare, return NaN
        return {
            'direct_cosine': float('nan'),
            'cka_clean': float('nan'),
            'cka_harm': float('nan'),
            'procrustes_cosine': float('nan'),
        }

    # 1. Direct cosine similarity (no alignment)
    direct_cosine = torch.dot(dir_a, dir_b).item()

    # 2. CKA on clean and harm embeddings
    cka_clean = compute_cka(clean_a, clean_b)
    cka_harm = compute_cka(harm_a, harm_b)

    # 3. Procrustes-aligned cosine similarity
    procrustes_cosine = compute_procrustes_similarity(clean_a, clean_b, dir_a, dir_b)

    return {
        'direct_cosine': direct_cosine,
        'cka_clean': cka_clean,
        'cka_harm': cka_harm,
        'procrustes_cosine': procrustes_cosine,
    }


def compute_similarity_matrix(config: Config, layer_idx: int = 16) -> Dict[str, pd.DataFrame]:
    """Compute 20x20 similarity matrices for all metrics"""
    cache_dir = config.cache_dir
    n_models = len(MODEL_NAMES)

    # Check which models have cached embeddings
    available_models = []
    for name in MODEL_NAMES:
        if (get_cache_path(name, 'clean', cache_dir).exists() and
            get_cache_path(name, 'harm', cache_dir).exists()):
            available_models.append(name)
        else:
            print(f"Warning: {name} not cached, skipping...")

    n_available = len(available_models)
    print(f"\nComputing similarity matrix for {n_available} models at layer {layer_idx}")

    # Initialize matrices
    metrics = ['direct_cosine', 'cka_clean', 'cka_harm', 'procrustes_cosine']
    matrices = {m: np.zeros((n_available, n_available)) for m in metrics}

    # Compute pairwise similarities
    for i, model_a in enumerate(tqdm(available_models, desc="Computing similarities")):
        for j, model_b in enumerate(available_models):
            if i == j:
                # Self-similarity = 1.0
                for m in metrics:
                    matrices[m][i, j] = 1.0
            elif j > i:
                # Compute similarity (symmetric)
                try:
                    sims = compute_pairwise_similarity(model_a, model_b, cache_dir, layer_idx)
                    for m in metrics:
                        matrices[m][i, j] = sims[m]
                        matrices[m][j, i] = sims[m]
                except Exception as e:
                    print(f"  Error computing {model_a} vs {model_b}: {e}")
                    for m in metrics:
                        matrices[m][i, j] = float('nan')
                        matrices[m][j, i] = float('nan')

    # Convert to DataFrames
    result = {}
    for m in metrics:
        df = pd.DataFrame(matrices[m], index=available_models, columns=available_models)
        result[m] = df

    return result


def save_matrices(matrices: Dict[str, pd.DataFrame], output_dir: str, layer_idx: int):
    """Save similarity matrices to CSV and plot heatmaps"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for metric_name, df in matrices.items():
        # Save CSV
        csv_path = output_path / f"similarity_{metric_name}_layer{layer_idx}.csv"
        df.to_csv(csv_path)
        print(f"Saved {csv_path}")

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(14, 12))
        mask = np.isnan(df.values)
        sns.heatmap(
            df, annot=True, fmt='.2f', cmap='RdYlGn',
            mask=mask, ax=ax, vmin=-1, vmax=1,
            annot_kws={'size': 7}
        )
        ax.set_title(f'Geometric Similarity: {metric_name} (Layer {layer_idx})')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        fig_path = output_path / f"heatmap_{metric_name}_layer{layer_idx}.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved {fig_path}")


def correlate_with_asr(
    similarity_matrices: Dict[str, pd.DataFrame],
    asr_path: str,
    output_dir: str
):
    """Correlate geometric similarity with ASR matrix"""
    # Load ASR matrix
    asr_df = pd.read_csv(asr_path, index_col=0)
    print(f"Loaded ASR matrix: {asr_df.shape}")

    output_path = Path(output_dir)
    results = []

    for metric_name, sim_df in similarity_matrices.items():
        # Align indices
        common_models = list(set(sim_df.index) & set(asr_df.index))
        if len(common_models) < 3:
            print(f"Not enough common models for {metric_name}")
            continue

        sim_aligned = sim_df.loc[common_models, common_models]
        asr_aligned = asr_df.loc[common_models, common_models]

        # Extract upper triangle (excluding diagonal)
        mask = np.triu(np.ones_like(sim_aligned, dtype=bool), k=1)
        sim_values = sim_aligned.values[mask]
        asr_values = asr_aligned.values[mask]

        # Remove NaN pairs
        valid_mask = ~(np.isnan(sim_values) | np.isnan(asr_values))
        sim_values = sim_values[valid_mask]
        asr_values = asr_values[valid_mask]

        if len(sim_values) < 3:
            continue

        # Compute correlations
        pearson_r, pearson_p = pearsonr(sim_values, asr_values)
        spearman_r, spearman_p = spearmanr(sim_values, asr_values)

        results.append({
            'metric': metric_name,
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'n_pairs': len(sim_values)
        })

        print(f"\n{metric_name}:")
        print(f"  Pearson:  r={pearson_r:.4f}, p={pearson_p:.4f}")
        print(f"  Spearman: r={spearman_r:.4f}, p={spearman_p:.4f}")

        # Scatter plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(sim_values, asr_values, alpha=0.6)
        ax.set_xlabel(f'Geometric Similarity ({metric_name})')
        ax.set_ylabel('Attack Success Rate (ASR)')
        ax.set_title(f'{metric_name} vs ASR\nPearson r={pearson_r:.3f}, Spearman r={spearman_r:.3f}')

        # Add trend line
        z = np.polyfit(sim_values, asr_values, 1)
        p = np.poly1d(z)
        x_line = np.linspace(sim_values.min(), sim_values.max(), 100)
        ax.plot(x_line, p(x_line), 'r--', alpha=0.8)

        plt.tight_layout()
        fig_path = output_path / f"correlation_{metric_name}.png"
        plt.savefig(fig_path, dpi=150)
        plt.close()

    # Save correlation results
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_path / "correlation_results.csv", index=False)
        print(f"\nSaved correlation results to {output_path / 'correlation_results.csv'}")

        # Print summary
        print("\n" + "="*60)
        print("CORRELATION SUMMARY")
        print("="*60)
        print(results_df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description='Compute geometric similarity matrix for GCG transferability',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--extract-all', action='store_true',
                        help='Extract embeddings for all 20 models')
    parser.add_argument('--extract-models', nargs='+', choices=MODEL_NAMES,
                        help='Extract embeddings for specific models only')
    parser.add_argument('--compute-matrix', action='store_true',
                        help='Compute 20x20 similarity matrix')
    parser.add_argument('--correlate', action='store_true',
                        help='Correlate with ASR matrix')
    parser.add_argument('--asr-path', type=str, default='../outputs/asr_matrix.csv',
                        help='Path to ASR matrix CSV')
    parser.add_argument('--layer', type=int, default=16,
                        help='Layer index to use for similarity computation')
    parser.add_argument('--cache-dir', type=str, default='./embeddings_cache')
    parser.add_argument('--output-dir', type=str, default='./geometry_output')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--list-cached', action='store_true',
                        help='List which models have cached embeddings')

    args = parser.parse_args()

    config = Config(
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size
    )

    # List cached models
    if args.list_cached:
        print("\nCached models:")
        for name in MODEL_NAMES:
            clean_exists = get_cache_path(name, 'clean', args.cache_dir).exists()
            harm_exists = get_cache_path(name, 'harm', args.cache_dir).exists()
            status = "OK" if (clean_exists and harm_exists) else "MISSING"
            print(f"  {name}: {status}")
        return

    # Extract embeddings
    if args.extract_all:
        extract_all_models(config)
    elif args.extract_models:
        extract_all_models(config, args.extract_models)

    # Compute similarity matrix
    similarity_matrices = None
    if args.compute_matrix:
        similarity_matrices = compute_similarity_matrix(config, layer_idx=args.layer)
        save_matrices(similarity_matrices, args.output_dir, args.layer)

    # Correlate with ASR
    if args.correlate:
        if similarity_matrices is None:
            # Load from saved files
            output_path = Path(args.output_dir)
            similarity_matrices = {}
            for metric in ['direct_cosine', 'cka_clean', 'cka_harm', 'procrustes_cosine']:
                csv_path = output_path / f"similarity_{metric}_layer{args.layer}.csv"
                if csv_path.exists():
                    similarity_matrices[metric] = pd.read_csv(csv_path, index_col=0)

        if similarity_matrices:
            correlate_with_asr(similarity_matrices, args.asr_path, args.output_dir)
        else:
            print("No similarity matrices found. Run --compute-matrix first.")


if __name__ == "__main__":
    main()
