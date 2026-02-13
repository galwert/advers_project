#!/usr/bin/env python3
"""
Refusal Direction Transfer Analysis using Procrustes Alignment

This script:
1. Loads Llama-2 and Vicuna models
2. Extracts embeddings from WikiText (clean) and HarmBench (malicious) prompts
3. Computes refusal directions at each layer
4. Performs Procrustes alignment between model spaces
5. Transfers refusal directions and evaluates effectiveness
"""
import io

import requests
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
import json
from scipy.linalg import orthogonal_procrustes
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

ADVBENCH_URL = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"

@dataclass
class ModelConfig:
    """Configuration for model loading"""
    name: str
    path: str
    max_length: int = 512
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class ExperimentConfig:
    """Configuration for the experiment"""
    num_wikitext: int = 520
    num_harmbench: int = 520
    num_gcg: int = 100  # GCG examples to use (model_index=0 for Llama-2)
    layer_range: Optional[Tuple[int, int]] = None  # None = all layers
    batch_size: int = 8
    use_gcg: bool = True
    gcg_path: str = "../outputs/advbench_suffixes_all_models_fixed.csv"
    output_dir: str = "./refusal_analysis_output"
    save_embeddings: bool = True
    compute_all_metrics: bool = True


class EmbeddingExtractor:
    """Extracts hidden state embeddings from transformer models"""

    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

        # Get number of layers
        if hasattr(model.config, 'num_hidden_layers'):
            self.num_layers = model.config.num_hidden_layers
        else:
            self.num_layers = len(model.model.layers)

    @torch.no_grad()
    def extract_embeddings(
            self,
            texts: List[str],
            layers: Optional[List[int]] = None,
            batch_size: int = 8,
            position: str = "last"  # "last", "first", or "mean"
    ) -> Dict[int, torch.Tensor]:
        """
        Extract embeddings from specified layers

        Args:
            texts: List of input texts
            layers: List of layer indices to extract (None = all layers)
            batch_size: Batch size for processing
            position: Which token position to use ("last", "first", "mean")

        Returns:
            Dictionary mapping layer_idx -> tensor of shape [num_texts, hidden_dim]
        """
        # Ensure texts is a list of strings
        if not isinstance(texts, list):
            raise ValueError(f"texts must be a list, got {type(texts)}")
        for idx, t in enumerate(texts):
            if not isinstance(t, str):
                try:
                    texts[idx] = str(t)
                except Exception:
                    raise ValueError(f"Element at index {idx} in texts is not convertible to string: {t}")
        if layers is None:
            layers = list(range(self.num_layers))

        # Storage for embeddings
        embeddings = {layer: [] for layer in layers}

        # Process in batches
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting embeddings"):
            batch_texts = texts[i:i + batch_size]

            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)

            # Forward pass with output_hidden_states
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )

            # Extract embeddings from each requested layer
            for layer_idx in layers:
                # hidden_states is tuple: (embed_layer, layer_0, ..., layer_n)
                # So layer_idx corresponds to hidden_states[layer_idx + 1]
                hidden_state = outputs.hidden_states[layer_idx + 1]  # [batch, seq_len, hidden_dim]

                # Get embeddings based on position strategy
                if position == "last":
                    # Use last non-padding token
                    attention_mask = inputs['attention_mask']
                    seq_lengths = attention_mask.sum(dim=1) - 1  # -1 for 0-indexing
                    batch_embeds = hidden_state[torch.arange(hidden_state.size(0)), seq_lengths]
                elif position == "first":
                    batch_embeds = hidden_state[:, 0, :]  # First token (usually BOS)
                elif position == "mean":
                    attention_mask = inputs['attention_mask'].unsqueeze(-1)
                    batch_embeds = (hidden_state * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
                else:
                    raise ValueError(f"Unknown position: {position}")

                embeddings[layer_idx].append(batch_embeds.cpu())

        # Concatenate all batches
        for layer_idx in layers:
            embeddings[layer_idx] = torch.cat(embeddings[layer_idx], dim=0)

        return embeddings


def load_wikitext(num_samples: int = 520, min_length: int = 50) -> List[str]:
    """Load WikiText examples"""
    print(f"Loading {num_samples} WikiText examples...")
    dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")

    texts = []
    for item in dataset:
        text = item['text'].strip()
        if len(text) >= min_length and not text.startswith('='):  # Filter headers
            texts.append(text)
        if len(texts) >= num_samples:
            break

    return texts[:num_samples]


def load_harmbench(num_samples: int = 520) -> List[str]:
    """Load HarmBench prompts"""

    print(f"[*] Downloading AdvBench from {ADVBENCH_URL}...")
    try:
        response = requests.get(ADVBENCH_URL)
        print(f"[DEBUG] HTTP status code: {response.status_code}")
        if response.status_code != 200:
            print(f"[-] HTTP error: {response.status_code} - {response.text[:200]}")
            raise RuntimeError(f"Failed to download: HTTP {response.status_code}")
        df = pd.read_csv(io.StringIO(response.text))
        df = df.rename(columns={"goal": "prompt"})
        subset = df.head(num_samples)
        print(f"[+] Loaded {len(subset)} behaviors from AdvBench.")
        # Return just the prompt strings
        return subset['prompt'].astype(str).tolist()
    except Exception as e:
        import traceback
        print(f"[-] Failed to download AdvBench: {repr(e)}")
        traceback.print_exc()
        return []

def load_gcg_prompts(csv_path: str, model_index: int = 0, num_samples: int = 100) -> List[str]:
    """Load GCG-attacked prompts for specific model"""
    print(f"Loading GCG prompts from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Filter for specific model
    model_df = df[df['model_index'] == model_index].head(num_samples)

    # Combine prompt + suffix
    gcg_prompts = []
    for _, row in model_df.iterrows():
        full_prompt = f"{row['prompt']} {row['suffix']}"
        gcg_prompts.append(full_prompt)

    print(f"Loaded {len(gcg_prompts)} GCG prompts")
    return gcg_prompts


def compute_refusal_direction(
        clean_embeds: torch.Tensor,
        malicious_embeds: torch.Tensor,
        method: str = "mean_diff",
        normalize: bool = True
) -> torch.Tensor:
    """
    Compute refusal direction vector

    Args:
        clean_embeds: [num_clean, hidden_dim]
        malicious_embeds: [num_malicious, hidden_dim]
        method: "mean_diff", "pca", or "contrastive"
        normalize: Whether to normalize the direction vector

    Returns:
        Direction vector of shape [hidden_dim]
    """
    if method == "mean_diff":
        # Simple mean difference
        direction = malicious_embeds.mean(dim=0) - clean_embeds.mean(dim=0)

    elif method == "pca":
        # PCA on the difference vectors
        diff_vectors = []
        for mal_embed in malicious_embeds:
            diffs = mal_embed.unsqueeze(0) - clean_embeds
            diff_vectors.append(diffs.mean(dim=0))
        diff_vectors = torch.stack(diff_vectors)

        # Get first principal component
        U, S, V = torch.pca_lowrank(diff_vectors, q=1)
        direction = V[:, 0]

    elif method == "contrastive":
        # Contrastive approach: maximize separation
        all_embeds = torch.cat([clean_embeds, malicious_embeds], dim=0)
        labels = torch.cat([
            torch.zeros(len(clean_embeds)),
            torch.ones(len(malicious_embeds))
        ])

        # Compute between-class scatter direction
        mean_clean = clean_embeds.mean(dim=0)
        mean_mal = malicious_embeds.mean(dim=0)
        direction = mean_mal - mean_clean

    else:
        raise ValueError(f"Unknown method: {method}")

    if normalize:
        direction = direction / (torch.norm(direction) + 1e-8)

    return direction


def compute_procrustes_alignment(
        source_embeds: torch.Tensor,
        target_embeds: torch.Tensor
) -> Tuple[np.ndarray, float]:
    """
    Compute optimal orthogonal Procrustes rotation matrix

    Args:
        source_embeds: [num_anchors, hidden_dim] from source model
        target_embeds: [num_anchors, hidden_dim] from target model

    Returns:
        R: Rotation matrix [hidden_dim, hidden_dim]
        disparity: Frobenius norm of alignment error
    """
    # Convert to numpy
    X = source_embeds.cpu().numpy().astype(np.float64)  # Use float64 for numerical stability
    Y = target_embeds.cpu().numpy().astype(np.float64)

    # Check for inf/nan in input
    if not np.isfinite(X).all() or not np.isfinite(Y).all():
        print(
            f"Warning: Input contains inf/nan values. X finite: {np.isfinite(X).all()}, Y finite: {np.isfinite(Y).all()}")
        # Replace inf/nan with 0
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)

    # Center the data
    X_centered = X - X.mean(axis=0, keepdims=True)
    Y_centered = Y - Y.mean(axis=0, keepdims=True)

    # Normalize to unit variance to prevent overflow
    # Add small epsilon to avoid division by zero
    eps = 1e-8
    X_std = X_centered.std(axis=0, keepdims=True) + eps
    Y_std = Y_centered.std(axis=0, keepdims=True) + eps

    X_normalized = X_centered / X_std
    Y_normalized = Y_centered / Y_std

    # Check again after normalization
    if not np.isfinite(X_normalized).all() or not np.isfinite(Y_normalized).all():
        print(f"Warning: Normalized data contains inf/nan. Skipping problematic dimensions.")
        X_normalized = np.nan_to_num(X_normalized, nan=0.0, posinf=0.0, neginf=0.0)
        Y_normalized = np.nan_to_num(Y_normalized, nan=0.0, posinf=0.0, neginf=0.0)

    try:
        # Compute Procrustes rotation on normalized data
        R, disparity = orthogonal_procrustes(X_normalized, Y_normalized)

        # Check if R contains inf/nan
        if not np.isfinite(R).all():
            print(f"Warning: Rotation matrix contains inf/nan. Using identity matrix.")
            R = np.eye(X.shape[1])
            disparity = float('inf')

    except Exception as e:
        print(f"Error in Procrustes computation: {e}. Using identity matrix.")
        R = np.eye(X.shape[1])
        disparity = float('inf')

    return R, disparity


def evaluate_direction_similarity(
        dir1: torch.Tensor,
        dir2: torch.Tensor
) -> Dict[str, float]:
    """Compute similarity metrics between two direction vectors"""
    # Cosine similarity
    cosine_sim = torch.dot(dir1, dir2) / (torch.norm(dir1) * torch.norm(dir2) + 1e-8)

    # Euclidean distance
    euclidean_dist = torch.norm(dir1 - dir2).item()

    # Angular distance
    angular_dist = torch.acos(torch.clamp(cosine_sim, -1.0, 1.0)).item()

    return {
        "cosine_similarity": cosine_sim.item(),
        "euclidean_distance": euclidean_dist,
        "angular_distance": angular_dist
    }


def analyze_layer_wise_transfer(
        source_embeddings: Dict[str, Dict[int, torch.Tensor]],
        target_embeddings: Dict[str, Dict[int, torch.Tensor]],
        layers: List[int],
        output_dir: Path
) -> pd.DataFrame:
    """
    Analyze refusal direction transfer across all layers

    Returns DataFrame with metrics for each layer
    """
    results = []

    for layer_idx in tqdm(layers, desc="Analyzing layers"):
        # Extract embeddings for this layer
        source_clean = source_embeddings['clean'][layer_idx]
        source_harm = source_embeddings['harm'][layer_idx]
        target_clean = target_embeddings['clean'][layer_idx]
        target_harm = target_embeddings['harm'][layer_idx]

        # Compute statistics about embeddings
        source_clean_norm = torch.norm(source_clean, dim=1).mean().item()
        source_harm_norm = torch.norm(source_harm, dim=1).mean().item()
        target_clean_norm = torch.norm(target_clean, dim=1).mean().item()
        target_harm_norm = torch.norm(target_harm, dim=1).mean().item()

        # Check for numerical issues
        if not torch.isfinite(source_clean).all():
            print(f"Warning: Layer {layer_idx} source_clean contains inf/nan")
        if not torch.isfinite(source_harm).all():
            print(f"Warning: Layer {layer_idx} source_harm contains inf/nan")
        if not torch.isfinite(target_clean).all():
            print(f"Warning: Layer {layer_idx} target_clean contains inf/nan")
        if not torch.isfinite(target_harm).all():
            print(f"Warning: Layer {layer_idx} target_harm contains inf/nan")

        # Compute refusal directions
        source_direction = compute_refusal_direction(source_clean, source_harm)
        target_direction = compute_refusal_direction(target_clean, target_harm)

        # Compute Procrustes alignment using clean data as anchors
        R, disparity = compute_procrustes_alignment(source_clean, target_clean)

        # Transfer source direction to target space
        R_tensor = torch.tensor(R, dtype=source_direction.dtype)
        transferred_direction = R_tensor @ source_direction

        # Evaluate similarity between transferred and native target direction
        similarity_metrics = evaluate_direction_similarity(
            transferred_direction,
            target_direction
        )

        # Additional metrics
        result = {
            'layer': layer_idx,
            'procrustes_disparity': disparity,
            'cosine_similarity': similarity_metrics['cosine_similarity'],
            'euclidean_distance': similarity_metrics['euclidean_distance'],
            'angular_distance': similarity_metrics['angular_distance'],
            'source_direction_norm': torch.norm(source_direction).item(),
            'target_direction_norm': torch.norm(target_direction).item(),
            'transferred_direction_norm': torch.norm(transferred_direction).item(),
            'source_clean_mean_norm': source_clean_norm,
            'source_harm_mean_norm': source_harm_norm,
            'target_clean_mean_norm': target_clean_norm,
            'target_harm_mean_norm': target_harm_norm
        }

        # If GCG data available, also compute with GCG
        if 'gcg' in source_embeddings:
            source_gcg = source_embeddings['gcg'][layer_idx]
            gcg_direction = compute_refusal_direction(source_clean, source_gcg)
            transferred_gcg = R_tensor @ gcg_direction

            gcg_metrics = evaluate_direction_similarity(transferred_gcg, target_direction)
            result['gcg_cosine_similarity'] = gcg_metrics['cosine_similarity']
            result['gcg_angular_distance'] = gcg_metrics['angular_distance']

        results.append(result)

    df = pd.DataFrame(results)

    # Save results
    output_path = output_dir / "layer_wise_analysis.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved layer-wise analysis to {output_path}")

    return df


def plot_analysis_results(df: pd.DataFrame, output_dir: Path):
    """Create visualization plots for the analysis"""

    # Create main figure with 3x2 subplots
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))

    # Plot 1: Cosine similarity across layers
    ax = axes[0, 0]
    ax.plot(df['layer'], df['cosine_similarity'], marker='o', label='HarmBench')
    if 'gcg_cosine_similarity' in df.columns:
        ax.plot(df['layer'], df['gcg_cosine_similarity'], marker='s', label='GCG')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('Refusal Direction Similarity Across Layers')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.3, label='Good threshold')
    ax.axhline(y=0.7, color='g', linestyle='--', alpha=0.3, label='Excellent threshold')

    # Plot 2: Procrustes disparity
    ax = axes[0, 1]
    ax.plot(df['layer'], df['procrustes_disparity'], marker='o', color='red')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Procrustes Disparity')
    ax.set_title('Embedding Space Alignment Quality')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')  # Log scale since disparity can vary widely

    # Plot 3: Angular distance
    ax = axes[1, 0]
    ax.plot(df['layer'], df['angular_distance'], marker='o', label='HarmBench')
    if 'gcg_angular_distance' in df.columns:
        ax.plot(df['layer'], df['gcg_angular_distance'], marker='s', label='GCG')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Angular Distance (radians)')
    ax.set_title('Angular Distance Between Directions')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Direction norms
    ax = axes[1, 1]
    ax.plot(df['layer'], df['source_direction_norm'], marker='o', label='Source')
    ax.plot(df['layer'], df['target_direction_norm'], marker='s', label='Target')
    ax.plot(df['layer'], df['transferred_direction_norm'], marker='^', label='Transferred')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Direction Norm')
    ax.set_title('Direction Vector Magnitudes')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 5: Embedding norms (source)
    ax = axes[2, 0]
    if 'source_clean_mean_norm' in df.columns:
        ax.plot(df['layer'], df['source_clean_mean_norm'], marker='o', label='Source Clean')
        ax.plot(df['layer'], df['source_harm_mean_norm'], marker='s', label='Source Harmful')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Mean Embedding Norm')
        ax.set_title('Source Model: Embedding Magnitudes')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

    # Plot 6: Embedding norms (target)
    ax = axes[2, 1]
    if 'target_clean_mean_norm' in df.columns:
        ax.plot(df['layer'], df['target_clean_mean_norm'], marker='o', label='Target Clean')
        ax.plot(df['layer'], df['target_harm_mean_norm'], marker='s', label='Target Harmful')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Mean Embedding Norm')
        ax.set_title('Target Model: Embedding Magnitudes')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(output_dir / "analysis_plots.png", dpi=300, bbox_inches='tight')
    print(f"Saved plots to {output_dir / 'analysis_plots.png'}")
    plt.close()

    # Create heatmap of metrics
    fig, ax = plt.subplots(figsize=(12, 8))

    metrics_to_plot = ['cosine_similarity', 'procrustes_disparity', 'angular_distance']
    # Normalize disparity for better visualization
    df_viz = df.copy()
    if df_viz['procrustes_disparity'].max() > 0:
        df_viz['procrustes_disparity'] = df_viz['procrustes_disparity'] / df_viz['procrustes_disparity'].max()

    data_for_heatmap = df_viz[['layer'] + metrics_to_plot].set_index('layer').T

    sns.heatmap(data_for_heatmap, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax, cbar_kws={'label': 'Value'})
    ax.set_title('Layer-wise Metrics Heatmap (Disparity normalized)')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Metric')

    plt.tight_layout()
    plt.savefig(output_dir / "metrics_heatmap.png", dpi=300, bbox_inches='tight')
    print(f"Saved heatmap to {output_dir / 'metrics_heatmap.png'}")
    plt.close()


def main():
    # Configuration
    config = ExperimentConfig()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Model configurations
    source_config = ModelConfig(
        name="llama2",
        path="meta-llama/Llama-2-7b-chat-hf"  # Adjust path as needed
    )

    target_config = ModelConfig(
        name="vicuna",
        path="lmsys/vicuna-7b-v1.5"  # Adjust path as needed
    )

    print("=" * 80)
    print("REFUSAL DIRECTION TRANSFER ANALYSIS")
    print("=" * 80)

    # Load datasets
    print("\n" + "=" * 80)
    print("LOADING DATASETS")
    print("=" * 80)
    wikitext = load_wikitext(config.num_wikitext)
    harmbench = load_harmbench(config.num_harmbench)

    datasets = {
        'clean': wikitext,
        'harm': harmbench
    }

    if config.use_gcg and Path(config.gcg_path).exists():
        gcg_prompts = load_gcg_prompts(config.gcg_path, model_index=0, num_samples=config.num_gcg)
        datasets['gcg'] = gcg_prompts

    print(f"\nDataset sizes:")
    for name, data in datasets.items():
        print(f"  {name}: {len(data)} examples")

    # Load models
    print("\n" + "=" * 80)
    print("LOADING MODELS")
    print("=" * 80)

    print(f"\nLoading source model: {source_config.name}")
    source_model = AutoModelForCausalLM.from_pretrained(
        source_config.path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    source_tokenizer = AutoTokenizer.from_pretrained(source_config.path)
    source_tokenizer.pad_token = source_tokenizer.eos_token

    print(f"\nLoading target model: {target_config.name}")
    target_model = AutoModelForCausalLM.from_pretrained(
        target_config.path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    target_tokenizer = AutoTokenizer.from_pretrained(target_config.path)
    target_tokenizer.pad_token = target_tokenizer.eos_token

    # Initialize extractors
    source_extractor = EmbeddingExtractor(source_model, source_tokenizer, source_config.device)
    target_extractor = EmbeddingExtractor(target_model, target_tokenizer, target_config.device)

    print(f"\nSource model layers: {source_extractor.num_layers}")
    print(f"Target model layers: {target_extractor.num_layers}")

    # Determine layers to analyze
    if config.layer_range:
        layers = list(range(config.layer_range[0], config.layer_range[1]))
    else:
        layers = list(range(min(source_extractor.num_layers, target_extractor.num_layers)))

    print(f"Analyzing layers: {layers[0]} to {layers[-1]} ({len(layers)} total)")

    # Extract embeddings
    print("\n" + "=" * 80)
    print("EXTRACTING EMBEDDINGS")
    print("=" * 80)

    source_embeddings = {}
    target_embeddings = {}

    for dataset_name, texts in datasets.items():
        print(f"\n--- {dataset_name.upper()} dataset ---")

        print(f"Extracting from source model ({source_config.name})...")
        source_embeddings[dataset_name] = source_extractor.extract_embeddings(
            texts, layers=layers, batch_size=config.batch_size
        )

        print(f"Extracting from target model ({target_config.name})...")
        target_embeddings[dataset_name] = target_extractor.extract_embeddings(
            texts, layers=layers, batch_size=config.batch_size
        )

    # Save embeddings if requested
    if config.save_embeddings:
        print("\n" + "=" * 80)
        print("SAVING EMBEDDINGS")
        print("=" * 80)

        embeddings_dir = output_dir / "embeddings"
        embeddings_dir.mkdir(exist_ok=True)

        for dataset_name in datasets.keys():
            torch.save(
                source_embeddings[dataset_name],
                embeddings_dir / f"source_{dataset_name}_embeddings.pt"
            )
            torch.save(
                target_embeddings[dataset_name],
                embeddings_dir / f"target_{dataset_name}_embeddings.pt"
            )

        print(f"Saved embeddings to {embeddings_dir}")

    # Analyze layer-wise transfer
    print("\n" + "=" * 80)
    print("ANALYZING LAYER-WISE TRANSFER")
    print("=" * 80)

    results_df = analyze_layer_wise_transfer(
        source_embeddings,
        target_embeddings,
        layers,
        output_dir
    )

    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    print("\nBest layers by cosine similarity:")
    top_layers = results_df.nlargest(5, 'cosine_similarity')[['layer', 'cosine_similarity', 'angular_distance']]
    print(top_layers.to_string(index=False))

    print("\nWorst layers by Procrustes disparity:")
    best_alignment = results_df.nsmallest(5, 'procrustes_disparity')[
        ['layer', 'procrustes_disparity', 'cosine_similarity']]
    print(best_alignment.to_string(index=False))

    if 'gcg_cosine_similarity' in results_df.columns:
        print("\nBest layers for GCG direction transfer:")
        top_gcg = results_df.nlargest(5, 'gcg_cosine_similarity')[
            ['layer', 'gcg_cosine_similarity', 'gcg_angular_distance']]
        print(top_gcg.to_string(index=False))

    # Create visualizations
    print("\n" + "=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)

    plot_analysis_results(results_df, output_dir)

    # Save summary report
    summary = {
        'config': {
            'source_model': source_config.name,
            'target_model': target_config.name,
            'num_wikitext': config.num_wikitext,
            'num_harmbench': config.num_harmbench,
            'num_layers_analyzed': len(layers),
            'use_gcg': config.use_gcg
        },
        'best_layer': int(results_df.loc[results_df['cosine_similarity'].idxmax(), 'layer']),
        'best_cosine_similarity': float(results_df['cosine_similarity'].max()),
        'mean_cosine_similarity': float(results_df['cosine_similarity'].mean()),
        'best_alignment_layer': int(results_df.loc[results_df['procrustes_disparity'].idxmin(), 'layer']),
        'best_alignment_disparity': float(results_df['procrustes_disparity'].min())
    }

    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")
    print(f"  - Layer-wise analysis: layer_wise_analysis.csv")
    print(f"  - Visualizations: analysis_plots.png, metrics_heatmap.png")
    print(f"  - Summary: summary.json")
    if config.save_embeddings:
        print(f"  - Embeddings: embeddings/")


if __name__ == "__main__":
    main()