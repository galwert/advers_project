import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from sklearn.decomposition import PCA
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_sentences(n=100, min_len=50, max_len=200):
    """Load sentences from wikitext."""
    dataset = load_dataset("wikitext", "wikitext-2-v1", split="train")
    sentences = [x['text'] for x in dataset if min_len < len(x['text']) < max_len][:n]
    return sentences


def load_model(model_name):
    """Load model and tokenizer."""
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    tok.pad_token = tok.eos_token
    return model, tok





def get_all_embeddings(model, tokenizer, sentences):
    """Get last token embeddings for all sentences, all layers.

    Returns: [n_layers, n_sentences, hidden_dim]
    """
    n_layers = None
    layer_embeddings = None

    for i, sentence in enumerate(sentences):
        inputs = tokenizer(sentence, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)

        if n_layers is None:
            n_layers = len(out.hidden_states)
            layer_embeddings = [[] for _ in range(n_layers)]

        for layer_idx, layer_hidden in enumerate(out.hidden_states):
            last_token = layer_hidden[0, -1, :].cpu()
            layer_embeddings[layer_idx].append(last_token)

        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(sentences)}")

    # Stack: list of lists -> [n_layers, n_sentences, hidden_dim]
    result = torch.stack([torch.stack(layer) for layer in layer_embeddings])
    return result


def apply_pca(embeddings, target_dim=128):
    """Apply PCA to embeddings.

    Args:
        embeddings: [n_layers, n_sentences, hidden_dim]
        target_dim: output dimension

    Returns:
        reduced: [n_layers, n_sentences, target_dim]
        explained_variance: [n_layers] - variance explained per layer
    """
    n_layers, n_sentences, hidden_dim = embeddings.shape
    reduced = []
    explained_variance = []

    for layer_idx in range(n_layers):
        X = embeddings[layer_idx].numpy()  # [n_sentences, hidden_dim]

        pca = PCA(n_components=target_dim)
        X_reduced = pca.fit_transform(X)

        reduced.append(torch.tensor(X_reduced))
        explained_variance.append(pca.explained_variance_ratio_.sum())

    reduced = torch.stack(reduced)  # [n_layers, n_sentences, target_dim]
    return reduced, explained_variance


def affine_align_score(X, Y):
    """Compute alignment score using full affine transformation.

    Args:
        X: [n_samples, dim] - source embeddings
        Y: [n_samples, dim] - target embeddings

    Returns:
        score: mean cosine similarity after alignment
    """
    n = X.shape[0]

    # Augment X with ones column for bias: [n, dim] -> [n, dim+1]
    X_aug = torch.cat([X.float(), torch.ones(n, 1)], dim=1)

    # Solve: X_aug @ W = Y (W includes rotation + scale + shift)
    W, _, _, _ = torch.linalg.lstsq(X_aug, Y.float())

    # Apply transformation
    X_aligned = X_aug @ W

    # Cosine similarity
    X_norm = torch.nn.functional.normalize(X_aligned, dim=1)
    Y_norm = torch.nn.functional.normalize(Y.float(), dim=1)

    score = (X_norm * Y_norm).sum(dim=1).mean().item()
    return score


def compute_layer_alignment_matrix(emb_a_pca, emb_b_pca):
    """Compute alignment between all layer pairs.

    Args:
        emb_a_pca: [n_layers_a, n_samples, dim]
        emb_b_pca: [n_layers_b, n_samples, dim]

    Returns:
        matrix: [n_layers_a - 1, n_layers_b - 1] (skip layer 0)
    """
    n_layers_a = emb_a_pca.shape[0]
    n_layers_b = emb_b_pca.shape[0]

    # Skip layer 0
    matrix = np.zeros((n_layers_a - 1, n_layers_b - 1))

    for i in range(1, n_layers_a):
        for j in range(1, n_layers_b):
            X = emb_a_pca[i]  # [n_samples, dim]
            Y = emb_b_pca[j]  # [n_samples, dim]

            score = affine_align_score(X, Y)
            matrix[i - 1, j - 1] = score

        print(f"Layer {i}/{n_layers_a - 1} done")

    return matrix


def get_top_k_alignments(alignment_matrix, k=10):
    """Find layer pairs with highest alignment scores.

    Args:
        alignment_matrix: [n_layers_a - 1, n_layers_b - 1]
        k: number of top pairs to return

    Returns:
        list of (layer_a, layer_b, score) tuples
    """
    n_a, n_b = alignment_matrix.shape

    # Flatten and get top-k indices
    flat = alignment_matrix.flatten()
    top_k_flat = np.argsort(flat)[::-1][:k]

    results = []
    for idx in top_k_flat:
        i = idx // n_b
        j = idx % n_b
        layer_a = i + 1  # +1 because we skipped layer 0
        layer_b = j + 1
        score = alignment_matrix[i, j]
        results.append((layer_a, layer_b, score))

    return results

if __name__ == "__main__":
    print(DEVICE)
    sentences = load_sentences(n=500)
    print(f"Loaded {len(sentences)} sentences")

    model_a, tok_a = load_model("meta-llama/Llama-2-7b-chat-hf")
    model_b, tok_b = load_model("Qwen/Qwen2-0.5B")

    print("\nExtracting Model A embeddings...")
    emb_a = get_all_embeddings(model_a, tok_a, sentences)
    print(f"Model A raw: {emb_a.shape}")

    print("\nExtracting Model B embeddings...")
    emb_b = get_all_embeddings(model_b, tok_b, sentences)
    print(f"Model B raw: {emb_b.shape}")

    # PCA to shared dimension
    target_dim = 48
    emb_a_pca, var_a = apply_pca(emb_a, target_dim)
    emb_b_pca, var_b = apply_pca(emb_b, target_dim)

    print(f"\nAfter PCA (dim={target_dim}):")
    print(f"Model A: {emb_a_pca.shape}")
    print(f"Model B: {emb_b_pca.shape}")

    print(f"\nExplained variance per layer:")

    print(f"Model A")
    for val in var_a:
        print(f"{val:.3f}", end=" ")
    print("\nModel B")
    for val in var_b:
        print(f"{val:.3f}", end=" ")
    print()
    alignment_matrix = compute_layer_alignment_matrix(emb_a_pca, emb_b_pca)
    print(f"Shape: {alignment_matrix.shape}")  # [32, 24]

    top_pairs = get_top_k_alignments(alignment_matrix, k=800)

    print("Top-10 layer alignments:")
    print(f"{'Model A':<10} {'Model B':<10} {'Score':<10}")
    print("-" * 30)
    for layer_a, layer_b, score in top_pairs:
        print(f"{layer_a:<10} {layer_b:<10} {score:.4f}")
