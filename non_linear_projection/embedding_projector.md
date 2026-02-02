# Embedding Projector

Learn a mapping between embedding spaces of two LLMs while preserving geometric structure.

## Problem

Given:
- Model A embeddings: $x \in \mathbb{R}^{d_A}$
- Model B embeddings: $y \in \mathbb{R}^{d_B}$
- Paired data from same inputs

Learn $f_\theta: \mathbb{R}^{d_A} \rightarrow \mathbb{R}^{d_B}$

## Architecture

Shallow MLP with normalized output:

$$f_\theta(x) = \text{normalize}\left(W_2 \cdot \text{GELU}(W_1 x + b_1) + b_2\right)$$

## Loss Function

**Alignment** (map A close to B):

$$\mathcal{L}_{\text{align}} = \frac{1}{N}\sum_{i=1}^{N} \left(1 - \cos(f_\theta(x_i), y_i)\right)$$

**Geometry preservation** (preserve A's structure):

$$\mathcal{L}_{\text{geom}} = \frac{1}{N^2}\sum_{i,j} \left( \cos(f_\theta(x_i), f_\theta(x_j)) - \cos(x_i, x_j) \right)^2$$

**Total**:

$$\mathcal{L} = \mathcal{L}_{\text{align}} + \lambda \mathcal{L}_{\text{geom}}$$

## Training

1. Extract hidden states (layer 16) from both models on same sentences
2. Train projector with AdamW
3. Tune $\lambda$ to balance alignment vs geometry preservation

## Hyperparameters

| Param | Default |
|-------|---------|
| $d_H$ | 4096 |
| $\lambda$ | 0.1 |
| lr | 1e-4 |
| batch_size | 64 |
