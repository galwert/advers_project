#!/usr/bin/env python3
"""
Comprehensive diagnostic for llama2→llama2 sanity check failure

This will help identify why cosine similarity is ~0.47 instead of ~1.0
"""

import torch
import numpy as np
from pathlib import Path
import sys

if len(sys.argv) < 2:
    print("Usage: python diagnose_sanity_check.py <embeddings_dir>")
    sys.exit(1)

embeddings_dir = Path(sys.argv[1])

print("="*80)
print("DIAGNOSING SANITY CHECK FAILURE")
print("="*80)

# Load embeddings
print("\nLoading embeddings...")
source_clean = torch.load(embeddings_dir / "source_clean_embeddings.pt", map_location='cpu')
target_clean = torch.load(embeddings_dir / "target_clean_embeddings.pt", map_location='cpu')
source_harm = torch.load(embeddings_dir / "source_harm_embeddings.pt", map_location='cpu')
target_harm = torch.load(embeddings_dir / "target_harm_embeddings.pt", map_location='cpu')

# Pick a representative layer
layer = 15

print(f"\nAnalyzing layer {layer}...")
sc = source_clean[layer]
tc = target_clean[layer]
sh = source_harm[layer]
th = target_harm[layer]

print(f"\nShapes:")
print(f"  source_clean: {sc.shape}")
print(f"  target_clean: {tc.shape}")
print(f"  source_harm: {sh.shape}")
print(f"  target_harm: {th.shape}")

# Check if clean embeddings are identical
clean_identical = torch.allclose(sc, tc, rtol=1e-4, atol=1e-6)
harm_identical = torch.allclose(sh, th, rtol=1e-4, atol=1e-6)

print(f"\n{'='*80}")
print(f"IDENTITY CHECK:")
print(f"  Clean embeddings identical: {clean_identical}")
print(f"  Harm embeddings identical: {harm_identical}")

if not clean_identical:
    diff = (sc - tc).abs()
    print(f"  Clean max diff: {diff.max():.2e}")
    print(f"  Clean mean diff: {diff.mean():.2e}")

if not harm_identical:
    diff = (sh - th).abs()
    print(f"  Harm max diff: {diff.max():.2e}")
    print(f"  Harm mean diff: {diff.mean():.2e}")

# Compute refusal directions
print(f"\n{'='*80}")
print(f"REFUSAL DIRECTION COMPUTATION:")

source_direction = (sh.mean(dim=0) - sc.mean(dim=0))
target_direction = (th.mean(dim=0) - tc.mean(dim=0))

source_norm = torch.norm(source_direction)
target_norm = torch.norm(target_direction)

print(f"  Source direction norm: {source_norm:.4f}")
print(f"  Target direction norm: {target_norm:.4f}")

# Cosine similarity between directions
cosine_sim = torch.dot(source_direction, target_direction) / (source_norm * target_norm)
print(f"  Cosine similarity: {cosine_sim:.6f}")

if clean_identical and harm_identical:
    print(f"\n  ✓ Embeddings are identical → directions should be identical")
    print(f"  ✓ Cosine similarity should be 1.0")
    if cosine_sim < 0.99:
        print(f"  ✗ BUT IT'S NOT! Something is wrong with the analysis code!")
else:
    print(f"\n  ✗ Embeddings are NOT identical")
    print(f"  This explains why cosine similarity is {cosine_sim:.4f} instead of 1.0")

# Check the separation between clean and harm
print(f"\n{'='*80}")
print(f"CLUSTER SEPARATION:")

# Within-group similarity
clean_within = torch.cdist(sc[:10], sc[:10]).mean()
harm_within = torch.cdist(sh[:10], sh[:10]).mean()

# Between-group distance
clean_harm_between = torch.cdist(sc[:10], sh[:10]).mean()

print(f"  Clean-to-clean distance (source): {clean_within:.4f}")
print(f"  Harm-to-harm distance (source): {harm_within:.4f}")
print(f"  Clean-to-harm distance (source): {clean_harm_between:.4f}")

separation_ratio = clean_harm_between / max(clean_within, harm_within)
print(f"  Separation ratio: {separation_ratio:.4f}")

if separation_ratio < 1.5:
    print(f"  ⚠️  WARNING: Clusters are not well-separated!")
    print(f"  This could mean the model doesn't distinguish malicious/clean prompts at this layer")

# Test Procrustes manually
print(f"\n{'='*80}")
print(f"TESTING PROCRUSTES ALIGNMENT:")

from scipy.linalg import orthogonal_procrustes

# Original approach (without normalization)
X = sc.numpy()
Y = tc.numpy()

X_centered = X - X.mean(axis=0)
Y_centered = Y - Y.mean(axis=0)

try:
    R_orig, disp_orig = orthogonal_procrustes(X_centered, Y_centered)
    print(f"  Original Procrustes disparity: {disp_orig:.2e}")
    print(f"  ✓ No overflow (layer {layer} is safe)")
except Exception as e:
    print(f"  ✗ Original Procrustes failed: {e}")
    print(f"  This would have been the overflow error")

# With normalization (current approach)
X_std = X_centered.std(axis=0) + 1e-8
Y_std = Y_centered.std(axis=0) + 1e-8
X_norm = X_centered / X_std
Y_norm = Y_centered / Y_std

try:
    R_norm, disp_norm = orthogonal_procrustes(X_norm, Y_norm)
    print(f"  Normalized Procrustes disparity: {disp_norm:.2e}")
except Exception as e:
    print(f"  ✗ Normalized Procrustes failed: {e}")

# Apply rotation and check transfer
if clean_identical and harm_identical:
    print(f"\n  NOTE: Since embeddings are identical, R should be identity matrix")
    R_diff_from_identity = np.abs(R_norm - np.eye(R_norm.shape[0])).max()
    print(f"  Max deviation from identity: {R_diff_from_identity:.2e}")

# Now transfer the direction and check
source_dir_np = source_direction.numpy()
transferred = R_norm @ source_dir_np
transferred_torch = torch.from_numpy(transferred)

transfer_cosine = torch.dot(transferred_torch, target_direction) / (torch.norm(transferred_torch) * target_norm)
print(f"\n  Cosine similarity after transfer: {transfer_cosine:.6f}")

print(f"\n{'='*80}")
print(f"DIAGNOSIS:")
print(f"{'='*80}")

if clean_identical and harm_identical:
    if cosine_sim > 0.99:
        print("✓ Everything looks good!")
        print("  Embeddings are identical and directions match")
    else:
        print("✗ PROBLEM FOUND:")
        print("  Embeddings are identical but directions don't match")
        print("  This is a bug in the refusal direction computation or comparison")
else:
    print("✗ PROBLEM FOUND:")
    print("  Source and target embeddings are NOT identical")
    print("\nPossible causes:")
    print("  1. Model was loaded twice with different random states")
    print("  2. Dropout or other randomness not disabled")
    print("  3. Different tokenization or prompt preprocessing")
    print("  4. Non-deterministic operations (like torch.bmm on some GPUs)")
    print("\nTo fix:")
    print("  • Ensure model.eval() is called")
    print("  • Set torch.manual_seed() and torch.cuda.manual_seed_all()")
    print("  • Use torch.backends.cudnn.deterministic = True")
    print("  • OR: For true sanity check, extract embeddings only once")
    print("    and compare: source_clean vs source_clean (perfect match)")