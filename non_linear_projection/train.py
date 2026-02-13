import torch
from torch.utils.data import DataLoader, TensorDataset
import h5py
import argparse
from model import EmbeddingProjector, loss_fn


def load_embeddings(path, first_model, second_model, layer_idx=None):
    """Load embeddings from H5 file. Supports both old and multi-layer formats."""
    with h5py.File(path, 'r') as f:
        if layer_idx is not None:
            # Multi-layer format: model_name/layer_N
            key_a = f"{first_model}/layer_{layer_idx}"
            key_b = f"{second_model}/layer_{layer_idx}"
            if key_a in f and key_b in f:
                emb_a = torch.from_numpy(f[key_a][:])
                emb_b = torch.from_numpy(f[key_b][:])
                return emb_a, emb_b

        # Fallback: old format (model_name directly)
        emb_a = torch.from_numpy(f[first_model][:])
        emb_b = torch.from_numpy(f[second_model][:])
    return emb_a, emb_b


def train(
        embeddings_path,
        first_model,
        second_model,
        layer_idx=None,
        d_h=4096,
        batch_size=64,
        lr=1e-4,
        epochs=100,
        lam=0.1,
        device='cuda'
):
    # Load data
    emb_a, emb_b = load_embeddings(embeddings_path, first_model, second_model, layer_idx)
    d_a, d_b = emb_a.shape[1], emb_b.shape[1]

    layer_str = f" (layer {layer_idx})" if layer_idx is not None else ""
    print(f"[*] Training projector{layer_str}: {d_a} -> {d_b}, {len(emb_a)} samples")

    dataset = TensorDataset(emb_a, emb_b)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model
    proj = EmbeddingProjector(d_a, d_b, d_h).to(device)
    optim = torch.optim.AdamW(proj.parameters(), lr=lr)

    # Train
    for epoch in range(epochs):
        total_loss = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            loss = loss_fn(proj, x, y, lam)

            optim.zero_grad()
            loss.backward()
            optim.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs} | Loss: {total_loss / len(loader):.4f}")

    return proj


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train projector(s)")
    parser.add_argument("--first_model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--second_model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--embeddings_path", type=str, default="llama2_llama3.h5",
                        help="Path to embeddings H5 file")
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated layer indices (e.g., 8,16,24). If not set, trains single projector from old-format H5.")
    parser.add_argument("--d_h", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lam", type=float, default=0.1)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.layers:
        # Multi-layer: train one projector per layer
        layers = [int(x) for x in args.layers.split(",")]
        print(f"[+] Training projectors for layers: {layers}")

        for layer_idx in layers:
            print(f"\n{'='*60}")
            print(f"TRAINING PROJECTOR FOR LAYER {layer_idx}")
            print(f"{'='*60}")

            proj = train(
                embeddings_path=args.embeddings_path,
                first_model=args.first_model,
                second_model=args.second_model,
                layer_idx=layer_idx,
                d_h=args.d_h,
                batch_size=args.batch_size,
                lr=args.lr,
                epochs=args.epochs,
                lam=args.lam,
                device=device,
            )

            save_path = f"projector_layer_{layer_idx}.pt"
            torch.save(proj.state_dict(), save_path)
            print(f"[+] Saved: {save_path}")

        print(f"\n[+] All projectors trained: {['projector_layer_{}.pt'.format(l) for l in layers]}")
    else:
        # Single projector (backward compatible)
        proj = train(
            embeddings_path=args.embeddings_path,
            first_model=args.first_model,
            second_model=args.second_model,
            d_h=args.d_h,
            batch_size=args.batch_size,
            lr=args.lr,
            epochs=args.epochs,
            lam=args.lam,
            device=device,
        )
        torch.save(proj.state_dict(), "projector.pt")
        print("[+] Saved: projector.pt")
