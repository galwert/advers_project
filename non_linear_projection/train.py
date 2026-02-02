import torch
from torch.utils.data import DataLoader, TensorDataset
import h5py
from model import EmbeddingProjector, loss_fn


def load_embeddings(path, first_model, second_model):
    with h5py.File(path, 'r') as f:
        emb_a = torch.from_numpy(f[first_model][:])
        emb_b = torch.from_numpy(f[second_model][:])
    return emb_a, emb_b


def train(
        embeddings_path,
        first_model,
        second_model,
        d_h=4096,
        batch_size=64,
        lr=1e-4,
        epochs=100,
        lam=0.1,
        device='cuda'
):
    # Load data
    emb_a, emb_b = load_embeddings(embeddings_path, first_model, second_model)
    d_a, d_b = emb_a.shape[1], emb_b.shape[1]

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
    model_a = "meta-llama/Llama-2-7b-chat-hf"
    model_b = "mistralai/Mistral-7B-Instruct-v0.1"

    proj = train(
        embeddings_path="embeddings.h5",
        first_model=model_a,
        second_model=model_b,
        d_h=4096,
        batch_size=64,
        epochs=100,
        lam=0.1
    )

    torch.save(proj.state_dict(), "projector.pt")