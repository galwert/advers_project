import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os

# --- CONFIG (ADJUSTED FOR MISTRAL) ---
# We are now bridging two DIFFERENT architectures
SOURCE_FILE = "src_llama.pt"  # File created by the harvester
TARGET_FILE = "tgt_mistral.pt"  # File created by the harvester
DATA_DIR = "alignment_data"  # The folder where you saved them
DEVICE = "cuda"

# --- 1. LOAD DATA ---
print(f"[*] Loading alignment data from {DATA_DIR}...")

try:
    src_emb = torch.load(os.path.join(DATA_DIR, SOURCE_FILE)).float()
    tgt_emb = torch.load(os.path.join(DATA_DIR, TARGET_FILE)).float()
except FileNotFoundError:
    print("❌ ERROR: Could not find the .pt files.")
    print("   Did you run the 'Harvester' script I gave you in the previous step?")
    print("   It creates 'alignment_data/src_llama.pt' and 'tgt_mistral.pt'")
    exit()

# Normalize (Critical for cosine similarity)
src_emb = torch.nn.functional.normalize(src_emb, p=2, dim=1).to(DEVICE)
tgt_emb = torch.nn.functional.normalize(tgt_emb, p=2, dim=1).to(DEVICE)

print(f"   Source Shape (Llama):   {src_emb.shape}")
print(f"   Target Shape (Mistral): {tgt_emb.shape}")


# --- 2. MODEL DEFINITION ---
# We use a Linear Transformation first.
# If this fails, we can switch back to the MLP, but start simple.
class ConceptTranslator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # A simple rotation matrix (Linear Layer)
        # This tries to rotate Llama's brain to match Mistral's
        self.matrix = nn.Linear(input_dim, input_dim, bias=False)

    def forward(self, x):
        return self.matrix(x)


dim = src_emb.shape[1]
adapter = ConceptTranslator(dim).to(DEVICE)

# We use MSE to force the vectors to overlap
criterion = nn.MSELoss()
optimizer = optim.Adam(adapter.parameters(), lr=5e-4)

# --- 3. TRAINING LOOP ---
BATCH_SIZE = 128
# Create dataset
dataset = TensorDataset(src_emb, tgt_emb)
# Split into Train (90%) and Test (10%) to see if it generalizes
train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

EPOCHS = 100
print("\n[*] Starting Alignment Training (Llama -> Mistral)...")

for epoch in range(EPOCHS):
    adapter.train()
    total_loss = 0

    for batch_src, batch_tgt in loader:
        optimizer.zero_grad()

        # 1. Project Llama to Mistral Space
        aligned_src = adapter(batch_src)

        # 2. Calculate Distance
        loss = criterion(aligned_src, batch_tgt)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Validation Step (Check the score!)
    if epoch % 10 == 0:
        adapter.eval()
        with torch.no_grad():
            # Get test set
            test_src, test_tgt = next(iter(test_loader))
            test_aligned = adapter(test_src)

            # Calculate Cosine Similarity (The Alignment Score)
            sim = torch.nn.functional.cosine_similarity(test_aligned, test_tgt).mean().item()
            print(f"Epoch {epoch:03d} | Loss: {total_loss / len(loader):.6f} | Val Alignment: {sim:.4f}")

# Save
torch.save(adapter.state_dict(), "llama_to_mistral.pth")
print("\n[SUCCESS] Matrix saved as 'llama_to_mistral.pth'")