import os
import sys
import json

# Ensure repository root is on pythonpath
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import torch.nn as nn
from torch.optim import AdamW

from src.translator.tokenization import SimpleTokenizer
from src.translator.encoder import TextToManifoldEncoder
from src.reasoning_engine.manifold import RelationalManifold

def train_manifold(epochs: int = 5, lr: float = 3e-4, device_str: str = "cuda"):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"[Stage 1] Pretraining Manifold on device: {device}")

    # Load corpus
    data_path = os.path.join(os.path.dirname(__file__), "../data/knowledge_graph/physics_principles.json")
    with open(data_path, "r") as f:
        data = json.load(f)
    texts = [item["statement"] for item in data]

    tokenizer = SimpleTokenizer(vocab_size=2000)
    tokenizer.build_vocab(texts)

    encoder = TextToManifoldEncoder(vocab_size=2000, latent_dim=256).to(device)
    manifold = RelationalManifold(latent_dim=256).to(device)

    optimizer = AdamW(list(encoder.parameters()) + list(manifold.parameters()), lr=lr)
    reconstruction_loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0.0
        for text in texts:
            tokens, _ = tokenizer.encode(text)
            tok_tensor = torch.tensor([tokens], dtype=torch.long, device=device)

            optimizer.zero_grad()
            # Encode text into 256D concept representation
            encoded = encoder(tok_tensor)
            pooled = encoder.pool_state(encoded)

            # Pass through dynamic relational manifold
            nodes, adj = manifold(pooled)
            readout = manifold.readout(nodes)

            # Auto-encoding consistency loss: manifold readout should faithfully retain input concept
            loss = reconstruction_loss_fn(readout, pooled)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(texts)
        print(f"Epoch {epoch + 1}/{epochs} - Manifold Reconstruction Loss: {avg_loss:.4f}")

    os.makedirs("logs/training_runs", exist_ok=True)
    torch.save({
        "encoder": encoder.state_dict(),
        "manifold": manifold.state_dict()
    }, "logs/training_runs/stage1_manifold.pt")
    print("[Stage 1] Checkpoint saved to logs/training_runs/stage1_manifold.pt")

if __name__ == "__main__":
    train_manifold()
