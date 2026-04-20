import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json

class GraphSampleDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.files = sorted(list(self.data_dir.glob("*.pt")))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return torch.load(self.files[idx], weights_only=False)

def train(model, train_loader, val_loader, epochs=30, lr=0.001, device='cpu', pos_weight=None):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if pos_weight is not None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
    else:
        criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float('inf')
    history = {"train_loss": [], "val_loss": []}

    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            # batch_data is a list of samples from __getitem__
            # Each sample is {"graphs": [G1, ..., G10], "label": L}
            
            # Reorganize list of sequences into sequence of Batches
            B = len(batch_data)
            T = len(batch_data[0]["graphs"])
            
            labels = torch.tensor([s["label"] for s in batch_data], dtype=torch.float32).to(device).unsqueeze(1)
            
            # For each timestep, create a Batch of B graphs
            batched_seq = []
            for t in range(T):
                graphs_at_t = [s["graphs"][t] for s in batch_data]
                batched_seq.append(Batch.from_data_list(graphs_at_t).to(device))
            
            optimizer.zero_grad()
            outputs = model(batched_seq)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_data in val_loader:
                B = len(batch_data)
                T = len(batch_data[0]["graphs"])
                labels = torch.tensor([s["label"] for s in batch_data], dtype=torch.float32).to(device).unsqueeze(1)
                batched_seq = []
                for t in range(T):
                    graphs_at_t = [s["graphs"][t] for s in batch_data]
                    batched_seq.append(Batch.from_data_list(graphs_at_t).to(device))
                
                outputs = model(batched_seq)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        history["val_loss"].append(avg_val_loss)
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "models/best_model.pt")

    return history

if __name__ == "__main__":
    from src.models.simple_gnn import SimpleMCADModel
    # Mock training for architecture check
    model = SimpleMCADModel()
    # This would be called by a main training script
