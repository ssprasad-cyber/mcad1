import torch
from torch.utils.data import DataLoader
from src.models.simple_gnn import SimpleMCADModel
from src.training.trainer import train, GraphSampleDataset
from src.evaluation.evaluate import evaluate
import os
from pathlib import Path

def run_overfit():
    # 1. Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 2. Data
    data_dir = "data/graphs/overfit"
    dataset = GraphSampleDataset(data_dir)
    print(f"Dataset size: {len(dataset)}")
    
    # Split into train/val (for overfitting, we can just use the same set or a small split)
    # Let's use 80/20 split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, collate_fn=lambda x: x)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False, collate_fn=lambda x: x)
    
    # 3. Model
    model = SimpleMCADModel(node_in_channels=7, edge_in_channels=4, hidden_channels=64)
    
    # 4. Train
    print("Starting training...")
    # Since it's a small dataset, we can use more epochs
    history = train(
        model, 
        train_loader, 
        val_loader, 
        epochs=50, 
        lr=0.001, 
        device=device,
        pos_weight=1.0 # Balanced classes in our overfit set roughly
    )
    
    # 5. Evaluate
    print("Final evaluation:")
    results = evaluate(model, val_loader, device=device)
    print(results)

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    run_overfit()
