import torch
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
from torch_geometric.data import Batch
from pathlib import Path
from tqdm import tqdm
import json

def evaluate(model, test_loader, device='cpu'):
    model.eval()
    model.to(device)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_data in tqdm(test_loader, desc="Evaluating"):
            B = len(batch_data)
            T = len(batch_data[0]["graphs"])
            labels = torch.tensor([s["label"] for s in batch_data], dtype=torch.float32).unsqueeze(1)
            
            batched_seq = []
            for t in range(T):
                graphs_at_t = [s["graphs"][t] for s in batch_data]
                batched_seq.append(Batch.from_data_list(graphs_at_t).to(device))
            
            outputs = model(batched_seq)
            probs = torch.sigmoid(outputs).cpu().numpy()
            
            all_preds.extend(probs)
            all_labels.extend(labels.numpy())
            
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    auc = roc_auc_score(all_labels, all_preds)
    
    # Binary predictions with 0.5 threshold
    binary_preds = (all_preds > 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, binary_preds, average='binary')
    
    results = {
        "roc_auc": auc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    
    return results

if __name__ == "__main__":
    # Test evaluation logic
    pass
