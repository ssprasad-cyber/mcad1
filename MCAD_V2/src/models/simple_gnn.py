import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class GATEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, heads=2):
        super(GATEncoder, self).__init__()
        self.conv1 = GATConv(in_channels, 32, heads=heads)
        self.conv2 = GATConv(32 * heads, out_channels, heads=1)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        
        # Global pooling to get a graph-level embedding
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x = global_mean_pool(x, batch)
        return x

class SimpleMCADModel(nn.Module):
    def __init__(self, node_in_channels=7, edge_in_channels=4, hidden_channels=64, num_classes=1):
        super(SimpleMCADModel, self).__init__()
        
        # 1. Graph Encoder (per frame)
        self.encoder = GATEncoder(node_in_channels, hidden_channels)
        
        # 2. Temporal Aggregator
        self.gru = nn.GRU(hidden_channels, hidden_channels, batch_first=True)
        
        # 3. Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )

    def forward(self, graphs_list):
        """
        graphs_list: List of T Batch objects (one per timestep)
        OR a single batch containing all timesteps (more efficient)
        For simplicity, we'll assume we pass a sequence of frame graph embeddings.
        """
        # Encode each frame in the sequence
        # Assuming input is a list of T graphs, each with batch size B
        embeddings = []
        for graph in graphs_list:
            # graph is a Batch object from torch_geometric
            emb = self.encoder(graph.x, graph.edge_index, batch=graph.batch)
            embeddings.append(emb) # (B, hidden_channels)
            
        # Stack to (B, T, hidden_channels)
        x = torch.stack(embeddings, dim=1)
        
        # GRU
        _, h_n = self.gru(x) # h_n is (1, B, hidden_channels)
        x = h_n.squeeze(0) # (B, hidden_channels)
        
        # Classify
        logits = self.classifier(x)
        return logits

if __name__ == "__main__":
    model = SimpleMCADModel()
    print(model)
    # Test with dummy data
    from torch_geometric.data import Batch, Data
    dummy_data = Data(x=torch.randn(5, 7), edge_index=torch.tensor([[0, 1], [1, 0]]))
    dummy_batch = Batch.from_data_list([dummy_data])
    seq = [dummy_batch] * 10
    out = model(seq)
    print(f"Output shape: {out.shape}")
