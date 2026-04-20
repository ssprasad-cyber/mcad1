import torch
from torch_geometric.data import Data
import numpy as np

class GraphConstructor:
    def __init__(self, distance_threshold=0.3):
        self.distance_threshold = distance_threshold

    def build_frame_graph(self, person_features):
        """
        Builds a graph for a single frame.
        person_features: np.array of shape (N, 7) [x, y, w, h, vx, vy, mag]
        """
        N = person_features.shape[0]
        if N == 0:
            # Empty graph
            return Data(
                x=torch.zeros((0, 7), dtype=torch.float32),
                edge_index=torch.empty((2, 0), dtype=torch.long),
                edge_attr=torch.empty((0, 4), dtype=torch.float32)
            )

        # Node features
        x = torch.tensor(person_features, dtype=torch.float32)

        # Edges based on distance
        edge_index = []
        edge_attr = []

        # Coordinates are at indices 0, 1
        coords = person_features[:, :2]
        velocities = person_features[:, 4:6]

        for i in range(N):
            for j in range(N):
                if i == j: continue
                
                dist = np.linalg.norm(coords[i] - coords[j])
                if dist < self.distance_threshold:
                    edge_index.append([i, j])
                    
                    # Relative features
                    rel_v = velocities[i] - velocities[j]
                    rel_mag = np.linalg.norm(rel_v)
                    edge_attr.append([dist, rel_v[0], rel_v[1], rel_mag])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        if edge_index.numel() == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 4), dtype=torch.float32)
        else:
            edge_attr = torch.tensor(edge_attr, dtype=torch.float32)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def build_sequence_graphs(self, all_window_features):
        """
        Converts list of frame features to list of Data objects.
        """
        graphs = []
        for frame_feats in all_window_features:
            graphs.append(self._to_tensor_graph(frame_feats))
        return graphs

    def _to_tensor_graph(self, frame_feats):
        # Helper to ensure we have a Data object even for empty frames
        if frame_feats.shape[0] == 0:
             return Data(
                x=torch.zeros((1, 7), dtype=torch.float32), # At least one dummy node to avoid crashes
                edge_index=torch.empty((2, 0), dtype=torch.long),
                edge_attr=torch.empty((0, 4), dtype=torch.float32)
            )
        return self.build_frame_graph(frame_feats)

if __name__ == "__main__":
    constructor = GraphConstructor()
    # Dummy features for 2 people
    feats = np.array([
        [0.5, 0.5, 0.1, 0.2, 0.01, 0.01, 0.014],
        [0.6, 0.6, 0.1, 0.2, -0.01, -0.01, 0.014]
    ])
    graph = constructor.build_frame_graph(feats)
    print(f"Nodes: {graph.x.shape}, Edges: {graph.edge_index.shape}")
