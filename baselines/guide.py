"""
GUIDE: Graph UnSupervised Anomaly Detection via Influential Features

Reference: https://github.com/pygod-team/pygod
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pygod.nn import GUIDEBase
from typing import Tuple


class GUIDE(nn.Module):
    """
    GUIDE: Graph UnSupervised Anomaly Detection via Influential Features
    
    Based on the PyGOD implementation.
    """
    
    def __init__(self, x_dim, s_dim, hid_a, hid_s, num_layers, dropout, act=F.relu):
        super().__init__()
        self.model = GUIDEBase(
            dim_a=x_dim,
            dim_s=s_dim,
            hid_a=hid_a,
            hid_s=hid_s,
            num_layers=num_layers,
            dropout=dropout,
            act=act
        )

    def forward(self, x, s, edge_index) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning reconstructed features and structures."""
        return self.model(x, s, edge_index)

    def get_embedding(self, x, s, edge_index) -> torch.Tensor:
        """Get embeddings for CLE."""
        x_recon, _ = self.model(x, s, edge_index)
        return x_recon

    def loss_func(self, x, x_, s, s_, alpha=0.5):
        """Compute GUIDE loss function."""
        diff_attribute = torch.pow(x - x_, 2)
        attribute_errors = torch.sqrt(torch.sum(diff_attribute, 1))

        diff_structure = torch.pow(s - s_, 2)
        structure_errors = torch.sqrt(torch.sum(diff_structure, 1))

        score = alpha * attribute_errors + (1 - alpha) * structure_errors
        loss = torch.mean(score)

        return score, loss


def calculate_structural_features(data, use_complex_motif=None, graphlet_size=4, selected_motif=True, cache_dir=None):
    """Calculate structural features for GUIDE with automatic complexity selection."""
    num_nodes = data.x.shape[0]

    if use_complex_motif is None:
        use_complex_motif = num_nodes <= 5000

    if use_complex_motif:
        print(f"Complex motif calculation requested but currently disabled. Using simplified calculation.")
        s = calculate_simplified_statistics(data)
    else:
        print("Using simplified structural statistics (fast mode)")
        s = calculate_simplified_statistics(data)

    return s


def calculate_simplified_statistics(data):
    """Simplified structural feature calculation using basic graph statistics."""
    from torch_geometric.utils import degree

    edge_index = data.edge_index
    num_nodes = data.x.shape[0]

    node_degrees = degree(edge_index[0], num_nodes=num_nodes)

    s = []

    deg_norm = node_degrees / (node_degrees.max() + 1e-8)
    s.append(deg_norm.unsqueeze(1))

    s.append((node_degrees / (num_nodes - 1)).unsqueeze(1))

    median_deg = torch.median(node_degrees)
    high_deg = (node_degrees > median_deg).float()
    s.append(high_deg.unsqueeze(1))

    deg_std = torch.std(node_degrees.float())
    if deg_std > 0:
        deg_zscore = (node_degrees.float() - torch.mean(node_degrees.float())) / deg_std
        s.append(torch.sigmoid(deg_zscore).unsqueeze(1))

    s = torch.cat(s, dim=1)
    print(f"Simplified statistics calculated. Feature dimension: {s.shape[1]}")

    return s
