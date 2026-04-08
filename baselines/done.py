"""
DONE: Deep Outlier Aware Attributed Network Embedding

Reference: https://github.com/Vincent-Liu-GitHub/DONE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, MLP
from typing import Tuple


class NeighDiff(MessagePassing):
    """Compute neighborhood difference for each edge."""
    
    def __init__(self):
        super().__init__(aggr='mean')

    def forward(self, h, edge_index):
        return self.propagate(edge_index, h=h)

    def message(self, h_i, h_j, edge_index):
        return torch.sum(torch.pow(h_i - h_j, 2), dim=1, keepdim=True)


class DONE(nn.Module):
    """
    DONE: Deep Outlier Aware Attributed Network Embedding
    
    Uses separate encoders/decoders for attributes and structure.
    """
    
    def __init__(self, x_dim, s_dim, hid_dim, num_layers, dropout, act=F.leaky_relu):
        super().__init__()

        decoder_layers = int(num_layers / 2)
        encoder_layers = num_layers - decoder_layers

        self.attr_encoder = MLP(
            in_channels=x_dim,
            hidden_channels=hid_dim,
            out_channels=hid_dim,
            num_layers=encoder_layers,
            dropout=dropout,
            act=act
        )

        self.attr_decoder = MLP(
            in_channels=hid_dim,
            hidden_channels=hid_dim,
            out_channels=x_dim,
            num_layers=decoder_layers,
            dropout=dropout,
            act=act
        )

        self.struct_encoder = MLP(
            in_channels=s_dim,
            hidden_channels=hid_dim,
            out_channels=hid_dim,
            num_layers=encoder_layers,
            dropout=dropout,
            act=act
        )

        self.struct_decoder = MLP(
            in_channels=hid_dim,
            hidden_channels=hid_dim,
            out_channels=s_dim,
            num_layers=decoder_layers,
            dropout=dropout,
            act=act
        )

        self.neigh_diff = NeighDiff()

    def forward(self, x, s, edge_index) -> Tuple:
        h_a = self.attr_encoder(x)
        x_ = self.attr_decoder(h_a)
        dna = self.neigh_diff(h_a, edge_index).squeeze()
        h_s = self.struct_encoder(s)
        s_ = self.struct_decoder(h_s)
        dns = self.neigh_diff(h_s, edge_index).squeeze()

        return x_, s_, h_a, h_s, dna, dns

    def get_embedding(self, x, s, edge_index) -> torch.Tensor:
        """Get attribute embeddings."""
        h_a = self.attr_encoder(x)
        return h_a

    def loss_func(self, x, x_, s, s_, h_a, h_s, dna, dns):
        """Compute DONE loss (equations from paper)."""
        dx = torch.sum(torch.pow(x - x_, 2), 1)
        tmp = 0.2 * dx + 0.2 * dna
        oa = tmp / (torch.sum(tmp) )

        ds = torch.sum(torch.pow(s - s_, 2), 1)
        tmp = 0.2 * ds + 0.2 * dns
        os = tmp / (torch.sum(tmp) )

        dc = torch.sum(torch.pow(h_a - h_s, 2), 1)
        oc = dc / (torch.sum(dc) )

        loss_prox_a = torch.mean(torch.log(torch.pow(oa, -1)) * dx)
        loss_hom_a = torch.mean(torch.log(torch.pow(oa, -1)) * dna)
        loss_prox_s = torch.mean(torch.log(torch.pow(os, -1)) * ds)
        loss_hom_s = torch.mean(torch.log(torch.pow(os, -1)) * dns)
        loss_c = torch.mean(torch.log(torch.pow(oc, -1)) * dc)

        loss = 0.2 * loss_prox_a + 0.2 * loss_hom_a + 0.2 * loss_prox_s + 0.2 * loss_hom_s + 0.2 * loss_c
        score = (oa + os + oc) / 3
        
        return score, loss
