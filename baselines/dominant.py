"""
DOMINANT: Deep Anomaly Detection on Attributed Networks

Reference: https://arxiv.org/abs/1806.02371
"""

import torch
import torch.nn as nn
import scipy.sparse as sp
import numpy as np
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
from typing import Tuple, Optional


class GraphConvolution(Module):
    """Graph Convolutional Layer."""
    
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        return output


class Encoder(nn.Module):
    """DOMINANT Encoder with 2 GCN layers."""
    
    def __init__(self, nfeat, nhid, dropout):
        super().__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        return x


class AttributeDecoder(nn.Module):
    """Decoder for attribute reconstruction."""
    
    def __init__(self, nfeat, nhid, dropout):
        super().__init__()
        self.gc1 = GraphConvolution(nhid, nhid)
        self.gc2 = GraphConvolution(nhid, nfeat)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        return x


class StructureDecoder(nn.Module):
    """Decoder for structure reconstruction."""
    
    def __init__(self, nhid, dropout):
        super().__init__()
        self.gc1 = GraphConvolution(nhid, nhid)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = x @ x.T
        return x


class Dominant(nn.Module):
    """
    DOMINANT: Deep Anomaly Detection on Attributed Networks
    
    Uses shared encoder for structure and attribute reconstruction.
    """
    
    def __init__(self, feat_size, hidden_size, dropout=0.3):
        super().__init__()
        self.shared_encoder = Encoder(feat_size, hidden_size, dropout)
        self.attr_decoder = AttributeDecoder(feat_size, hidden_size, dropout)
        self.struct_decoder = StructureDecoder(hidden_size, dropout)

    def forward(self, x, adj) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
        - struct_reconstructed: Reconstructed adjacency matrix
        - x_hat: Reconstructed features
        - x: Node embeddings
        """
        x = self.shared_encoder(x, adj)
        x_hat = self.attr_decoder(x, adj)
        struct_reconstructed = self.struct_decoder(x, adj)
        return struct_reconstructed, x_hat, x

    def get_embedding(self, x, adj) -> torch.Tensor:
        """Get node embeddings."""
        return self.shared_encoder(x, adj)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def dominant_loss_func(
    adj_label: torch.Tensor,
    A_hat: torch.Tensor,
    attrs: torch.Tensor,
    X_hat: torch.Tensor,
    alpha: float = 0.8
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute DOMINANT loss.
    
    Parameters:
    - adj_label: Ground truth adjacency matrix (with self-loops)
    - A_hat: Reconstructed adjacency matrix
    - attrs: Original node features
    - X_hat: Reconstructed features
    - alpha: Weight for attribute loss
    
    Returns:
    - reconstruction_errors: Per-node anomaly scores
    - cost: Mean reconstruction loss
    - structure_cost: Structure reconstruction loss
    - attribute_cost: Attribute reconstruction loss
    """
    diff_attribute = torch.pow(X_hat - attrs, 2)
    attribute_reconstruction_errors = torch.sqrt(torch.sum(diff_attribute, 1))
    attribute_cost = torch.mean(attribute_reconstruction_errors)

    diff_structure = torch.pow(A_hat - adj_label, 2)
    structure_reconstruction_errors = torch.sqrt(torch.sum(diff_structure, 1))
    structure_cost = torch.mean(structure_reconstruction_errors)

    reconstruction_errors = alpha * attribute_reconstruction_errors + (1 - alpha) * structure_reconstruction_errors
    cost = alpha * attribute_cost + (1 - alpha) * structure_cost

    return reconstruction_errors, cost, structure_cost, attribute_cost


class DOMINANTTrainer:
    """Trainer for DOMINANT + CLE."""
    
    def __init__(
        self,
        feat_size: int,
        hidden_size: int,
        dropout: float = 0.3,
        ae_lr: float = 5e-3,
        cle_lr: float = 1e-4,
        cle_weight_decay: float = 5e-4,
        cle_T: int = 400,
        cle_num_bins: int = 1,
        cle_hidden: list = None,
        alpha: float = 0.8,
        device=None
    ):
        if cle_hidden is None:
            cle_hidden = [256, 512, 256]
            
        self.alpha = alpha
        
        # Initialize DOMINANT model
        self.model = Dominant(feat_size, hidden_size, dropout)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=ae_lr)
        
        # Will initialize CLE later when we have embeddings
        self.cle_model = None
        self.cle_optimizer = None
        
        # CLE settings
        self.cle_lr = cle_lr
        self.cle_weight_decay = cle_weight_decay
        self.cle_T = cle_T
        self.cle_num_bins = cle_num_bins
        self.cle_hidden = cle_hidden
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.model.to(self.device)
        self.emb_ref = None
        self.loss_normalizer = None
        self.normalize_method = None
        
    def _prepare_data(self, data):
        """Prepare data for training."""
        from torch_geometric.utils import to_dense_adj
        
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        y = data.y.bool().cpu().numpy()
        adj = to_dense_adj(edge_index)[0].cpu().numpy()
        
        # Add self-loops and normalize
        adj_selfloop = adj + sp.eye(adj.shape[0])
        adj_norm = normalize_adj(adj_selfloop).toarray()
        
        adj_label = torch.FloatTensor(adj_selfloop).to(self.device)
        adj = torch.FloatTensor(adj_norm).to(self.device)
        x = x.to(self.device)
        
        return x, adj, adj_label, y, edge_index
    
    def init_cle(self, emb_dim: int, emb_ref: torch.Tensor, noise_flow=None):
        """Initialize CLE model."""
        from core.cle_models import CLERegression, CLEMLP
        
        self.emb_ref = emb_ref
        self.cle_model = CLERegression(
            hidden_size=self.cle_hidden,
            epochs=300,
            batch_size=64,
            lr=self.cle_lr,
            weight_decay=self.cle_weight_decay,
            T=self.cle_T,
            num_bins=self.cle_num_bins,
            device=self.device
        )
        
        # Initialize CLE MLP
        self.cle_model.model = CLEMLP([emb_dim] + self.cle_hidden, num_bins=self.cle_num_bins).to(self.device)
        self.cle_optimizer = torch.optim.Adam(
            self.cle_model.model.parameters(),
            lr=self.cle_lr,
            weight_decay=self.cle_weight_decay
        )
        
        if noise_flow is not None:
            self.cle_model.noise_flow = noise_flow
            
    def get_embedding(self, x, adj):
        """Get embeddings from DOMINANT."""
        return self.model.get_embedding(x, adj)
    
    def get_score(self, x, adj, adj_label):
        """Get anomaly scores from DOMINANT."""
        _, _, emb = self.model(x, adj)
        A_hat, X_hat, _ = self.model(x, adj)
        scores, _, _, _ = dominant_loss_func(adj_label, A_hat, x, X_hat, self.alpha)
        return scores.detach().cpu().numpy(), emb
