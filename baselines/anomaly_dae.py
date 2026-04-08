"""
AnomalyDAE Model Definition

Reference: https://github.com GrailOfTheMind/AnomalyDAE
"""

import torch
import torch.nn as nn
import numpy as np
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from typing import Tuple


class NodeAttention(nn.Module):
    """Node-level attention mechanism."""
    
    def __init__(self, in_sz, out_sz, nb_nodes, dropout=0.):
        super().__init__()
        self.in_sz = in_sz
        self.out_sz = out_sz
        self.nb_nodes = nb_nodes
        self.dropout = dropout

        self.conv_seq = nn.Conv1d(in_sz, out_sz, 1, bias=False)
        self.conv_f1 = nn.Conv1d(out_sz, 1, 1)
        self.conv_f2 = nn.Conv1d(out_sz, 1, 1)

    def forward(self, inputs, adj):
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(0)

        seq_fts = self.conv_seq(inputs.transpose(1, 2)).transpose(1, 2)
        f_1_t = self.conv_f1(seq_fts.transpose(1, 2))
        f_2_t = self.conv_f2(seq_fts.transpose(1, 2))
        f_1 = f_1_t.view(self.nb_nodes, 1)
        f_2 = f_2_t.view(self.nb_nodes, 1)

        adj_dense = adj.to_dense() if adj.is_sparse else adj
        f_1 = adj_dense * f_1
        f_2 = adj_dense * f_2.t()
        logits = f_1 + f_2
        coefs = F.softmax(logits, dim=1)
        seq_fts_squeezed = seq_fts.squeeze(1)
        vals = torch.mm(coefs, seq_fts_squeezed)
        return vals


class InnerDecoder(nn.Module):
    """Inner product decoder."""
    
    def __init__(self, act_struc=F.sigmoid, act_attr=lambda x: x):
        super().__init__()
        self.act_struc = act_struc
        self.act_attr = act_attr

    def forward(self, inputs):
        z_u, z_a = inputs
        z_u_t = z_u.t()
        structure_outputs = self.act_struc(torch.mm(z_u, z_u_t))
        z_a_t = z_a.t()
        attr_outputs = self.act_attr(torch.mm(z_u, z_a_t))
        return structure_outputs, attr_outputs


class Dense(nn.Module):
    """Dense layer with optional sparse input support."""
    
    def __init__(self, input_dim, output_dim, act=lambda x: x, sparse_inputs=False, dropout=0.):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.act = act
        self.sparse_inputs = sparse_inputs
        self.dropout = dropout
        self.weight = Parameter(torch.FloatTensor(input_dim, output_dim))
        self.bias = Parameter(torch.FloatTensor(output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs):
        if self.sparse_inputs:
            output = torch.spmm(inputs, self.weight)
        else:
            output = torch.mm(inputs, self.weight)
        output = output + self.bias
        if self.act is not None:
            output = self.act(output)
        if self.dropout > 0:
            output = F.dropout(output, self.dropout, training=self.training)
        return output


class AnomalyDAE(nn.Module):
    """
    AnomalyDAE: Dual Autoencoder for Anomaly Detection
    
    Uses node attention and dual decoders for structure and attribute reconstruction.
    """
    
    def __init__(self, num_features, num_nodes, hidden1, hidden2, decoder_act=[F.sigmoid, lambda x: x], dropout=0.):
        super().__init__()
        self.num_features = num_features
        self.num_nodes = num_nodes
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.dropout = dropout

        self.struct_dense1 = Dense(num_features, hidden1, act=F.relu, sparse_inputs=True, dropout=dropout)
        self.attr_dense1 = Dense(num_nodes, hidden1, act=F.relu, sparse_inputs=False, dropout=dropout)
        self.attr_dense2 = Dense(hidden1, hidden2, act=lambda x: x, dropout=dropout)
        self.node_attention = NodeAttention(in_sz=hidden1, out_sz=hidden2, nb_nodes=num_nodes, dropout=dropout)
        self.inner_decoder = InnerDecoder(act_struc=decoder_act[0], act_attr=decoder_act[1])

    def forward(self, features, adj) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning structure_recon, attribute_recon, embeddings."""
        hidden1 = self.struct_dense1(features)
        hidden1_expanded = hidden1.unsqueeze(1)
        embeddings_s = self.node_attention(hidden1_expanded, adj)

        features_t = features.t()
        hidden2 = self.attr_dense1(features_t)
        embeddings_a = self.attr_dense2(hidden2)

        structure_recon, attribute_recon = self.inner_decoder((embeddings_s, embeddings_a))

        return structure_recon, attribute_recon, embeddings_s

    def get_embedding(self, features, adj) -> torch.Tensor:
        """Get structure embeddings."""
        hidden1 = self.struct_dense1(features)
        hidden1_expanded = hidden1.unsqueeze(1)
        return self.node_attention(hidden1_expanded, adj)


def anomaly_dae_loss_func(adj, A_hat, attrs, X_hat, alpha=0.8):
    """Compute AnomalyDAE loss."""
    device = attrs.device
    if adj.device != device:
        adj = adj.to(device)

    diff_attribute = torch.pow(X_hat - attrs, 2)
    attribute_reconstruction_errors = torch.sqrt(torch.sum(diff_attribute, 1))
    attribute_cost = torch.mean(attribute_reconstruction_errors)

    diff_structure = torch.pow(A_hat - adj, 2)
    structure_reconstruction_errors = torch.sqrt(torch.sum(diff_structure, 1))
    structure_cost = torch.mean(structure_reconstruction_errors)

    reconstruction_errors = alpha * attribute_reconstruction_errors + (1 - alpha) * structure_reconstruction_errors
    cost = alpha * attribute_cost + (1 - alpha) * structure_cost

    return reconstruction_errors, cost, structure_cost, attribute_cost
