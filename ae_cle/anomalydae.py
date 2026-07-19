"""AnomalyDAE model and training functions for AnomalyDAE+CLE joint training.

AnomalyDAE (Fan et al., ICASSP 2020): Dual Autoencoder for anomaly detection on attributed networks.
Uses a structure attention branch + attribute MLP branch with shared inner-product decoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import math
import time
from datetime import datetime, timedelta
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_geometric.utils import to_dense_adj
from torch.optim import Adam

from .cle import CLERegression, LinearFlowNoise, MLP
from .utils import (_normalize_vector, _center_cols, _normalize_cols,
    _procrustes_align, _sign_fix, _align_embedding, _compute_combined_score,
    compute_all_metrics, LossNormalizer, format_time_precise, format_timedelta_precise)


# ==================== AnomalyDAE Model Components ====================

class NodeAttention(nn.Module):
    """Graph attention via Conv1d + adjacency-masked softmax (from original AnomalyDAE)."""
    def __init__(self, in_sz, out_sz, nb_nodes, dropout=0.):
        super(NodeAttention, self).__init__()
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
    """Inner-product decoder for structure and attribute reconstruction."""
    def __init__(self, act_struc=F.sigmoid, act_attr=lambda x: x):
        super(InnerDecoder, self).__init__()
        self.act_struc = act_struc
        self.act_attr = act_attr

    def forward(self, inputs):
        z_u, z_a = inputs
        structure_outputs = self.act_struc(torch.mm(z_u, z_u.t()))
        attr_outputs = self.act_attr(torch.mm(z_u, z_a.t()))
        return structure_outputs, attr_outputs


class Dense(nn.Module):
    """Simple dense layer."""
    def __init__(self, input_dim, output_dim, act=lambda x: x, sparse_inputs=False, dropout=0.):
        super(Dense, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.act = act
        self.sparse_inputs = sparse_inputs
        self.dropout = dropout

        self.weight = Parameter(torch.FloatTensor(input_dim, output_dim))
        self.bias = Parameter(torch.FloatTensor(output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
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
    """PyTorch implementation of AnomalyDAE (Fan et al., ICASSP 2020).

    Dual autoencoder with structure attention + attribute MLP branches.
    """
    def __init__(self, num_features, num_nodes, hidden1, hidden2, decoder_act=None, dropout=0.):
        super(AnomalyDAE, self).__init__()
        if decoder_act is None:
            decoder_act = [F.sigmoid, lambda x: x]
        self.num_features = num_features
        self.num_nodes = num_nodes
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.dropout = dropout

        # Structure branch
        self.struct_dense1 = Dense(num_features, hidden1, act=F.relu, sparse_inputs=True, dropout=dropout)
        # Attribute branch
        self.attr_dense1 = Dense(num_nodes, hidden1, act=F.relu, sparse_inputs=False, dropout=dropout)
        self.attr_dense2 = Dense(hidden1, hidden2, act=lambda x: x, dropout=dropout)
        # Attention for structure
        self.node_attention = NodeAttention(in_sz=hidden1, out_sz=hidden2, nb_nodes=num_nodes, dropout=dropout)
        # Decoder
        self.inner_decoder = InnerDecoder(act_struc=decoder_act[0], act_attr=decoder_act[1])

    def forward(self, features, adj):
        # Structure branch
        hidden1 = self.struct_dense1(features)
        hidden1_expanded = hidden1.unsqueeze(1)
        embeddings_s = self.node_attention(hidden1_expanded, adj)

        # Attribute branch
        features_t = features.t()
        hidden2 = self.attr_dense1(features_t)
        embeddings_a = self.attr_dense2(hidden2)

        # Decode
        structure_recon, attribute_recon = self.inner_decoder((embeddings_s, embeddings_a))
        return structure_recon, attribute_recon, embeddings_s


# ==================== Utility Functions ====================

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def ae_loss_func(adj, A_hat, attrs, X_hat, alpha):
    """Unsupervised AE reconstruction loss for AnomalyDAE.

    Returns
    -------
    reconstruction_errors : torch.Tensor (n_nodes,) — per-node anomaly scores
    cost : torch.Tensor scalar — mean cost for training
    structure_cost : torch.Tensor scalar
    attribute_cost : torch.Tensor scalar
    """
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


# ==================== Training Functions ====================

def _train_single_joint_anomalydae(data, epochs=100, ae_hidden=64, ae_dropout=0.3,
                                    cle_hidden=None, device=None, seed=42,
                                    lamda1=0.5, lamda2=0.5,
                                    normalize_loss=True, normalize_method='exponential_moving_average',
                                    normalize_scores=True, score_norm_method='min_max',
                                    use_embedding_transform=True, joint_training=True, verbose=True,
                                    use_adaptive_prior=True):
    """Train a single AnomalyDAE(+CLE) model on given graph.

    Parallel to _train_single_joint in training.py, but uses AnomalyDAE instead of DOMINANT.

    Parameters
    ----------
    data : PyG Data
    epochs : int
    ae_hidden : int — base hidden size (hidden1 = ae_hidden*2, hidden2 = ae_hidden)
    ae_dropout : float — dropout rate for AnomalyDAE layers
    cle_hidden : list[int] | None
    device : torch.device | None
    seed : int
    lamda1 : float — training-time CLE weight
    lamda2 : float — eval-time CLE weight
    normalize_loss : bool
    normalize_method : str
    normalize_scores : bool
    score_norm_method : str
    use_embedding_transform : bool
    joint_training : bool
    verbose : bool

    Returns
    -------
    ae_score : np.ndarray (n_nodes,)
    cle_score : np.ndarray (n_nodes,) or None
    combined_score : np.ndarray (n_nodes,)
    """
    if cle_hidden is None:
        cle_hidden = [256, 512, 256]
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(seed)
    np.random.seed(seed)

    x = data.x.to(device)
    edge_index = data.edge_index

    # Process adjacency
    adj_tensor = to_dense_adj(edge_index, max_num_nodes=x.size(0))[0]
    adj_np = adj_tensor.cpu().numpy()
    adj_np_selfloop = adj_np + sp.eye(adj_np.shape[0])
    adj_norm = normalize_adj(adj_np_selfloop).toarray()
    adj_norm = torch.FloatTensor(adj_norm).to(device)
    adj_label = torch.FloatTensor(adj_np_selfloop).to(device)

    # ---- AnomalyDAE model ----
    hidden1 = ae_hidden * 2
    hidden2 = ae_hidden
    ae_model = AnomalyDAE(num_features=x.size(1), num_nodes=x.size(0),
                           hidden1=hidden1, hidden2=hidden2, dropout=ae_dropout).to(device)
    ae_optimizer = torch.optim.Adam(ae_model.parameters(), lr=5e-3)

    # ---- CLE model (only if joint training) ----
    cle_model = None
    cle_optimizer = None
    emb_ref = None
    loss_normalizer = None

    if joint_training:
        cle_model = CLERegression(
            hidden_size=cle_hidden, epochs=300, batch_size=64,
            lr=1e-4, weight_decay=5e-4, T=400, num_bins=1, device=device
        )

        # Reference embedding for alignment
        ae_model.eval()
        with torch.no_grad():
            _, _, emb0 = ae_model(x, adj_norm)
            if use_embedding_transform:
                emb_ref = _normalize_cols(_center_cols(emb0))
            else:
                emb_ref = emb0
        ae_model.train()

        # Fit noise flow on reference embedding
        if use_adaptive_prior:
            flow = LinearFlowNoise(dim=emb_ref.shape[1], ridge=1e-3, device=device, dtype=emb_ref.dtype)
            flow.fit(emb_ref)
            cle_model.noise_flow = flow.eval()

        if normalize_loss:
            loss_normalizer = LossNormalizer(method=normalize_method)

    # ---- Training loop ----
    for epoch in range(epochs):
        ae_model.train()
        if joint_training and cle_model.model is not None:
            cle_model.model.train()

        ae_optimizer.zero_grad()
        if cle_optimizer is not None:
            cle_optimizer.zero_grad()

        # AE forward
        A_hat, X_hat, emb = ae_model(x, adj_norm)
        # AnomalyDAE loss_func returns (errors, cost, struct_cost, attr_cost)
        ae_loss, ae_loss_mean, _, _ = ae_loss_func(adj_label, A_hat, x, X_hat, alpha=0.8)

        if joint_training:
            # Align embedding
            if use_embedding_transform:
                emb_n = _normalize_cols(_center_cols(emb))
                emb_aligned, _ = _procrustes_align(emb_n, emb_ref)
                emb_aligned = _sign_fix(emb_aligned, emb_ref)
            else:
                emb_aligned = emb

            # CLE forward (initialize model on first batch if needed)
            if cle_model.model is None:
                cle_model.model = MLP(
                    [emb_aligned.shape[-1]] + cle_hidden,
                    num_bins=cle_model.num_bins
                ).to(device)
                cle_optimizer = Adam(cle_model.model.parameters(), lr=1e-4, weight_decay=5e-4)

            t = torch.randint(0, cle_model.T, (emb_aligned.shape[0],), device=device).long()
            cle_loss = cle_model.compute_loss(emb_aligned.detach(), t)

            # Joint loss
            if normalize_loss:
                ae_norm, cle_norm = loss_normalizer.normalize(ae_loss_mean, cle_loss)
                joint_loss = ae_norm + lamda1 * cle_norm
            else:
                joint_loss = ae_loss_mean + lamda1 * cle_loss

            joint_loss.backward()
            ae_optimizer.step()
            if cle_optimizer is not None:
                cle_optimizer.step()

            if verbose and (epoch % 20 == 0 or epoch == epochs - 1):
                print("  Epoch {:04d} | AE: {:.5f} | CLE: {:.5f} | Joint: {:.5f}".format(
                    epoch, ae_loss_mean.item(), cle_loss.item(), joint_loss.item()))
        else:
            # AE only
            ae_loss_mean.backward()
            ae_optimizer.step()

            if verbose and (epoch % 20 == 0 or epoch == epochs - 1):
                print("  Epoch {:04d} | AE Loss: {:.5f}".format(epoch, ae_loss_mean.item()))

    # ---- Get scores ----
    ae_model.eval()
    with torch.no_grad():
        A_hat, X_hat, emb = ae_model(x, adj_norm)
        ae_loss, _, _, _ = ae_loss_func(adj_label, A_hat, x, X_hat, alpha=0.8)
        ae_score = ae_loss.detach().cpu().numpy()

        if joint_training:
            cle_model.model.eval()
            if use_embedding_transform:
                emb_eval_aligned = _align_embedding(emb, emb_ref)
            else:
                emb_eval_aligned = emb
            cle_score = cle_model.predict_score(emb_eval_aligned)

            if normalize_scores:
                ae_score_n = _normalize_vector(ae_score, method=score_norm_method)
                cle_score_n = _normalize_vector(cle_score, method=score_norm_method)
            else:
                ae_score_n = ae_score
                cle_score_n = cle_score

            combined_score = 1.0 * ae_score_n + lamda2 * cle_score_n
        else:
            cle_score = None
            if normalize_scores:
                combined_score = _normalize_vector(ae_score, method=score_norm_method)
            else:
                combined_score = ae_score

    return ae_score, cle_score, combined_score


def train_joint_anomalydae_cle(data, epochs=100, ae_hidden=64, ae_dropout=0.3,
                                cle_hidden=None, batch_size=64, device=None,
                                normalize_loss=True, normalize_method='exponential_moving_average',
                                lamda1=0.5, lamda2=0.5, normalize_scores=True,
                                score_norm_method='min_max', joint_training=True,
                                dataset_name='unknown', use_embedding_transform=True,
                           use_adaptive_prior=True):
    """Joint training of AnomalyDAE + CLE models.

    Parallel to train_joint_ae_cle in training.py.

    Parameters
    ----------
    data : PyG Data
    epochs : int
    ae_hidden : int — base hidden size (hidden1 = ae_hidden*2, hidden2 = ae_hidden)
    ae_dropout : float
    cle_hidden : list[int]
    batch_size : int
    device : torch.device | None
    normalize_loss : bool
    normalize_method : str
    lamda1 : float
    lamda2 : float
    normalize_scores : bool
    score_norm_method : str
    joint_training : bool
    dataset_name : str
    use_embedding_transform : bool

    Returns
    -------
    If joint_training:
        (ae_model, cle_model, combined_metrics_dict)
    Else:
        (ae_model, ae_metrics_dict)
    """
    if cle_hidden is None:
        cle_hidden = [256, 512, 256]
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if device.type == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        device = torch.device('cpu')

    print("Using device:", device)
    print("Loading dataset: {}".format(dataset_name))
    print("Dataset info: {} nodes, {} features, {} anomalies".format(
        data.x.shape[0], data.x.shape[1], data.y.sum().item()))
    print("Base model: AnomalyDAE")

    training_start_time = time.time()
    print("Training started at: {}".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

    x = data.x
    y = data.y.bool()
    edge_index = data.edge_index

    # Process adjacency
    adj_tensor = to_dense_adj(edge_index, max_num_nodes=x.size(0))[0]
    adj_np = adj_tensor.cpu().numpy()
    adj_np_selfloop = adj_np + sp.eye(adj_np.shape[0])
    adj_norm = normalize_adj(adj_np_selfloop).toarray()
    adj_label = torch.FloatTensor(adj_np_selfloop)
    adj_norm = torch.FloatTensor(adj_norm)

    # Initialize AnomalyDAE model
    hidden1 = ae_hidden * 2
    hidden2 = ae_hidden
    ae_model = AnomalyDAE(num_features=x.size(1), num_nodes=x.size(0),
                           hidden1=hidden1, hidden2=hidden2, dropout=ae_dropout)
    ae_optimizer = torch.optim.Adam(ae_model.parameters(), lr=5e-3)

    # Move to device
    adj_norm = adj_norm.to(device)
    adj_label = adj_label.to(device)
    x = x.to(device)
    ae_model = ae_model.to(device)

    # Initialize CLE model (only if joint training)
    cle_model = None
    cle_optimizer = None
    emb_ref = None
    loss_normalizer = None

    if joint_training:
        cle_model = CLERegression(
            hidden_size=cle_hidden, epochs=300, batch_size=batch_size,
            lr=1e-4, weight_decay=5e-4, T=400, num_bins=1, device=device
        )
        loss_normalizer = LossNormalizer(method=normalize_method) if normalize_loss else None

    print("\n" + "=" * 60)
    if joint_training:
        print("Phase 1: Joint Training AnomalyDAE + CLE (Unsupervised)")
        print("AE:1.0, CLE:lamda1={}".format(lamda1))
    else:
        print("Phase 1: Training AnomalyDAE Only (Unsupervised)")
    if normalize_loss and joint_training:
        print("Loss Normalization: {}".format(normalize_method))
    print("=" * 60)

    # Reference embedding for alignment
    if joint_training:
        ae_model.eval()
        with torch.no_grad():
            _, _, emb0 = ae_model(x, adj_norm)
            if use_embedding_transform:
                emb_ref = _normalize_cols(_center_cols(emb0))
            else:
                emb_ref = emb0
        ae_model.train()

        if use_adaptive_prior:
            flow = LinearFlowNoise(dim=emb_ref.shape[1], ridge=1e-3, device=device, dtype=emb_ref.dtype)
            flow.fit(emb_ref)
            cle_model.noise_flow = flow.eval()

    # Training loop
    epoch_times = []
    for epoch in range(epochs):
        epoch_start_time = time.time()

        ae_model.train()
        ae_optimizer.zero_grad()

        if joint_training:
            if cle_model.model is not None:
                cle_model.model.train()
            if cle_model.model is not None and cle_optimizer is None:
                cle_optimizer = Adam(cle_model.model.parameters(), lr=1e-4, weight_decay=5e-4)
            if cle_optimizer is not None:
                cle_optimizer.zero_grad()

        # AE forward
        A_hat, X_hat, emb = ae_model(x, adj_norm)
        ae_loss, ae_loss_mean, _, _ = ae_loss_func(adj_label, A_hat, x, X_hat, alpha=0.8)

        if joint_training:
            # Align embedding
            if use_embedding_transform:
                emb_n = _normalize_cols(_center_cols(emb))
                emb_aligned, _ = _procrustes_align(emb_n, emb_ref)
                emb_aligned = _sign_fix(emb_aligned, emb_ref)
            else:
                emb_aligned = emb

            if cle_model.model is None:
                cle_model.model = MLP(
                    [emb_aligned.shape[-1]] + cle_hidden,
                    num_bins=cle_model.num_bins
                ).to(device)
                cle_optimizer = Adam(cle_model.model.parameters(), lr=1e-4, weight_decay=5e-4)

            t = torch.randint(0, cle_model.T, (emb_aligned.shape[0],), device=device).long()
            cle_loss = cle_model.compute_loss(emb_aligned.detach(), t)

            if normalize_loss:
                ae_norm, cle_norm = loss_normalizer.normalize(ae_loss_mean, cle_loss)
                joint_loss = ae_norm + lamda1 * cle_norm
            else:
                joint_loss = ae_loss_mean + lamda1 * cle_loss

            joint_loss.backward()
            ae_optimizer.step()
            if cle_optimizer is not None:
                cle_optimizer.step()

            if epoch % 20 == 0 or epoch == epochs - 1:
                print("Epoch: {:04d} | AE Loss: {:.5f} | CLE Loss: {:.5f} | Joint Loss: {:.5f}".format(
                    epoch, ae_loss_mean.item(), cle_loss.item(), joint_loss.item()))
        else:
            ae_loss_mean.backward()
            ae_optimizer.step()
            if epoch % 20 == 0 or epoch == epochs - 1:
                print("Epoch: {:04d} | AE Loss: {:.5f}".format(epoch, ae_loss_mean.item()))

        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)

        # Periodic evaluation
        if epoch % 20 == 0 or epoch == epochs - 1:
            ae_model.eval()
            with torch.no_grad():
                A_hat, X_hat, emb = ae_model(x, adj_norm)
                ae_loss, _, _, _ = ae_loss_func(adj_label, A_hat, x, X_hat, alpha=0.8)
                ae_score = ae_loss.detach().cpu().numpy()

                if normalize_scores:
                    ae_score_disp = _normalize_vector(ae_score, method=score_norm_method)
                else:
                    ae_score_disp = ae_score

                y_np = y.cpu().numpy()
                ae_metrics = compute_all_metrics(y_np, ae_score_disp)

                if joint_training:
                    cle_model.model.eval()
                    if use_embedding_transform:
                        emb_eval_aligned = _align_embedding(emb, emb_ref)
                    else:
                        emb_eval_aligned = emb
                    cle_score = cle_model.predict_score(emb_eval_aligned)
                    combined_score = _compute_combined_score(
                        ae_score_disp, cle_score, normalize_scores, score_norm_method, lamda2)

                    cle_metrics = compute_all_metrics(y_np, cle_score)
                    combined_metrics = compute_all_metrics(y_np, combined_score)

                    print("  -> AE:       AUC:{:.4f} AUPRC:{:.4f} R@K:{:.4f} P@K:{:.4f}".format(
                        ae_metrics['auc'], ae_metrics['auprc'],
                        ae_metrics['recall_at_k'], ae_metrics['precision_at_k']))
                    print("  -> CLE:      AUC:{:.4f} AUPRC:{:.4f} R@K:{:.4f} P@K:{:.4f}".format(
                        cle_metrics['auc'], cle_metrics['auprc'],
                        cle_metrics['recall_at_k'], cle_metrics['precision_at_k']))
                    print("  -> Combined: AUC:{:.4f} AUPRC:{:.4f} R@K:{:.4f} P@K:{:.4f}".format(
                        combined_metrics['auc'], combined_metrics['auprc'],
                        combined_metrics['recall_at_k'], combined_metrics['precision_at_k']))
                else:
                    print("  -> AE: AUC:{:.4f} AUPRC:{:.4f} R@K:{:.4f} P@K:{:.4f}".format(
                        ae_metrics['auc'], ae_metrics['auprc'],
                        ae_metrics['recall_at_k'], ae_metrics['precision_at_k']))

                elapsed_time = time.time() - training_start_time
                avg_epoch_time = np.mean(epoch_times[-20:]) if epoch_times else 0
                eta_seconds = avg_epoch_time * (epochs - epoch - 1)
                eta_str = str(timedelta(seconds=int(eta_seconds)))
                print("  -> Time: {}/epoch (avg), Elapsed: {}, ETA: {}".format(
                    format_time_precise(avg_epoch_time),
                    format_time_precise(elapsed_time), eta_str))

    # ==================== Final Evaluation ====================
    print("\n" + "=" * 60)
    print("Phase 2: Final Evaluation")
    print("=" * 60)

    ae_model.eval()
    with torch.no_grad():
        A_hat, X_hat, emb = ae_model(x, adj_norm)
        ae_loss, _, _, _ = ae_loss_func(adj_label, A_hat, x, X_hat, alpha=0.8)
        ae_score = ae_loss.detach().cpu().numpy()

        if normalize_scores:
            ae_score = _normalize_vector(ae_score, method=score_norm_method)

        y_np = y.cpu().numpy()

        if joint_training:
            if cle_model.model is None:
                raise ValueError("CLE model was not properly initialized during training")
            cle_model.model.eval()

            if use_embedding_transform:
                emb_eval_aligned = _align_embedding(emb, emb_ref)
            else:
                emb_eval_aligned = emb
            cle_score = cle_model.predict_score(emb_eval_aligned)
            combined_score = _compute_combined_score(
                ae_score, cle_score, normalize_scores, score_norm_method, lamda2)

            ae_metrics = compute_all_metrics(y_np, ae_score)
            cle_metrics = compute_all_metrics(y_np, cle_score)
            combined_metrics = compute_all_metrics(y_np, combined_score)

            print("\nFinal Results:")
            print("  AE Model:")
            print("    AUC: {:.6f} | AUPRC: {:.6f} | Recall@K: {:.6f} | Precision@K: {:.6f}".format(
                ae_metrics['auc'], ae_metrics['auprc'],
                ae_metrics['recall_at_k'], ae_metrics['precision_at_k']))
            print("  CLE Model:")
            print("    AUC: {:.6f} | AUPRC: {:.6f} | Recall@K: {:.6f} | Precision@K: {:.6f}".format(
                cle_metrics['auc'], cle_metrics['auprc'],
                cle_metrics['recall_at_k'], cle_metrics['precision_at_k']))
            print("  Combined Model:")
            print("    AUC: {:.6f} | AUPRC: {:.6f} | Recall@K: {:.6f} | Precision@K: {:.6f}".format(
                combined_metrics['auc'], combined_metrics['auprc'],
                combined_metrics['recall_at_k'], combined_metrics['precision_at_k']))
            print("  (K = {} anomalies)".format(int(y_np.sum())))

            return ae_model, cle_model, combined_metrics
        else:
            ae_metrics = compute_all_metrics(y_np, ae_score)
            print("\nFinal Results:")
            print("  AE Model:")
            print("    AUC: {:.6f} | AUPRC: {:.6f} | Recall@K: {:.6f} | Precision@K: {:.6f}".format(
                ae_metrics['auc'], ae_metrics['auprc'],
                ae_metrics['recall_at_k'], ae_metrics['precision_at_k']))
            print("  (K = {} anomalies)".format(int(y_np.sum())))

            return ae_model, ae_metrics
