"""DOMINANT model and training functions for DOMINANT+CLE joint training.

DOMINANT (Ding et al., 2019): GCN-based autoencoder for anomaly detection
on attributed networks. Reconstructs both adjacency matrix A and node features X
via a shared GCN encoder with separate structure/attribute decoders.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import math
import numpy as np
import scipy.sparse as sp
import time
from datetime import datetime, timedelta
from torch_geometric.utils import to_dense_adj
from torch.optim import Adam

from .cle import CLERegression, LinearFlowNoise, MLP
from .utils import (_normalize_vector, _center_cols, _normalize_cols,
    _procrustes_align, _sign_fix, _align_embedding, _compute_combined_score,
    compute_all_metrics, LossNormalizer, format_time_precise)


# ==================== DOMINANT Model Components ====================

class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class Encoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(Encoder, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        return x


class Attribute_Decoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(Attribute_Decoder, self).__init__()
        self.gc1 = GraphConvolution(nhid, nhid)
        self.gc2 = GraphConvolution(nhid, nfeat)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        return x


class Structure_Decoder(nn.Module):
    def __init__(self, nhid, dropout):
        super(Structure_Decoder, self).__init__()
        self.gc1 = GraphConvolution(nhid, nhid)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = x @ x.T
        return x


class Dominant(nn.Module):
    def __init__(self, feat_size, hidden_size, dropout):
        super(Dominant, self).__init__()
        self.shared_encoder = Encoder(feat_size, hidden_size, dropout)
        self.attr_decoder = Attribute_Decoder(feat_size, hidden_size, dropout)
        self.struct_decoder = Structure_Decoder(hidden_size, dropout)

    def forward(self, x, adj):
        x = self.shared_encoder(x, adj)
        x_hat = self.attr_decoder(x, adj)
        struct_reconstructed = self.struct_decoder(x, adj)
        return struct_reconstructed, x_hat, x


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def ae_loss_func(adj, A_hat, attrs, X_hat, alpha):
    """AE loss function: alpha * attr_error + (1-alpha) * struct_error."""
    diff_attribute = torch.pow(X_hat - attrs, 2)
    attribute_reconstruction_errors = torch.sqrt(torch.sum(diff_attribute, 1))
    attribute_cost = torch.mean(attribute_reconstruction_errors)

    diff_structure = torch.pow(A_hat - adj, 2)
    structure_reconstruction_errors = torch.sqrt(torch.sum(diff_structure, 1))
    structure_cost = torch.mean(structure_reconstruction_errors)

    cost = alpha * attribute_reconstruction_errors + (1 - alpha) * structure_reconstruction_errors

    return cost, structure_cost, attribute_cost


# ==================== Training Functions ====================

def _train_single_joint(data, epochs=100, ae_hidden=64, cle_hidden=None,
                        device=None, seed=42, lamda1=0.5, lamda2=0.5,
                        normalize_loss=True, normalize_method='exponential_moving_average',
                        normalize_scores=True, score_norm_method='min_max',
                        use_embedding_transform=True, joint_training=True, verbose=True,
                        eval_x=None, target_x=None,
                        dropout=0.3, lr_ae=5e-3, struct_weight=0.8):
    """Train a single DOMINANT(+CLE) model on given graph.

    Parameters
    ----------
    data : PyG Data
    epochs : int
    ae_hidden : int
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
    dropout : float
    lr_ae : float
    struct_weight : float — α for structure vs attribute balance

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
    target = target_x.to(device) if target_x is not None else x
    edge_index = data.edge_index

    # Process adjacency
    adj_tensor = to_dense_adj(edge_index, max_num_nodes=x.size(0))[0]
    adj_np = adj_tensor.cpu().numpy()
    adj_np_selfloop = adj_np + sp.eye(adj_np.shape[0])
    adj_norm = normalize_adj(adj_np_selfloop).toarray()
    adj_norm = torch.FloatTensor(adj_norm).to(device)
    adj_label = torch.FloatTensor(adj_np_selfloop).to(device)

    # ---- AE model ----
    ae_model = Dominant(feat_size=x.size(1), hidden_size=ae_hidden, dropout=dropout).to(device)
    ae_optimizer = torch.optim.Adam(ae_model.parameters(), lr=lr_ae)

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

        ae_model.eval()
        with torch.no_grad():
            _, _, emb0 = ae_model(x, adj_norm)
            if use_embedding_transform:
                emb_ref = _normalize_cols(_center_cols(emb0))
            else:
                emb_ref = emb0
        ae_model.train()

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
        ae_loss, _, _ = ae_loss_func(adj_label, A_hat, target, X_hat, struct_weight)
        ae_loss_mean = torch.mean(ae_loss)

        if joint_training:
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

            if verbose and (epoch % 20 == 0 or epoch == epochs - 1):
                print("  Epoch {:04d} | AE: {:.5f} | CLE: {:.5f} | Joint: {:.5f}".format(
                    epoch, ae_loss_mean.item(), cle_loss.item(), joint_loss.item()))
        else:
            ae_loss_mean.backward()
            ae_optimizer.step()

            if verbose and (epoch % 20 == 0 or epoch == epochs - 1):
                print("  Epoch {:04d} | AE Loss: {:.5f}".format(epoch, ae_loss_mean.item()))

    # ---- Get scores ----
    eval_feat = eval_x.to(device) if eval_x is not None else x
    eval_target = target_x.to(device) if target_x is not None else eval_feat
    ae_model.eval()
    with torch.no_grad():
        A_hat, X_hat, emb = ae_model(eval_feat, adj_norm)
        ae_loss, _, _ = ae_loss_func(adj_label, A_hat, eval_target, X_hat, struct_weight)
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


def train_joint_ae_cle(data, epochs=100, ae_hidden=64, cle_hidden=None,
                       batch_size=64, device=None, normalize_loss=True,
                       normalize_method='exponential_moving_average',
                       lamda1=0.5, lamda2=0.5, normalize_scores=True,
                       score_norm_method='min_max', joint_training=True,
                       dataset_name='unknown', use_embedding_transform=True,
                       dropout=0.3, lr_ae=5e-3, struct_weight=0.8):
    """Joint training of DOMINANT + CLE models.

    Parameters
    ----------
    data : PyG Data
    epochs : int
    ae_hidden : int — hidden size for AE
    cle_hidden : list[int]
    batch_size : int
    device : torch.device | None
    normalize_loss : bool
    normalize_method : str
    lamda1 : float — training-time CLE weight
    lamda2 : float — eval-time CLE weight
    normalize_scores : bool
    score_norm_method : str
    joint_training : bool
    dataset_name : str
    use_embedding_transform : bool
    dropout : float
    lr_ae : float
    struct_weight : float — α for structure vs attribute balance

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
    print("Base model: DOMINANT")

    training_start_time = time.time()
    print("Training started at: {}".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

    x = data.x.to(device)
    y = data.y.bool()
    edge_index = data.edge_index
    adj = to_dense_adj(edge_index)[0]

    # Process adjacency
    adj_np = adj.cpu().numpy()
    adj_np_selfloop = adj_np + sp.eye(adj_np.shape[0])
    adj_norm = normalize_adj(adj_np_selfloop).toarray()
    adj = torch.FloatTensor(adj_norm).to(device)
    adj_label = torch.FloatTensor(adj_np_selfloop).to(device)

    # Initialize AE model
    ae_model = Dominant(feat_size=x.size(1), hidden_size=ae_hidden, dropout=dropout).to(device)
    ae_optimizer = torch.optim.Adam(ae_model.parameters(), lr=lr_ae)
    cle_optimizer = None

    # Initialize CLE model (only if joint training)
    cle_model = None
    if joint_training:
        cle_model = CLERegression(
            hidden_size=cle_hidden, epochs=300, batch_size=batch_size,
            lr=1e-4, weight_decay=5e-4, T=400, num_bins=1, device=device
        )

    print("\n" + "=" * 60)
    if joint_training:
        print("Phase 1: Joint Training DOMINANT + CLE (Unsupervised)")
        print("AE:1.0, CLE:lamda1={}".format(lamda1))
    else:
        print("Phase 1: Training DOMINANT Only (Unsupervised)")
    if normalize_loss and joint_training:
        print("Loss Normalization: {}".format(normalize_method))
    print("=" * 60)

    loss_normalizer = LossNormalizer(method=normalize_method) if (normalize_loss and joint_training) else None

    # Reference embedding for alignment
    emb_ref = None
    if joint_training:
        ae_model.eval()
        with torch.no_grad():
            A0, X0, emb0 = ae_model(x, adj)
            if use_embedding_transform:
                emb_ref = _normalize_cols(_center_cols(emb0))
            else:
                emb_ref = emb0
        ae_model.train()

        flow = LinearFlowNoise(dim=emb_ref.shape[1], ridge=1e-3, device=device, dtype=emb_ref.dtype)
        flow.fit(emb_ref)
        cle_model.noise_flow = flow.eval()

    # Training loop
    epoch_times = []
    for epoch in range(epochs):
        epoch_start_time = time.time()

        ae_model.train()
        if joint_training and cle_model.model is not None:
            cle_model.model.train()

        ae_optimizer.zero_grad()
        if joint_training:
            if cle_model.model is not None and cle_optimizer is None:
                cle_optimizer = Adam(cle_model.model.parameters(), lr=1e-4, weight_decay=5e-4)
            if cle_optimizer is not None:
                cle_optimizer.zero_grad()

        # AE forward
        A_hat, X_hat, emb = ae_model(x, adj)
        ae_loss, _, _ = ae_loss_func(adj_label, A_hat, x, X_hat, struct_weight)
        ae_loss_mean = torch.mean(ae_loss)

        if joint_training:
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
                A_hat, X_hat, emb = ae_model(x, adj)
                ae_loss, _, _ = ae_loss_func(adj_label, A_hat, x, X_hat, struct_weight)
                ae_score = ae_loss.detach().cpu().numpy()

                if normalize_scores:
                    ae_score_disp = _normalize_vector(ae_score, method=score_norm_method)
                else:
                    ae_score_disp = ae_score

                y_np = y.cpu().numpy()
                ae_m = compute_all_metrics(y_np, ae_score_disp)

                if joint_training:
                    cle_model.model.eval()
                    if use_embedding_transform:
                        emb_eval_aligned = _align_embedding(emb, emb_ref)
                    else:
                        emb_eval_aligned = emb
                    cle_score = cle_model.predict_score(emb_eval_aligned)
                    combined_score = _compute_combined_score(
                        ae_score_disp, cle_score, normalize_scores, score_norm_method, lamda2)

                    cle_m = compute_all_metrics(y_np, cle_score)
                    combined_m = compute_all_metrics(y_np, combined_score)

                    print("  -> AE:       AUC:{auc:.4f} AUPRC:{auprc:.4f} P@{k}:{precision_at_k:.4f} R@{k}:{recall_at_k:.4f}".format(**ae_m))
                    print("  -> CLE:      AUC:{auc:.4f} AUPRC:{auprc:.4f} P@{k}:{precision_at_k:.4f} R@{k}:{recall_at_k:.4f}".format(**cle_m))
                    print("  -> Combined: AUC:{auc:.4f} AUPRC:{auprc:.4f} P@{k}:{precision_at_k:.4f} R@{k}:{recall_at_k:.4f}".format(**combined_m))
                else:
                    print("  -> AE: AUC:{auc:.4f} AUPRC:{auprc:.4f} P@{k}:{precision_at_k:.4f} R@{k}:{recall_at_k:.4f}".format(**ae_m))

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
        A_hat, X_hat, emb = ae_model(x, adj)
        ae_loss, _, _ = ae_loss_func(adj_label, A_hat, x, X_hat, struct_weight)
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
            print("  AE:       AUC:{auc:.6f}  AUPRC:{auprc:.6f}  P@{k}:{precision_at_k:.4f}  R@{k}:{recall_at_k:.4f}".format(**ae_metrics))
            print("  CLE:      AUC:{auc:.6f}  AUPRC:{auprc:.6f}  P@{k}:{precision_at_k:.4f}  R@{k}:{recall_at_k:.4f}".format(**cle_metrics))
            print("  Combined: AUC:{auc:.6f}  AUPRC:{auprc:.6f}  P@{k}:{precision_at_k:.4f}  R@{k}:{recall_at_k:.4f}".format(**combined_metrics))
            print("  (K = {} anomalies)".format(int(y_np.sum())))

            return ae_model, cle_model, combined_metrics
        else:
            ae_metrics = compute_all_metrics(y_np, ae_score)
            print("\nFinal Results:")
            print("  AE: AUC:{auc:.6f}  AUPRC:{auprc:.6f}  P@{k}:{precision_at_k:.4f}  R@{k}:{recall_at_k:.4f}".format(**ae_metrics))
            print("  (K = {} anomalies)".format(int(y_np.sum())))

            return ae_model, ae_metrics
