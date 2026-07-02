"""GAD-NR model and training functions for GAD-NR+CLE joint training.

GAD-NR (Graph Anomaly Detection via Neighborhood Reconstruction):
Autoencoder that reconstructs node features, degree, and neighborhood structure.
Uses GNN encoder (GCN/GIN/GAT/SAGE) + PairNorm for stable embeddings.

Embedding for CLE = l1 (encoder output after GNN + PairNorm).
Anomaly score = multi-task reconstruction error per node.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from datetime import datetime, timedelta
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch_geometric.nn import GCNConv, GINConv, SAGEConv, GATConv, PNAConv

from .cle import CLERegression, LinearFlowNoise, MLP as CLEMLP
from .utils import (_normalize_vector, _center_cols, _normalize_cols,
    _procrustes_align, _sign_fix, _align_embedding, _compute_combined_score,
    compute_all_metrics, LossNormalizer, format_time_precise, format_timedelta_precise)


# ==================== GAD-NR Model Components ====================

class PairNorm(nn.Module):
    """PairNorm normalization layer for stabilizing GNN embeddings."""
    def __init__(self, mode='PN', scale=1):
        assert mode in ['None', 'PN', 'PN-SI', 'PN-SCS']
        super(PairNorm, self).__init__()
        self.mode = mode
        self.scale = scale

    def forward(self, x):
        if self.mode == 'None':
            return x
        col_mean = x.mean(dim=0)
        if self.mode == 'PN':
            x = x - col_mean
            rownorm_mean = (1e-6 + x.pow(2).sum(dim=1).mean()).sqrt()
            x = self.scale * x / rownorm_mean
        if self.mode == 'PN-SI':
            x = x - col_mean
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual
        if self.mode == 'PN-SCS':
            x = x - col_mean
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual - col_mean
        return x


class GADNR_MLP(nn.Module):
    """Internal MLP for GAD-NR (separate from CLE's MLP)."""
    def __init__(self, layer_num, input_dim, hidden_dim, output_dim):
        super(GADNR_MLP, self).__init__()
        assert layer_num >= 1
        if layer_num == 1:
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            layers = [nn.Linear(input_dim, hidden_dim)]
            for _ in range(layer_num - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Linear(hidden_dim, output_dim))
            self.linear = nn.Sequential(*layers)

    def forward(self, x):
        return self.linear(x)


class FNN(nn.Module):
    """Feed-forward network for GAD-NR decoders."""
    def __init__(self, input_dim, hidden_dim, output_dim, layer_num):
        super(FNN, self).__init__()
        assert layer_num >= 1
        if layer_num == 1:
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            layers = [nn.Linear(input_dim, hidden_dim)]
            for _ in range(layer_num - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Linear(hidden_dim, output_dim))
            self.linear = nn.Sequential(*layers)

    def forward(self, x):
        return self.linear(x)


class MLP_generator(nn.Module):
    """MLP for neighborhood sample generation."""
    def __init__(self, input_dim, output_dim):
        super(MLP_generator, self).__init__()
        self.linear1 = nn.Linear(input_dim, output_dim)
        self.linear2 = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class GNNStructEncoder(nn.Module):
    """GAD-NR Autoencoder: encodes node features via GNN, decodes via
    neighborhood + feature + degree reconstruction.

    Embedding for CLE = l1 (encoder output after 2 GNN layers + PairNorm).
    """

    def __init__(self, in_dim0, in_dim, hidden_dim, layer_num, sample_size, device,
                 neighbor_num_list, GNN_name="GCN", norm_mode="PN-SCS", norm_scale=20,
                 lambda_loss1=0.001, lambda_loss2=0.8, lambda_loss3=0.5):
        super(GNNStructEncoder, self).__init__()

        self.mlp0 = nn.Linear(in_dim0, hidden_dim)
        self.norm = PairNorm(norm_mode, norm_scale)
        self.out_dim = hidden_dim
        self.lambda_loss1 = lambda_loss1
        self.lambda_loss2 = lambda_loss2
        self.lambda_loss3 = lambda_loss3

        # GNN Encoder layers
        if GNN_name == "GIN":
            self.linear1 = GADNR_MLP(layer_num, hidden_dim, hidden_dim, hidden_dim)
            self.graphconv1 = GINConv(self.linear1)
            self.linear2 = GADNR_MLP(layer_num, hidden_dim, hidden_dim, hidden_dim)
            self.graphconv2 = GINConv(self.linear2)
        elif GNN_name == "GCN":
            self.graphconv1 = GCNConv(hidden_dim, hidden_dim)
            self.graphconv2 = GCNConv(hidden_dim, hidden_dim)
        elif GNN_name == "GAT":
            self.graphconv1 = GATConv(hidden_dim, hidden_dim)
            self.graphconv2 = GATConv(hidden_dim, hidden_dim)
        else:  # SAGE
            self.graphconv1 = SAGEConv(hidden_dim, hidden_dim, aggr='mean')
            self.graphconv2 = SAGEConv(hidden_dim, hidden_dim, aggr='mean')

        self.neighbor_num_list = neighbor_num_list
        self.tot_node = len(neighbor_num_list)

        # Gaussian parameters for neighborhood generation
        self.gaussian_mean = nn.Parameter(
            torch.FloatTensor(sample_size, hidden_dim).uniform_(-0.5 / hidden_dim, 0.5 / hidden_dim)).to(device)
        self.gaussian_log_sigma = nn.Parameter(
            torch.FloatTensor(sample_size, hidden_dim).uniform_(-0.5 / hidden_dim, 0.5 / hidden_dim)).to(device)

        # MLP-based neighborhood sampling parameters
        self.mlp_gaussian_mean = nn.Parameter(
            torch.FloatTensor(hidden_dim).uniform_(-0.5 / hidden_dim, 0.5 / hidden_dim)).to(device)
        self.mlp_gaussian_log_sigma = nn.Parameter(
            torch.FloatTensor(hidden_dim).uniform_(-0.5 / hidden_dim, 0.5 / hidden_dim)).to(device)

        self.mlp_mean = FNN(hidden_dim, hidden_dim, hidden_dim, 3)
        self.mlp_sigma = FNN(hidden_dim, hidden_dim, hidden_dim, 3)
        self.softplus = nn.Softplus()

        # Aggregation layers for neighborhood statistics
        self.mean_agg = SAGEConv(hidden_dim, hidden_dim, aggr='mean', normalize=False)
        deg_tensor = torch.tensor(neighbor_num_list, dtype=torch.long, device=device)
        self.std_agg = PNAConv(hidden_dim, hidden_dim, aggregators=["std"], scalers=["identity"], deg=deg_tensor)
        self.layer1_generator = MLP_generator(hidden_dim, hidden_dim)

        # Decoders
        self.degree_decoder = FNN(hidden_dim, hidden_dim, 1, 4)
        self.feature_decoder = FNN(hidden_dim, hidden_dim, in_dim, 3)
        self.degree_loss_func = nn.MSELoss()
        self.feature_loss_func = nn.MSELoss()
        self.init_projection = FNN(in_dim, hidden_dim, hidden_dim, 1)
        self.in_dim = in_dim
        self.sample_size = sample_size

    def forward_encoder(self, x, edge_index):
        """Forward pass through encoder only — returns latent embeddings."""
        h0 = self.mlp0(x)
        l1 = self.graphconv1(h0, edge_index)
        l1 = self.norm(l1)
        return l1, h0

    def reconstruction_neighbors2(self, l1, h0, edge_index):
        """Neighborhood reconstruction using simplified approach for stability."""
        try:
            if not torch.isfinite(l1).all() or not torch.isfinite(h0).all():
                tot_nodes = l1.shape[0]
                fallback_loss = torch.ones(tot_nodes, device=l1.device) * 0.1
                return torch.mean(fallback_loss), fallback_loss

            mean_neigh = self.mean_agg(h0, edge_index).detach()

            if not torch.isfinite(mean_neigh).all():
                tot_nodes = l1.shape[0]
                fallback_loss = torch.ones(tot_nodes, device=l1.device) * 0.1
                return torch.mean(fallback_loss), fallback_loss

            # Generate neighborhood samples
            self_embedding = l1.unsqueeze(0).repeat(self.sample_size, 1, 1)
            generated_mean = self.mlp_mean(self_embedding)
            generated_sigma = self.mlp_sigma(self_embedding)
            generated_sigma = torch.clamp(generated_sigma, min=-5.0, max=5.0)

            std_z = torch.randn_like(generated_mean, device=l1.device)
            std_z = torch.clamp(std_z, min=-3.0, max=3.0)
            var = generated_mean + torch.exp(torch.clamp(generated_sigma, max=2.0)) * std_z
            nhij = self.layer1_generator(var)
            nhij = torch.where(torch.isfinite(nhij), nhij, torch.zeros_like(nhij))

            generated_mean_final = torch.mean(nhij, dim=0)

            if generated_mean_final.shape != mean_neigh.shape:
                tot_nodes = l1.shape[0]
                fallback_loss = torch.ones(tot_nodes, device=l1.device) * 0.1
                return torch.mean(fallback_loss), fallback_loss

            diff = generated_mean_final - mean_neigh
            neigh_loss = torch.mean(diff ** 2, dim=1)
            neigh_loss = torch.where(torch.isfinite(neigh_loss), neigh_loss, torch.ones_like(neigh_loss) * 0.1)
            neigh_loss = torch.clamp(neigh_loss, min=1e-8, max=100.0)

            reg_loss = 0.01 * torch.mean(torch.clamp(generated_sigma ** 2, max=10.0))
            total_loss = neigh_loss + reg_loss

            if not torch.isfinite(total_loss).all():
                tot_nodes = l1.shape[0]
                fallback_loss = torch.ones(tot_nodes, device=l1.device) * 0.1
                return torch.mean(fallback_loss), fallback_loss

            return torch.mean(total_loss), total_loss

        except Exception:
            tot_nodes = l1.shape[0]
            fallback_loss = torch.ones(tot_nodes, device=l1.device) * 0.1
            return torch.mean(fallback_loss), fallback_loss

    def neighbor_decoder(self, l1, ground_truth_degree_matrix, h0, device, x, edge_index):
        """Complete decoder: neighborhood + feature + degree reconstruction."""
        tot_nodes = l1.shape[0]

        # Degree decoder
        degree_logits = self.degree_decoder(l1).squeeze(-1)
        ground_truth_degree_matrix = ground_truth_degree_matrix.float()
        degree_loss = self.degree_loss_func(degree_logits, ground_truth_degree_matrix)
        degree_loss_per_node = (degree_logits - ground_truth_degree_matrix).pow(2)

        # Feature decoder
        feature_logits = self.feature_decoder(l1)
        feature_loss = self.feature_loss_func(feature_logits, x)
        feature_loss_per_node = (feature_logits - x).pow(2).mean(1)

        # Neighborhood reconstruction
        h_loss, h_loss_per_node = self.reconstruction_neighbors2(l1, h0, edge_index)

        # Combine losses
        loss = self.lambda_loss1 * h_loss + degree_loss * self.lambda_loss3 + self.lambda_loss2 * feature_loss
        loss_per_node = (self.lambda_loss1 * h_loss_per_node +
                         degree_loss_per_node * self.lambda_loss3 +
                         self.lambda_loss2 * feature_loss_per_node)

        return loss, loss_per_node, h_loss_per_node, degree_loss_per_node, feature_loss_per_node

    def forward(self, edge_index, x, ground_truth_degree_matrix, device):
        """Complete forward pass — returns loss and per-node anomaly scores."""
        l1, h0 = self.forward_encoder(x, edge_index)
        loss, loss_per_node, h_loss, degree_loss, feature_loss = self.neighbor_decoder(
            l1, ground_truth_degree_matrix, h0, device, x, edge_index)
        return loss, loss_per_node, h_loss, degree_loss, feature_loss


# ==================== Utility Functions ====================

def build_neighbor_dict(edge_index, num_nodes):
    """Build neighbor dictionary from edge index."""
    neighbor_dict = {i: [] for i in range(num_nodes)}
    edge_index_np = edge_index.cpu().numpy()

    for i in range(edge_index_np.shape[1]):
        src, dst = edge_index_np[0, i], edge_index_np[1, i]
        if dst not in neighbor_dict[src]:
            neighbor_dict[src].append(dst)
        if src not in neighbor_dict[dst]:
            neighbor_dict[dst].append(src)

    return neighbor_dict


def compute_degree_matrix(edge_index, num_nodes):
    """Compute degree of each node."""
    degrees = torch.zeros(num_nodes)
    edge_index_np = edge_index.cpu()

    for i in range(edge_index_np.shape[1]):
        src, dst = edge_index_np[0, i], edge_index_np[1, i]
        degrees[src] += 1
        degrees[dst] += 1

    return degrees


# ==================== Training Functions ====================

def _train_single_joint_gadnr(data, epochs=100, gadnr_hidden=64, sample_size=10,
                               encoder='GCN', cle_hidden=None, device=None, seed=42,
                               lamda1=0.5, lamda2=0.5,
                               normalize_loss=True, normalize_method='exponential_moving_average',
                               normalize_scores=True, score_norm_method='min_max',
                               use_embedding_transform=True, joint_training=True, verbose=True):
    """Train a single GAD-NR(+CLE) model on given graph.

    Parallel to _train_single_joint (DOMINANT), _train_single_joint_anomalydae,
    and _train_single_joint_guide.

    GAD-NR uses raw edge_index (not preprocessed adjacency) and requires
    neighbor structures. Embedding for CLE = encoder output l1.

    Parameters
    ----------
    data : PyG Data
    epochs : int
    gadnr_hidden : int — hidden dimension for GNN encoder
    sample_size : int — neighborhood sample size
    encoder : str — GNN encoder type ('GCN', 'GIN', 'GAT', 'SAGE')
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
    edge_index = data.edge_index.to(device)
    num_nodes = x.shape[0]

    # Precompute neighbor structures
    neighbor_dict = build_neighbor_dict(data.edge_index, num_nodes)
    neighbor_num_list = [len(neighbor_dict[i]) for i in range(num_nodes)]
    ground_truth_degree_matrix = compute_degree_matrix(data.edge_index, num_nodes).to(device)

    # ---- GAD-NR model ----
    ae_model = GNNStructEncoder(
        in_dim0=x.shape[1],
        in_dim=x.shape[1],
        hidden_dim=gadnr_hidden,
        layer_num=2,
        sample_size=sample_size,
        device=device,
        neighbor_num_list=neighbor_num_list,
        GNN_name=encoder,
        lambda_loss1=0.001,
        lambda_loss2=0.8,
        lambda_loss3=0.5
    ).to(device)
    ae_optimizer = torch.optim.Adam(ae_model.parameters(), lr=1e-3)

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

        # Reference embedding: encoder output l1
        ae_model.eval()
        with torch.no_grad():
            l1, h0 = ae_model.forward_encoder(x, edge_index)
            if use_embedding_transform:
                emb_ref = _normalize_cols(_center_cols(l1))
            else:
                emb_ref = l1
        ae_model.train()

        # Fit noise flow on reference embedding
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
        try:
            ae_loss, ae_loss_per_node, _, _, _ = ae_model(
                edge_index, x, ground_truth_degree_matrix, device)
            ae_loss_mean = torch.mean(ae_loss)

            if torch.isnan(ae_loss_mean) or torch.isinf(ae_loss_mean):
                if verbose:
                    print("  Epoch {:04d} | GAD-NR loss is NaN/Inf, skipping".format(epoch))
                continue

            if torch.isnan(ae_loss_per_node).any() or torch.isinf(ae_loss_per_node).any():
                if verbose:
                    print("  Epoch {:04d} | GAD-NR loss_per_node has NaN/Inf, skipping".format(epoch))
                continue

        except Exception as e:
            if verbose:
                print("  Epoch {:04d} | GAD-NR forward error: {}".format(epoch, e))
            continue

        if joint_training:
            # Align embedding (l1) to reference
            l1, h0 = ae_model.forward_encoder(x, edge_index)
            if use_embedding_transform:
                emb_n = _normalize_cols(_center_cols(l1))
                emb_aligned, _ = _procrustes_align(emb_n, emb_ref)
                emb_aligned = _sign_fix(emb_aligned, emb_ref)
            else:
                emb_aligned = l1

            # Initialize CLE model on first batch if needed
            if cle_model.model is None:
                cle_model.model = CLEMLP(
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
            torch.nn.utils.clip_grad_norm_(ae_model.parameters(), max_norm=1.0)
            ae_optimizer.step()
            if cle_optimizer is not None:
                torch.nn.utils.clip_grad_norm_(cle_model.model.parameters(), max_norm=1.0)
                cle_optimizer.step()

            if verbose and (epoch % 20 == 0 or epoch == epochs - 1):
                print("  Epoch {:04d} | GAD-NR: {:.5f} | CLE: {:.5f} | Joint: {:.5f}".format(
                    epoch, ae_loss_mean.item(), cle_loss.item(), joint_loss.item()))
        else:
            # AE only
            ae_loss_mean.backward()
            torch.nn.utils.clip_grad_norm_(ae_model.parameters(), max_norm=1.0)
            ae_optimizer.step()

            if verbose and (epoch % 20 == 0 or epoch == epochs - 1):
                print("  Epoch {:04d} | GAD-NR Loss: {:.5f}".format(epoch, ae_loss_mean.item()))

    # ---- Get scores ----
    ae_model.eval()
    with torch.no_grad():
        ae_loss, ae_loss_per_node, _, _, _ = ae_model(
            edge_index, x, ground_truth_degree_matrix, device)
        ae_score = ae_loss_per_node.detach().cpu().numpy()

        if ae_score.ndim > 1:
            ae_score = ae_score.flatten()

        if joint_training:
            cle_model.model.eval()
            l1, h0 = ae_model.forward_encoder(x, edge_index)
            if use_embedding_transform:
                emb_eval_aligned = _align_embedding(l1, emb_ref)
            else:
                emb_eval_aligned = l1
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


def train_joint_gadnr_cle(data, epochs=100, gadnr_hidden=64, sample_size=10,
                           encoder='GCN', cle_hidden=None, batch_size=64, device=None,
                           normalize_loss=True, normalize_method='exponential_moving_average',
                           lamda1=0.5, lamda2=0.5, normalize_scores=True,
                           score_norm_method='min_max', joint_training=True,
                           dataset_name='unknown', use_embedding_transform=True):
    """Joint training of GAD-NR + CLE models.

    Parallel to train_joint_ae_cle (DOMINANT), train_joint_anomalydae_cle,
    and train_joint_guide_cle.

    Parameters
    ----------
    data : PyG Data
    epochs : int
    gadnr_hidden : int — hidden dimension
    sample_size : int — neighborhood sample size
    encoder : str — GNN encoder ('GCN', 'GIN', 'GAT', 'SAGE')
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
    print("Base model: GAD-NR (encoder: {})".format(encoder))

    training_start_time = time.time()
    print("Training started at: {}".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

    x = data.x.to(device)
    y = data.y.bool()
    edge_index = data.edge_index.to(device)
    num_nodes = x.shape[0]

    # Precompute neighbor structures
    neighbor_dict = build_neighbor_dict(data.edge_index, num_nodes)
    neighbor_num_list = [len(neighbor_dict[i]) for i in range(num_nodes)]
    ground_truth_degree_matrix = compute_degree_matrix(data.edge_index, num_nodes).to(device)

    # Initialize GAD-NR model
    ae_model = GNNStructEncoder(
        in_dim0=x.shape[1],
        in_dim=x.shape[1],
        hidden_dim=gadnr_hidden,
        layer_num=2,
        sample_size=sample_size,
        device=device,
        neighbor_num_list=neighbor_num_list,
        GNN_name=encoder,
        lambda_loss1=0.001,
        lambda_loss2=0.8,
        lambda_loss3=0.5
    ).to(device)
    ae_optimizer = torch.optim.Adam(ae_model.parameters(), lr=1e-3)

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
        print("Phase 1: Joint Training GAD-NR + CLE (Unsupervised)")
        print("GAD-NR:1.0, CLE:lamda1={}".format(lamda1))
    else:
        print("Phase 1: Training GAD-NR Only (Unsupervised)")
    if normalize_loss and joint_training:
        print("Loss Normalization: {}".format(normalize_method))
    print("=" * 60)

    # Reference embedding for alignment
    if joint_training:
        ae_model.eval()
        with torch.no_grad():
            l1, h0 = ae_model.forward_encoder(x, edge_index)
            if use_embedding_transform:
                emb_ref = _normalize_cols(_center_cols(l1))
            else:
                emb_ref = l1
        ae_model.train()

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
        try:
            ae_loss, ae_loss_per_node, _, _, _ = ae_model(
                edge_index, x, ground_truth_degree_matrix, device)
            ae_loss_mean = torch.mean(ae_loss)

            if torch.isnan(ae_loss_mean) or torch.isinf(ae_loss_mean):
                print("Warning: GAD-NR loss is NaN/Inf at epoch {}, skipping".format(epoch))
                continue
            if torch.isnan(ae_loss_per_node).any() or torch.isinf(ae_loss_per_node).any():
                print("Warning: GAD-NR loss_per_node has NaN/Inf at epoch {}, skipping".format(epoch))
                continue
        except Exception as e:
            print("Error in GAD-NR forward at epoch {}: {}".format(epoch, e))
            continue

        if joint_training:
            # Align embedding
            l1, h0 = ae_model.forward_encoder(x, edge_index)
            if use_embedding_transform:
                emb_n = _normalize_cols(_center_cols(l1))
                emb_aligned, _ = _procrustes_align(emb_n, emb_ref)
                emb_aligned = _sign_fix(emb_aligned, emb_ref)
            else:
                emb_aligned = l1

            if cle_model.model is None:
                cle_model.model = CLEMLP(
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
            torch.nn.utils.clip_grad_norm_(ae_model.parameters(), max_norm=1.0)
            ae_optimizer.step()
            if cle_optimizer is not None:
                torch.nn.utils.clip_grad_norm_(cle_model.model.parameters(), max_norm=1.0)
                cle_optimizer.step()

            if epoch % 20 == 0 or epoch == epochs - 1:
                print("Epoch: {:04d} | GAD-NR Loss: {:.5f} | CLE Loss: {:.5f} | Joint Loss: {:.5f}".format(
                    epoch, ae_loss_mean.item(), cle_loss.item(), joint_loss.item()))
        else:
            ae_loss_mean.backward()
            torch.nn.utils.clip_grad_norm_(ae_model.parameters(), max_norm=1.0)
            ae_optimizer.step()
            if epoch % 20 == 0 or epoch == epochs - 1:
                print("Epoch: {:04d} | GAD-NR Loss: {:.5f}".format(epoch, ae_loss_mean.item()))

        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)

        # Periodic evaluation
        if epoch % 20 == 0 or epoch == epochs - 1:
            ae_model.eval()
            with torch.no_grad():
                ae_loss, ae_loss_per_node, _, _, _ = ae_model(
                    edge_index, x, ground_truth_degree_matrix, device)

                if torch.isnan(ae_loss_per_node).any() or torch.isinf(ae_loss_per_node).any():
                    continue

                ae_score = ae_loss_per_node.detach().cpu().numpy()
                if ae_score.ndim > 1:
                    ae_score = ae_score.flatten()

                if normalize_scores:
                    ae_score_disp = _normalize_vector(ae_score, method=score_norm_method)
                else:
                    ae_score_disp = ae_score

                y_np = y.cpu().numpy()
                ae_metrics = compute_all_metrics(y_np, ae_score_disp)

                if joint_training:
                    cle_model.model.eval()
                    l1, h0 = ae_model.forward_encoder(x, edge_index)
                    if use_embedding_transform:
                        emb_eval_aligned = _align_embedding(l1, emb_ref)
                    else:
                        emb_eval_aligned = l1
                    cle_score = cle_model.predict_score(emb_eval_aligned)
                    combined_score = _compute_combined_score(
                        ae_score_disp, cle_score, normalize_scores, score_norm_method, lamda2)

                    cle_metrics = compute_all_metrics(y_np, cle_score)
                    combined_metrics = compute_all_metrics(y_np, combined_score)

                    print("  -> GAD-NR:   AUC:{:.4f} AUPRC:{:.4f} R@K:{:.4f} P@K:{:.4f}".format(
                        ae_metrics['auc'], ae_metrics['auprc'],
                        ae_metrics['recall_at_k'], ae_metrics['precision_at_k']))
                    print("  -> CLE:      AUC:{:.4f} AUPRC:{:.4f} R@K:{:.4f} P@K:{:.4f}".format(
                        cle_metrics['auc'], cle_metrics['auprc'],
                        cle_metrics['recall_at_k'], cle_metrics['precision_at_k']))
                    print("  -> Combined: AUC:{:.4f} AUPRC:{:.4f} R@K:{:.4f} P@K:{:.4f}".format(
                        combined_metrics['auc'], combined_metrics['auprc'],
                        combined_metrics['recall_at_k'], combined_metrics['precision_at_k']))
                else:
                    print("  -> GAD-NR: AUC:{:.4f} AUPRC:{:.4f} R@K:{:.4f} P@K:{:.4f}".format(
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
        ae_loss, ae_loss_per_node, _, _, _ = ae_model(
            edge_index, x, ground_truth_degree_matrix, device)
        ae_score = ae_loss_per_node.detach().cpu().numpy()

        if ae_score.ndim > 1:
            ae_score = ae_score.flatten()

        if normalize_scores:
            ae_score = _normalize_vector(ae_score, method=score_norm_method)

        y_np = y.cpu().numpy()

        if joint_training:
            if cle_model.model is None:
                raise ValueError("CLE model was not properly initialized during training")
            cle_model.model.eval()

            l1, h0 = ae_model.forward_encoder(x, edge_index)
            if use_embedding_transform:
                emb_eval_aligned = _align_embedding(l1, emb_ref)
            else:
                emb_eval_aligned = l1
            cle_score = cle_model.predict_score(emb_eval_aligned)
            combined_score = _compute_combined_score(
                ae_score, cle_score, normalize_scores, score_norm_method, lamda2)

            ae_metrics = compute_all_metrics(y_np, ae_score)
            cle_metrics = compute_all_metrics(y_np, cle_score)
            combined_metrics = compute_all_metrics(y_np, combined_score)

            print("\nFinal Results:")
            print("  GAD-NR Model:")
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
            print("  GAD-NR Model:")
            print("    AUC: {:.6f} | AUPRC: {:.6f} | Recall@K: {:.6f} | Precision@K: {:.6f}".format(
                ae_metrics['auc'], ae_metrics['auprc'],
                ae_metrics['recall_at_k'], ae_metrics['precision_at_k']))
            print("  (K = {} anomalies)".format(int(y_np.sum())))

            return ae_model, ae_metrics
