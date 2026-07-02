"""GUIDE model and training functions for GUIDE+CLE joint training.

GUIDE (Graph Anomaly Detection via Motif-based Autoencoder):
Motif-aware graph autoencoder that reconstructs both node attributes and
structural features (motif/GDD or simplified graph statistics).

Uses GUIDEBase from PyGOD as the core AE. Embedding for CLE = reconstructed
attribute features x_hat (decoder output).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from datetime import datetime, timedelta
from torch.optim import Adam
from torch_geometric.utils import degree

from pygod.nn import GUIDEBase

from .cle import CLERegression, LinearFlowNoise, MLP
from .utils import (_normalize_vector, _center_cols, _normalize_cols,
    _procrustes_align, _sign_fix, _align_embedding, _compute_combined_score,
    compute_all_metrics, LossNormalizer, format_time_precise, format_timedelta_precise)


# ==================== GUIDE Model Wrapper ====================

class GUIDE_Base(nn.Module):
    """Thin wrapper around PyGOD's GUIDEBase for joint training.

    GUIDEBase: motif-aware autoencoder that reconstructs node attributes x
    and structural features s (motif counts or simplified statistics).

    Forward: (x, s, edge_index) -> (x_hat, s_hat)
    """

    def __init__(self, x_dim, s_dim, hid_a, hid_s, num_layers, dropout, act):
        super(GUIDE_Base, self).__init__()
        self.model = GUIDEBase(
            dim_a=x_dim,
            dim_s=s_dim,
            hid_a=hid_a,
            hid_s=hid_s,
            num_layers=num_layers,
            dropout=dropout,
            act=act
        )

    def forward(self, x, s, edge_index):
        """Forward pass through GUIDE model.

        Returns
        -------
        x_ : torch.Tensor — reconstructed attribute features
        s_ : torch.Tensor — reconstructed structural features
        """
        return self.model(x, s, edge_index)

    def loss_func(self, x, x_, s, s_, alpha=0.5):
        """Compute GUIDE reconstruction loss.

        Returns
        -------
        score : torch.Tensor (n_nodes,) — per-node anomaly score
        loss : torch.Tensor scalar — mean cost for training
        """
        diff_attribute = torch.pow(x - x_, 2)
        attribute_errors = torch.sqrt(torch.sum(diff_attribute, 1))

        diff_structure = torch.pow(s - s_, 2)
        structure_errors = torch.sqrt(torch.sum(diff_structure, 1))

        score = alpha * attribute_errors + (1 - alpha) * structure_errors
        loss = torch.mean(score)

        return score, loss


# ==================== Structural Feature Computation ====================

def calculate_simplified_statistics(data):
    """Simplified structural feature calculation using basic graph statistics.

    Fast but less accurate than full motif/GDD calculation.
    Used as default for GUIDE's structural input features.

    Parameters
    ----------
    data : PyG Data

    Returns
    -------
    s : torch.Tensor (n_nodes, s_dim)
    """
    edge_index = data.edge_index
    num_nodes = data.x.shape[0]

    node_degrees = degree(edge_index[0], num_nodes=num_nodes)

    s = []

    # Feature 1: Node degree (normalized)
    deg_norm = node_degrees / (node_degrees.max() + 1e-8)
    s.append(deg_norm.unsqueeze(1))

    # Feature 2: Degree centrality approximation
    s.append((node_degrees / max(num_nodes - 1, 1)).unsqueeze(1))

    # Feature 3: Binary degree indicator (high/low degree)
    median_deg = torch.median(node_degrees)
    high_deg = (node_degrees > median_deg).float()
    s.append(high_deg.unsqueeze(1))

    # Feature 4: Degree variance indicator
    deg_std = torch.std(node_degrees.float())
    if deg_std > 0:
        deg_zscore = (node_degrees.float() - torch.mean(node_degrees.float())) / deg_std
        s.append(torch.sigmoid(deg_zscore).unsqueeze(1))

    s = torch.cat(s, dim=1)
    return s


def calculate_structural_features(data, use_complex_motif=None, graphlet_size=4,
                                   selected_motif=True, cache_dir=None):
    """Calculate structural features for GUIDE with automatic complexity selection.

    Parameters
    ----------
    data : PyG Data
    use_complex_motif : bool or None
        True: use full motif/GDD calculation (GUIDEBase.calc_gdd)
        False: use simplified statistics
        None: auto-select based on dataset size (default)
    graphlet_size : int
    selected_motif : bool
    cache_dir : str or None

    Returns
    -------
    s : torch.Tensor (n_nodes, s_dim)
    """
    num_nodes = data.x.shape[0]

    if use_complex_motif is None:
        use_complex_motif = num_nodes <= 1000

    if use_complex_motif:
        print("Computing complex motif features via GUIDEBase.calc_gdd...")
        try:
            s = GUIDEBase.calc_gdd(
                data,
                cache_dir=cache_dir,
                graphlet_size=graphlet_size,
                selected_motif=selected_motif
            )
            # calc_gdd may produce fewer rows than data.x if nodes are isolated
            # (e.g., auxiliary nodes not yet connected). Pad with simplified features.
            if s.shape[0] < num_nodes:
                n_missing = num_nodes - s.shape[0]
                print("  -> {} nodes missing from motif output, padding with simplified features".format(n_missing))
                pad_s = calculate_simplified_statistics(data)
                # Assume missing nodes are at the end (auxiliary nodes convention)
                s = torch.cat([s, pad_s[-n_missing:]], dim=0)
            print("  -> motif features shape: {}".format(s.shape))
        except Exception as e:
            print("  -> calc_gdd failed ({}), falling back to simplified statistics".format(e))
            s = calculate_simplified_statistics(data)
    else:
        s = calculate_simplified_statistics(data)

    return s


# ==================== Training Functions ====================

def _train_single_joint_guide(data, epochs=100, guide_hidden_a=64, guide_hidden_s=4,
                               guide_num_layers=4, guide_dropout=0.0, guide_alpha=0.5,
                               cle_hidden=None, device=None, seed=42,
                               lamda1=0.5, lamda2=0.5,
                               normalize_loss=True, normalize_method='exponential_moving_average',
                               normalize_scores=True, score_norm_method='min_max',
                               use_embedding_transform=True, joint_training=True, verbose=True):
    """Train a single GUIDE(+CLE) model on given graph.

    Parallel to _train_single_joint (DOMINANT) and _train_single_joint_anomalydae.

    GUIDE uses raw edge_index (not preprocessed adjacency) and requires
    structural features s. Embedding for CLE = reconstructed x_hat.

    Parameters
    ----------
    data : PyG Data
    epochs : int
    guide_hidden_a : int — attribute encoder hidden size
    guide_hidden_s : int — structure encoder hidden size
    guide_num_layers : int — number of layers
    guide_dropout : float — dropout rate
    guide_alpha : float — weight balancing attribute vs structure loss
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

    # Compute structural features
    s = calculate_structural_features(data)
    if torch.isnan(s).any():
        if verbose:
            print(f"  [WARNING] NaN in structural features: {torch.isnan(s).sum().item()} values, replacing with 0")
        s = torch.nan_to_num(s, nan=0.0)
    s = s.to(device)

    # ---- GUIDE model ----
    ae_model = GUIDE_Base(
        x_dim=x.size(1),
        s_dim=s.size(1),
        hid_a=guide_hidden_a,
        hid_s=guide_hidden_s,
        num_layers=guide_num_layers,
        dropout=guide_dropout,
        act=F.relu
    ).to(device)
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

        # Reference embedding: reconstructed features x_hat from initial forward pass
        ae_model.eval()
        with torch.no_grad():
            x0, s0 = ae_model(x, s, edge_index)
            if use_embedding_transform:
                emb_ref = _normalize_cols(_center_cols(x0))
            else:
                emb_ref = x0
        ae_model.train()

        # Fit noise flow on reference embedding
        flow = LinearFlowNoise(dim=emb_ref.shape[1], ridge=1e-3, device=device, dtype=emb_ref.dtype)
        flow.fit(emb_ref)
        cle_model.noise_flow = flow.eval()

        if normalize_loss:
            loss_normalizer = LossNormalizer(method=normalize_method)

    nan_count = 0
    max_nan_epochs = 5

    # ---- Training loop ----
    for epoch in range(epochs):
        ae_model.train()
        if joint_training and cle_model.model is not None:
            cle_model.model.train()

        ae_optimizer.zero_grad()
        if cle_optimizer is not None:
            cle_optimizer.zero_grad()

        # AE forward: GUIDE reconstructs x and s
        x_, s_ = ae_model(x, s, edge_index)
        guide_score, guide_loss = ae_model.loss_func(x, x_, s, s_, guide_alpha)
        guide_loss_mean = torch.mean(guide_score)

        if joint_training:
            # Align embedding (x_hat) to reference before feeding to CLE
            if use_embedding_transform:
                emb_n = _normalize_cols(_center_cols(x_))
                emb_aligned, _ = _procrustes_align(emb_n, emb_ref)
                emb_aligned = _sign_fix(emb_aligned, emb_ref)
            else:
                emb_aligned = x_

            # Initialize CLE model on first batch if needed
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
                guide_norm, cle_norm = loss_normalizer.normalize(guide_loss_mean, cle_loss)
                joint_loss = guide_norm + lamda1 * cle_norm
            else:
                joint_loss = guide_loss_mean + lamda1 * cle_loss

            # NaN guard: detect and skip backward to avoid corrupting parameters
            try:
                loss_is_nan = torch.isnan(joint_loss).any() if isinstance(joint_loss, torch.Tensor) else False
                loss_is_inf = torch.isinf(joint_loss).any() if isinstance(joint_loss, torch.Tensor) else False
            except Exception:
                loss_is_nan = False
                loss_is_inf = False

            if loss_is_nan or loss_is_inf:
                if verbose:
                    print("  [WARNING] Invalid joint loss at epoch {}: NaN={}, Inf={}".format(
                        epoch, loss_is_nan, loss_is_inf))
                    print("    GUIDE loss: {:.5f} | CLE loss: {:.5f}".format(
                        guide_loss_mean.item(), cle_loss.item()))
                nan_count += 1

                # If GUIDE loss itself is NaN, reinitialize GUIDE model and emb_ref
                if isinstance(guide_loss_mean, torch.Tensor) and torch.isnan(guide_loss_mean):
                    if verbose:
                        print("  GUIDE loss is NaN, reinitializing GUIDE model...")
                    ae_model = GUIDE_Base(
                        x_dim=x.size(1), s_dim=s.size(1),
                        hid_a=guide_hidden_a, hid_s=guide_hidden_s,
                        num_layers=guide_num_layers, dropout=guide_dropout, act=F.relu
                    ).to(device)
                    ae_optimizer = torch.optim.Adam(ae_model.parameters(), lr=5e-3)
                    # Recompute reference embedding
                    ae_model.eval()
                    with torch.no_grad():
                        x0, s0 = ae_model(x, s, edge_index)
                        if use_embedding_transform:
                            emb_ref = _normalize_cols(_center_cols(x0))
                        else:
                            emb_ref = x0
                    ae_model.train()

                # Reset CLE if NaN persists too long
                if nan_count >= max_nan_epochs:
                    if verbose:
                        print("  Too many NaN epochs ({}), resetting CLE model...".format(nan_count))
                    if cle_model.model is not None:
                        del cle_model.model
                    cle_model.model = None
                    cle_optimizer = None
                    nan_count = 0

                # Skip this backward step
                if cle_optimizer is not None:
                    cle_optimizer.zero_grad()
                ae_optimizer.zero_grad()
                continue

            nan_count = 0  # Reset counter on healthy epoch

            joint_loss.backward()
            torch.nn.utils.clip_grad_norm_(ae_model.parameters(), max_norm=1.0)
            ae_optimizer.step()
            if cle_optimizer is not None:
                torch.nn.utils.clip_grad_norm_(cle_model.model.parameters(), max_norm=1.0)
                cle_optimizer.step()

            if verbose and (epoch % 20 == 0 or epoch == epochs - 1):
                print("  Epoch {:04d} | GUIDE: {:.5f} | CLE: {:.5f} | Joint: {:.5f}".format(
                    epoch, guide_loss_mean.item(), cle_loss.item(), joint_loss.item()))
        else:
            # AE only
            guide_loss_mean.backward()
            torch.nn.utils.clip_grad_norm_(ae_model.parameters(), max_norm=1.0)
            ae_optimizer.step()

            if verbose and (epoch % 20 == 0 or epoch == epochs - 1):
                print("  Epoch {:04d} | GUIDE Loss: {:.5f}".format(epoch, guide_loss_mean.item()))

    # ---- Get scores ----
    ae_model.eval()
    with torch.no_grad():
        x_, s_ = ae_model(x, s, edge_index)
        # Detect NaN in reconstructed outputs
        if torch.isnan(x_).any() or torch.isnan(s_).any():
            n_nan_x = torch.isnan(x_).sum().item()
            n_nan_s = torch.isnan(s_).sum().item()
            if verbose:
                print(f"  [WARNING] NaN in GUIDE output (final): x_={n_nan_x}, s_={n_nan_s}")
            x_ = torch.nan_to_num(x_, nan=0.0)
            s_ = torch.nan_to_num(s_, nan=0.0)
        ae_score, _ = ae_model.loss_func(x, x_, s, s_, guide_alpha)
        ae_score = ae_score.detach().cpu().numpy()
        if np.isnan(ae_score).any():
            n_nan_score = np.isnan(ae_score).sum()
            if verbose:
                print(f"  [WARNING] NaN in GUIDE anomaly scores (final): {n_nan_score} values, replacing with median")
            ae_score = np.nan_to_num(ae_score, nan=float(np.median(ae_score[np.isfinite(ae_score)])) if np.isfinite(ae_score).any() else 0.0)

        if joint_training:
            cle_model.model.eval()
            if use_embedding_transform:
                emb_eval_aligned = _align_embedding(x_, emb_ref)
            else:
                emb_eval_aligned = x_
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


def train_joint_guide_cle(data, epochs=100, guide_hidden_a=64, guide_hidden_s=4,
                           guide_num_layers=4, guide_dropout=0.0, guide_alpha=0.5,
                           cle_hidden=None, batch_size=64, device=None,
                           normalize_loss=True, normalize_method='exponential_moving_average',
                           lamda1=0.5, lamda2=0.5, normalize_scores=True,
                           score_norm_method='min_max', joint_training=True,
                           dataset_name='unknown', use_embedding_transform=True):
    """Joint training of GUIDE + CLE models.

    Parallel to train_joint_ae_cle (DOMINANT) and train_joint_anomalydae_cle.

    Parameters
    ----------
    data : PyG Data
    epochs : int
    guide_hidden_a : int — attribute encoder hidden size
    guide_hidden_s : int — structure encoder hidden size
    guide_num_layers : int
    guide_dropout : float
    guide_alpha : float — attribute/structure loss balance
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
    print("Base model: GUIDE")

    training_start_time = time.time()
    print("Training started at: {}".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

    x = data.x.to(device)
    y = data.y.bool()
    edge_index = data.edge_index.to(device)

    # Compute structural features
    s = calculate_structural_features(data)
    if torch.isnan(s).any():
        print("  [WARNING] NaN in structural features: {} values, replacing with 0".format(
            torch.isnan(s).sum().item()))
        s = torch.nan_to_num(s, nan=0.0)
    s = s.to(device)
    print("Structural features shape: {}".format(s.shape))

    # Initialize GUIDE model
    ae_model = GUIDE_Base(
        x_dim=x.size(1),
        s_dim=s.size(1),
        hid_a=guide_hidden_a,
        hid_s=guide_hidden_s,
        num_layers=guide_num_layers,
        dropout=guide_dropout,
        act=F.relu
    ).to(device)
    ae_optimizer = torch.optim.Adam(ae_model.parameters(), lr=5e-3)

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
        print("Phase 1: Joint Training GUIDE + CLE (Unsupervised)")
        print("GUIDE:1.0, CLE:lamda1={}".format(lamda1))
    else:
        print("Phase 1: Training GUIDE Only (Unsupervised)")
    if normalize_loss and joint_training:
        print("Loss Normalization: {}".format(normalize_method))
    print("=" * 60)

    # Reference embedding for alignment
    if joint_training:
        ae_model.eval()
        with torch.no_grad():
            x0, s0 = ae_model(x, s, edge_index)
            if use_embedding_transform:
                emb_ref = _normalize_cols(_center_cols(x0))
            else:
                emb_ref = x0
        ae_model.train()

        flow = LinearFlowNoise(dim=emb_ref.shape[1], ridge=1e-3, device=device, dtype=emb_ref.dtype)
        flow.fit(emb_ref)
        cle_model.noise_flow = flow.eval()

    # Training loop
    nan_count = 0
    max_nan_epochs = 5
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
        x_, s_ = ae_model(x, s, edge_index)
        guide_score, guide_loss = ae_model.loss_func(x, x_, s, s_, guide_alpha)
        guide_loss_mean = torch.mean(guide_score)

        if joint_training:
            # Align embedding
            if use_embedding_transform:
                emb_n = _normalize_cols(_center_cols(x_))
                emb_aligned, _ = _procrustes_align(emb_n, emb_ref)
                emb_aligned = _sign_fix(emb_aligned, emb_ref)
            else:
                emb_aligned = x_

            if cle_model.model is None:
                cle_model.model = MLP(
                    [emb_aligned.shape[-1]] + cle_hidden,
                    num_bins=cle_model.num_bins
                ).to(device)
                cle_optimizer = Adam(cle_model.model.parameters(), lr=1e-4, weight_decay=5e-4)

            t = torch.randint(0, cle_model.T, (emb_aligned.shape[0],), device=device).long()
            cle_loss = cle_model.compute_loss(emb_aligned.detach(), t)

            if normalize_loss:
                guide_norm, cle_norm = loss_normalizer.normalize(guide_loss_mean, cle_loss)
                joint_loss = guide_norm + lamda1 * cle_norm
            else:
                joint_loss = guide_loss_mean + lamda1 * cle_loss

            # NaN guard
            try:
                loss_is_nan = torch.isnan(joint_loss).any() if isinstance(joint_loss, torch.Tensor) else False
                loss_is_inf = torch.isinf(joint_loss).any() if isinstance(joint_loss, torch.Tensor) else False
            except Exception:
                loss_is_nan = False
                loss_is_inf = False

            if loss_is_nan or loss_is_inf:
                print("  [WARNING] Invalid joint loss at epoch {}: NaN={}, Inf={}".format(
                    epoch, loss_is_nan, loss_is_inf))
                nan_count += 1

                if isinstance(guide_loss_mean, torch.Tensor) and torch.isnan(guide_loss_mean):
                    print("  GUIDE loss is NaN, reinitializing GUIDE model...")
                    ae_model = GUIDE_Base(
                        x_dim=x.size(1), s_dim=s.size(1),
                        hid_a=guide_hidden_a, hid_s=guide_hidden_s,
                        num_layers=guide_num_layers, dropout=guide_dropout, act=F.relu
                    ).to(device)
                    ae_optimizer = torch.optim.Adam(ae_model.parameters(), lr=5e-3)
                    ae_model.eval()
                    with torch.no_grad():
                        x0, s0 = ae_model(x, s, edge_index)
                        if use_embedding_transform:
                            emb_ref = _normalize_cols(_center_cols(x0))
                        else:
                            emb_ref = x0
                    ae_model.train()

                if nan_count >= max_nan_epochs:
                    print("  Too many NaN epochs ({}), resetting CLE model...".format(nan_count))
                    if cle_model.model is not None:
                        del cle_model.model
                    cle_model.model = None
                    cle_optimizer = None
                    nan_count = 0

                if cle_optimizer is not None:
                    cle_optimizer.zero_grad()
                ae_optimizer.zero_grad()
                continue

            nan_count = 0

            joint_loss.backward()
            torch.nn.utils.clip_grad_norm_(ae_model.parameters(), max_norm=1.0)
            ae_optimizer.step()
            if cle_optimizer is not None:
                torch.nn.utils.clip_grad_norm_(cle_model.model.parameters(), max_norm=1.0)
                cle_optimizer.step()

            if epoch % 20 == 0 or epoch == epochs - 1:
                print("Epoch: {:04d} | GUIDE Loss: {:.5f} | CLE Loss: {:.5f} | Joint Loss: {:.5f}".format(
                    epoch, guide_loss_mean.item(), cle_loss.item(), joint_loss.item()))
        else:
            guide_loss_mean.backward()
            torch.nn.utils.clip_grad_norm_(ae_model.parameters(), max_norm=1.0)
            ae_optimizer.step()
            if epoch % 20 == 0 or epoch == epochs - 1:
                print("Epoch: {:04d} | GUIDE Loss: {:.5f}".format(epoch, guide_loss.item()))

        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)

        # Periodic evaluation
        if epoch % 20 == 0 or epoch == epochs - 1:
            ae_model.eval()
            with torch.no_grad():
                x_eval, s_eval = ae_model(x, s, edge_index)
                ae_score, _ = ae_model.loss_func(x, x_eval, s, s_eval, guide_alpha)
                ae_score = ae_score.detach().cpu().numpy()

                if normalize_scores:
                    ae_score_disp = _normalize_vector(ae_score, method=score_norm_method)
                else:
                    ae_score_disp = ae_score

                y_np = y.cpu().numpy()
                ae_metrics = compute_all_metrics(y_np, ae_score_disp)

                if joint_training:
                    cle_model.model.eval()
                    if use_embedding_transform:
                        emb_eval_aligned = _align_embedding(x_eval, emb_ref)
                    else:
                        emb_eval_aligned = x_eval
                    cle_score = cle_model.predict_score(emb_eval_aligned)
                    combined_score = _compute_combined_score(
                        ae_score_disp, cle_score, normalize_scores, score_norm_method, lamda2)

                    cle_metrics = compute_all_metrics(y_np, cle_score)
                    combined_metrics = compute_all_metrics(y_np, combined_score)

                    print("  -> GUIDE:    AUC:{:.4f} AUPRC:{:.4f} R@K:{:.4f} P@K:{:.4f}".format(
                        ae_metrics['auc'], ae_metrics['auprc'],
                        ae_metrics['recall_at_k'], ae_metrics['precision_at_k']))
                    print("  -> CLE:      AUC:{:.4f} AUPRC:{:.4f} R@K:{:.4f} P@K:{:.4f}".format(
                        cle_metrics['auc'], cle_metrics['auprc'],
                        cle_metrics['recall_at_k'], cle_metrics['precision_at_k']))
                    print("  -> Combined: AUC:{:.4f} AUPRC:{:.4f} R@K:{:.4f} P@K:{:.4f}".format(
                        combined_metrics['auc'], combined_metrics['auprc'],
                        combined_metrics['recall_at_k'], combined_metrics['precision_at_k']))
                else:
                    print("  -> GUIDE: AUC:{:.4f} AUPRC:{:.4f} R@K:{:.4f} P@K:{:.4f}".format(
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
        x_final, s_final = ae_model(x, s, edge_index)
        ae_score, _ = ae_model.loss_func(x, x_final, s, s_final, guide_alpha)
        ae_score = ae_score.detach().cpu().numpy()

        if normalize_scores:
            ae_score = _normalize_vector(ae_score, method=score_norm_method)

        y_np = y.cpu().numpy()

        if joint_training:
            if cle_model.model is None:
                raise ValueError("CLE model was not properly initialized during training")
            cle_model.model.eval()

            if use_embedding_transform:
                emb_eval_aligned = _align_embedding(x_final, emb_ref)
            else:
                emb_eval_aligned = x_final
            cle_score = cle_model.predict_score(emb_eval_aligned)
            combined_score = _compute_combined_score(
                ae_score, cle_score, normalize_scores, score_norm_method, lamda2)

            ae_metrics = compute_all_metrics(y_np, ae_score)
            cle_metrics = compute_all_metrics(y_np, cle_score)
            combined_metrics = compute_all_metrics(y_np, combined_score)

            print("\nFinal Results:")
            print("  GUIDE Model:")
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
            print("  GUIDE Model:")
            print("    AUC: {:.6f} | AUPRC: {:.6f} | Recall@K: {:.6f} | Precision@K: {:.6f}".format(
                ae_metrics['auc'], ae_metrics['auprc'],
                ae_metrics['recall_at_k'], ae_metrics['precision_at_k']))
            print("  (K = {} anomalies)".format(int(y_np.sum())))

            return ae_model, ae_metrics
