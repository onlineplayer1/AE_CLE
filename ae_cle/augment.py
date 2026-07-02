"""Data augmentation: auxiliary node generation for ensemble learning."""

import torch
import numpy as np
from torch_geometric.utils import to_dense_adj
from torch_geometric.data import Data as PyGData


def add_auxiliary_nodes(data, n_aux, feature_method='outlier_tail',
                        edge_method='random_connect', n_connections=5,
                        k_std=3.0, seed=None):
    """Add synthetic auxiliary anomaly nodes to the graph.

    Construct artificial anomalous nodes without modifying the original graph,
    used for ensemble diversity.

    Parameters
    ----------
    data : PyGData
    n_aux : int or float
        If 0 < n_aux < 1, treated as ratio (e.g. 0.05 = 5% of original nodes);
        if n_aux >= 1, treated as absolute count.
    feature_method : str
        'outlier_tail' (default): mean + k*std in random directions (classic 3-sigma outlier)
        'gaussian_noise': generate with mean + 2*std Gaussian noise
        'perturb_existing': copy real node features + 3*std perturbation
        'smote_outlier' (GraphSMOTE-inspired): interpolate two real nodes then push to tail
        'neighbor_dissimilar' (SAWGAD-inspired): generate features dissimilar to anchor's neighbors
        'feature_shuffle': copy real node features then randomly shuffle feature dimensions
    edge_method : str
        'random_connect' (default): each auxiliary node connects to n_connections random real nodes
        'isolated': no edges added
        'clique': auxiliary nodes form a clique among themselves
        'low_similarity_connect' (SAWGAD-inspired): connect to real nodes with lowest cosine similarity
    n_connections : int, number of real nodes each auxiliary node connects to
    k_std : float, outlier magnitude multiplier (default 3.0, i.e. 3-sigma)
    seed : int or None

    Returns
    -------
    augmented_data : PyGData, containing original nodes + auxiliary nodes
    aux_indices : np.ndarray, auxiliary node indices (always at the end)
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    n_orig = data.x.shape[0]
    n_feat = data.x.shape[1]
    device = data.x.device

    if 0 < n_aux < 1:
        n_aux = max(1, int(round(n_aux * n_orig)))

    x_np = data.x.cpu().numpy()
    feat_mean = x_np.mean(axis=0)
    feat_std = x_np.std(axis=0)
    feat_std = np.clip(feat_std, 1e-8, None)

    if feature_method == 'outlier_tail':
        aux_feats = np.zeros((n_aux, n_feat))
        for i in range(n_aux):
            direction = np.random.randn(n_feat)
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            offset = k_std * direction * feat_std
            sign = np.random.choice([-1.0, 1.0], size=n_feat)
            aux_feats[i] = feat_mean + sign * offset

    elif feature_method == 'gaussian_noise':
        aux_feats = np.random.randn(n_aux, n_feat) * feat_std * 2.0 + feat_mean

    elif feature_method == 'perturb_existing':
        chosen_idx = np.random.choice(n_orig, n_aux, replace=True)
        noise = np.random.randn(n_aux, n_feat) * feat_std * 3.0
        aux_feats = x_np[chosen_idx] + noise

    elif feature_method == 'smote_outlier':
        aux_feats = np.zeros((n_aux, n_feat))
        for i in range(n_aux):
            a, b = np.random.choice(n_orig, 2, replace=False)
            alpha = np.random.uniform(0, 1)
            interp = alpha * x_np[a] + (1 - alpha) * x_np[b]
            direction = np.random.randn(n_feat)
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            aux_feats[i] = interp + k_std * direction * feat_std

    elif feature_method == 'neighbor_dissimilar':
        adj_dense = to_dense_adj(data.edge_index, max_num_nodes=n_orig)[0].cpu().numpy()
        aux_feats = np.zeros((n_aux, n_feat))
        for i in range(n_aux):
            anchor = np.random.choice(n_orig)
            neighbors = np.where(adj_dense[anchor] > 0)[0]
            if len(neighbors) == 0:
                neighbor_mean = feat_mean
            else:
                neighbor_mean = x_np[neighbors].mean(axis=0)
            direction = np.random.randn(n_feat)
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            aux_feats[i] = neighbor_mean + k_std * direction * feat_std

    elif feature_method == 'feature_shuffle':
        chosen_idx = np.random.choice(n_orig, n_aux, replace=True)
        aux_feats = x_np[chosen_idx].copy()
        for i in range(n_aux):
            np.random.shuffle(aux_feats[i])
    else:
        raise ValueError(f"Unknown feature_method: {feature_method}")

    aux_feats = np.clip(aux_feats, -1e6, 1e6)

    edge_index_np = data.edge_index.cpu().numpy()
    new_edges = []

    if edge_method == 'random_connect':
        n_conn = min(n_connections, n_orig)
        for i in range(n_aux):
            aux_id = n_orig + i
            targets = np.random.choice(n_orig, size=n_conn, replace=False)
            for t in targets:
                new_edges.append([aux_id, t])
                new_edges.append([t, aux_id])

    elif edge_method == 'isolated':
        pass

    elif edge_method == 'clique':
        for i in range(n_aux):
            for j in range(i + 1, n_aux):
                aux_i = n_orig + i
                aux_j = n_orig + j
                new_edges.append([aux_i, aux_j])
                new_edges.append([aux_j, aux_i])

    elif edge_method == 'low_similarity_connect':
        from sklearn.metrics.pairwise import cosine_similarity
        n_conn = min(n_connections, n_orig)
        sim = cosine_similarity(aux_feats, x_np)
        for i in range(n_aux):
            targets = np.argsort(sim[i])[:n_conn]
            aux_id = n_orig + i
            for t in targets:
                new_edges.append([aux_id, t])
                new_edges.append([t, aux_id])
    else:
        raise ValueError(f"Unknown edge_method: {edge_method}")

    if len(new_edges) > 0:
        new_edges_np = np.array(new_edges).T
        combined_edges = np.hstack([edge_index_np, new_edges_np])
    else:
        combined_edges = edge_index_np

    edge_index_aug = torch.tensor(combined_edges, dtype=torch.long, device=device)

    x_aug = torch.cat([
        data.x,
        torch.tensor(aux_feats, dtype=data.x.dtype, device=device)
    ], dim=0)

    y_aug = torch.cat([
        data.y,
        torch.full((n_aux,), -1, dtype=torch.long, device=device)
    ], dim=0)

    augmented_data = PyGData(x=x_aug, y=y_aug, edge_index=edge_index_aug)
    aux_indices = np.arange(n_orig, n_orig + n_aux)

    return augmented_data, aux_indices
