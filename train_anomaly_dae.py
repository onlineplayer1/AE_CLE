"""
Training function for AnomalyDAE + CLE.
"""

import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.optim import Adam
from torch_geometric.utils import to_dense_adj
import scipy.sparse as sp
import time

from baselines.anomaly_dae import AnomalyDAE, anomaly_dae_loss_func
from core.cle_models import CLERegression, CLEMLP, LinearFlowNoise
from core.cle_utils import (
    LossNormalizer, normalize_vector, align_embedding,
    compute_combined_score, _center_cols, _normalize_cols
)


def train_anomaly_dae_cle(
    data,
    epochs=100,
    ae_hidden=64,
    cle_hidden=None,
    batch_size=64,
    device=None,
    normalize_loss=True,
    normalize_method='exponential_moving_average',
    lamda1=0.5,
    lamda2=0.5,
    normalize_scores=True,
    score_norm_method='min_max',
    joint_training=True,
    dataset_name='unknown',
    ae_dropout=0.3,
    alpha=0.8,
    ae_lr=5e-3,
    ae_weight_decay=0.0,
    cle_lr=1e-4,
    cle_weight_decay=5e-4,
    cle_T=200
):
    """Train AnomalyDAE + CLE model."""
    from core.cle_utils import _procrustes_align, _sign_fix
    
    if cle_hidden is None:
        cle_hidden = [256, 512, 256]
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    print(f"Dataset: {dataset_name}")
    
    # Prepare data
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.bool()
    adj = to_dense_adj(edge_index)[0].cpu().numpy()
    adj_selfloop = adj + sp.eye(adj.shape[0])
    adj_norm = sp.coo_matrix(adj_selfloop)
    rowsum = np.array(adj_norm.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj_norm = adj_norm.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo().toarray()
    
    adj_label = torch.FloatTensor(adj_selfloop).to(device)
    adj = torch.FloatTensor(adj_norm).to(device)
    
    # Initialize model
    hidden1 = ae_hidden * 2
    hidden2 = ae_hidden
    ae_model = AnomalyDAE(
        num_features=x.size(1),
        num_nodes=x.size(0),
        hidden1=hidden1,
        hidden2=hidden2,
        dropout=ae_dropout
    ).to(device)
    ae_optimizer = torch.optim.Adam(ae_model.parameters(), lr=ae_lr, weight_decay=ae_weight_decay)
    
    # Loss normalizer and flow
    loss_normalizer = LossNormalizer(method=normalize_method) if (normalize_loss and joint_training) else None
    emb_ref, flow = None, None
    
    if joint_training:
        ae_model.eval()
        with torch.no_grad():
            _, _, emb0 = ae_model(x, adj)
            emb_ref = _normalize_cols(_center_cols(emb0))
        ae_model.train()
        flow = LinearFlowNoise(dim=emb_ref.shape[1], ridge=1e-3, device=device, dtype=emb_ref.dtype)
        flow.fit(emb_ref)
    
    cle_model = None
    cle_optimizer = None
    
    print(f"\n{'='*60}")
    print(f"Joint Training AnomalyDAE + CLE" if joint_training else "Training AnomalyDAE Only")
    print(f"{'='*60}")
    
    for epoch in range(epochs):
        ae_model.train()
        ae_optimizer.zero_grad()
        
        if joint_training and cle_model and cle_model.model:
            cle_model.model.train()
            if cle_optimizer is None:
                cle_optimizer = Adam(cle_model.model.parameters(), lr=cle_lr, weight_decay=cle_weight_decay)
            cle_optimizer.zero_grad()
        
        # AE forward
        A_hat, X_hat, emb = ae_model(x, adj)
        ae_loss, ae_loss_mean, _, _ = anomaly_dae_loss_func(adj_label, A_hat, x, X_hat, alpha)
        
        if joint_training:
            emb_n = _normalize_cols(_center_cols(emb))
            emb_aligned, _ = _procrustes_align(emb_n, emb_ref)
            emb_aligned = _sign_fix(emb_aligned, emb_ref)
            
            if cle_model is None:
                cle_model = CLERegression(hidden_size=cle_hidden, epochs=300, batch_size=batch_size,
                                         lr=cle_lr, weight_decay=cle_weight_decay, T=cle_T, num_bins=1, device=device)
                cle_model.noise_flow = flow.eval()
            
            if cle_model.model is None:
                cle_model.model = CLEMLP([emb_aligned.shape[-1]] + cle_hidden, num_bins=1).to(device)
                cle_optimizer = Adam(cle_model.model.parameters(), lr=cle_lr, weight_decay=cle_weight_decay)
            
            t = torch.randint(0, cle_model.T, (emb_aligned.shape[0],), device=device).long()
            cle_loss = cle_model.compute_loss(emb_aligned.detach(), t)
            
            if normalize_loss and loss_normalizer:
                ae_loss_n, cle_loss_n = loss_normalizer.normalize(ae_loss_mean, cle_loss)
                joint_loss = ae_loss_n + lamda1 * cle_loss_n
            else:
                joint_loss = ae_loss_mean + lamda1 * cle_loss
            
            joint_loss.backward()
            ae_optimizer.step()
            if cle_optimizer:
                cle_optimizer.step()
            
            if epoch % 20 == 0 or epoch == epochs - 1:
                print(f"Epoch: {epoch:04d} | AE: {ae_loss_mean.item():.5f} | CLE: {cle_loss.item():.5f}")
        else:
            ae_loss_mean.backward()
            ae_optimizer.step()
        
        # Evaluation
        if epoch % 20 == 0 or epoch == epochs - 1:
            ae_model.eval()
            with torch.no_grad():
                A_hat, X_hat, emb = ae_model(x, adj)
                ae_loss, _, _, _ = anomaly_dae_loss_func(adj_label, A_hat, x, X_hat, alpha)
                ae_score = ae_loss.cpu().numpy()
                if normalize_scores:
                    ae_score = normalize_vector(ae_score, method=score_norm_method)
                y_np = y.cpu().numpy()
                ae_auc = roc_auc_score(y_np, ae_score)
                
                if joint_training and cle_model and cle_model.model:
                    cle_model.model.eval()
                    emb_eval = align_embedding(emb, emb_ref)
                    cle_score = cle_model.predict_score(emb_eval)
                    combined = compute_combined_score(ae_score, cle_score, normalize_scores, score_norm_method, lamda2)
                    print(f"  -> AE: {ae_auc:.4f} | Combined: {roc_auc_score(y_np, combined):.4f}")
                else:
                    print(f"  -> AUC: {ae_auc:.4f}")
    
    # Final evaluation
    ae_model.eval()
    with torch.no_grad():
        A_hat, X_hat, emb = ae_model(x, adj)
        ae_loss, _, _, _ = anomaly_dae_loss_func(adj_label, A_hat, x, X_hat, alpha)
        ae_score = ae_loss.cpu().numpy()
        if normalize_scores:
            ae_score = normalize_vector(ae_score, method=score_norm_method)
        y_np = y.cpu().numpy()
        ae_auc = roc_auc_score(y_np, ae_score)
        
        if joint_training and cle_model and cle_model.model:
            cle_model.model.eval()
            emb_eval = align_embedding(emb, emb_ref)
            cle_score = cle_model.predict_score(emb_eval)
            combined = compute_combined_score(ae_score, cle_score, normalize_scores, score_norm_method, lamda2)
            final_auc = roc_auc_score(y_np, combined)
            print(f"\nFinal: AE={ae_auc:.6f}, Combined={final_auc:.6f}")
            return ae_model, cle_model, final_auc
    
    return ae_model, None, ae_auc
