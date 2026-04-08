"""
Training functions for different AE + CLE models.

Each function implements the complete training pipeline for a specific model.
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.optim import Adam
from torch_geometric.utils import to_dense_adj
import scipy.sparse as sp
import time
from datetime import datetime, timedelta

from core.cle_models import CLERegression, CLEMLP, LinearFlowNoise
from core.cle_utils import (
    LossNormalizer,
    normalize_vector,
    align_embedding,
    compute_combined_score,
    format_time_precise,
    format_timedelta_precise,
    _center_cols,
    _normalize_cols
)


def train_dominant_cle(
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
    use_embedding_transform=True,
    ae_dropout=0.3,
    alpha=0.8,
    cle_lr=1e-4,
    cle_weight_decay=5e-4,
    cle_T=200
):
    """
    Train DOMINANT + CLE model.
    
    This is a complete training function that includes:
    - DOMINANT model initialization
    - CLE model initialization
    - Joint training loop
    - Periodic evaluation
    - Final evaluation
    """
    from baselines.dominant import Dominant, dominant_loss_func, normalize_adj as dom_normalize_adj
    
    if cle_hidden is None:
        cle_hidden = [256, 512, 256]
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    print(f"Dataset: {dataset_name}")
    print(f"Nodes: {data.x.shape[0]}, Features: {data.x.shape[1]}, Anomalies: {data.y.sum().item()}")
    
    # Prepare data
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.bool()
    adj = to_dense_adj(edge_index)[0].cpu().numpy()
    adj_selfloop = adj + sp.eye(adj.shape[0])
    adj_norm = dom_normalize_adj(adj_selfloop).toarray()
    adj_label = torch.FloatTensor(adj_selfloop).to(device)
    adj = torch.FloatTensor(adj_norm).to(device)
    
    # Initialize models
    ae_model = Dominant(feat_size=x.size(1), hidden_size=ae_hidden, dropout=ae_dropout).to(device)
    ae_optimizer = torch.optim.Adam(ae_model.parameters(), lr=5e-3)
    
    # Loss normalizer
    loss_normalizer = LossNormalizer(method=normalize_method) if (normalize_loss and joint_training) else None
    
    # Reference embedding and flow
    emb_ref = None
    flow = None
    
    if joint_training:
        ae_model.eval()
        with torch.no_grad():
            _, _, emb0 = ae_model(x, adj)
            if use_embedding_transform:
                emb_ref = _normalize_cols(_center_cols(emb0))
            else:
                emb_ref = emb0
        ae_model.train()
        
        flow = LinearFlowNoise(dim=emb_ref.shape[1], ridge=1e-3, device=device, dtype=emb_ref.dtype)
        flow.fit(emb_ref)
    
    cle_model = None
    cle_optimizer = None
    epoch_times = []
    training_start_time = time.time()
    
    print(f"\n{'='*60}")
    if joint_training:
        print(f"Joint Training DOMINANT + CLE | lamda1={lamda1}")
    else:
        print(f"Training DOMINANT Only")
    print(f"{'='*60}")
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        ae_model.train()
        ae_optimizer.zero_grad()
        
        if joint_training and cle_model and cle_model.model:
            cle_model.model.train()
            if cle_optimizer is None:
                cle_optimizer = Adam(cle_model.model.parameters(), lr=cle_lr, weight_decay=cle_weight_decay)
            cle_optimizer.zero_grad()
        
        # AE forward
        A_hat, X_hat, emb = ae_model(x, adj)
        ae_loss, _, _, _ = dominant_loss_func(adj_label, A_hat, x, X_hat, alpha)
        ae_loss_mean = ae_loss.mean()
        
        if joint_training:
            # Align embedding
            if use_embedding_transform:
                emb_n = _normalize_cols(_center_cols(emb))
                from core.cle_utils import _procrustes_align, _sign_fix
                emb_aligned, _ = _procrustes_align(emb_n, emb_ref)
                emb_aligned = _sign_fix(emb_aligned, emb_ref)
            else:
                emb_aligned = emb
            
            # Initialize CLE
            if cle_model is None:
                cle_model = CLERegression(
                    hidden_size=cle_hidden,
                    epochs=300,
                    batch_size=batch_size,
                    lr=cle_lr,
                    weight_decay=cle_weight_decay,
                    T=cle_T,
                    num_bins=1,
                    device=device
                )
                cle_model.noise_flow = flow.eval()
            
            if cle_model.model is None:
                cle_model.model = CLEMLP([emb_aligned.shape[-1]] + cle_hidden, num_bins=1).to(device)
                cle_optimizer = Adam(cle_model.model.parameters(), lr=cle_lr, weight_decay=cle_weight_decay)
            
            # CLE forward
            t = torch.randint(0, cle_model.T, (emb_aligned.shape[0],), device=device).long()
            cle_loss = cle_model.compute_loss(emb_aligned.detach(), t)
            
            # Normalize and combine losses
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
                print(f"Epoch: {epoch:04d} | AE Loss: {ae_loss_mean.item():.5f} | CLE Loss: {cle_loss.item():.5f} | Joint Loss: {joint_loss.item():.5f}")
        else:
            ae_loss_mean.backward()
            ae_optimizer.step()
            
            if epoch % 20 == 0 or epoch == epochs - 1:
                print(f"Epoch: {epoch:04d} | AE Loss: {ae_loss_mean.item():.5f}")
        
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        
        # Periodic evaluation
        if epoch % 20 == 0 or epoch == epochs - 1:
            ae_model.eval()
            with torch.no_grad():
                A_hat, X_hat, emb = ae_model(x, adj)
                ae_loss, _, _, _ = dominant_loss_func(adj_label, A_hat, x, X_hat, alpha)
                ae_score = ae_loss.cpu().numpy()
                
                if normalize_scores:
                    ae_score = normalize_vector(ae_score, method=score_norm_method)
                
                y_np = y.cpu().numpy()
                ae_auc = roc_auc_score(y_np, ae_score)
                
                if joint_training and cle_model and cle_model.model:
                    cle_model.model.eval()
                    emb_eval = align_embedding(emb, emb_ref) if use_embedding_transform else emb
                    cle_score = cle_model.predict_score(emb_eval)
                    combined_score = compute_combined_score(ae_score, cle_score, normalize_scores, score_norm_method, lamda2)
                    cle_auc = roc_auc_score(y_np, cle_score)
                    combined_auc = roc_auc_score(y_np, combined_score)
                    print(f"  -> AE AUC: {ae_auc:.4f} | CLE AUC: {cle_auc:.4f} | Combined AUC: {combined_auc:.4f}")
                else:
                    print(f"  -> AE AUC: {ae_auc:.4f}")
    
    # Final evaluation
    print(f"\n{'='*60}")
    print("Final Evaluation")
    print(f"{'='*60}")
    
    ae_model.eval()
    with torch.no_grad():
        A_hat, X_hat, emb = ae_model(x, adj)
        ae_loss, _, _, _ = dominant_loss_func(adj_label, A_hat, x, X_hat, alpha)
        ae_score = ae_loss.cpu().numpy()
        
        if normalize_scores:
            ae_score = normalize_vector(ae_score, method=score_norm_method)
        
        y_np = y.cpu().numpy()
        ae_auc = roc_auc_score(y_np, ae_score)
        
        if joint_training and cle_model and cle_model.model:
            cle_model.model.eval()
            emb_eval = align_embedding(emb, emb_ref) if use_embedding_transform else emb
            cle_score = cle_model.predict_score(emb_eval)
            combined_score = compute_combined_score(ae_score, cle_score, normalize_scores, score_norm_method, lamda2)
            cle_auc = roc_auc_score(y_np, cle_score)
            combined_auc = roc_auc_score(y_np, combined_score)
            
            print(f"\nFinal Results:")
            print(f"  AE AUC:       {ae_auc:.6f}")
            print(f"  CLE AUC:      {cle_auc:.6f}")
            print(f"  Combined AUC: {combined_auc:.6f}")
            
            return ae_model, cle_model, combined_auc
        else:
            print(f"\nFinal Results:")
            print(f"  AE AUC: {ae_auc:.6f}")
            return ae_model, None, ae_auc
