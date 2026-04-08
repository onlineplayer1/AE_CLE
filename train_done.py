"""
Training function for DONE + CLE.
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.optim import Adam
from torch_geometric.utils import to_dense_adj
import time

from baselines.done import DONE
from core.cle_models import CLERegression, CLEMLP, LinearFlowNoise
from core.cle_utils import (
    LossNormalizer, normalize_vector, align_embedding,
    compute_combined_score, _center_cols, _normalize_cols, _procrustes_align, _sign_fix
)


def train_done_cle(
    data,
    epochs=100,
    done_hidden=64,
    done_num_layers=4,
    done_dropout=0.0,
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
    cle_lr=1e-4,
    cle_weight_decay=5e-4,
    cle_T=200
):
    """Train DONE + CLE model."""
    
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
    s = to_dense_adj(edge_index)[0].to(device)
    
    # Initialize model
    ae_model = DONE(
        x_dim=x.size(1),
        s_dim=s.size(1),
        hid_dim=done_hidden,
        num_layers=done_num_layers,
        dropout=done_dropout,
        act=F.leaky_relu
    ).to(device)
    ae_optimizer = torch.optim.Adam(ae_model.parameters(), lr=5e-3)
    
    # Loss normalizer and flow
    loss_normalizer = LossNormalizer(method=normalize_method) if (normalize_loss and joint_training) else None
    emb_ref, flow = None, None
    
    if joint_training:
        ae_model.eval()
        with torch.no_grad():
            _, _, h_a0, _, _, _ = ae_model(x, s, edge_index)
            emb_ref = _normalize_cols(_center_cols(h_a0))
        ae_model.train()
        flow = LinearFlowNoise(dim=emb_ref.shape[1], ridge=1e-3, device=device, dtype=emb_ref.dtype)
        flow.fit(emb_ref)
    
    cle_model = None
    cle_optimizer = None
    
    print(f"\n{'='*60}")
    print(f"Joint Training DONE + CLE" if joint_training else "Training DONE Only")
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
        x_, s_, h_a, h_s, dna, dns = ae_model(x, s, edge_index)
        done_score, done_loss = ae_model.loss_func(x, x_, s, s_, h_a, h_s, dna, dns)
        done_loss_mean = done_score.mean()
        
        if joint_training:
            emb_n = _normalize_cols(_center_cols(h_a))
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
                done_loss_n, cle_loss_n = loss_normalizer.normalize(done_loss_mean, cle_loss)
                joint_loss = done_loss_n + lamda1 * cle_loss_n
            else:
                joint_loss = done_loss_mean + lamda1 * cle_loss
            
            joint_loss.backward()
            ae_optimizer.step()
            if cle_optimizer:
                cle_optimizer.step()
            
            if epoch % 20 == 0 or epoch == epochs - 1:
                print(f"Epoch: {epoch:04d} | DONE: {done_loss_mean.item():.5f} | CLE: {cle_loss.item():.5f}")
        else:
            done_loss_mean.backward()
            ae_optimizer.step()
        
        # Evaluation
        if epoch % 20 == 0 or epoch == epochs - 1:
            ae_model.eval()
            with torch.no_grad():
                x_, s_, h_a, _, _, _ = ae_model(x, s, edge_index)
                done_score_eval, _ = ae_model.loss_func(x, x_, s, s_, h_a, h_a, dna, dns)
                done_score_np = done_score_eval.detach().cpu().numpy()
                if normalize_scores:
                    done_score_np = normalize_vector(done_score_np, method=score_norm_method)
                y_np = y.cpu().numpy()
                done_auc = roc_auc_score(y_np, done_score_np)
                
                if joint_training and cle_model and cle_model.model:
                    cle_model.model.eval()
                    emb_eval = align_embedding(h_a, emb_ref)
                    cle_score = cle_model.predict_score(emb_eval)
                    combined = compute_combined_score(done_score_np, cle_score, normalize_scores, score_norm_method, lamda2)
                    print(f"  -> DONE: {done_auc:.4f} | Combined: {roc_auc_score(y_np, combined):.4f}")
                else:
                    print(f"  -> AUC: {done_auc:.4f}")
    
    # Final evaluation
    ae_model.eval()
    with torch.no_grad():
        x_, s_, h_a, _, _, _ = ae_model(x, s, edge_index)
        done_score_final, _ = ae_model.loss_func(x, x_, s, s_, h_a, h_a, dna, dns)
        done_score_np = done_score_final.detach().cpu().numpy()
        if normalize_scores:
            done_score_np = normalize_vector(done_score_np, method=score_norm_method)
        y_np = y.cpu().numpy()
        done_auc = roc_auc_score(y_np, done_score_np)
        
        if joint_training and cle_model and cle_model.model:
            cle_model.model.eval()
            emb_eval = align_embedding(h_a, emb_ref)
            cle_score = cle_model.predict_score(emb_eval)
            combined = compute_combined_score(done_score_np, cle_score, normalize_scores, score_norm_method, lamda2)
            final_auc = roc_auc_score(y_np, combined)
            print(f"\nFinal: DONE={done_auc:.6f}, Combined={final_auc:.6f}")
            return ae_model, cle_model, final_auc
    
    return ae_model, None, done_auc
