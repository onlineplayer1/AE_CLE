"""
Training function for GAD-NR + CLE.
"""

import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.optim import Adam
import time

from baselines.gadnr import GNNStructEncoder, build_neighbor_dict, compute_degree_matrix
from core.cle_models import CLERegression, CLEMLP, LinearFlowNoise
from core.cle_utils import (
    LossNormalizer, normalize_vector, align_embedding,
    compute_combined_score, _center_cols, _normalize_cols
)


def train_gadnr_cle(
    data,
    epochs=100,
    gadnr_hidden=64,
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
    sample_size=10,
    encoder="GCN",
    cle_lr=1e-4,
    cle_weight_decay=5e-4,
    cle_T=200
):
    """Train GAD-NR + CLE model."""
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
    
    # Build neighbor structures
    num_nodes = x.shape[0]
    neighbor_dict = build_neighbor_dict(edge_index, num_nodes)
    neighbor_num_list = [len(neighbors) for neighbors in neighbor_dict.values()]
    ground_truth_degree_matrix = compute_degree_matrix(edge_index, num_nodes).to(device)
    
    # Initialize model
    ae_model = GNNStructEncoder(
        in_dim0=x.shape[1],
        in_dim=x.shape[1],
        hidden_dim=gadnr_hidden,
        layer_num=2,
        sample_size=sample_size,
        device=device,
        neighbor_num_list=neighbor_num_list,
        GNN_name=encoder,
        lambda_loss1=0.01,
        lambda_loss2=0.1,
        lambda_loss3=0.8
    ).to(device)
    ae_optimizer = torch.optim.Adam(ae_model.parameters(), lr=1e-3)
    
    # Loss normalizer and flow
    loss_normalizer = LossNormalizer(method=normalize_method) if (normalize_loss and joint_training) else None
    emb_ref, flow = None, None
    
    if joint_training:
        ae_model.eval()
        with torch.no_grad():
            l1, h0 = ae_model.forward_encoder(x, edge_index)
            emb_ref = _normalize_cols(_center_cols(l1))
        ae_model.train()
        flow = LinearFlowNoise(dim=emb_ref.shape[1], ridge=1e-3, device=device, dtype=emb_ref.dtype)
        flow.fit(emb_ref)
    
    cle_model = None
    cle_optimizer = None
    
    print(f"\n{'='*60}")
    print(f"Joint Training GAD-NR + CLE" if joint_training else "Training GAD-NR Only")
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
        try:
            gadnr_loss, gadnr_loss_per_node, h_loss, degree_loss, feature_loss = ae_model(
                edge_index, x, ground_truth_degree_matrix, neighbor_dict, device)
            gadnr_loss_mean = torch.mean(gadnr_loss)
            
            if torch.isnan(gadnr_loss_mean) or torch.isinf(gadnr_loss_mean):
                print(f"Warning: GAD-NR loss is NaN/Inf at epoch {epoch}")
                continue
        except Exception as e:
            print(f"Error in GAD-NR forward pass at epoch {epoch}: {e}")
            continue
        
        if joint_training:
            l1, h0 = ae_model.forward_encoder(x, edge_index)
            emb_n = _normalize_cols(_center_cols(l1))
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
                gadnr_loss_n, cle_loss_n = loss_normalizer.normalize(gadnr_loss_mean, cle_loss)
                joint_loss = gadnr_loss_n + lamda1 * cle_loss_n
            else:
                joint_loss = gadnr_loss_mean + lamda1 * cle_loss
            
            joint_loss.backward()
            torch.nn.utils.clip_grad_norm_(ae_model.parameters(), max_norm=1.0)
            ae_optimizer.step()
            if cle_optimizer:
                torch.nn.utils.clip_grad_norm_(cle_model.model.parameters(), max_norm=1.0)
                cle_optimizer.step()
            
            if epoch % 20 == 0 or epoch == epochs - 1:
                print(f"Epoch: {epoch:04d} | GAD-NR: {gadnr_loss_mean.item():.5f} | CLE: {cle_loss.item():.5f}")
        else:
            gadnr_loss_mean.backward()
            torch.nn.utils.clip_grad_norm_(ae_model.parameters(), max_norm=1.0)
            ae_optimizer.step()
        
        # Evaluation
        if epoch % 20 == 0 or epoch == epochs - 1:
            ae_model.eval()
            with torch.no_grad():
                try:
                    gadnr_loss, gadnr_loss_per_node, _, _, _ = ae_model(
                        edge_index, x, ground_truth_degree_matrix, neighbor_dict, device)
                    gadnr_score = gadnr_loss_per_node.detach().cpu().numpy()
                    
                    if np.isnan(gadnr_score).any() or np.isinf(gadnr_score).any():
                        print(f"Warning: GAD-NR score contains NaN/Inf during evaluation at epoch {epoch}")
                        continue
                    
                    if gadnr_score.ndim > 1:
                        gadnr_score = gadnr_score.flatten()
                    
                    if normalize_scores:
                        gadnr_score = normalize_vector(gadnr_score, method=score_norm_method)
                    
                    y_np = y.cpu().numpy()
                    gadnr_auc = roc_auc_score(y_np, gadnr_score)
                    
                    if joint_training and cle_model and cle_model.model:
                        cle_model.model.eval()
                        l1, h0 = ae_model.forward_encoder(x, edge_index)
                        emb_eval = align_embedding(l1, emb_ref)
                        cle_score = cle_model.predict_score(emb_eval)
                        combined = compute_combined_score(gadnr_score, cle_score, normalize_scores, score_norm_method, lamda2)
                        print(f"  -> GAD-NR: {gadnr_auc:.4f} | Combined: {roc_auc_score(y_np, combined):.4f}")
                    else:
                        print(f"  -> AUC: {gadnr_auc:.4f}")
                except Exception as e:
                    print(f"Error during evaluation at epoch {epoch}: {e}")
    
    # Final evaluation
    ae_model.eval()
    with torch.no_grad():
        gadnr_loss, gadnr_loss_per_node, _, _, _ = ae_model(
            edge_index, x, ground_truth_degree_matrix, neighbor_dict, device)
        gadnr_score = gadnr_loss_per_node.detach().cpu().numpy()
        
        if np.isnan(gadnr_score).any() or np.isinf(gadnr_score).any():
            raise ValueError("Training produced NaN/Inf scores")
        
        if gadnr_score.ndim > 1:
            gadnr_score = gadnr_score.flatten()
        
        if normalize_scores:
            gadnr_score = normalize_vector(gadnr_score, method=score_norm_method)
        
        y_np = y.cpu().numpy()
        gadnr_auc = roc_auc_score(y_np, gadnr_score)
        
        if joint_training and cle_model and cle_model.model:
            cle_model.model.eval()
            l1, h0 = ae_model.forward_encoder(x, edge_index)
            emb_eval = align_embedding(l1, emb_ref)
            cle_score = cle_model.predict_score(emb_eval)
            combined = compute_combined_score(gadnr_score, cle_score, normalize_scores, score_norm_method, lamda2)
            final_auc = roc_auc_score(y_np, combined)
            print(f"\nFinal: GAD-NR={gadnr_auc:.6f}, Combined={final_auc:.6f}")
            return ae_model, cle_model, final_auc
    
    return ae_model, None, gadnr_auc
