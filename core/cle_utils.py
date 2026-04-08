"""
CLE Utilities

Contains utility functions for loss normalization, embedding alignment,
score combination, and timing helpers.
"""

import torch
import numpy as np
from typing import Optional


# ==================== Timing Utilities ====================

def format_time_precise(seconds: float) -> str:
    """Format time with millisecond precision."""
    if seconds < 1.0:
        return f"{seconds:.1f}s"
    elif seconds < 60.0:
        return f"{seconds:.3f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        if hours > 0:
            return f"{hours}:{minutes:02d}:{secs:06.3f}"
        else:
            return f"{minutes}:{secs:06.3f}"


def format_timedelta_precise(td) -> str:
    """Format timedelta with millisecond precision."""
    total_seconds = td.total_seconds()
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60

    if hours > 0:
        return f"{hours}:{minutes:02d}:{seconds:06.3f}"
    elif minutes > 0:
        return f"{minutes}:{seconds:06.3f}"
    else:
        return f"{seconds:.3f}s"


# ==================== Score Normalization ====================

def normalize_vector(vec: np.ndarray, method: str = 'min_max') -> np.ndarray:
    """
    Normalize a 1D numpy array to a comparable range.
    
    Parameters:
    - min_max: scale each vector to [0, 1]
    - z_score: standardize to zero-mean unit-std, then squash with sigmoid
    - rank: convert to fractional ranks in [0, 1]
    """
    if not isinstance(vec, np.ndarray):
        vec = np.asarray(vec)
    if vec.ndim != 1:
        vec = vec.reshape(-1)
    eps = 1e-8

    if method == 'min_max':
        vmin = np.min(vec)
        vmax = np.max(vec)
        denom = (vmax - vmin) if (vmax - vmin) > eps else eps
        return (vec - vmin) / denom
    elif method == 'z_score':
        mean = np.mean(vec)
        std = np.std(vec)
        std = std if std > eps else eps
        z = (vec - mean) / std
        return 1 / (1 + np.exp(-z))
    elif method == 'rank':
        order = vec.argsort()
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(len(vec))
        return ranks / max(len(vec) - 1, 1)
    else:
        return vec


# ==================== Embedding Alignment ====================

def _center_cols(E: torch.Tensor) -> torch.Tensor:
    """Center columns of embedding matrix."""
    return E - E.mean(dim=0, keepdim=True)


def _normalize_cols(E: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize columns of embedding matrix to unit norm."""
    norms = torch.norm(E, dim=0, keepdim=True).clamp_min(eps)
    return E / norms


def _procrustes_align(E_cur: torch.Tensor, E_ref: torch.Tensor):
    """
    Orthogonal Procrustes alignment.
    
    Finds R = argmin ||E_cur R - E_ref||_F, s.t. R^T R = I
    """
    M = E_cur.T @ E_ref
    U, _, Vh = torch.linalg.svd(M, full_matrices=False)
    R = U @ Vh
    E_aligned = E_cur @ R
    return E_aligned, R


def _sign_fix(E_aligned: torch.Tensor, E_ref: torch.Tensor) -> torch.Tensor:
    """Fix sign ambiguity in aligned embeddings."""
    corr = (E_aligned * E_ref).sum(dim=0, keepdim=True)
    s = torch.sign(corr)
    s[s == 0] = 1
    return E_aligned * s


def align_embedding(
    emb: torch.Tensor,
    emb_ref: torch.Tensor,
    print_stats: bool = False,
    prefix: str = ""
) -> torch.Tensor:
    """
    Align embedding to reference using Procrustes analysis.
    
    Process: center -> normalize -> procrustes -> sign fix
    """
    emb_n = _normalize_cols(_center_cols(emb))
    emb_aligned, _ = _procrustes_align(emb_n, emb_ref)
    emb_aligned = _sign_fix(emb_aligned, emb_ref)

    if print_stats:
        emb_np = emb_aligned.detach().cpu().numpy()
        print(f"{prefix}emb aligned | shape: {emb_np.shape} | "
              f"mean: {float(emb_np.mean()):.6f} | std: {float(emb_np.std()):.6f}")

    return emb_aligned


# ==================== Score Combination ====================

def compute_combined_score(
    ae_score: np.ndarray,
    cle_score: np.ndarray,
    normalize_scores: bool,
    score_norm_method: str,
    lamda2: float
) -> np.ndarray:
    """
    Compute combined anomaly score from AE and CLE scores.
    
    Combined score = ae_score + lamda2 * cle_score
    """
    if normalize_scores:
        ae_score_n = normalize_vector(ae_score, method=score_norm_method)
        cle_score_n = normalize_vector(cle_score, method=score_norm_method)
    else:
        ae_score_n = ae_score
        cle_score_n = cle_score

    combined_score = 1.0 * ae_score_n + lamda2 * cle_score_n
    return combined_score


# ==================== Loss Normalization ====================

class LossNormalizer:
    """
    Normalize losses from different models to the same scale.
    
    Supports multiple normalization methods:
    - exponential_moving_average: EMA-based scaling
    - running_average: Running average-based scaling
    - min_max: Min-max normalization
    - z_score: Z-score normalization
    """
    
    def __init__(self, method: str = 'exponential_moving_average', alpha: float = 0.9):
        self.method = method
        self.alpha = alpha
        self.ae_loss_ema = None
        self.cle_loss_ema = None
        self.ae_losses = []
        self.cle_losses = []
    
    def normalize(self, ae_loss, cle_loss):
        """
        Normalize AE and CLE losses to the same scale.
        
        Returns:
        - normalized_ae_loss, normalized_cle_loss
        """
        ae_loss_val = ae_loss.item() if isinstance(ae_loss, torch.Tensor) else ae_loss
        cle_loss_val = cle_loss.item() if isinstance(cle_loss, torch.Tensor) else cle_loss

        self.ae_losses.append(ae_loss_val)
        self.cle_losses.append(cle_loss_val)
        
        if self.method == 'exponential_moving_average':
            return self._ema_normalize(ae_loss, cle_loss)
        elif self.method == 'running_average':
            return self._running_avg_normalize(ae_loss, cle_loss)
        elif self.method == 'min_max':
            return self._min_max_normalize(ae_loss, cle_loss)
        elif self.method == 'z_score':
            return self._z_score_normalize(ae_loss, cle_loss)
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")
    
    def _ema_normalize(self, ae_loss, cle_loss):
        """Exponential moving average normalization."""
        ae_loss_val = ae_loss.item() if isinstance(ae_loss, torch.Tensor) else ae_loss
        cle_loss_val = cle_loss.item() if isinstance(cle_loss, torch.Tensor) else cle_loss

        if self.ae_loss_ema is None:
            self.ae_loss_ema = ae_loss_val
            self.cle_loss_ema = cle_loss_val
        else:
            self.ae_loss_ema = self.alpha * self.ae_loss_ema + (1 - self.alpha) * ae_loss_val
            self.cle_loss_ema = self.alpha * self.cle_loss_ema + (1 - self.alpha) * cle_loss_val

        ae_scale = max(self.ae_loss_ema, 1e-8)
        cle_scale = max(self.cle_loss_ema, 1e-8)
        target_scale = (ae_scale + cle_scale) / 2

        normalized_ae = ae_loss * (target_scale / ae_scale)
        normalized_cle = cle_loss * (target_scale / cle_scale)

        return normalized_ae, normalized_cle
    
    def _running_avg_normalize(self, ae_loss, cle_loss):
        """Running average normalization."""
        ae_loss_val = ae_loss.item() if isinstance(ae_loss, torch.Tensor) else ae_loss
        cle_loss_val = cle_loss.item() if isinstance(cle_loss, torch.Tensor) else cle_loss

        ae_avg = np.mean(self.ae_losses) if self.ae_losses else ae_loss_val
        cle_avg = np.mean(self.cle_losses) if self.cle_losses else cle_loss_val

        ae_scale = max(ae_avg, 1e-8)
        cle_scale = max(cle_avg, 1e-8)
        target_scale = (ae_scale + cle_scale) / 2

        normalized_ae = ae_loss * (target_scale / ae_scale)
        normalized_cle = cle_loss * (target_scale / cle_scale)

        return normalized_ae, normalized_cle
    
    def _min_max_normalize(self, ae_loss, cle_loss):
        """Min-max normalization."""
        ae_loss_val = ae_loss.item() if isinstance(ae_loss, torch.Tensor) else ae_loss
        cle_loss_val = cle_loss.item() if isinstance(cle_loss, torch.Tensor) else cle_loss

        if len(self.ae_losses) < 2:
            return ae_loss, cle_loss

        ae_min, ae_max = min(self.ae_losses), max(self.ae_losses)
        cle_min, cle_max = min(self.cle_losses), max(self.cle_losses)

        ae_range = max(ae_max - ae_min, 1e-8)
        cle_range = max(cle_max - cle_min, 1e-8)

        ae_normalized = (ae_loss_val - ae_min) / ae_range
        cle_normalized = (cle_loss_val - cle_min) / cle_range

        return (
            torch.tensor(ae_normalized, device=ae_loss.device, dtype=ae_loss.dtype),
            torch.tensor(cle_normalized, device=cle_loss.device, dtype=cle_loss.dtype)
        )
    
    def _z_score_normalize(self, ae_loss, cle_loss):
        """Z-score normalization."""
        ae_loss_val = ae_loss.item() if isinstance(ae_loss, torch.Tensor) else ae_loss
        cle_loss_val = cle_loss.item() if isinstance(cle_loss, torch.Tensor) else cle_loss

        if len(self.ae_losses) < 2:
            return ae_loss, cle_loss

        ae_mean = np.mean(self.ae_losses)
        ae_std = np.std(self.ae_losses) if np.std(self.ae_losses) > 0 else 1e-8

        cle_mean = np.mean(self.cle_losses)
        cle_std = np.std(self.cle_losses) if np.std(self.cle_losses) > 0 else 1e-8

        ae_normalized = (ae_loss_val - ae_mean) / ae_std
        cle_normalized = (cle_loss_val - cle_mean) / cle_std

        return (
            torch.tensor(ae_normalized, device=ae_loss.device, dtype=ae_loss.dtype),
            torch.tensor(cle_normalized, device=cle_loss.device, dtype=cle_loss.dtype)
        )


# ==================== Parameter Loading ====================

def load_best_params(dataset_name: str, params_dir: str = 'optuna_results') -> Optional[dict]:
    """
    Load best parameters for a given dataset if available.
    
    Returns best_params dict if found, None otherwise.
    """
    import os
    import json

    params_file = os.path.join(params_dir, f'best_params_{dataset_name}.json')

    if os.path.exists(params_file):
        try:
            with open(params_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            print(f"Loaded best parameters for dataset '{dataset_name}' from {params_file}")
            print(f"Best AUC from tuning: {results['best_value']:.6f}")
            return results['best_params']
        except Exception as e:
            print(f"Warning: Could not load best parameters from {params_file}: {e}")
            print("Using default parameters instead.")

    return None


def convert_cle_hidden_params(params: dict) -> dict:
    """
    Convert separate cle_hidden1/2/3 params to cle_hidden list.
    
    Handles compatibility between Optuna tuning format and model format.
    Only does parameter name conversion, no removal.
    """
    if params is None:
        return {}
    
    # Handle anomlay_dae parameter naming (lr -> ae_lr, weight_decay -> ae_weight_decay)
    if 'weight_decay' in params and 'ae_weight_decay' not in params:
        params['ae_weight_decay'] = params.pop('weight_decay')
    
    if 'lr' in params and 'ae_lr' not in params:
        params['ae_lr'] = params.pop('lr')
    
    # Convert cle_hidden1/2/3 to cle_hidden list
    if 'cle_hidden1' in params and 'cle_hidden2' in params and 'cle_hidden3' in params:
        try:
            cle_hidden_list = [
                int(params.get('cle_hidden1')),
                int(params.get('cle_hidden2')),
                int(params.get('cle_hidden3'))
            ]
            params['cle_hidden'] = cle_hidden_list
            params.pop('cle_hidden1', None)
            params.pop('cle_hidden2', None)
            params.pop('cle_hidden3', None)
        except Exception:
            pass
    
    return params


def filter_model_params(params: dict, model_name: str) -> dict:
    """
    Filter out unsupported parameters for a specific model.
    """
    if params is None:
        return {}
    
    unsupported = {
        'dominant': ['done_hidden', 'done_num_layers', 'done_dropout', 'done_act',
                      'guide_hidden_a', 'guide_hidden_s', 'guide_num_layers', 
                      'guide_dropout', 'guide_alpha', 'use_complex_motif',
                      'gadnr_hidden', 'sample_size', 'encoder'],
        'anomaly_dae': ['done_hidden', 'done_num_layers', 'done_dropout', 'done_act',
                         'guide_hidden_a', 'guide_hidden_s', 'guide_num_layers', 
                         'guide_dropout', 'guide_alpha', 'use_complex_motif',
                         'gadnr_hidden', 'sample_size', 'encoder', 'optimizer'],
        'done': ['ae_hidden', 'ae_dropout', 'alpha',
                 'guide_hidden_a', 'guide_hidden_s', 'guide_num_layers', 
                 'guide_dropout', 'guide_alpha', 'use_complex_motif',
                 'gadnr_hidden', 'sample_size', 'encoder', 'optimizer'],
        'guide': ['ae_hidden', 'ae_dropout', 'alpha',
                  'done_hidden', 'done_num_layers', 'done_dropout', 'done_act',
                  'gadnr_hidden', 'sample_size', 'encoder', 'optimizer'],
        'gadnr': ['ae_hidden', 'ae_dropout', 'alpha',
                  'done_hidden', 'done_num_layers', 'done_dropout', 'done_act',
                  'guide_hidden_a', 'guide_hidden_s', 'guide_num_layers', 
                  'guide_dropout', 'guide_alpha', 'use_complex_motif', 'optimizer']
    }
    
    to_remove = unsupported.get(model_name, [])
    for key in to_remove:
        params.pop(key, None)
    
    return params