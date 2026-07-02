"""Utility functions for time formatting, score normalization, and embedding alignment."""

import torch
import numpy as np


def format_time_precise(seconds):
    """Format time with millisecond precision"""
    if seconds < 1.0:
        return ".1f"
    elif seconds < 60.0:
        return ".3f"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        if hours > 0:
            return f"{hours}:{minutes:02d}:{secs:06.3f}"
        else:
            return f"{minutes}:{secs:06.3f}"


def format_timedelta_precise(td):
    """Format timedelta with millisecond precision"""
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


def _normalize_vector(vec: np.ndarray, method: str = 'min_max'):
    """
    Normalize a 1D numpy array to a comparable range.
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


# ---- Embedding alignment helpers ----

def _center_cols(E: torch.Tensor) -> torch.Tensor:
    return E - E.mean(dim=0, keepdim=True)


def _normalize_cols(E: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    norms = torch.norm(E, dim=0, keepdim=True).clamp_min(eps)
    return E / norms


def _procrustes_align(E_cur: torch.Tensor, E_ref: torch.Tensor):
    """Orthogonal Procrustes: find R = argmin ||E_cur R - E_ref||_F, s.t. R^T R = I"""
    # Guard against NaN/Inf in inputs (can happen with bad hyperparams during tuning)
    if not torch.isfinite(E_cur).all() or not torch.isfinite(E_ref).all():
        return E_cur, torch.eye(E_cur.shape[1], device=E_cur.device, dtype=E_cur.dtype)
    M = E_cur.T @ E_ref
    if not torch.isfinite(M).all():
        return E_cur, torch.eye(E_cur.shape[1], device=E_cur.device, dtype=E_cur.dtype)
    reg = 1e-10 * torch.eye(M.shape[0], device=M.device, dtype=M.dtype)
    try:
        U, _, Vh = torch.linalg.svd(M + reg, full_matrices=False)
    except RuntimeError:
        try:
            U, _, Vh = torch.linalg.svd(M.cpu() + reg.cpu(), full_matrices=False)
            U, Vh = U.to(M.device), Vh.to(M.device)
        except RuntimeError:
            return E_cur, torch.eye(E_cur.shape[1], device=E_cur.device, dtype=E_cur.dtype)
    R = U @ Vh
    E_aligned = E_cur @ R
    return E_aligned, R


def _sign_fix(E_aligned: torch.Tensor, E_ref: torch.Tensor) -> torch.Tensor:
    corr = (E_aligned * E_ref).sum(dim=0, keepdim=True)
    s = torch.sign(corr)
    s[s == 0] = 1
    return E_aligned * s


def _align_embedding(emb: torch.Tensor, emb_ref: torch.Tensor, print_stats: bool = False, prefix: str = ""):
    """Align embedding to reference using Procrustes analysis."""
    emb_n = _normalize_cols(_center_cols(emb))
    emb_aligned, _ = _procrustes_align(emb_n, emb_ref)
    emb_aligned = _sign_fix(emb_aligned, emb_ref)

    if print_stats:
        emb_np = emb_aligned.detach().cpu().numpy()
        print(f"{prefix}emb aligned | shape: {emb_np.shape} | mean: {float(emb_np.mean()):.6f} | std: {float(emb_np.std()):.6f}")

    return emb_aligned


def compute_all_metrics(y_true, y_score, k=None):
    """Compute AUC-ROC, AUPRC, Precision@K, Recall@K.

    K defaults to the number of true anomalies (standard in GAD literature).

    Parameters
    ----------
    y_true : np.ndarray (bool or int)
        Binary ground-truth labels.
    y_score : np.ndarray (float)
        Anomaly scores (higher = more anomalous).
    k : int or None
        Top-K cutoff. If None, defaults to number of true anomalies.

    Returns
    -------
    dict with keys: auc, auprc, precision_at_k, recall_at_k, k
    """
    from sklearn.metrics import roc_auc_score, average_precision_score

    if not isinstance(y_true, np.ndarray):
        y_true = np.asarray(y_true)
    if not isinstance(y_score, np.ndarray):
        y_score = np.asarray(y_score)

    y_true = y_true.astype(np.int32) if y_true.dtype == bool else y_true.copy()
    n_anom = int(y_true.sum())
    if k is None:
        k = max(1, n_anom)

    # Guard against NaN/Inf in anomaly scores (can happen with numerical instability)
    nan_mask = ~np.isfinite(y_score)
    if nan_mask.any():
        n_nan = int(nan_mask.sum())
        print(f"  [WARNING] {n_nan} NaN/Inf values in anomaly scores, replacing with median of finite values")
        finite_vals = y_score[np.isfinite(y_score)]
        fill_val = float(np.median(finite_vals)) if len(finite_vals) > 0 else 0.0
        y_score = y_score.copy()
        y_score[nan_mask] = fill_val

    auc = float(roc_auc_score(y_true, y_score))
    auprc = float(average_precision_score(y_true, y_score))

    # Precision@K / Recall@K
    top_k_idx = np.argsort(y_score)[-k:]
    y_pred_topk = np.zeros_like(y_true)
    y_pred_topk[top_k_idx] = 1
    tp = int((y_pred_topk & y_true).sum())

    precision_at_k = tp / k if k > 0 else 0.0
    recall_at_k = tp / n_anom if n_anom > 0 else 0.0

    return {
        'auc': auc, 'auprc': auprc,
        'precision_at_k': precision_at_k, 'recall_at_k': recall_at_k,
        'k': k
    }


def _compute_combined_score(ae_score: np.ndarray, cle_score: np.ndarray,
                           normalize_scores: bool, score_norm_method: str, lamda2: float):
    """Compute combined anomaly score with optional normalization."""
    if normalize_scores:
        ae_score_n = _normalize_vector(ae_score, method=score_norm_method)
        cle_score_n = _normalize_vector(cle_score, method=score_norm_method)
    else:
        ae_score_n = ae_score
        cle_score_n = cle_score

    combined_score = 1.0 * ae_score_n + lamda2 * cle_score_n
    return combined_score


class LossNormalizer:
    """
    Normalize loss to comparable scales.
    Supports multiple normalization methods.
    """
    def __init__(self, method='exponential_moving_average', alpha=0.9):
        """
        Parameters:
        - method: 'exponential_moving_average', 'running_average', 'min_max', 'z_score'
        - alpha: EMA smoothing coefficient
        """
        self.method = method
        self.alpha = alpha
        self.ae_loss_ema = None
        self.cle_loss_ema = None
        self.ae_losses = []
        self.cle_losses = []

    def normalize(self, ae_loss, cle_loss):
        """
        Normalize two losses to comparable scales.

        Returns:
        - normalized_ae_loss: normalized AE loss
        - normalized_cle_loss: normalized CLE loss
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
        """Exponential moving average normalization"""
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
        """Running average normalization"""
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
        """Min-max normalization (preserves gradient graph)"""
        if len(self.ae_losses) < 2:
            return ae_loss, cle_loss

        ae_min = min(self.ae_losses)
        ae_max = max(self.ae_losses)
        cle_min = min(self.cle_losses)
        cle_max = max(self.cle_losses)

        ae_range = max(ae_max - ae_min, 1e-8)
        cle_range = max(cle_max - cle_min, 1e-8)

        ae_normalized = (ae_loss - ae_min) / ae_range
        cle_normalized = (cle_loss - cle_min) / cle_range

        return ae_normalized, cle_normalized

    def _z_score_normalize(self, ae_loss, cle_loss):
        """Z-score normalization (preserves gradient graph)"""
        if len(self.ae_losses) < 2:
            return ae_loss, cle_loss

        ae_mean = np.mean(self.ae_losses)
        ae_std = np.std(self.ae_losses) if np.std(self.ae_losses) > 0 else 1e-8

        cle_mean = np.mean(self.cle_losses)
        cle_std = np.std(self.cle_losses) if np.std(self.cle_losses) > 0 else 1e-8

        ae_normalized = (ae_loss - ae_mean) / ae_std
        cle_normalized = (cle_loss - cle_mean) / cle_std

        return ae_normalized, cle_normalized
