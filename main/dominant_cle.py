"""
Joint training of DOMINANT + CLE models
- Use DOMINANT to get embedding (with anomaly reconstruction)
- Feed embedding into CLE for training
- Combine losses with configurable weights
- After training, combine anomaly scores with configurable weights
- Calculate and output AUC

DOMINANT provides embeddings for better CLE performance
"""

import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import math
from pygod.utils import load_data
from torch_geometric.utils import to_dense_adj
import torch.nn.functional as F
from torch.optim import Adam
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib

# 设置论文级绘图字体：Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12          # 全局字号
plt.rcParams['axes.titlesize'] = 14     # 标题字号
plt.rcParams['axes.labelsize'] = 14     # 轴标签字号
plt.rcParams['legend.fontsize'] = 12    # 图例字号


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


# ==================== AE Model ====================

class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
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
    """AE loss function"""
    diff_attribute = torch.pow(X_hat - attrs, 2)
    attribute_reconstruction_errors = torch.sqrt(torch.sum(diff_attribute, 1))
    attribute_cost = torch.mean(attribute_reconstruction_errors)

    diff_structure = torch.pow(A_hat - adj, 2)
    structure_reconstruction_errors = torch.sqrt(torch.sum(diff_structure, 1))
    structure_cost = torch.mean(structure_reconstruction_errors)

    cost = alpha * attribute_reconstruction_errors + (1 - alpha) * structure_reconstruction_errors

    return cost, structure_cost, attribute_cost


# ==================== CLE Model ====================

class MLP(nn.Module):
    def __init__(self, hidden_sizes, num_bins=7):
        super().__init__()
        self.hidden_sizes = hidden_sizes
        self.activation = nn.ReLU()

        layers = []
        for i in range(1, len(self.hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))

        if num_bins > 1:
            layers.append(nn.Linear(hidden_sizes[-1], num_bins))
            self.softmax = nn.Softmax(dim=1)
        else:
            layers.append(nn.Linear(hidden_sizes[-1], 1))
            self.softmax = lambda x: x

        self.layers = nn.ModuleList(layers)
        # Lower dropout to reduce prediction variance
        self.drop = torch.nn.Dropout(p=0.3, inplace=False)

    def forward(self, x):
        x = self.activation(self.layers[0](x))

        for layer in self.layers[1:-1]:
            x = self.activation(layer(x))
            x = self.drop(x)

        return self.softmax(self.layers[-1](x))


def binning(t, T=300, num_bins=30, device='cpu'):
    """Gives the bin number for a given t"""
    return torch.maximum(
        torch.minimum(
            torch.floor(t * num_bins / T).to(device),
            torch.tensor(num_bins - 1).to(device)
        ),
        torch.tensor(0).to(device)
    ).long()


# -------- Minimal linear normalizing flow for noise (affine Gaussian flow) --------
class LinearFlowNoise(nn.Module):
    """A simple linear normalizing flow: z ~ N(0,I), eps = z @ L^T, where L L^T ≈ Cov(X).
    Fit L by Cholesky of empirical covariance of centered X.
    """
    def __init__(self, dim, ridge=1e-3, device=None, dtype=torch.float32):
        super().__init__()
        self.dim = dim
        self.ridge = ridge
        self.register_buffer('L', torch.eye(dim, dtype=dtype, device=device))
        self.device = device
        self.dtype = dtype

    @torch.no_grad()
    def fit(self, X: torch.Tensor):
        # X expected shape [N, D], assume already centered/normalized as desired
        N, D = X.shape
        Xc = X - X.mean(dim=0, keepdim=True)
        Cov = (Xc.T @ Xc) / max(N - 1, 1)
        Cov = Cov + self.ridge * torch.eye(D, device=X.device, dtype=X.dtype)
        try:
            L = torch.linalg.cholesky(Cov)
        except RuntimeError:
            # fallback: eigen decomposition
            evals, evecs = torch.linalg.eigh(Cov)
            evals = torch.clamp(evals, min=self.ridge)
            L = evecs @ torch.diag(torch.sqrt(evals))
        self.L = L

    @torch.no_grad()
    def sample_like(self, X: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        Z = torch.randn(X.shape, device=X.device, dtype=X.dtype, generator=generator)
        # eps = Z @ L^T
        return Z @ self.L.T


class CLE():
    def __init__(self, seed=0, model_name="CLE", hidden_size=[256, 512, 256], epochs=400,
                 batch_size=64, lr=1e-4, weight_decay=5e-4, T=400, num_bins=7, device=None,
                 deterministic_noise=True, noise_seed=None):
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.T = T
        self.num_bins = num_bins

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.seed = seed

        # Optional flow-based noise
        self.noise_flow: LinearFlowNoise | None = None

        # Control noise determinism for debugging
        self.deterministic_noise = deterministic_noise
        self.noise_seed = noise_seed if noise_seed is not None else self.seed
        try:
            self.noise_gen = torch.Generator(device=self.device)
        except TypeError:
            self.noise_gen = torch.Generator()
        self.noise_gen.manual_seed(int(self.noise_seed) if self.noise_seed is not None else 0)

        betas = torch.linspace(0.0001, 0.01, T)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        self.alphas_cumprod = alphas_cumprod

        def forward_noise(x_0, t, drift=False):
            """
            线性差值噪声注入（Linear interpolation noise）：
            对于每个样本计算插值因子 s = t / (T-1)，返回
                x_t = (1 - s) * x_0 + s * eps
            其中 eps 来自 noise_flow（若存在）或标准高斯。
            保留 `drift` 参数作为 API 兼容性，但行为与非 drift 分支一致（均为线性差值）。
            """
            # 采样噪声：优先使用已拟合的线性流，否则使用标准高斯
            if self.noise_flow is not None:
                eps = self.noise_flow.sample_like(x_0, generator=self.noise_gen if self.deterministic_noise else None)
            else:
                eps = torch.randn(x_0.shape, generator=(self.noise_gen if self.deterministic_noise else None), device=self.device, dtype=x_0.dtype)

            # 计算插值系数 s in [0,1]
            T_minus1 = max(self.T - 1, 1)
            s = (t.to(torch.float32) / float(T_minus1)).to(self.device).unsqueeze(1)

            x0_dev = x_0.to(self.device)
            eps_dev = eps.to(self.device)

            # 线性插值：x_t = (1-s) * x0 + s * eps
            return ((1.0 - s) * x0_dev + s * eps_dev).to(torch.float32)

        self.forward_noise = forward_noise
        self.model = None

    def compute_loss(self, x_0, t):
        """Compute CLE loss"""
        x_noisy = self.forward_noise(x_0, t)
        t_pred = self.model(x_noisy)
        # If using discrete bins (classification), use cross-entropy on binned targets.
        # If num_bins == 1 we treat the task as regression: predict continuous normalized time in [0,1].
        if getattr(self, "num_bins", 1) > 1:
            target = binning(t, T=self.T, device=self.device, num_bins=self.num_bins)
            Loss = nn.CrossEntropyLoss()(t_pred, target)
            return Loss
        else:
            # Regression target: normalize t to [0,1] by dividing by (T-1)
            T_minus1 = max(self.T - 1, 1)
            t_norm = t.to(self.device).to(torch.float32) / float(T_minus1)
            pred = t_pred.squeeze()
            Loss = nn.MSELoss()(pred, t_norm)
            return Loss

    def predict_score(self, X):
        """Predict anomaly score"""
        preds = []
        self.model.eval()

        pred_t = self.model(X.to(self.device).to(torch.float32))
        preds.append(pred_t.cpu().detach().numpy())

        preds = np.concatenate(preds, axis=0)

        # For classification (num_bins>1) return expected bin index as score.
        # For regression (num_bins==1) return predicted continuous time scaled to [0, T-1].
        if getattr(self, "num_bins", 1) > 1:
            preds = np.matmul(preds, np.arange(0, preds.shape[-1]))
        else:
            preds = preds.squeeze()
            # preds are predicted normalized time in [0,1]; scale back to original timestep range
            T_minus1 = max(self.T - 1, 1)
            preds = preds * float(T_minus1)

        return preds


class CLERegression(CLE):
    def __init__(self, seed=0, model_name="CLE_regression", hidden_size=[256, 512, 256],
                 epochs=400, batch_size=64, lr=1e-4, weight_decay=5e-4, T=400, num_bins=7, device=None):
        # Allow num_bins == 1 to enable regression mode (predict continuous time).
        # Allow num_bins == 1 to enable regression mode (predict continuous time).
        super().__init__(seed, model_name, hidden_size, epochs, batch_size, lr, weight_decay, T, num_bins, device)


# ==================== Joint Training ====================

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
        # squash to (0,1) to be comparable with min-max
        return 1 / (1 + np.exp(-z))
    elif method == 'rank':
        # fractional ranks in [0,1]
        order = vec.argsort()
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(len(vec))
        return ranks / max(len(vec) - 1, 1)
    else:
        # fallback: no normalization
        return vec

# ---- Embedding alignment helpers ----

def _center_cols(E: torch.Tensor) -> torch.Tensor:
    return E - E.mean(dim=0, keepdim=True)

def _normalize_cols(E: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    norms = torch.norm(E, dim=0, keepdim=True).clamp_min(eps)
    return E / norms

def _procrustes_align(E_cur: torch.Tensor, E_ref: torch.Tensor):
    """Orthogonal Procrustes: find R = argmin ||E_cur R - E_ref||_F, s.t. R^T R = I"""
    M = E_cur.T @ E_ref
    U, _, Vh = torch.linalg.svd(M, full_matrices=False)
    R = U @ Vh
    E_aligned = E_cur @ R
    return E_aligned, R

def _sign_fix(E_aligned: torch.Tensor, E_ref: torch.Tensor) -> torch.Tensor:
    # Ensure each dim direction matches reference
    corr = (E_aligned * E_ref).sum(dim=0, keepdim=True)
    s = torch.sign(corr)
    s[s == 0] = 1
    return E_aligned * s

class LossNormalizer:
    """
    归一化 loss 到同一数量级
    支持多种归一化方法
    """
    def __init__(self, method='exponential_moving_average', alpha=0.9):
        """
        Parameters:
        - method: 'exponential_moving_average', 'running_average', 'min_max', 'z_score'
        - alpha: EMA 平滑系数
        """
        self.method = method
        self.alpha = alpha
        self.ae_loss_ema = None
        self.cle_loss_ema = None
        self.ae_losses = []
        self.cle_losses = []
    
    def normalize(self, ae_loss, cle_loss):
        """
        归一化两个 loss 到同一数量级

        Returns:
        - normalized_ae_loss: 归一化后的 AE loss
        - normalized_cle_loss: 归一化后的 CLE loss
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
        """指数移动平均归一化"""
        ae_loss_val = ae_loss.item() if isinstance(ae_loss, torch.Tensor) else ae_loss
        cle_loss_val = cle_loss.item() if isinstance(cle_loss, torch.Tensor) else cle_loss

        if self.ae_loss_ema is None:
            self.ae_loss_ema = ae_loss_val
            self.cle_loss_ema = cle_loss_val
        else:
            self.ae_loss_ema = self.alpha * self.ae_loss_ema + (1 - self.alpha) * ae_loss_val
            self.cle_loss_ema = self.alpha * self.cle_loss_ema + (1 - self.alpha) * cle_loss_val

        # 防止除以零
        ae_scale = max(self.ae_loss_ema, 1e-8)
        cle_scale = max(self.cle_loss_ema, 1e-8)

        # 归一化到相同的量级
        target_scale = (ae_scale + cle_scale) / 2

        normalized_ae = ae_loss * (target_scale / ae_scale)
        normalized_cle = cle_loss * (target_scale / cle_scale)

        return normalized_ae, normalized_cle
    
    def _running_avg_normalize(self, ae_loss, cle_loss):
        """运行平均归一化"""
        ae_loss_val = ae_loss.item() if isinstance(ae_loss, torch.Tensor) else ae_loss
        cle_loss_val = cle_loss.item() if isinstance(cle_loss, torch.Tensor) else cle_loss

        ae_avg = np.mean(self.ae_losses) if self.ae_losses else ae_loss_val
        cle_avg = np.mean(self.cle_losses) if self.cle_losses else cle_loss_val

        # 防止除以零
        ae_scale = max(ae_avg, 1e-8)
        cle_scale = max(cle_avg, 1e-8)

        target_scale = (ae_scale + cle_scale) / 2

        normalized_ae = ae_loss * (target_scale / ae_scale)
        normalized_cle = cle_loss * (target_scale / cle_scale)

        return normalized_ae, normalized_cle
    
    def _min_max_normalize(self, ae_loss, cle_loss):
        """最小最大归一化"""
        ae_loss_val = ae_loss.item() if isinstance(ae_loss, torch.Tensor) else ae_loss
        cle_loss_val = cle_loss.item() if isinstance(cle_loss, torch.Tensor) else cle_loss

        if len(self.ae_losses) < 2:
            return ae_loss, cle_loss

        ae_min, ae_max = min(self.ae_losses), max(self.ae_losses)
        cle_min, cle_max = min(self.cle_losses), max(self.cle_losses)

        ae_range = max(ae_max - ae_min, 1e-8)
        cle_range = max(cle_max - cle_min, 1e-8)

        # 归一化到 [0, 1]
        ae_normalized = (ae_loss_val - ae_min) / ae_range
        cle_normalized = (cle_loss_val - cle_min) / cle_range

        return torch.tensor(ae_normalized, device=ae_loss.device, dtype=ae_loss.dtype), \
               torch.tensor(cle_normalized, device=cle_loss.device, dtype=cle_loss.dtype)
    
    def _z_score_normalize(self, ae_loss, cle_loss):
        """Z-score 归一化"""
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

        return torch.tensor(ae_normalized, device=ae_loss.device, dtype=ae_loss.dtype), \
               torch.tensor(cle_normalized, device=cle_loss.device, dtype=cle_loss.dtype)


def _align_embedding(emb: torch.Tensor, emb_ref: torch.Tensor, print_stats: bool = False, prefix: str = ""):
    """Align embedding to reference using Procrustes analysis."""
    emb_n = _normalize_cols(_center_cols(emb))
    emb_aligned, _ = _procrustes_align(emb_n, emb_ref)
    emb_aligned = _sign_fix(emb_aligned, emb_ref)

    if print_stats:
        emb_np = emb_aligned.detach().cpu().numpy()
        print(f"{prefix}emb aligned | shape: {emb_np.shape} | mean: {float(emb_np.mean()):.6f} | std: {float(emb_np.std()):.6f}")
        

    return emb_aligned


def _compute_combined_score(ae_score: np.ndarray, cle_score: np.ndarray,
                           normalize_scores: bool, score_norm_method: str, lamda2: float):
    """Compute combined anomaly score with optional normalization."""
    if normalize_scores:
        ae_score_n = _normalize_vector(ae_score, method=score_norm_method)
        cle_score_n = _normalize_vector(cle_score, method=score_norm_method)
    else:
        ae_score_n = ae_score
        cle_score_n = cle_score

    # Combined score with configurable weights (AE:1.0, CLE:lamda2)
    combined_score = 1.0 * ae_score_n + lamda2 * cle_score_n
    return combined_score


def train_joint_ae_cle(data, epochs=100, ae_hidden=64, cle_hidden=[256, 512, 256],
                       batch_size=64, device=None, normalize_loss=True, normalize_method='exponential_moving_average',
                       lamda1=0.5, lamda2=0.5, normalize_scores=True, score_norm_method='min_max',
                       joint_training=True, dataset_name='unknown', use_embedding_transform=True):
    """
    Joint training of DOMINANT + CLE models with configurable weights and loss normalization

    Parameters:
    - data: dataset (PyTorch Geometric Data object)
    - epochs: AE training epochs (positive integer)
    - ae_hidden: AE hidden size (positive integer)
    - cle_hidden: CLE hidden sizes (list of positive integers)
    - batch_size: batch size (positive integer)
    - device: computing device (torch.device or None)
    - normalize_loss: whether to normalize losses (default: True)
    - normalize_method: normalization method ('exponential_moving_average', 'running_average', 'min_max', 'z_score')
    - lamda1: training-time weight for CLE loss (AE loss weight fixed to 1.0). Default 0.5
    - lamda2: evaluation-time weight for CLE score (AE score weight fixed to 1.0). Default 0.5
    - joint_training: whether to perform joint AE+CLE training (default: True). If False, only train AE.
    - use_embedding_transform: whether to use embedding transform (center+normalize+Procrustes+sign_fix). Default True.
        When True: emb_final = sign_fix(procrustes_align(normalize(center(emb))), emb_ref)
        When False: emb_final = emb (use raw embedding directly)
    """

    # Parameter validation
    if data is None:
        raise ValueError("data cannot be None")

    if not hasattr(data, 'x') or not hasattr(data, 'y') or not hasattr(data, 'edge_index'):
        raise ValueError("data must be a PyTorch Geometric Data object with x, y, and edge_index attributes")

    if not isinstance(epochs, int) or epochs <= 0:
        raise ValueError("epochs must be a positive integer")

    if not isinstance(ae_hidden, int) or ae_hidden <= 0:
        raise ValueError("ae_hidden must be a positive integer")

    if not isinstance(cle_hidden, list) or len(cle_hidden) == 0:
        raise ValueError("cle_hidden must be a non-empty list of positive integers")

    for hidden_dim in cle_hidden:
        if not isinstance(hidden_dim, int) or hidden_dim <= 0:
            raise ValueError("All elements in cle_hidden must be positive integers")

    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    valid_normalize_methods = ['exponential_moving_average', 'running_average', 'min_max', 'z_score']
    if normalize_method not in valid_normalize_methods:
        raise ValueError(f"normalize_method must be one of {valid_normalize_methods}")

    if not isinstance(lamda1, (int, float)) or lamda1 < 0:
        raise ValueError("lamda1 must be a non-negative number")

    if not isinstance(lamda2, (int, float)) or lamda2 < 0:
        raise ValueError("lamda2 must be a non-negative number")

    if not isinstance(use_embedding_transform, bool):
        raise ValueError("use_embedding_transform must be a boolean")

    print(f"Use embedding transform: {use_embedding_transform}")

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Validate device availability
    if device.type == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        device = torch.device('cpu')
    
    print("Using device:", device)
    print("Loading dataset: {}".format(dataset_name))
    print("Dataset info: {} nodes, {} features, {} anomalies".format(
        data.x.shape[0], data.x.shape[1], data.y.sum().item()))

    # Record training start time
    training_start_time = time.time()
    print(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    x = data.x
    y = data.y.bool()
    edge_index = data.edge_index
    adj = to_dense_adj(edge_index)[0]

    # Process adjacency matrix
    adj_np = adj.cpu().numpy()
    adj_np_selfloop = adj_np + sp.eye(adj_np.shape[0])
    adj_norm = normalize_adj(adj_np_selfloop)
    adj_norm = adj_norm.toarray()

    adj_label = torch.FloatTensor(adj_np_selfloop)
    adj = torch.FloatTensor(adj_norm)

    # Initialize models
    ae_model = Dominant(feat_size=x.size(1), hidden_size=ae_hidden, dropout=0.3)
    ae_optimizer = torch.optim.Adam(ae_model.parameters(), lr=5e-3)
    cle_optimizer = None  # Will be initialized when CLE model is created

    # Move to device
    adj = adj.to(device)
    adj_label = adj_label.to(device)
    x = x.to(device)
    ae_model = ae_model.to(device)

    # Initialize CLE model (only if joint training)
    cle_model = None
    if joint_training:
        cle_model = CLERegression(
            hidden_size=cle_hidden,
            epochs=300,
            batch_size=batch_size,
            lr=1e-4,
            weight_decay=5e-4,
            T=400,
            num_bins=1,
            device=device
        )

    print("\n" + "=" * 60)
    if joint_training:
        print(f"Phase 1: Joint Training DOMINANT + CLE (Unsupervised)")
        print(f"AE:1.0, CLE:lamda1={lamda1}")
    else:
        print(f"Phase 1: Training DOMINANT Only (Unsupervised)")
    if normalize_loss and joint_training:
        print(f"Loss Normalization: {normalize_method}")
    print("=" * 60)

    # Initialize loss normalizer
    loss_normalizer = LossNormalizer(method=normalize_method) if (normalize_loss and joint_training) else None

    # Reference embedding for alignment (only needed for joint training)
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

        # Fit a simple linear flow on emb_ref (center+normalize beforehand)
        flow = LinearFlowNoise(dim=emb_ref.shape[1], ridge=1e-3, device=device, dtype=emb_ref.dtype)
        flow.fit(emb_ref)
        cle_model.noise_flow = flow.eval()

    # Training loop (AE only or joint AE+CLE)
    epoch_times = []  # Store time for each epoch
    for epoch in range(epochs):
        epoch_start_time = time.time()
        try:
            ae_model.train()
            if joint_training:
                if cle_model.model is not None:
                    cle_model.model.train()

            ae_optimizer.zero_grad()

            # Initialize CLE optimizer if needed
            if joint_training:
                if cle_model.model is not None and cle_optimizer is None:
                    cle_optimizer = Adam(cle_model.model.parameters(), lr=1e-4, weight_decay=5e-4)

                if cle_optimizer is not None:
                    cle_optimizer.zero_grad()

            # AE forward pass
            A_hat, X_hat, emb = ae_model(x, adj)
            ae_loss, struct_loss, feat_loss = ae_loss_func(adj_label, A_hat, x, X_hat, 0.8)
            ae_loss_mean = torch.mean(ae_loss)

            if joint_training:
                # Align embedding to reference before feeding into CLE
                if use_embedding_transform:
                    emb_n = _normalize_cols(_center_cols(emb))
                    emb_aligned, _ = _procrustes_align(emb_n, emb_ref)
                    emb_aligned = _sign_fix(emb_aligned, emb_ref)
                else:
                    emb_aligned = emb  # Use raw embedding directly

                # CLE forward pass (using aligned embedding)
                if cle_model.model is None:
                    cle_model.model = MLP([emb_aligned.shape[-1]] + cle_hidden, num_bins=cle_model.num_bins).to(device)
                    cle_optimizer = Adam(cle_model.model.parameters(), lr=1e-4, weight_decay=5e-4)

                # Sample timesteps
                t = torch.randint(0, cle_model.T, (emb_aligned.shape[0],), device=device).long()
                cle_loss = cle_model.compute_loss(emb_aligned.detach(), t)

                # Normalize losses if enabled
                if normalize_loss:
                    ae_loss_normalized, cle_loss_normalized = loss_normalizer.normalize(ae_loss_mean, cle_loss)
                    joint_loss = ae_loss_normalized + lamda1 * cle_loss_normalized
                else:
                    joint_loss = ae_loss_mean + lamda1 * cle_loss

                joint_loss.backward()
                ae_optimizer.step()
                if cle_optimizer is not None:
                    cle_optimizer.step()

                print("Epoch: {:04d} | AE Loss: {:.5f} | CLE Loss: {:.5f} | Joint Loss: {:.5f}".format(
                    epoch, ae_loss_mean.item(), cle_loss.item(), joint_loss.item()))
            else:
                # AE only training
                ae_loss_mean.backward()
                ae_optimizer.step()

                print("Epoch: {:04d} | AE Loss: {:.5f}".format(epoch, ae_loss_mean.item()))

        except RuntimeError as e:
            print(f"Runtime error during training at epoch {epoch}: {str(e)}")
            print("This might be due to CUDA out of memory or invalid tensor operations.")
            raise
        except ValueError as e:
            print(f"Value error during training at epoch {epoch}: {str(e)}")
            print("This might be due to invalid input values or tensor shapes.")
            raise
        except Exception as e:
            print(f"Unexpected error during training at epoch {epoch}: {type(e).__name__}: {str(e)}")
            raise

        # Record epoch time
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)

        # Periodic evaluation
        if epoch % 20 == 0 or epoch == epochs - 1:
            ae_model.eval()
            with torch.no_grad():
                A_hat, X_hat, emb = ae_model(x, adj)
                ae_loss, struct_loss, feat_loss = ae_loss_func(adj_label, A_hat, x, X_hat, 0.8)
                ae_score = ae_loss.detach().cpu().numpy()

                # Normalize scores if enabled
                if normalize_scores:
                    ae_score = _normalize_vector(ae_score, method=score_norm_method)

                # Calculate AUC
                y_np = y.cpu().numpy()
                ae_auc = roc_auc_score(y_np, ae_score)

                if joint_training:
                    # CLE evaluation (align emb before prediction)
                    cle_model.model.eval()

                    if use_embedding_transform:
                        emb_eval_aligned = _align_embedding(emb, emb_ref)
                    else:
                        emb_eval_aligned = emb  # Use raw embedding directly
                    cle_score = cle_model.predict_score(emb_eval_aligned)
                    combined_score = _compute_combined_score(ae_score, cle_score, normalize_scores, score_norm_method, lamda2)

                    # Calculate AUC (use raw scores for individual AUCs, combined uses possibly normalized)
                    cle_auc = roc_auc_score(y_np, cle_score)
                    combined_auc = roc_auc_score(y_np, combined_score)

                    print("  -> AE AUC: {:.4f} | CLE AUC: {:.4f} | Combined AUC: {:.4f}".format(
                        ae_auc, cle_auc, combined_auc))
                else:
                    print("  -> AE AUC: {:.4f}".format(ae_auc))

                # Show timing information
                elapsed_time = time.time() - training_start_time
                avg_epoch_time = np.mean(epoch_times[-20:]) if epoch_times else 0  # Last 20 epochs average
                eta_seconds = avg_epoch_time * (epochs - epoch - 1)
                eta_str = str(timedelta(seconds=int(eta_seconds)))

                print("  -> Time: {}/epoch (avg), Elapsed: {}, ETA: {}".format(
                    format_time_precise(avg_epoch_time), format_time_precise(elapsed_time), eta_str))

    # ==================== Final Evaluation ====================
    print("\n" + "=" * 60)
    print("Phase 2: Final Evaluation")
    print("=" * 60)

    try:
        ae_model.eval()

        with torch.no_grad():
            A_hat, X_hat, emb = ae_model(x, adj)
            ae_loss, struct_loss, feat_loss = ae_loss_func(adj_label, A_hat, x, X_hat, 0.8)
            ae_score = ae_loss.detach().cpu().numpy()

            # Normalize scores if enabled
            if normalize_scores:
                ae_score = _normalize_vector(ae_score, method=score_norm_method)

            # Calculate AUC
            y_np = y.cpu().numpy()
            ae_auc = roc_auc_score(y_np, ae_score)

            if joint_training:
                if cle_model.model is None:
                    raise ValueError("CLE model was not properly initialized during training")
                cle_model.model.eval()

                if use_embedding_transform:
                    emb_eval_aligned = _align_embedding(emb, emb_ref, print_stats=False)
                else:
                    emb_eval_aligned = emb  # Use raw embedding directly
                cle_score = cle_model.predict_score(emb_eval_aligned)
                combined_score = _compute_combined_score(ae_score, cle_score, normalize_scores, score_norm_method, lamda2)

                # Calculate AUC (use raw scores for individual AUCs, combined uses possibly normalized)
                cle_auc = roc_auc_score(y_np, cle_score)
                combined_auc = roc_auc_score(y_np, combined_score)

                print("\nFinal Results:")
                print("  AE AUC:       {:.6f}".format(ae_auc))
                print("  CLE AUC:      {:.6f}".format(cle_auc))
                print("  Combined AUC: {:.6f}".format(combined_auc))

                return ae_model, cle_model, combined_auc
            else:
                print("\nFinal Results:")
                print("  AE AUC: {:.6f}".format(ae_auc))

                return ae_model, ae_auc

    except RuntimeError as e:
        print(f"Runtime error during final evaluation: {str(e)}")
        print("This might be due to CUDA out of memory or invalid tensor operations.")
        raise
    except ValueError as e:
        print(f"Value error during final evaluation: {str(e)}")
        print("This might be due to invalid input values or tensor shapes.")
        raise
    except Exception as e:
        print(f"Unexpected error during final evaluation: {type(e).__name__}: {str(e)}")
        raise

    # Print training summary with timing
    total_training_time = time.time() - training_start_time
    print(f"\nTraining completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Dataset: {dataset_name}")
    print(f"Training mode: {'Joint AE+CLE' if joint_training else 'AE only'}")
    print(f"Total epochs: {epochs}")
    print(f"Average epoch time: {format_time_precise(np.mean(epoch_times))}")
    print(f"Total training time: {format_timedelta_precise(timedelta(seconds=total_training_time))}")
    print(f"Peak memory usage: {torch.cuda.max_memory_allocated(device)/1024**3:.2f}GB" if device.type == 'cuda' else "N/A")
    print("=" * 60)


def load_best_params(dataset_name, params_dir='optuna_results'):
    """
    Load best parameters for a given dataset if available
    Returns best_params dict if found, None otherwise
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train DOMINANT+CLE joint model (Unsupervised)')
    parser.add_argument('--dataset', type=str, default='disney', help='Dataset name')
    parser.add_argument('--params_dir', type=str, default='optuna_results', help='Directory containing best parameters')
    parser.add_argument('--use_best_params', action='store_true', help='Use best parameters from optuna tuning')
    parser.add_argument('--joint_training', action='store_true', default=True, help='Perform joint AE+CLE training (default: True)')
    parser.add_argument('--ae_only', action='store_true', help='Train AE only (equivalent to setting joint_training=False)')
    parser.add_argument('--n_runs', type=int, default=10, help='Number of runs for averaging results (default: 10)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--use_embedding_transform', action='store_true', default=True,
                        help='Use embedding transform (center+normalize+Procrustes+sign_fix). Default: True. Use --no_use_embedding_transform to disable.')
    parser.add_argument('--no_use_embedding_transform', dest='use_embedding_transform', action='store_false',
                        help='Disable embedding transform, use raw embedding directly')

    args = parser.parse_args()

    # Handle AE only mode
    if args.ae_only:
        args.joint_training = False

    # Load data
    print(f"Loading dataset: {args.dataset}")
    data = load_data(args.dataset)

    # Default parameters
    params = {
        'epochs': 100,
        'ae_hidden': 64,
        'cle_hidden': [256, 512, 256],
        'batch_size': 64,
        'normalize_loss': True,
        'normalize_method': 'exponential_moving_average',
        'lamda1': 0.5,
        'lamda2': 0.5,
        'normalize_scores': True,
        'score_norm_method': 'min_max'
    }

    # Try to load best parameters if requested
    if args.use_best_params:
        best_params = load_best_params(args.dataset, args.params_dir)
        if best_params:
            # Update default params with best params
            params.update(best_params)
            print("Using optimized parameters from Optuna tuning.")
        else:
            print("No optimized parameters found, using default parameters.")
    else:
        print("Using default parameters (use --use_best_params to load optimized parameters).")

    print("\nParameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")

    print(f"\nRunning {args.n_runs} time(s) for averaging results...")

    # Record total experiment start time
    experiment_start_time = time.time()
    print(f"Experiment started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Run multiple times and collect results
    auc_scores = []
    run_times = []  # Store time for each run
    import random

    for run in range(args.n_runs):
        run_start_time = time.time()
        print(f"\n{'='*60}")
        print(f"Run {run + 1}/{args.n_runs}")
        print('='*60)

        # Set seed for reproducibility but different for each run
        seed = args.seed + run
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        print(f"Random seed: {seed}")

        # Train with parameters
        if args.joint_training:
            ae_model, cle_model, final_auc = train_joint_ae_cle(
                data=data,
                epochs=params['epochs'],
                ae_hidden=params['ae_hidden'],
                cle_hidden=params['cle_hidden'],
                batch_size=params['batch_size'],
                normalize_loss=params['normalize_loss'],
                normalize_method=params['normalize_method'],
                lamda1=params['lamda1'],
                lamda2=params['lamda2'],
                normalize_scores=params['normalize_scores'],
                score_norm_method=params['score_norm_method'],
                joint_training=args.joint_training,
                dataset_name=args.dataset,
                use_embedding_transform=args.use_embedding_transform
            )
        else:
            ae_model, final_auc = train_joint_ae_cle(
                data=data,
                epochs=params['epochs'],
                ae_hidden=params['ae_hidden'],
                cle_hidden=params['cle_hidden'],  # Not used in AE-only mode
                batch_size=params['batch_size'],
                normalize_loss=params['normalize_loss'],
                normalize_method=params['normalize_method'],
                lamda1=params['lamda1'],
                lamda2=params['lamda2'],
                normalize_scores=params['normalize_scores'],
                score_norm_method=params['score_norm_method'],
                joint_training=args.joint_training,
                dataset_name=args.dataset,
                use_embedding_transform=args.use_embedding_transform  # Kept for consistency
            )

        run_time = time.time() - run_start_time
        run_times.append(run_time)

        auc_scores.append(final_auc)
        print(f"Run {run + 1} AUC: {final_auc:.6f} (Time: {format_timedelta_precise(timedelta(seconds=run_time))})")

    # Calculate statistics (excluding the 2 worst AUC scores and first run time)
    auc_scores = np.array(auc_scores)

    # Sort scores and exclude the 2 smallest (worst) ones
    sorted_scores = np.sort(auc_scores)
    filtered_scores = sorted_scores[2:]  # Keep all except the first 2 (smallest)

    mean_auc = np.mean(filtered_scores)
    std_auc = np.std(filtered_scores)

    # Calculate average run time (excluding first run if more than 1 run)
    if len(run_times) > 1:
        # Exclude the first run time to avoid warm-up overhead
        filtered_run_times = run_times[1:]
        avg_run_time = np.mean(filtered_run_times)
        run_time_note = f" (excluding first run, based on {len(filtered_run_times)} runs)"
    else:
        filtered_run_times = run_times
        avg_run_time = np.mean(filtered_run_times)
        run_time_note = ""

    # Calculate total experiment time
    total_experiment_time = time.time() - experiment_start_time

    print(f"\nExperiment completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Training mode: {'Joint AE+CLE' if not args.ae_only else 'AE only'}")
    print(f"Number of runs: {args.n_runs}")
    print(f"Individual AUC scores: {[f'{score:.6f}' for score in auc_scores]}")
    print(f"Individual run times: {[f'{format_timedelta_precise(timedelta(seconds=t))}' for t in run_times]}")
    if len(run_times) > 1:
        print(f"Filtered run times (excluding first): {[f'{format_timedelta_precise(timedelta(seconds=t))}' for t in filtered_run_times]}")
    print(f"Mean AUC: {mean_auc:.6f}")
    print(f"Std AUC:  {std_auc:.6f} ")
    print(f"95% CI:   [{mean_auc - 1.96*std_auc:.6f}, {mean_auc + 1.96*std_auc:.6f}] (95%置信区间)")
    print(f"AUC: {mean_auc:.6f} ± {std_auc:.6f} ")
    print(f"Average run time: {format_timedelta_precise(timedelta(seconds=avg_run_time))}{run_time_note}")
    print(f"Total experiment time: {format_timedelta_precise(timedelta(seconds=total_experiment_time))}")
    print("=" * 60)


# ==================== t-SNE Visualization ====================

def visualize_latent_space(features, embeddings, labels, save_path='tsne.pdf'):
    """
    t-SNE 可视化函数：对比原始特征空间和学到的嵌入空间中正常点与异常点的分布

    Parameters:
    - features: 原始节点特征 (N, D)
    - embeddings: 模型学到的 Latent Representations (N, d)
    - labels: 节点标签 (N, )，0 是正常点，1 是异常点
    - save_path: 图片保存路径 (支持 .pdf, .png 等格式)

    Features:
    - 自动采样：如果 N > 4000，保留所有异常点后随机采样正常点
    - 论文级排版：包含 (a) Raw Attributes 和 (b) Learned Embeddings 两个子图
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    # ==================== 参数设置 ====================
    MAX_SAMPLES = 4000  # 最大采样数量

    # 转换为 numpy array
    features = np.array(features)
    embeddings = np.array(embeddings)
    labels = np.array(labels)

    N = len(labels)
    print(f"Total samples: {N}")

    # ==================== 采样策略 ====================
    # 必须保留所有异常点 (Label=1)，再从正常点 (Label=0) 中补齐
    anomaly_mask = (labels == 1)
    normal_mask = (labels == 0)

    n_anomaly = np.sum(anomaly_mask)
    n_normal = np.sum(normal_mask)

    print(f"Normal samples: {n_normal}, Anomaly samples: {n_anomaly}")

    if N <= MAX_SAMPLES:
        # 不需要采样，使用全部数据
        indices = np.arange(N)
        print(f"Using all {N} samples (N <= {MAX_SAMPLES})")
    else:
        # 需要采样：保留所有异常点，从正常点中随机采样
        anomaly_indices = np.where(anomaly_mask)[0]

        # 计算需要从正常点中采样的数量
        n_normal_to_sample = MAX_SAMPLES - n_anomaly

        if n_normal_to_sample < 0:
            # 异常点数量已经超过最大值，保留所有异常点
            print(f"Warning: Anomaly samples ({n_anomaly}) > MAX_SAMPLES ({MAX_SAMPLES}), "
                  f"using all anomalies only")
            indices = anomaly_indices
        else:
            # 从正常点中随机采样
            normal_indices = np.where(normal_mask)[0]
            sampled_normal_indices = np.random.choice(
                normal_indices,
                size=min(n_normal_to_sample, n_normal),
                replace=False
            )

            # 合并索引
            indices = np.concatenate([anomaly_indices, sampled_normal_indices])
            np.random.shuffle(indices)

        print(f"Sampled {len(indices)} samples "
              f"(all {n_anomaly} anomalies + {len(indices) - n_anomaly} normals)")

    # 应用采样
    features_sampled = features[indices]
    embeddings_sampled = embeddings[indices]
    labels_sampled = labels[indices]

    # 分离正常点和异常点的索引
    normal_idx = np.where(labels_sampled == 0)[0]
    anomaly_idx = np.where(labels_sampled == 1)[0]

    print(f"Plotting: {len(normal_idx)} normal, {len(anomaly_idx)} anomaly samples")

    # ==================== t-SNE 降维 ====================
    print("Running t-SNE on features...")
    tsne_features = TSNE(
        n_components=2,
        perplexity=min(30, len(features_sampled) - 1),
        random_state=42,
        max_iter=1000,
        learning_rate='auto',
        init='pca'
    ).fit_transform(features_sampled)

    print("Running t-SNE on embeddings...")
    tsne_embeddings = TSNE(
        n_components=2,
        perplexity=min(30, len(embeddings_sampled) - 1),
        random_state=42,
        max_iter=1000,
        learning_rate='auto',
        init='pca'
    ).fit_transform(embeddings_sampled)

    # ==================== 绘图 ====================
    # 设置论文级绘图风格
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'legend.fontsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
    })

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 颜色配置
    COLOR_NORMAL = '#1f77b4'  # 蓝色 - 正常点
    COLOR_ANOMALY = '#d62728'  # 红色 - 异常点

    # 子图 1: Raw Attributes
    ax1 = axes[0]
    ax1.set_title("(a) Raw Attributes", fontsize=14, fontweight='bold')

    # 先画正常点（蓝色，半透明，小点）
    ax1.scatter(
        tsne_features[normal_idx, 0],
        tsne_features[normal_idx, 1],
        c=COLOR_NORMAL,
        alpha=0.3,
        s=15,
        label='Normal',
        rasterized=True  # 减少文件大小
    )

    # 再画异常点（红色，不透明，大点，三角形）
    ax1.scatter(
        tsne_features[anomaly_idx, 0],
        tsne_features[anomaly_idx, 1],
        c=COLOR_ANOMALY,
        alpha=0.9,
        s=50,
        marker='^',
        edgecolors='black',
        linewidths=0.5,
        label='Anomaly',
        zorder=10  # 确保异常点在最上层
    )

    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.grid(False)

    # 子图 2: Learned Embeddings
    ax2 = axes[1]
    ax2.set_title("(b) Learned Embeddings", fontsize=14, fontweight='bold')

    # 先画正常点（蓝色，半透明，小点）
    ax2.scatter(
        tsne_embeddings[normal_idx, 0],
        tsne_embeddings[normal_idx, 1],
        c=COLOR_NORMAL,
        alpha=0.3,
        s=15,
        label='Normal',
        rasterized=True
    )

    # 再画异常点（红色，不透明，大点，三角形）
    ax2.scatter(
        tsne_embeddings[anomaly_idx, 0],
        tsne_embeddings[anomaly_idx, 1],
        c=COLOR_ANOMALY,
        alpha=0.9,
        s=50,
        marker='^',
        edgecolors='black',
        linewidths=0.5,
        label='Anomaly',
        zorder=10
    )

    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.legend(loc='upper right', framealpha=0.9)
    ax2.grid(False)

    # 调整布局并保存
    plt.tight_layout()

    # 保存高清图片
    if save_path.endswith('.pdf'):
        plt.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300)
    else:
        plt.savefig(save_path, bbox_inches='tight', dpi=300, facecolor='white')

    print(f"Figure saved to: {save_path}")
    plt.show()

    return fig


def visualize_from_model(data, ae_model, cle_model=None, save_path='tsne.pdf', device=None):
    """
    从训练好的模型提取嵌入并进行 t-SNE 可视化

    Parameters:
    - data: PyTorch Geometric Data 对象
    - ae_model: 训练好的 AE 模型
    - cle_model: 训练好的 CLE 模型 (可选)
    - save_path: 图片保存路径
    - device: 计算设备
    """
    import torch
    from torch_geometric.utils import to_dense_adj
    import scipy.sparse as sp
    import numpy as np

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 提取特征
    x = data.x.to(device)
    edge_index = data.edge_index
    adj = to_dense_adj(edge_index)[0]

    # 处理邻接矩阵
    adj_np = adj.cpu().numpy()
    adj_np_selfloop = adj_np + sp.eye(adj_np.shape[0])
    adj_norm = normalize_adj(adj_np_selfloop)
    adj_norm = adj_norm.toarray()
    adj = torch.FloatTensor(adj_norm).to(device)

    # 提取嵌入
    ae_model.eval()
    with torch.no_grad():
        if cle_model is not None:
            # 联合模型：使用原始嵌入
            A_hat, X_hat, embeddings = ae_model(x, adj)
        else:
            # 仅 AE 模型
            A_hat, X_hat, embeddings = ae_model(x, adj)

    embeddings = embeddings.cpu().numpy()
    features = data.x.cpu().numpy()
    labels = data.y.cpu().numpy()

    # 调用可视化函数
    return visualize_latent_space(features, embeddings, labels, save_path)


if __name__ == "__main__":
    # 测试可视化函数
    import argparse

    parser = argparse.ArgumentParser(description='t-SNE Visualization for GAD Models')
    parser.add_argument('--dataset', type=str, default='disney', help='Dataset name')
    parser.add_argument('--model_path', type=str, default=None, help='Path to saved model')
    parser.add_argument('--save_path', type=str, default='tsne.pdf', help='Output path for visualization')

    args = parser.parse_args()

    # 加载数据
    data = load_data(args.dataset)
    print(f"Dataset: {args.dataset}")
    print(f"Nodes: {data.x.shape[0]}, Features: {data.x.shape[1]}")
    print(f"Anomalies: {data.y.sum().item()}")

    # 使用默认参数训练模型
    print("\nTraining model...")
    ae_model, cle_model, auc = train_joint_ae_cle(
        data=data,
        epochs=50,  # 少量 epoch 用于演示
        ae_hidden=64,
        cle_hidden=[256, 512, 256],
        batch_size=64,
        joint_training=(cle_model is not None),
        dataset_name=args.dataset
    )

    # 进行可视化
    print("\nGenerating t-SNE visualization...")
    visualize_from_model(
        data=data,
        ae_model=ae_model,
        cle_model=cle_model,
        save_path=args.save_path
    )
