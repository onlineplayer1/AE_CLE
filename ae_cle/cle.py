"""CLE (Continuous-time Likelihood Estimation) module for denoising-based anomaly detection."""

import torch
import torch.nn as nn
import numpy as np


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
        N, D = X.shape
        Xc = X - X.mean(dim=0, keepdim=True)
        Cov = (Xc.T @ Xc) / max(N - 1, 1)
        Cov = Cov + self.ridge * torch.eye(D, device=X.device, dtype=X.dtype)
        try:
            L = torch.linalg.cholesky(Cov)
        except RuntimeError:
            evals, evecs = torch.linalg.eigh(Cov)
            evals = torch.clamp(evals, min=self.ridge)
            L = evecs @ torch.diag(torch.sqrt(evals))
        self.L = L

    @torch.no_grad()
    def sample_like(self, X: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        Z = torch.randn(X.shape, device=X.device, dtype=X.dtype, generator=generator)
        return Z @ self.L.T


class CLE():
    def __init__(self, seed=0, hidden_size=[256, 512, 256], epochs=400,
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

        self.noise_flow: LinearFlowNoise | None = None

        self.deterministic_noise = deterministic_noise
        self.noise_seed = noise_seed if noise_seed is not None else self.seed
        try:
            self.noise_gen = torch.Generator(device=self.device)
        except TypeError:
            self.noise_gen = torch.Generator()
        self.noise_gen.manual_seed(int(self.noise_seed) if self.noise_seed is not None else 0)

        def forward_noise(x_0, t):
            if self.noise_flow is not None:
                eps = self.noise_flow.sample_like(x_0, generator=self.noise_gen if self.deterministic_noise else None)
            else:
                eps = torch.randn(x_0.shape, generator=(self.noise_gen if self.deterministic_noise else None), device=self.device, dtype=x_0.dtype)

            T_minus1 = max(self.T - 1, 1)
            s = (t.to(torch.float32) / float(T_minus1)).to(self.device).unsqueeze(1)

            x0_dev = x_0.to(self.device)
            eps_dev = eps.to(self.device)

            return ((1.0 - s) * x0_dev + s * eps_dev).to(torch.float32)

        self.forward_noise = forward_noise
        self.model = None

    def compute_loss(self, x_0, t):
        """Compute CLE loss"""
        x_noisy = self.forward_noise(x_0, t)
        t_pred = self.model(x_noisy)
        if getattr(self, "num_bins", 1) > 1:
            target = binning(t, T=self.T, device=self.device, num_bins=self.num_bins)
            Loss = nn.CrossEntropyLoss()(t_pred, target)
            return Loss
        else:
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

        if getattr(self, "num_bins", 1) > 1:
            preds = np.matmul(preds, np.arange(0, preds.shape[-1]))
        else:
            preds = preds.squeeze()
            T_minus1 = max(self.T - 1, 1)
            preds = preds * float(T_minus1)

        return preds


class CLERegression(CLE):
    def __init__(self, seed=0, hidden_size=[256, 512, 256],
                 epochs=400, batch_size=64, lr=1e-4, weight_decay=5e-4, T=400, num_bins=7, device=None):
        super().__init__(seed, hidden_size, epochs, batch_size, lr, weight_decay, T, num_bins, device)
