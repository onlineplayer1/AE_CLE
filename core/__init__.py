"""
GAD-CLE: Graph Anomaly Detection with Curriculum Learning Enhancement

A modular framework for combining graph anomaly detection models with
curriculum learning-based denoising (CLE).

Models:
- DOMINANT: Deep Anomaly Detection on Attributed Networks
- AnomalyDAE: Anomaly Detection with Dual Autoencoders
- DONE: Deep Outlier Aware Attributed Network Embedding
- GUIDE: Graph UnSupervised Anomaly Detection via Influential Features
- GAD-NR: Graph Anomaly Detection via Neighborhood Reconstruction

Installation:
    pip install torch torch-geometric pygod scikit-learn numpy scipy optuna

Usage:
    python main.py --model dominant --dataset disney
    python main.py --model anomaly_dae --dataset enron --use_best_params

Author: Your Name
"""

__version__ = "1.0.0"

from .cle_models import CLE, CLERegression, CLEMLP, LinearFlowNoise
from .cle_utils import (
    LossNormalizer,
    normalize_vector,
    align_embedding,
    compute_combined_score
)

__all__ = [
    "CLE",
    "CLERegression",
    "CLEMLP",
    "LossNormalizer",
    "normalize_vector",
    "align_embedding",
    "compute_combined_score",
    "LinearFlowNoise",
]
