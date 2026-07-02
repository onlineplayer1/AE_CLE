"""AE + CLE: Joint graph anomaly detection with ensemble learning."""

# GNN layers (re-exported through dominant.py for unified interface)
from .dominant import (GraphConvolution, Encoder, Attribute_Decoder,
    Structure_Decoder, Dominant, normalize_adj, ae_loss_func,
    _train_single_joint, train_joint_ae_cle)

# CLE module
from .cle import MLP, binning, LinearFlowNoise, CLE, CLERegression

# Data augmentation
from .augment import add_auxiliary_nodes

# Utilities
from .utils import (format_time_precise, format_timedelta_precise,
    _normalize_vector, _center_cols, _normalize_cols, _procrustes_align,
    _sign_fix, _align_embedding, _compute_combined_score, compute_all_metrics,
    LossNormalizer)

# Training (AnomalyDAE)
from .anomalydae import (AnomalyDAE, _train_single_joint_anomalydae,
    train_joint_anomalydae_cle)

# Ensemble (DOMINANT)
from .ensemble import _train_one_aux_worker, train_auxiliary_node_ensemble

# Ensemble (AnomalyDAE)
from .ensemble import (_train_one_aux_worker_anomalydae,
    train_auxiliary_node_ensemble_anomalydae)

# Training (GUIDE)
from .guide import (GUIDE_Base, calculate_structural_features,
    calculate_simplified_statistics, _train_single_joint_guide,
    train_joint_guide_cle)

# Ensemble (GUIDE)
from .ensemble import (_train_one_aux_worker_guide,
    train_auxiliary_node_ensemble_guide)

# Training (GAD-NR)
from .gadnr import (GNNStructEncoder, build_neighbor_dict, compute_degree_matrix,
    _train_single_joint_gadnr, train_joint_gadnr_cle)

# Ensemble (GAD-NR)
from .ensemble import (_train_one_aux_worker_gadnr,
    train_auxiliary_node_ensemble_gadnr)

# Training (DONE)
from .done import (DONE_Base, _train_single_joint_done, train_joint_done_cle)

# Ensemble (DONE)
from .ensemble import (_train_one_aux_worker_done,
    train_auxiliary_node_ensemble_done)

# CLI utilities
from .cli import load_best_params
