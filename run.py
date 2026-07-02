#!/usr/bin/env python
"""
AE + CLE Graph Anomaly Detection
=================================
Joint training of Autoencoder + CLE with auxiliary node ensemble support.
Supports multiple base AE models: DOMINANT (GCN), AnomalyDAE (dual AE).

Usage:
    python run.py --base_model dominant --dataset weibo --auxiliary_node --use_best_params --n_runs 5
    python run.py --base_model anomalydae --dataset weibo --ae_only --n_runs 1

Tuning:
    python tune.py --base_model dominant --dataset weibo --auxiliary_node --n_trials 50
"""
from ae_cle.cli import main

if __name__ == "__main__":
    main()
