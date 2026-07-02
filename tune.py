"""
Optuna hyperparameter tuning for AE+CLE joint training.
Optimizes key hyperparameters to maximize AUC score and saves best parameters.

Supports two modes:
- Standard: tune train_joint_ae_cle parameters
- Auxiliary node ensemble: tune train_auxiliary_node_ensemble parameters
"""

import optuna
import torch
import numpy as np
import random
from pygod.utils import load_data
import json
from datetime import datetime
import os
import argparse


class OptunaTrainer:
    """Wrapper class for Optuna optimization of AE+CLE joint training"""

    def __init__(self, data, dataset_name, device=None, n_trials=50, timeout=None, study_name=None, seed=42,
                 use_auxiliary_node=False, ae_only=False, base_model='dominant', metric='auc',
                 n_seeds_per_trial=1):
        self.data = data
        self.dataset_name = dataset_name
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_trials = n_trials
        self.timeout = timeout
        self.seed = seed
        self.use_auxiliary_node = use_auxiliary_node
        self.ae_only = ae_only
        self.base_model = base_model
        self.metric = metric  # 'auc' or 'auprc'
        self.n_seeds_per_trial = n_seeds_per_trial  # average over N seeds to reduce variance
        if base_model == 'anomalydae':
            base_tag = '_anomalydae'
        elif base_model == 'guide':
            base_tag = '_guide'
        elif base_model == 'gadnr':
            base_tag = '_gadnr'
        elif base_model == 'done':
            base_tag = '_done'
        else:
            base_tag = ''
        if use_auxiliary_node:
            mode_suffix = base_tag + "_auxnode"
        elif ae_only:
            mode_suffix = base_tag + "_aeonly"
        else:
            mode_suffix = base_tag
        self.mode_suffix = mode_suffix
        if base_model == 'anomalydae':
            model_label = 'AnomalyDAE'
        elif base_model == 'guide':
            model_label = 'GUIDE'
        elif base_model == 'gadnr':
            model_label = 'GADNR'
        elif base_model == 'done':
            model_label = 'DONE'
        else:
            model_label = 'DOMINANT'
        self.study_name = study_name or f"{model_label.lower()}_cle_tuning_{dataset_name}{mode_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def objective(self, trial):
        """Optuna objective function — dispatches to ae_only, auxiliary-node, or standard mode"""
        if self.ae_only:
            return self._objective_ae_only(trial)
        elif self.use_auxiliary_node:
            return self._objective_auxiliary_node(trial)
        else:
            return self._objective_standard(trial)

    def _objective_ae_only(self, trial):
        """AE-only objective — tune AE hyperparameters only (no CLE)."""
        # AE parameters
        ae_hidden = trial.suggest_int('ae_hidden', 32, 128, step=16)
        epochs = trial.suggest_int('epochs', 100, 300, step=50)
        batch_size = trial.suggest_int('batch_size', 32, 128, step=32)
        if self.base_model == 'anomalydae':
            ae_dropout = trial.suggest_float('ae_dropout', 0.0, 0.5, step=0.1)
        elif self.base_model == 'guide':
            guide_dropout = trial.suggest_float('guide_dropout', 0.0, 0.5, step=0.1)
            guide_alpha = trial.suggest_float('guide_alpha', 0.1, 0.9, step=0.1)
        elif self.base_model == 'gadnr':
            sample_size = trial.suggest_int('sample_size', 5, 30, step=5)
            encoder = trial.suggest_categorical('encoder', ['GCN', 'GIN', 'GAT', 'SAGE'])
        elif self.base_model == 'done':
            done_dropout = trial.suggest_float('done_dropout', 0.0, 0.5, step=0.1)
            done_num_layers = trial.suggest_int('done_num_layers', 1, 4, step=1)
        else:
            dropout = trial.suggest_float('dropout', 0.0, 0.5, step=0.1)
            struct_weight = trial.suggest_float('struct_weight', 0.1, 0.9, step=0.1)
        normalize_scores = trial.suggest_categorical('normalize_scores', [True, False])
        score_norm_method = trial.suggest_categorical('score_norm_method', ['min_max', 'z_score', 'rank'])

        print(f"\n{'='*60}")
        print(f"Trial {trial.number}: Testing AE-only hyperparameters ({self.base_model})")
        print(f"{'='*60}")
        print(f"AE: hidden={ae_hidden}, epochs={epochs}, batch={batch_size}")
        if self.base_model == 'anomalydae':
            print(f"AE dropout: {ae_dropout}")
        elif self.base_model == 'guide':
            print(f"GUIDE dropout: {guide_dropout}, alpha: {guide_alpha}")
        elif self.base_model == 'gadnr':
            print(f"GAD-NR sample_size: {sample_size}, encoder: {encoder}")
        elif self.base_model == 'done':
            print(f"DONE dropout: {done_dropout}")
        else:
            print(f"DOMINANT dropout: {dropout}, struct_weight: {struct_weight}")
        print(f"Scores: normalize={normalize_scores}, method={score_norm_method}")

        all_scores = []
        for s in range(self.n_seeds_per_trial):
            run_seed = self.seed + trial.number * 1000 + s
            random.seed(run_seed)
            np.random.seed(run_seed)
            torch.manual_seed(run_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(run_seed)
            try:
                if self.base_model == 'anomalydae':
                    score = self._run_ae_only_anomalydae(ae_hidden, ae_dropout, epochs, batch_size, normalize_scores, score_norm_method)
                elif self.base_model == 'guide':
                    score = self._run_ae_only_guide(ae_hidden, guide_dropout, guide_alpha, epochs, batch_size, normalize_scores, score_norm_method)
                elif self.base_model == 'gadnr':
                    score = self._run_ae_only_gadnr(ae_hidden, sample_size, encoder, epochs, batch_size, normalize_scores, score_norm_method)
                elif self.base_model == 'done':
                    score = self._run_ae_only_done(ae_hidden, done_dropout, epochs, batch_size, normalize_scores, score_norm_method)
                else:
                    score = self._run_ae_only(ae_hidden, epochs, batch_size, normalize_scores, score_norm_method,
                                                dropout=dropout, struct_weight=struct_weight)
                all_scores.append(score)
            except Exception as e:
                print(f"  Seed {run_seed} failed: {e}")
                all_scores.append(0.0)
        avg_score = float(np.mean(all_scores))
        if self.n_seeds_per_trial > 1:
            print(f"AE-only {self.metric.upper()} (avg over {self.n_seeds_per_trial} seeds): {avg_score:.6f} (individual: {[f'{v:.4f}' for v in all_scores]})")
        else:
            print(f"AE-only {self.metric.upper()}: {avg_score:.6f}")
        return avg_score

    def _run_ae_only(self, ae_hidden, epochs, batch_size, normalize_scores, score_norm_method,
                      dropout=0.3, struct_weight=0.8):
        """Run AE-only training with given hyperparameters"""
        import ae_cle
        _, ae_metrics = ae_cle.train_joint_ae_cle(
            data=self.data,
            epochs=epochs,
            ae_hidden=ae_hidden,
            cle_hidden=[256, 512, 256],
            batch_size=batch_size,
            device=self.device,
            normalize_loss=False,
            normalize_method='exponential_moving_average',
            lamda1=0.0, lamda2=0.0,
            normalize_scores=normalize_scores,
            score_norm_method=score_norm_method,
            joint_training=False,
            dataset_name=self.dataset_name,
            use_embedding_transform=True,
            dropout=dropout, lr_ae=5e-3, struct_weight=struct_weight
        )
        return ae_metrics[self.metric]

    def _objective_standard(self, trial):
        """
        Standard (non-bagging) objective — tune train_joint_ae_cle.
        Only returns combined AUC if it outperforms AE-only baseline.
        """
        # ========== Hyperparameter Search Space ==========

        # AE parameters
        ae_hidden = trial.suggest_int('ae_hidden', 32, 128, step=16)  # 32, 48, 64, 80, 96, 112, 128

        # CLE parameters
        cle_hidden1 = trial.suggest_int('cle_hidden1', 128, 512, step=64)  # First hidden layer
        cle_hidden2 = trial.suggest_int('cle_hidden2', 256, 1024, step=128)  # Second hidden layer
        cle_hidden3 = trial.suggest_int('cle_hidden3', 128, 512, step=64)  # Third hidden layer
        cle_hidden = [cle_hidden1, cle_hidden2, cle_hidden3]

        # CLE training parameters
        cle_lr = trial.suggest_float('cle_lr', 1e-5, 1e-3, log=True)  # CLE learning rate
        cle_weight_decay = trial.suggest_float('cle_weight_decay', 1e-5, 1e-3, log=True)  # CLE weight decay
        cle_T = trial.suggest_int('cle_T', 200, 800, step=100)  # CLE time steps (T parameter)

        # Training parameters - use longer training for more stable results
        epochs = trial.suggest_int('epochs', 100, 300, step=50)  # 100, 150, 200, 250, 300
        batch_size = trial.suggest_int('batch_size', 32, 128, step=32)  # 32, 64, 96, 128

        # Loss and score combination weights
        lamda1 = trial.suggest_float('lamda1', 0.01, 2.0, step=0.01)  # Training-time CLE weight (expanded range)
        lamda2 = trial.suggest_float('lamda2', 0.01, 2.0, step=0.01)  # Evaluation-time CLE weight (expanded range)

        # Normalization methods (categorical choices)
        normalize_method = trial.suggest_categorical('normalize_method',
            ['exponential_moving_average', 'running_average', 'min_max', 'z_score'])
        score_norm_method = trial.suggest_categorical('score_norm_method',
            ['min_max', 'z_score', 'rank'])

        # Whether to use normalization
        normalize_loss = trial.suggest_categorical('normalize_loss', [True, False])
        normalize_scores = trial.suggest_categorical('normalize_scores', [True, False])

        # AnomalyDAE-specific: add ae_dropout
        if self.base_model == 'anomalydae':
            ae_dropout = trial.suggest_float('ae_dropout', 0.0, 0.5, step=0.1)
        # GUIDE-specific params
        if self.base_model == 'guide':
            guide_dropout = trial.suggest_float('guide_dropout', 0.0, 0.5, step=0.1)
            guide_alpha = trial.suggest_float('guide_alpha', 0.1, 0.9, step=0.1)
        # GAD-NR-specific params
        if self.base_model == 'gadnr':
            sample_size = trial.suggest_int('sample_size', 5, 30, step=5)
            encoder = trial.suggest_categorical('encoder', ['GCN', 'GIN', 'GAT', 'SAGE'])
        # DONE-specific params
        if self.base_model == 'done':
            done_dropout = trial.suggest_float('done_dropout', 0.0, 0.5, step=0.1)
        # DOMINANT-specific params
        if self.base_model == 'dominant':
            dropout = trial.suggest_float('dropout', 0.0, 0.5, step=0.1)
            struct_weight = trial.suggest_float('struct_weight', 0.1, 0.9, step=0.1)

        print(f"\n{'='*60}")
        print(f"Trial {trial.number}: Testing hyperparameters ({self.base_model})")
        print(f"{'='*60}")
        print(f"AE: hidden={ae_hidden}")
        if self.base_model == 'anomalydae':
            print(f"AE dropout: {ae_dropout}")
        elif self.base_model == 'guide':
            print(f"GUIDE dropout: {guide_dropout}, alpha: {guide_alpha}")
        elif self.base_model == 'gadnr':
            print(f"GAD-NR sample_size: {sample_size}, encoder: {encoder}")
        elif self.base_model == 'done':
            print(f"DONE dropout: {done_dropout}")
        else:
            print(f"DOMINANT dropout: {dropout}, struct_weight: {struct_weight}")
        print(f"CLE: hidden={cle_hidden}, lr={cle_lr:.2e}, wd={cle_weight_decay:.2e}, T={cle_T}")
        print(f"Training: epochs={epochs}, batch_size={batch_size}")
        print(f"Weights: lamda1={lamda1:.3f}, lamda2={lamda2:.3f}")
        print(f"Normalization: loss={normalize_loss} ({normalize_method}), scores={normalize_scores} ({score_norm_method})")

        all_scores = []
        for s in range(self.n_seeds_per_trial):
            run_seed = self.seed + trial.number * 1000 + s
            random.seed(run_seed)
            np.random.seed(run_seed)
            torch.manual_seed(run_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(run_seed)
            try:
                # ========== Phase 1: AE-only training (baseline) ==========
                if s == 0:
                    print(f"\n--- Phase 1: AE-only training (baseline) ---")
                if self.base_model == 'anomalydae':
                    ae_model, ae_auc = self._train_ae_only_anomalydae(
                        ae_hidden=ae_hidden, ae_dropout=ae_dropout,
                        epochs=epochs, batch_size=batch_size)
                elif self.base_model == 'guide':
                    ae_model, ae_auc = self._train_ae_only_guide(
                        ae_hidden=ae_hidden, guide_dropout=guide_dropout,
                        guide_alpha=guide_alpha, epochs=epochs, batch_size=batch_size)
                elif self.base_model == 'gadnr':
                    ae_model, ae_auc = self._train_ae_only_gadnr(
                        ae_hidden=ae_hidden, sample_size=sample_size,
                        encoder=encoder, epochs=epochs, batch_size=batch_size)
                elif self.base_model == 'done':
                    ae_model, ae_auc = self._train_ae_only_done(
                        ae_hidden=ae_hidden, done_dropout=done_dropout,
                        epochs=epochs, batch_size=batch_size)
                else:
                    ae_model, ae_auc = self._train_ae_only(
                        ae_hidden=ae_hidden, epochs=epochs, batch_size=batch_size,
                        dropout=dropout, struct_weight=struct_weight)

                # ========== Phase 2: Joint AE+CLE training ==========
                if self.base_model == 'anomalydae':
                    combined_auc = self._train_joint_with_params_anomalydae(
                        ae_hidden=ae_hidden, ae_dropout=ae_dropout, cle_hidden=cle_hidden,
                        cle_lr=cle_lr, cle_weight_decay=cle_weight_decay, cle_T=cle_T,
                        epochs=epochs, batch_size=batch_size, lamda1=lamda1, lamda2=lamda2,
                        normalize_loss=normalize_loss, normalize_method=normalize_method,
                        normalize_scores=normalize_scores, score_norm_method=score_norm_method)
                elif self.base_model == 'guide':
                    combined_auc = self._train_joint_with_params_guide(
                        ae_hidden=ae_hidden, guide_dropout=guide_dropout, guide_alpha=guide_alpha,
                        cle_hidden=cle_hidden, cle_lr=cle_lr, cle_weight_decay=cle_weight_decay, cle_T=cle_T,
                        epochs=epochs, batch_size=batch_size, lamda1=lamda1, lamda2=lamda2,
                        normalize_loss=normalize_loss, normalize_method=normalize_method,
                        normalize_scores=normalize_scores, score_norm_method=score_norm_method)
                elif self.base_model == 'gadnr':
                    combined_auc = self._train_joint_with_params_gadnr(
                        ae_hidden=ae_hidden, sample_size=sample_size, encoder=encoder,
                        cle_hidden=cle_hidden, cle_lr=cle_lr, cle_weight_decay=cle_weight_decay, cle_T=cle_T,
                        epochs=epochs, batch_size=batch_size, lamda1=lamda1, lamda2=lamda2,
                        normalize_loss=normalize_loss, normalize_method=normalize_method,
                        normalize_scores=normalize_scores, score_norm_method=score_norm_method)
                elif self.base_model == 'done':
                    combined_auc = self._train_joint_with_params_done(
                        ae_hidden=ae_hidden, done_dropout=done_dropout, cle_hidden=cle_hidden,
                        cle_lr=cle_lr, cle_weight_decay=cle_weight_decay, cle_T=cle_T,
                        epochs=epochs, batch_size=batch_size, lamda1=lamda1, lamda2=lamda2,
                        normalize_loss=normalize_loss, normalize_method=normalize_method,
                        normalize_scores=normalize_scores, score_norm_method=score_norm_method)
                else:
                    combined_auc = self._train_joint_with_params(
                        ae_hidden=ae_hidden, cle_hidden=cle_hidden, cle_lr=cle_lr,
                        cle_weight_decay=cle_weight_decay, cle_T=cle_T, epochs=epochs,
                        batch_size=batch_size, lamda1=lamda1, lamda2=lamda2,
                        normalize_loss=normalize_loss, normalize_method=normalize_method,
                        normalize_scores=normalize_scores, score_norm_method=score_norm_method,
                        dropout=dropout, struct_weight=struct_weight)

                score = combined_auc if combined_auc > ae_auc else ae_auc
                all_scores.append(score)
            except Exception as e:
                print(f"  Seed {run_seed} failed: {e}")
                all_scores.append(0.0)

        avg_score = float(np.mean(all_scores))
        if self.n_seeds_per_trial > 1:
            print(f"Standard {self.metric.upper()} (avg over {self.n_seeds_per_trial} seeds): {avg_score:.6f} (individual: {[f'{v:.4f}' for v in all_scores]})")
        else:
            print(f"Standard {self.metric.upper()}: {avg_score:.6f}")
        return avg_score

    def _objective_auxiliary_node(self, trial):
        """Auxiliary node ensemble objective — tune train_auxiliary_node_ensemble."""
        # ========== Auxiliary-node-specific hyperparameters ==========
        n_aux_models = trial.suggest_int('n_aux_models', 8, 20, step=1)  # ≥8 for stability (TKDE)
        # Use ratio when ratio mode selected, otherwise absolute count
        use_ratio = trial.suggest_categorical('aux_use_ratio', [True, False])
        if use_ratio:
            n_aux_nodes = trial.suggest_float('aux_node_ratio', 0.01, 0.25, step=0.01)
        else:
            n_aux_nodes = trial.suggest_int('n_aux_nodes', 1, 30, step=1)
        n_connections = trial.suggest_int('n_connections', 0, 20, step=1)
        k_std = trial.suggest_float('k_std', 0.5, 5.0, step=0.5)  # capped to avoid extreme outliers
        feature_method = trial.suggest_categorical('feature_method',
            ['outlier_tail', 'gaussian_noise', 'perturb_existing', 'smote_outlier',
             'neighbor_dissimilar', 'feature_shuffle'])
        edge_method = trial.suggest_categorical('edge_method',
            ['random_connect', 'isolated', 'clique', 'low_similarity_connect'])
        agg_method = trial.suggest_categorical('agg_method', ['mean', 'max', 'median'])

        # ========== AE parameters ==========
        ae_hidden = trial.suggest_int('ae_hidden', 16, 256, step=16)
        if self.base_model == 'anomalydae':
            ae_dropout = trial.suggest_float('ae_dropout', 0.1, 0.7, step=0.1)
        elif self.base_model == 'guide':
            guide_dropout = trial.suggest_float('guide_dropout', 0.0, 0.5, step=0.1)
            guide_alpha = trial.suggest_float('guide_alpha', 0.1, 0.9, step=0.1)
        elif self.base_model == 'gadnr':
            sample_size = trial.suggest_int('sample_size', 5, 30, step=5)
            encoder = trial.suggest_categorical('encoder', ['GCN', 'GIN', 'GAT', 'SAGE'])
        elif self.base_model == 'done':
            done_dropout = trial.suggest_float('done_dropout', 0.0, 0.5, step=0.1)
        else:
            dropout = trial.suggest_float('dropout', 0.1, 0.7, step=0.1)
            struct_weight = trial.suggest_float('struct_weight', 0.1, 0.9, step=0.1)

        # ========== CLE parameters ==========
        cle_hidden1 = trial.suggest_int('cle_hidden1', 128, 512, step=64)
        cle_hidden2 = trial.suggest_int('cle_hidden2', 256, 1024, step=128)
        cle_hidden3 = trial.suggest_int('cle_hidden3', 128, 512, step=64)
        cle_hidden = [cle_hidden1, cle_hidden2, cle_hidden3]

        # ========== Training parameters ==========
        epochs = trial.suggest_int('epochs', 80, 200, step=10)  # ≥80 for convergence stability
        lr_ae = trial.suggest_float('lr_ae', 1e-4, 1e-2, log=True)

        # ========== Loss and score combination weights ==========
        lamda1 = trial.suggest_float('lamda1', 0.01, 2.0, step=0.01)
        lamda2 = trial.suggest_float('lamda2', 0.01, 2.0, step=0.01)

        # ========== Normalization methods ==========
        normalize_method = trial.suggest_categorical('normalize_method',
            ['exponential_moving_average', 'running_average', 'min_max', 'z_score'])
        score_norm_method = trial.suggest_categorical('score_norm_method',
            ['min_max', 'z_score', 'rank'])
        normalize_loss = trial.suggest_categorical('normalize_loss', [True, False])
        normalize_scores = trial.suggest_categorical('normalize_scores', [True, False])

        # ========== Embedding transform (Procrustes alignment) ==========
        use_embedding_transform = trial.suggest_categorical('use_embedding_transform', [False])

        print("\n" + "=" * 60)
        print("Trial {}: Testing auxiliary node ensemble hyperparameters ({})".format(
            trial.number, self.base_model))
        print("=" * 60)
        n_aux_str = "{:.1f}%".format(n_aux_nodes * 100) if use_ratio else "{}".format(n_aux_nodes)
        print("Aux-Node: n_models={}, n_aux={}, feature={}, edge={}, n_conn={}, k_std={:.1f}, agg={}".format(
            n_aux_models, n_aux_str, feature_method, edge_method, n_connections, k_std, agg_method))
        print("AE: hidden={}, lr={:.2e}".format(ae_hidden, lr_ae))
        if self.base_model == 'anomalydae':
            print("  ae_dropout={:.1f}".format(ae_dropout))
        elif self.base_model == 'guide':
            print("  guide_dropout={:.1f}, guide_alpha={:.1f}".format(guide_dropout, guide_alpha))
        elif self.base_model == 'gadnr':
            print("  sample_size={}, encoder={}".format(sample_size, encoder))
        elif self.base_model == 'done':
            print("  done_dropout={:.1f}".format(done_dropout))
        else:
            print("  dropout={:.1f}, struct_w={:.1f}".format(dropout, struct_weight))
        print("CLE: hidden={}".format(cle_hidden))
        print("Training: epochs={} (per model)".format(epochs))
        print("Weights: lamda1={:.3f}, lamda2={:.3f}".format(lamda1, lamda2))
        print("Normalization: loss={} ({}), scores={} ({})".format(
            normalize_loss, normalize_method, normalize_scores, score_norm_method))
        print("Embedding transform (Procrustes): {}".format(use_embedding_transform))

        all_scores = []
        for s in range(self.n_seeds_per_trial):
            run_seed = self.seed + trial.number * 1000 + s
            try:
                if self.base_model == 'anomalydae':
                    ensemble_auc, scores, model_aucs = self._run_auxiliary_node_anomalydae(
                        n_models=n_aux_models, n_aux_nodes=n_aux_nodes,
                        feature_method=feature_method, edge_method=edge_method,
                        n_connections=n_connections, k_std=k_std, ae_hidden=ae_hidden,
                        ae_dropout=ae_dropout, cle_hidden=cle_hidden, epochs=epochs,
                        lamda1=lamda1, lamda2=lamda2, normalize_loss=normalize_loss,
                        normalize_method=normalize_method, normalize_scores=normalize_scores,
                        score_norm_method=score_norm_method,
                        use_embedding_transform=use_embedding_transform,
                        agg_method=agg_method, lr_ae=lr_ae)
                elif self.base_model == 'gadnr':
                    ensemble_auc, scores, model_aucs = self._run_auxiliary_node_gadnr(
                        n_models=n_aux_models, n_aux_nodes=n_aux_nodes,
                        feature_method=feature_method, edge_method=edge_method,
                        n_connections=n_connections, k_std=k_std, ae_hidden=ae_hidden,
                        sample_size=sample_size, encoder=encoder, cle_hidden=cle_hidden,
                        epochs=epochs, lamda1=lamda1, lamda2=lamda2,
                        normalize_loss=normalize_loss, normalize_method=normalize_method,
                        normalize_scores=normalize_scores, score_norm_method=score_norm_method,
                        use_embedding_transform=use_embedding_transform,
                        agg_method=agg_method, lr_ae=lr_ae)
                elif self.base_model == 'done':
                    ensemble_auc, scores, model_aucs = self._run_auxiliary_node_done(
                        n_models=n_aux_models, n_aux_nodes=n_aux_nodes,
                        feature_method=feature_method, edge_method=edge_method,
                        n_connections=n_connections, k_std=k_std, ae_hidden=ae_hidden,
                        done_dropout=done_dropout, cle_hidden=cle_hidden, epochs=epochs,
                        lamda1=lamda1, lamda2=lamda2, normalize_loss=normalize_loss,
                        normalize_method=normalize_method, normalize_scores=normalize_scores,
                        score_norm_method=score_norm_method,
                        use_embedding_transform=use_embedding_transform,
                        agg_method=agg_method, lr_ae=lr_ae)
                elif self.base_model == 'guide':
                    ensemble_auc, scores, model_aucs = self._run_auxiliary_node_guide(
                        n_models=n_aux_models, n_aux_nodes=n_aux_nodes,
                        feature_method=feature_method, edge_method=edge_method,
                        n_connections=n_connections, k_std=k_std, ae_hidden=ae_hidden,
                        guide_dropout=guide_dropout, guide_alpha=guide_alpha,
                        cle_hidden=cle_hidden, epochs=epochs, lamda1=lamda1, lamda2=lamda2,
                        normalize_loss=normalize_loss, normalize_method=normalize_method,
                        normalize_scores=normalize_scores, score_norm_method=score_norm_method,
                        use_embedding_transform=use_embedding_transform,
                        agg_method=agg_method, lr_ae=lr_ae)
                else:
                    ensemble_auc, scores, model_aucs = self._run_auxiliary_node(
                        n_models=n_aux_models, n_aux_nodes=n_aux_nodes,
                        feature_method=feature_method, edge_method=edge_method,
                        n_connections=n_connections, k_std=k_std, ae_hidden=ae_hidden,
                        cle_hidden=cle_hidden, epochs=epochs, lamda1=lamda1, lamda2=lamda2,
                        normalize_loss=normalize_loss, normalize_method=normalize_method,
                        normalize_scores=normalize_scores, score_norm_method=score_norm_method,
                        use_embedding_transform=use_embedding_transform,
                        agg_method=agg_method, dropout=dropout, lr_ae=lr_ae,
                        struct_weight=struct_weight)

                if self.metric == 'auprc':
                    from ae_cle.utils import compute_all_metrics
                    trial_value = compute_all_metrics(self.data.y.bool().cpu().numpy(), scores)['auprc']
                else:
                    trial_value = ensemble_auc
                all_scores.append(trial_value)
            except Exception as e:
                print("  Seed {} failed: {}".format(run_seed, e))
                all_scores.append(0.0)

        avg_score = float(np.mean(all_scores))
        if self.n_seeds_per_trial > 1:
            print("Trial {} Result: Ensemble {} (avg over {} seeds) = {:.4f}  (individual: {})".format(
                trial.number, self.metric.upper(), self.n_seeds_per_trial, avg_score,
                [f'{v:.4f}' for v in all_scores]))
        else:
            print("Trial {} Result: Ensemble {} = {:.4f}".format(
                trial.number, self.metric.upper(), avg_score))
        return avg_score

    def _run_auxiliary_node(self, n_models, n_aux_nodes, feature_method, edge_method,
                             n_connections, k_std, ae_hidden, cle_hidden, epochs,
                             lamda1, lamda2, normalize_loss, normalize_method,
                             normalize_scores, score_norm_method, use_embedding_transform=True,
                             agg_method='mean', dropout=0.3, lr_ae=5e-3, struct_weight=0.8):
        """Run auxiliary node ensemble with given hyperparameters"""
        import ae_cle

        ensemble_auc, scores, model_aucs, _ =ae_cle.train_auxiliary_node_ensemble(
            data=self.data,
            n_models=n_models,
            n_aux_nodes=n_aux_nodes,
            feature_method=feature_method,
            edge_method=edge_method,
            n_connections=n_connections,
            k_std=k_std,
            epochs=epochs,
            ae_hidden=ae_hidden,
            cle_hidden=cle_hidden,
            device=self.device,
            base_seed=self.seed,
            lamda1=lamda1,
            lamda2=lamda2,
            normalize_loss=normalize_loss,
            normalize_method=normalize_method,
            normalize_scores=normalize_scores,
            score_norm_method=score_norm_method,
            use_embedding_transform=use_embedding_transform,
            joint_training=True,
            verbose=False,
            agg_method=agg_method,
            dropout=dropout,
            lr_ae=lr_ae,
            struct_weight=struct_weight,
            parallel=False  # Optuna handles trial-level parallelism; keep per-trial sequential
        )

        return ensemble_auc, scores, model_aucs

    # ---- AnomalyDAE-specific helper methods ----

    def _run_ae_only_anomalydae(self, ae_hidden, ae_dropout, epochs, batch_size, normalize_scores, score_norm_method):
        """Run AnomalyDAE AE-only training with given hyperparameters"""
        import ae_cle
        _, ae_metrics = ae_cle.train_joint_anomalydae_cle(
            data=self.data,
            epochs=epochs,
            ae_hidden=ae_hidden,
            ae_dropout=ae_dropout,
            cle_hidden=[256, 512, 256],
            batch_size=batch_size,
            device=self.device,
            normalize_loss=False,
            normalize_method='exponential_moving_average',
            lamda1=0.0, lamda2=0.0,
            normalize_scores=normalize_scores,
            score_norm_method=score_norm_method,
            joint_training=False,
            dataset_name=self.dataset_name,
            use_embedding_transform=True
        )
        return ae_metrics[self.metric]

    def _train_ae_only_anomalydae(self, ae_hidden, ae_dropout, epochs, batch_size):
        """Run AnomalyDAE AE-only training for baseline AUC"""
        import ae_cle
        ae_model, ae_metrics = ae_cle.train_joint_anomalydae_cle(
            data=self.data,
            epochs=epochs,
            ae_hidden=ae_hidden,
            ae_dropout=ae_dropout,
            cle_hidden=[256, 512, 256],
            batch_size=batch_size,
            device=self.device,
            normalize_loss=False,
            normalize_method='exponential_moving_average',
            lamda1=0.0, lamda2=0.0,
            normalize_scores=True,
            score_norm_method='min_max',
            joint_training=False,
            dataset_name=self.dataset_name,
            use_embedding_transform=True
        )
        return ae_model, ae_metrics[self.metric]

    def _train_joint_with_params_anomalydae(self, ae_hidden, ae_dropout, cle_hidden, cle_lr, cle_weight_decay, cle_T,
                                             epochs, batch_size, lamda1, lamda2, normalize_loss, normalize_method,
                                             normalize_scores, score_norm_method):
        """Run AnomalyDAE+CLE joint training with specified parameters"""
        import ae_cle
        ae_model, cle_model, combined_metrics = ae_cle.train_joint_anomalydae_cle(
            data=self.data,
            epochs=epochs,
            ae_hidden=ae_hidden,
            ae_dropout=ae_dropout,
            cle_hidden=cle_hidden,
            batch_size=batch_size,
            device=self.device,
            normalize_loss=normalize_loss,
            normalize_method=normalize_method,
            lamda1=lamda1,
            lamda2=lamda2,
            normalize_scores=normalize_scores,
            score_norm_method=score_norm_method,
            joint_training=True,
            dataset_name=self.dataset_name,
            use_embedding_transform=True
        )
        return combined_metrics[self.metric]

    def _run_auxiliary_node_anomalydae(self, n_models, n_aux_nodes, feature_method, edge_method,
                                        n_connections, k_std, ae_hidden, ae_dropout, cle_hidden, epochs,
                                        lamda1, lamda2, normalize_loss, normalize_method,
                                        normalize_scores, score_norm_method, use_embedding_transform=True,
                                        agg_method='mean', lr_ae=5e-3):
        """Run AnomalyDAE auxiliary node ensemble with given hyperparameters"""
        import ae_cle
        ensemble_auc, scores, model_aucs, _ =ae_cle.train_auxiliary_node_ensemble_anomalydae(
            data=self.data,
            n_models=n_models,
            n_aux_nodes=n_aux_nodes,
            feature_method=feature_method,
            edge_method=edge_method,
            n_connections=n_connections,
            k_std=k_std,
            epochs=epochs,
            ae_hidden=ae_hidden,
            ae_dropout=ae_dropout,
            cle_hidden=cle_hidden,
            device=self.device,
            base_seed=self.seed,
            lamda1=lamda1,
            lamda2=lamda2,
            normalize_loss=normalize_loss,
            normalize_method=normalize_method,
            normalize_scores=normalize_scores,
            score_norm_method=score_norm_method,
            use_embedding_transform=use_embedding_transform,
            joint_training=True,
            verbose=False,
            agg_method=agg_method,
            lr_ae=lr_ae,
            parallel=False
        )
        return ensemble_auc, scores, model_aucs

    # ---- GUIDE-specific helper methods ----

    def _run_ae_only_guide(self, ae_hidden, guide_dropout, guide_alpha, epochs, batch_size, normalize_scores, score_norm_method):
        """Run GUIDE AE-only training with given hyperparameters"""
        import ae_cle
        _, ae_metrics = ae_cle.train_joint_guide_cle(
            data=self.data,
            epochs=epochs,
            guide_hidden_a=ae_hidden,
            guide_hidden_s=4,
            guide_num_layers=4,
            guide_dropout=guide_dropout,
            guide_alpha=guide_alpha,
            cle_hidden=[256, 512, 256],
            batch_size=batch_size,
            device=self.device,
            normalize_loss=False,
            normalize_method='exponential_moving_average',
            lamda1=0.0, lamda2=0.0,
            normalize_scores=normalize_scores,
            score_norm_method=score_norm_method,
            joint_training=False,
            dataset_name=self.dataset_name,
            use_embedding_transform=True
        )
        return ae_metrics[self.metric]

    def _train_ae_only_guide(self, ae_hidden, guide_dropout, guide_alpha, epochs, batch_size):
        """Run GUIDE AE-only training for baseline AUC"""
        import ae_cle
        ae_model, ae_metrics = ae_cle.train_joint_guide_cle(
            data=self.data,
            epochs=epochs,
            guide_hidden_a=ae_hidden,
            guide_hidden_s=4,
            guide_num_layers=4,
            guide_dropout=guide_dropout,
            guide_alpha=guide_alpha,
            cle_hidden=[256, 512, 256],
            batch_size=batch_size,
            device=self.device,
            normalize_loss=False,
            normalize_method='exponential_moving_average',
            lamda1=0.0, lamda2=0.0,
            normalize_scores=True,
            score_norm_method='min_max',
            joint_training=False,
            dataset_name=self.dataset_name,
            use_embedding_transform=True
        )
        return ae_model, ae_metrics[self.metric]

    def _train_joint_with_params_guide(self, ae_hidden, guide_dropout, guide_alpha, cle_hidden, cle_lr, cle_weight_decay, cle_T,
                                        epochs, batch_size, lamda1, lamda2, normalize_loss, normalize_method,
                                        normalize_scores, score_norm_method):
        """Run GUIDE+CLE joint training with specified parameters"""
        import ae_cle
        ae_model, cle_model, combined_metrics = ae_cle.train_joint_guide_cle(
            data=self.data,
            epochs=epochs,
            guide_hidden_a=ae_hidden,
            guide_hidden_s=4,
            guide_num_layers=4,
            guide_dropout=guide_dropout,
            guide_alpha=guide_alpha,
            cle_hidden=cle_hidden,
            batch_size=batch_size,
            device=self.device,
            normalize_loss=normalize_loss,
            normalize_method=normalize_method,
            lamda1=lamda1,
            lamda2=lamda2,
            normalize_scores=normalize_scores,
            score_norm_method=score_norm_method,
            joint_training=True,
            dataset_name=self.dataset_name,
            use_embedding_transform=True
        )
        return combined_metrics[self.metric]

    def _run_auxiliary_node_guide(self, n_models, n_aux_nodes, feature_method, edge_method,
                                   n_connections, k_std, ae_hidden, guide_dropout, guide_alpha,
                                   cle_hidden, epochs, lamda1, lamda2, normalize_loss,
                                   normalize_method, normalize_scores, score_norm_method,
                                   use_embedding_transform=True, agg_method='mean', lr_ae=5e-3):
        """Run GUIDE auxiliary node ensemble with given hyperparameters"""
        import ae_cle
        ensemble_auc, scores, model_aucs, _ =ae_cle.train_auxiliary_node_ensemble_guide(
            data=self.data,
            n_models=n_models,
            n_aux_nodes=n_aux_nodes,
            feature_method=feature_method,
            edge_method=edge_method,
            n_connections=n_connections,
            k_std=k_std,
            epochs=epochs,
            guide_hidden_a=ae_hidden,
            guide_hidden_s=4,
            guide_num_layers=4,
            guide_dropout=guide_dropout,
            guide_alpha=guide_alpha,
            cle_hidden=cle_hidden,
            device=self.device,
            base_seed=self.seed,
            lamda1=lamda1,
            lamda2=lamda2,
            normalize_loss=normalize_loss,
            normalize_method=normalize_method,
            normalize_scores=normalize_scores,
            score_norm_method=score_norm_method,
            use_embedding_transform=use_embedding_transform,
            joint_training=True,
            verbose=False,
            agg_method=agg_method,
            lr_ae=lr_ae,
            parallel=False
        )
        return ensemble_auc, scores, model_aucs

    # ---- GAD-NR-specific helper methods ----

    def _run_ae_only_gadnr(self, ae_hidden, sample_size, encoder, epochs, batch_size, normalize_scores, score_norm_method):
        """Run GAD-NR AE-only training with given hyperparameters"""
        import ae_cle
        _, ae_metrics = ae_cle.train_joint_gadnr_cle(
            data=self.data,
            epochs=epochs,
            gadnr_hidden=ae_hidden,
            sample_size=sample_size,
            encoder=encoder,
            cle_hidden=[256, 512, 256],
            batch_size=batch_size,
            device=self.device,
            normalize_loss=False,
            normalize_method='exponential_moving_average',
            lamda1=0.0, lamda2=0.0,
            normalize_scores=normalize_scores,
            score_norm_method=score_norm_method,
            joint_training=False,
            dataset_name=self.dataset_name,
            use_embedding_transform=True
        )
        return ae_metrics[self.metric]

    def _train_ae_only_gadnr(self, ae_hidden, sample_size, encoder, epochs, batch_size):
        """Run GAD-NR AE-only training for baseline AUC"""
        import ae_cle
        ae_model, ae_metrics = ae_cle.train_joint_gadnr_cle(
            data=self.data,
            epochs=epochs,
            gadnr_hidden=ae_hidden,
            sample_size=sample_size,
            encoder=encoder,
            cle_hidden=[256, 512, 256],
            batch_size=batch_size,
            device=self.device,
            normalize_loss=False,
            normalize_method='exponential_moving_average',
            lamda1=0.0, lamda2=0.0,
            normalize_scores=True,
            score_norm_method='min_max',
            joint_training=False,
            dataset_name=self.dataset_name,
            use_embedding_transform=True
        )
        return ae_model, ae_metrics[self.metric]

    def _train_joint_with_params_gadnr(self, ae_hidden, sample_size, encoder, cle_hidden, cle_lr, cle_weight_decay, cle_T,
                                        epochs, batch_size, lamda1, lamda2, normalize_loss, normalize_method,
                                        normalize_scores, score_norm_method):
        """Run GAD-NR+CLE joint training with specified parameters"""
        import ae_cle
        ae_model, cle_model, combined_metrics = ae_cle.train_joint_gadnr_cle(
            data=self.data,
            epochs=epochs,
            gadnr_hidden=ae_hidden,
            sample_size=sample_size,
            encoder=encoder,
            cle_hidden=cle_hidden,
            batch_size=batch_size,
            device=self.device,
            normalize_loss=normalize_loss,
            normalize_method=normalize_method,
            lamda1=lamda1,
            lamda2=lamda2,
            normalize_scores=normalize_scores,
            score_norm_method=score_norm_method,
            joint_training=True,
            dataset_name=self.dataset_name,
            use_embedding_transform=True
        )
        return combined_metrics[self.metric]

    def _run_auxiliary_node_gadnr(self, n_models, n_aux_nodes, feature_method, edge_method,
                                   n_connections, k_std, ae_hidden, sample_size, encoder,
                                   cle_hidden, epochs, lamda1, lamda2, normalize_loss,
                                   normalize_method, normalize_scores, score_norm_method,
                                   use_embedding_transform=True, agg_method='mean', lr_ae=5e-3):
        """Run GAD-NR auxiliary node ensemble with given hyperparameters"""
        import ae_cle
        ensemble_auc, scores, model_aucs, _ =ae_cle.train_auxiliary_node_ensemble_gadnr(
            data=self.data,
            n_models=n_models,
            n_aux_nodes=n_aux_nodes,
            feature_method=feature_method,
            edge_method=edge_method,
            n_connections=n_connections,
            k_std=k_std,
            epochs=epochs,
            gadnr_hidden=ae_hidden,
            sample_size=sample_size,
            encoder=encoder,
            cle_hidden=cle_hidden,
            device=self.device,
            base_seed=self.seed,
            lamda1=lamda1,
            lamda2=lamda2,
            normalize_loss=normalize_loss,
            normalize_method=normalize_method,
            normalize_scores=normalize_scores,
            score_norm_method=score_norm_method,
            use_embedding_transform=use_embedding_transform,
            joint_training=True,
            verbose=False,
            agg_method=agg_method,
            lr_ae=lr_ae,
            parallel=False
        )
        return ensemble_auc, scores, model_aucs

    # ---- DONE-specific helper methods ----

    def _run_ae_only_done(self, ae_hidden, done_dropout, epochs, batch_size, normalize_scores, score_norm_method):
        """Run DONE AE-only training with given hyperparameters"""
        import ae_cle
        _, ae_metrics = ae_cle.train_joint_done_cle(
            data=self.data,
            epochs=epochs,
            done_hidden=ae_hidden,
            done_num_layers=4,
            done_dropout=done_dropout,
            cle_hidden=[256, 512, 256],
            batch_size=batch_size,
            device=self.device,
            normalize_loss=False,
            normalize_method='exponential_moving_average',
            lamda1=0.0, lamda2=0.0,
            normalize_scores=normalize_scores,
            score_norm_method=score_norm_method,
            joint_training=False,
            dataset_name=self.dataset_name,
            use_embedding_transform=True
        )
        return ae_metrics[self.metric]

    def _train_ae_only_done(self, ae_hidden, done_dropout, epochs, batch_size):
        """Run DONE AE-only training for baseline AUC"""
        import ae_cle
        ae_model, ae_metrics = ae_cle.train_joint_done_cle(
            data=self.data,
            epochs=epochs,
            done_hidden=ae_hidden,
            done_num_layers=4,
            done_dropout=done_dropout,
            cle_hidden=[256, 512, 256],
            batch_size=batch_size,
            device=self.device,
            normalize_loss=False,
            normalize_method='exponential_moving_average',
            lamda1=0.0, lamda2=0.0,
            normalize_scores=True,
            score_norm_method='min_max',
            joint_training=False,
            dataset_name=self.dataset_name,
            use_embedding_transform=True
        )
        return ae_model, ae_metrics[self.metric]

    def _train_joint_with_params_done(self, ae_hidden, done_dropout, cle_hidden, cle_lr, cle_weight_decay, cle_T,
                                       epochs, batch_size, lamda1, lamda2, normalize_loss, normalize_method,
                                       normalize_scores, score_norm_method):
        """Run DONE+CLE joint training with specified parameters"""
        import ae_cle
        ae_model, cle_model, combined_metrics = ae_cle.train_joint_done_cle(
            data=self.data,
            epochs=epochs,
            done_hidden=ae_hidden,
            done_num_layers=4,
            done_dropout=done_dropout,
            cle_hidden=cle_hidden,
            batch_size=batch_size,
            device=self.device,
            normalize_loss=normalize_loss,
            normalize_method=normalize_method,
            lamda1=lamda1,
            lamda2=lamda2,
            normalize_scores=normalize_scores,
            score_norm_method=score_norm_method,
            joint_training=True,
            dataset_name=self.dataset_name,
            use_embedding_transform=True
        )
        return combined_metrics[self.metric]

    def _run_auxiliary_node_done(self, n_models, n_aux_nodes, feature_method, edge_method,
                                  n_connections, k_std, ae_hidden, done_dropout,
                                  cle_hidden, epochs, lamda1, lamda2, normalize_loss,
                                  normalize_method, normalize_scores, score_norm_method,
                                  use_embedding_transform=True, agg_method='mean', lr_ae=5e-3):
        """Run DONE auxiliary node ensemble with given hyperparameters"""
        import ae_cle
        ensemble_auc, scores, model_aucs, _ =ae_cle.train_auxiliary_node_ensemble_done(
            data=self.data,
            n_models=n_models,
            n_aux_nodes=n_aux_nodes,
            feature_method=feature_method,
            edge_method=edge_method,
            n_connections=n_connections,
            k_std=k_std,
            epochs=epochs,
            done_hidden=ae_hidden,
            done_num_layers=4,
            done_dropout=done_dropout,
            cle_hidden=cle_hidden,
            device=self.device,
            base_seed=self.seed,
            lamda1=lamda1,
            lamda2=lamda2,
            normalize_loss=normalize_loss,
            normalize_method=normalize_method,
            normalize_scores=normalize_scores,
            score_norm_method=score_norm_method,
            use_embedding_transform=use_embedding_transform,
            joint_training=True,
            verbose=False,
            agg_method=agg_method,
            lr_ae=lr_ae,
            parallel=False
        )
        return ensemble_auc, scores, model_aucs

    def _train_ae_only(self, ae_hidden, epochs, batch_size, dropout=0.3, struct_weight=0.8):
        """
        Run AE-only training to get baseline AUC
        """
        import ae_cle

        ae_model, ae_metrics = ae_cle.train_joint_ae_cle(
            data=self.data,
            epochs=epochs,
            ae_hidden=ae_hidden,
            cle_hidden=[256, 512, 256],  # Not used in AE-only mode
            batch_size=batch_size,
            device=self.device,
            normalize_loss=False,  # Not applicable for AE-only
            normalize_method='exponential_moving_average',
            lamda1=0.0,  # Not applicable for AE-only
            lamda2=0.0,  # Not applicable for AE-only
            normalize_scores=True,
            score_norm_method='min_max',
            joint_training=False,
            dataset_name=self.dataset_name,
            use_embedding_transform=True,  # Keep transform for AE consistency
            dropout=dropout, lr_ae=5e-3, struct_weight=struct_weight
        )

        return ae_model, ae_metrics[self.metric]

    def _train_joint_with_params(self, ae_hidden, cle_hidden, cle_lr, cle_weight_decay, cle_T,
                                 epochs, batch_size, lamda1, lamda2, normalize_loss, normalize_method,
                                 normalize_scores, score_norm_method, dropout=0.3, struct_weight=0.8):
        """
        Run joint AE+CLE training with specified parameters
        """
        import ae_cle

        ae_model, cle_model, combined_metrics = ae_cle.train_joint_ae_cle(
            data=self.data,
            epochs=epochs,
            ae_hidden=ae_hidden,
            cle_hidden=cle_hidden,
            batch_size=batch_size,
            device=self.device,
            normalize_loss=normalize_loss,
            normalize_method=normalize_method,
            lamda1=lamda1,
            lamda2=lamda2,
            normalize_scores=normalize_scores,
            score_norm_method=score_norm_method,
            joint_training=True,
            dataset_name=self.dataset_name,
            use_embedding_transform=True,  # Keep transform for consistency with baseline
            dropout=dropout, lr_ae=5e-3, struct_weight=struct_weight
        )

        return combined_metrics[self.metric]

    def run_optimization(self, direction='maximize', sampler=None, pruner=None):
        """
        Run Optuna optimization study
        """
        study = optuna.create_study(
            direction=direction,
            study_name=self.study_name,
            sampler=sampler or optuna.samplers.TPESampler(seed=self.seed),
            pruner=pruner or optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )

        print(f"\n{'='*60}")
        print(f"Starting Optuna optimization for dataset: {self.dataset_name}")
        print(f"Study name: {self.study_name}")
        print(f"Number of trials: {self.n_trials}")
        print(f"Timeout: {self.timeout} seconds" if self.timeout else "No timeout")
        print(f"{'='*60}\n")

        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )

        return study

    def save_results(self, study, output_dir=None):
        """Save optimization results. output_dir defaults based on base_model and metric."""
        _metric_suffix = '_auprc' if self.metric == 'auprc' else ''
        if output_dir is None:
            if self.base_model == 'anomalydae':
                output_dir = 'results/optuna_results_anomalydae' + _metric_suffix
            elif self.base_model == 'guide':
                output_dir = 'results/optuna_results_guide' + _metric_suffix
            elif self.base_model == 'gadnr':
                output_dir = 'results/optuna_results_gadnr' + _metric_suffix
            elif self.base_model == 'done':
                output_dir = 'results/optuna_results_done' + _metric_suffix
            else:
                output_dir = 'results/optuna_results_dominant' + _metric_suffix
        """
        Save optimization results to files
        """
        os.makedirs(output_dir, exist_ok=True)

        # Save best parameters
        best_params = study.best_params
        best_value = study.best_value

        results = {
            'dataset': self.dataset_name,
            'best_value': best_value,
            'best_params': best_params,
            'n_trials': len(study.trials),
            'study_name': self.study_name
        }

        # Save as JSON with dataset name (and mode suffix to avoid overwriting)
        json_filename = f'best_params_{self.dataset_name}{self.mode_suffix}.json'
        json_path = os.path.join(output_dir, json_filename)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # ========== Save detailed trial results ==========
        detailed_results = []
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                trial_info = {
                    'trial_number': trial.number,
                    'value': trial.value,
                    'params': trial.params,
                    'state': str(trial.state)
                }
                detailed_results.append(trial_info)

        # Save detailed results
        detailed_filename = f'trial_details_{self.dataset_name}{self.mode_suffix}.json'
        detailed_path = os.path.join(output_dir, detailed_filename)
        with open(detailed_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*60}")
        print(f"Optimization Results for {self.dataset_name}")
        print(f"{'='*60}")
        print(f"Best AUC: {best_value:.6f}")
        print(f"Number of trials: {len(study.trials)}")
        print(f"Successful trials: {len(detailed_results)}")
        print(f"\nBest Parameters:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")
        print(f"\nBest parameters saved to: {json_path}")
        print(f"Trial details saved to: {detailed_path}")
        print(f"{'='*60}")

        return results


def run_tuning(dataset_name, args):
    """Run Optuna tuning for a single dataset."""
    print("\n" + "=" * 60)
    print("Loading dataset: {}".format(dataset_name))
    print("=" * 60)
    data = load_data(dataset_name)
    n_nodes, n_feat, n_anom = data.x.shape[0], data.x.shape[1], data.y.sum().item()
    print("Dataset loaded: {} nodes, {} features, {} anomalies (ratio: {:.2f}%)".format(
        n_nodes, n_feat, n_anom, 100.0 * n_anom / n_nodes))

    if args.auxiliary_node:
        print("Mode: AUXILIARY NODE ENSEMBLE TUNING")
    print("Base model: {}".format(args.base_model.upper()))

    trainer = OptunaTrainer(
        data=data,
        dataset_name=dataset_name,
        n_trials=args.n_trials,
        timeout=args.timeout,
        study_name=args.study_name,
        seed=args.seed,
        use_auxiliary_node=args.auxiliary_node,
        ae_only=args.ae_only,
        base_model=args.base_model,
        metric=args.metric,
        n_seeds_per_trial=args.n_seeds_per_trial
    )

    study = trainer.run_optimization()
    trainer.save_results(study, output_dir=args.output_dir)
    return study.best_value


def main():
    """Main function — supports single dataset or batch mode with --datasets."""
    parser = argparse.ArgumentParser(description='Optuna hyperparameter tuning for AE+CLE')
    parser.add_argument('--dataset', type=str, default=None, help='Dataset name (single)')
    parser.add_argument('--datasets', type=str, default=None,
                        help='Comma-separated dataset names for batch tuning (e.g. "weibo,disney,enron")')
    parser.add_argument('--base_model', type=str, required=True, choices=['dominant', 'anomalydae', 'guide', 'gadnr', 'done'],
                        help='Base AE model: dominant, anomalydae, guide, gadnr, or done')
    parser.add_argument('--n_trials', type=int, default=50, help='Number of Optuna trials per dataset')
    parser.add_argument('--timeout', type=int, default=None, help='Timeout in seconds per dataset')
    parser.add_argument('--study_name', type=str, default=None, help='Optuna study name')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory (auto-set based on base_model if not specified)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--ae_only', action='store_true',
                        help='Tune AE-only (without CLE) hyperparameters')
    parser.add_argument('--auxiliary_node', action='store_true',
                        help='Tune auxiliary node ensemble parameters (add synthetic anomalous nodes)')
    parser.add_argument('--metric', type=str, default='auc', choices=['auc', 'auprc'],
                        help='Metric to optimize: auc or auprc (default: auc)')
    parser.add_argument('--n_seeds_per_trial', type=int, default=1,
                        help='Number of seeds to average per trial for stability (default: 1)')

    args = parser.parse_args()

    # Set default output_dir based on base_model and metric
    _metric_suffix = '_auprc' if args.metric == 'auprc' else ''
    if args.output_dir is None:
        if args.base_model == 'anomalydae':
            args.output_dir = 'results/optuna_results_anomalydae' + _metric_suffix
        elif args.base_model == 'guide':
            args.output_dir = 'results/optuna_results_guide' + _metric_suffix
        elif args.base_model == 'gadnr':
            args.output_dir = 'results/optuna_results_gadnr' + _metric_suffix
        elif args.base_model == 'done':
            args.output_dir = 'results/optuna_results_done' + _metric_suffix
        else:
            args.output_dir = 'results/optuna_results_dominant' + _metric_suffix

    # Resolve dataset list
    if args.datasets:
        dataset_list = [d.strip() for d in args.datasets.split(',') if d.strip()]
    elif args.dataset:
        dataset_list = [args.dataset]
    else:
        parser.error("Either --dataset or --datasets must be provided.")

    results = {}
    total_start = datetime.now()
    print("\n" + "=" * 60)
    print("BATCH TUNING: {} dataset(s)".format(len(dataset_list)))
    print("Datasets: {}".format(', '.join(dataset_list)))
    print("Trials per dataset: {}".format(args.n_trials))
    print("Start time: {}".format(total_start.strftime('%Y-%m-%d %H:%M:%S')))
    print("=" * 60)

    for i, ds in enumerate(dataset_list, 1):
        print("\n" + "█" * 60)
        print("█ [{}/{}] Dataset: {}".format(i, len(dataset_list), ds))
        print("█" * 60)

        try:
            best_auc = run_tuning(ds, args)
            results[ds] = {'status': 'OK', 'best_auc': best_auc}
        except Exception as e:
            print("ERROR tuning {}: {}".format(ds, e))
            import traceback
            traceback.print_exc()
            results[ds] = {'status': 'FAILED', 'error': str(e)}

    # Final summary
    total_time = datetime.now() - total_start
    print("\n" + "=" * 60)
    print("BATCH TUNING COMPLETE")
    print("=" * 60)
    print("Total time: {}".format(total_time))
    print("Results:")
    for ds, r in results.items():
        if r['status'] == 'OK':
            print("  {}  →  Best AUC: {:.6f}".format(ds, r['best_auc']))
        else:
            print("  {}  →  FAILED: {}".format(ds, r['error']))
    print("=" * 60)



if __name__ == "__main__":
    main()
