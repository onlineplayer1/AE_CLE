"""
GAD-CLE: Unified Training Interface

Usage:
    python main.py --model dominant --dataset disney
    python main.py --model anomaly_dae --dataset enron --use_best_params
"""

import argparse
import torch
import random
import numpy as np
import time
from datetime import datetime, timedelta

from train_dominant import train_dominant_cle
from train_anomaly_dae import train_anomaly_dae_cle
from train_done import train_done_cle
from train_guide import train_guide_cle
from train_gadnr import train_gadnr_cle

from core.cle_utils import (
    convert_cle_hidden_params,
    format_time_precise,
    format_timedelta_precise,
    load_best_params
)

from pygod.utils import load_data


TRAIN_FUNCTIONS = {
    'dominant': train_dominant_cle,
    'anomaly_dae': train_anomaly_dae_cle,
    'done': train_done_cle,
    'guide': train_guide_cle,
    'gadnr': train_gadnr_cle,
}


MODEL_DEFAULTS = {
    'dominant': {
        'ae_hidden': 64,
        'ae_dropout': 0.3,
        'alpha': 0.8
    },
    'anomaly_dae': {
        'ae_hidden': 64,
        'ae_dropout': 0.3,
        'alpha': 0.8
    },
    'done': {
        'done_hidden': 64,
        'done_num_layers': 4,
        'done_dropout': 0.0
    },
    'guide': {
        'guide_hidden_a': 64,
        'guide_hidden_s': 4,
        'guide_num_layers': 4,
        'guide_dropout': 0.0,
        'guide_alpha': 0.5
    },
    'gadnr': {
        'gadnr_hidden': 64,
        'sample_size': 10,
        'encoder': 'GCN'
    }
}


PARAM_DIR_MAP = {
    'dominant': 'params/dominant',
    'anomaly_dae': 'params/anomaly_dae',
    'done': 'params/done',
    'guide': 'params/guide',
    'gadnr': 'params/gadnr'
}


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def run_single_experiment(model_name, data, dataset_name, params, seed):
    """Run a single experiment."""
    set_seed(seed)
    train_fn = TRAIN_FUNCTIONS[model_name]
    _, _, final_auc = train_fn(
        data=data,
        dataset_name=dataset_name,
        **params
    )
    return final_auc


def run_experiment(model_name, dataset, params=None, n_runs=10, seed=42,
                 use_best_params=False, params_dir=None):
    """Run full experiment."""
    print(f"\n{'='*60}")
    print(f"Model: {model_name.upper()} + CLE")
    print(f"Dataset: {dataset}")
    print(f"{'='*60}")
    print(f"Loading dataset: {dataset}")
    data = load_data(dataset)
    
    default_params = {
        'epochs': 100,
        'cle_hidden': [256, 512, 256],
        'batch_size': 64,
        'normalize_loss': True,
        'normalize_method': 'exponential_moving_average',
        'lamda1': 0.5,
        'lamda2': 0.5,
        'normalize_scores': True,
        'score_norm_method': 'min_max',
        'joint_training': True
    }
    default_params.update(MODEL_DEFAULTS[model_name])
    
    if use_best_params and params_dir:
        best_params = load_best_params(dataset, params_dir)
        if best_params:
            best_params = convert_cle_hidden_params(best_params)
            default_params.update(best_params)
            print("Using optimized parameters from Optuna tuning.")
    
    if params:
        default_params.update(params)
    
    print("\nParameters:")
    for key, value in default_params.items():
        print(f"  {key}: {value}")
    
    print(f"\nRunning {n_runs} time(s)...")
    
    auc_scores = []
    run_times = []
    experiment_start_time = time.time()
    
    for run in range(n_runs):
        run_start_time = time.time()
        print(f"\n{'='*60}")
        print(f"Run {run + 1}/{n_runs}")
        print(f"{'='*60}")
        
        current_seed = seed + run
        print(f"Random seed: {current_seed}")
        
        try:
            final_auc = run_single_experiment(
                model_name=model_name,
                data=data,
                dataset_name=dataset,
                params=default_params,
                seed=current_seed
            )
        except Exception as e:
            print(f"Error during training: {e}")
            continue
        
        run_time = time.time() - run_start_time
        run_times.append(run_time)
        auc_scores.append(final_auc)
        
        print(f"Run {run + 1} AUC: {final_auc:.6f} (Time: {format_timedelta_precise(timedelta(seconds=run_time))})")
    
    if not auc_scores:
        print("No successful runs!")
        return None, None
    
    auc_scores = np.array(auc_scores)
    sorted_scores = np.sort(auc_scores)
    
    if len(sorted_scores) > 2:
        filtered_scores = sorted_scores[2:]
    else:
        filtered_scores = sorted_scores
    
    mean_auc = np.mean(filtered_scores)
    std_auc = np.std(filtered_scores)
    
    filtered_run_times = run_times[1:] if len(run_times) > 1 else run_times
    avg_run_time = np.mean(filtered_run_times)
    
    total_experiment_time = time.time() - experiment_start_time
    
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Dataset: {dataset}")
    print(f"Model: {model_name.upper()} + CLE")
    print(f"Number of runs: {n_runs}")
    print(f"Individual AUC scores: {[f'{score:.6f}' for score in auc_scores]}")
    if len(run_times) > 1:
        print(f"Individual run times: {[format_timedelta_precise(timedelta(seconds=t)) for t in run_times]}")
    print(f"Mean AUC: {mean_auc:.6f}")
    print(f"Std AUC:  {std_auc:.6f}")
    print(f"95% CI:   [{mean_auc - 1.96*std_auc:.6f}, {mean_auc + 1.96*std_auc:.6f}]")
    print(f"Average run time: {format_timedelta_precise(timedelta(seconds=avg_run_time))}")
    print(f"Total experiment time: {format_timedelta_precise(timedelta(seconds=total_experiment_time))}")
    print(f"{'='*60}")
    
    return mean_auc, std_auc


def main():
    parser = argparse.ArgumentParser(
        description='GAD-CLE: Graph Anomaly Detection with Curriculum Learning Enhancement',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train DOMINANT+CLE on Disney dataset
  python main.py --model dominant --dataset disney

  # Train AnomalyDAE+CLE with optimized parameters
  python main.py --model anomaly_dae --dataset enron --use_best_params

  # Train with custom epochs
  python main.py --model done --dataset reddit --epochs 200

  # Train GUIDE+CLE with custom hidden sizes
  python main.py --model guide --dataset blogback --cle_hidden 512,1024,512
        """
    )
    
    parser.add_argument('--model', type=str, default='dominant',
                       choices=['dominant', 'anomaly_dae', 'done', 'guide', 'gadnr'],
                       help='Model to use (default: dominant)')
    
    parser.add_argument('--dataset', type=str, default='disney',
                       help='Dataset name (default: disney)')
    
    parser.add_argument('--params_dir', type=str, default=None,
                       help='Directory containing best parameters')
    
    parser.add_argument('--use_best_params', action='store_true',
                       help='Use best parameters from Optuna tuning')
    
    parser.add_argument('--n_runs', type=int, default=10,
                       help='Number of runs for averaging results (default: 10)')
    
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    
    parser.add_argument('--ae_hidden', type=int, default=None,
                       help='AE/GNN hidden size')
    
    parser.add_argument('--cle_hidden', type=str, default=None,
                       help='CLE hidden sizes (comma-separated, e.g., 256,512,256)')
    
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size')
    
    parser.add_argument('--lamda1', type=float, default=None,
                       help='Training loss weight for CLE')
    
    parser.add_argument('--lamda2', type=float, default=None,
                       help='Evaluation score weight for CLE')
    
    parser.add_argument('--normalize_method', type=str, default=None,
                       choices=['exponential_moving_average', 'running_average', 'min_max', 'z_score'],
                       help='Loss normalization method')
    
    parser.add_argument('--ae_only', action='store_true',
                       help='Train AE model only (without CLE)')
    
    parser.add_argument('--use_embedding_transform', action='store_true', default=True,
                       help='Use embedding transform (center+normalize+Procrustes+sign_fix)')
    
    args = parser.parse_args()
    
    params = {}
    if args.epochs is not None:
        params['epochs'] = args.epochs
    if args.ae_hidden is not None:
        if args.model == 'done':
            params['done_hidden'] = args.ae_hidden
        elif args.model == 'guide':
            params['guide_hidden_a'] = args.ae_hidden
        elif args.model == 'gadnr':
            params['gadnr_hidden'] = args.ae_hidden
        else:
            params['ae_hidden'] = args.ae_hidden
    if args.cle_hidden is not None:
        params['cle_hidden'] = [int(x) for x in args.cle_hidden.split(',')]
    if args.batch_size is not None:
        params['batch_size'] = args.batch_size
    if args.lamda1 is not None:
        params['lamda1'] = args.lamda1
    if args.lamda2 is not None:
        params['lamda2'] = args.lamda2
    if args.normalize_method is not None:
        params['normalize_method'] = args.normalize_method
    if args.ae_only:
        params['joint_training'] = False
    
    if args.params_dir is None:
        args.params_dir = PARAM_DIR_MAP[args.model]
    
    run_experiment(
        model_name=args.model,
        dataset=args.dataset,
        params=params,
        n_runs=args.n_runs,
        seed=args.seed,
        use_best_params=args.use_best_params,
        params_dir=args.params_dir
    )


if __name__ == "__main__":
    main()
