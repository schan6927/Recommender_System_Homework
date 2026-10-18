from pathlib import Path

OUTPUT = Path('results')
OUTPUT.mkdir(exist_ok=True)
KS = [1, 5, 10, 100, 500]
COMMON = {
    'seed': 2026, 'reproducibility': True, 'use_gpu': False,
    'worker': 0, 'epochs': 300, 'stopping_step': 10, 'eval_step': 1,
    'train_batch_size': 2048, 'eval_batch_size': 4096,
    'learner': 'adam', 'learning_rate': 0.001, 'weight_decay': 0.0,
    'show_progress': False, 'save_dataset': False, 'save_dataloaders': False,
    'load_col': {'inter': ['user_id', 'item_id', 'rating', 'timestamp']},
    'eval_args': {'split': {'RS': [0.8, 0.1, 0.1]},
                  'group_by': 'user', 'order': 'RO', 'mode': 'full'},
    'metrics': ['NDCG', 'Recall', 'Precision'], 'topk': [10, 20],
    'valid_metric': 'NDCG@10', 'metric_decimal_place': 6,
    'train_neg_sample_args': {'distribution': 'uniform', 'sample_num': 1,
                              'alpha': 1.0, 'dynamic': False, 'candidate_num': 0},
}
