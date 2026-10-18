# Recommender Systems HW1 (Fall 2026)

Experiments on MovieLens-100K (`ml-100k`) with RecBole 1.2.1.

## Models
- **KNN**: ItemKNN and UserKNN (`knn_method=user`), k ∈ {1, 5, 10, 100, 500}
- **BPR**: embedding_size ∈ {32, 64, 128, 256} with reg_lambda = 0.1; reg_lambda ∈ {0.01, 0.1, 0.5} with embedding_size = 64.
  RecBole's BPR does not use `reg_lambda`, so `bpr_l2.py` adds an explicit L2 term (λ/2)(‖p_u‖² + ‖q_i‖² + ‖q_j‖²) averaged over the batch.
- **SLIM / EASE**: SLIMElastic (alpha=0.2, l1_ratio=0.02), EASE (reg_weight=250)

## Common setting (`settings.py`)
All ratings treated as implicit feedback; per-user random 80/10/10 split (seed 2026); full-sort evaluation;
metrics NDCG / Recall / Precision @10, @20; model selection by validation NDCG@10; CPU.

## Files
- `RS_HW1_Experiments.ipynb` — Colab notebook that runs everything
- `settings.py`, `bpr_l2.py`, `experiment.py` — shared config, BPR with L2, train/eval helper
- `run_knn.py`, `run_bpr.py`, `run_linear.py` — experiment scripts
- `results/` — `knn.csv`, `bpr.csv`, `linear.csv`, `comparison.csv`, plots (`knn.png`, `bpr.png`), per-run JSON and configs

## Run
Open the notebook in Colab and run all cells in order, or with Python 3.11 and the packages listed in the notebook:
```
python run_knn.py
python run_bpr.py
python run_linear.py
```
