from experiment import run_one, save_table
from settings import KS

results = []
for method in ['item', 'user']:
    for k in KS:
        result = run_one(f'{method}_k{k}', 'ItemKNN', k=k, knn_method=method)
        results.append(result)
        save_table(results, 'knn.csv')
