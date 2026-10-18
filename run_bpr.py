from experiment import run_one, save_table

settings = [(size, 0.1) for size in [32, 64, 128, 256]]
settings += [(64, 0.01), (64, 0.5)]
results = []
for size, regularization in settings:
    result = run_one(f'bpr_d{size}_r{regularization}', 'BPR',
                     embedding_size=size, reg_lambda=regularization)
    results.append(result)
    save_table(results, 'bpr.csv')
