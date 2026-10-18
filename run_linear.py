from experiment import run_one, save_table

results = []
results.append(run_one('slim', 'SLIMElastic', alpha=0.2, l1_ratio=0.02,
                       positive_only=True, hide_item=True))
save_table(results, 'linear.csv')
results.append(run_one('ease', 'EASE', reg_weight=250.0))
save_table(results, 'linear.csv')
