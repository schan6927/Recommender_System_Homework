import hashlib
import json
import time
import importlib.metadata as metadata
import numpy as np
import pandas as pd
import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import get_model, get_trainer, init_seed
from settings import COMMON, OUTPUT
from bpr_l2 import BPRL2

torch.set_num_threads(4)

def run_one(run_id, model_name, rerun=False, **parameters):
    result_path = OUTPUT / (run_id + '.json')
    signature = {'model': model_name, 'parameters': parameters, 'common': COMMON}
    if result_path.exists() and not rerun:
        previous = json.loads(result_path.read_text())
        if previous['signature'] != signature:
            raise ValueError('설정이 바뀌었습니다. OUTPUT을 새 폴더로 지정하세요.')
        print('완료 결과 재사용:', run_id, flush=True)
        return previous

    config = Config(model=model_name, dataset='ml-100k', config_dict={
        **COMMON, **parameters, 'checkpoint_dir': str(OUTPUT / 'checkpoints' / run_id)})
    init_seed(COMMON['seed'], True)
    dataset = create_dataset(config)
    train, valid, test = data_preparation(config, dataset)
    hashes, counts = [], []
    for loader in (train, valid, test):
        data = loader.dataset.inter_feat
        pairs = np.column_stack([data[config['USER_ID_FIELD']].cpu().numpy(),
                                 data[config['ITEM_ID_FIELD']].cpu().numpy()])
        hashes.append(hashlib.sha256(pairs.astype('<i8').tobytes()).hexdigest())
        counts.append(len(pairs))
    split_path = OUTPUT / 'split.json'
    if split_path.exists() and json.loads(split_path.read_text())['hashes'] != hashes:
        raise ValueError('데이터 분할이 다릅니다. 실험을 중단합니다.')
    split_path.write_text(json.dumps({'hashes': hashes, 'counts': counts}, indent=2))

    init_seed(COMMON['seed'], True)
    cls = BPRL2 if model_name == 'BPR' else get_model(model_name)
    start = time.perf_counter()
    model = cls(config, train.dataset).to(config['device'])
    if model_name == 'ItemKNN':
        assert model.method == parameters['knn_method'], 'KNN 방식이 반영되지 않았습니다.'
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    _, validation = trainer.fit(train, valid, saved=True, show_progress=False)
    test_scores = trainer.evaluate(test, load_best_model=True, show_progress=False)
    row = {'id': run_id, 'model': model_name, **parameters,
           'seconds': time.perf_counter() - start,
           **{'valid_' + k: float(v) for k, v in validation.items()},
           **{'test_' + k: float(v) for k, v in test_scores.items()}}
    result = {'signature': signature, 'row': row, 'split_hashes': hashes}
    temporary = result_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2))
    temporary.replace(result_path)
    (OUTPUT / (run_id + '_config.txt')).write_text(str(config))
    print(run_id, test_scores, flush=True)
    return result

def save_table(results, name):
    table = pd.DataFrame([result['row'] for result in results])
    table.to_csv(OUTPUT / name, index=False)
    print(table.to_string(index=False), flush=True)
    versions = {p: metadata.version(p) for p in ['recbole','torch','numpy','ray','scipy']}
    (OUTPUT / 'versions.json').write_text(json.dumps(versions, indent=2))
