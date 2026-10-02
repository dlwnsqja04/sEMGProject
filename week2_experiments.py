"""15가지 윈도우 → 3개 모델 → 연장 학습 → 한 번의 최종 테스트."""
from __future__ import annotations

import csv
import hashlib
import inspect
import json

import numpy as np
import torch

from experiments.data import OUT, load_split, partition, prepare
from experiments.training import create_model, evaluate, rank, score_predictions, train
from week2_data import code_signature
from week2_signal import ROOT


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = partition()
    protocol = {'widths_ms': [100, 200, 300, 400, 600], 'hop_fractions': [.25, .5, 1.],
                'screening_model': 'resnet18', 'screening_epochs': 5,
                'models': ['resnet18', 'mobilenet_v3_small', 'densenet161'], 'model_epochs': 10,
                'final_max_epochs': 40, 'early_stopping_patience': 10,
                'selection_order': ['validation_trial_macro_f1', 'validation_window_macro_f1', 'lower_validation_loss'],
                'split_trials': {'train': 160, 'val': 40, 'test': 50},
                'outer_split_seed': 42, 'inner_split_seed': 43, 'training_seed': 42,
                'batch_size': 16, 'weights': None, 'optimizer': 'Adam', 'learning_rate': .001,
                'lr_scheduler': 'ReduceLROnPlateau(val_loss), factor=0.5, patience=3, min_lr=1e-5',
                'task': 'five known subject identities A-E, not gesture recognition',
                'scope': 'staged search, single split and seed; no claim of global optimum',
                'duplicate_note': 'E/e (38).csv and E/e (50).csv have identical bytes, both test; also report 49 unique signals',
                'torch_version': str(torch.__version__)}
    source_paths = [ROOT / name for name in ('week2_experiments.py', 'week2_signal.py',
                                            'experiments/data.py', 'experiments/training.py')]
    protocol['fingerprint'] = hashlib.sha256(
        b''.join(code_signature(p) for p in source_paths) + json.dumps(rows, sort_keys=True).encode()).hexdigest()
    protocol_path = OUT / 'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise RuntimeError('실험 코드/데이터가 변경되었습니다. 이전 결과를 보관한 뒤 새 결과 폴더를 사용하세요.')
    protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2))
    with (OUT / 'trial_split.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / 'split_code.txt').write_text(inspect.getsource(partition), encoding='utf-8')
    torch.set_num_threads(4)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Device: {device}; fixed trial split: 160 train / 40 validation / 50 test', flush=True)
    screening = []
    for width in protocol['widths_ms']:
        for fraction in protocol['hop_fractions']:
            hop = int(width * fraction)
            folder = prepare(width, hop, rows)
            screening.append(train('resnet18', width, hop, folder, 'window', 5, device))
            (OUT / 'window_comparison.json').write_text(json.dumps(screening, indent=2))
    selected_window = max(screening, key=lambda r: rank(r['best']))
    width, hop = selected_window['width'], selected_window['hop']
    print(f'SELECTED WINDOW: {width}/{hop} by validation only', flush=True)
    folder = prepare(width, hop, rows)
    models = []
    for name in protocol['models']:
        models.append(train(name, width, hop, folder, 'model', 10, device))
        (OUT / 'model_comparison.json').write_text(json.dumps(models, indent=2))
    selected_model = max(models, key=lambda r: rank(r['best']))
    final = train(selected_model['name'], width, hop, folder, 'model', 40, device, patience=10)
    (OUT / 'selection_locked.json').write_text(json.dumps(final, indent=2))
    evaluation_file = OUT / 'test_evaluation.json'
    if not evaluation_file.exists():
        prepare(width, hop, rows, splits=('test',))
        saved = torch.load(OUT / 'runs' / f'model_{final["name"]}_w{width}_h{hop}' / 'best.pt',
                           map_location='cpu', weights_only=True)
        model = create_model(final['name']).to(device)
        model.load_state_dict(saved['model'])
        test = load_split(folder, 'test')
        metrics, probabilities = evaluate(model, test, device)
        metrics['test_loss'] = metrics.pop('val_loss')
        metrics.update({'test_trials': 50, 'test_windows': len(test[1]), 'selected_epoch': final['best']['epoch'],
                        'model': final['name'], 'width': width, 'hop': hop,
                        'test_used_for_selection': False})
        test_rows = [r for r in rows if r['split'] == 'test']
        seen, unique_ids = set(), []
        for i, row in enumerate(test_rows):
            if row['sha256'] not in seen:
                unique_ids.append(i)
                seen.add(row['sha256'])
        mask = np.isin(test[2], unique_ids)
        metrics['unique_signal_evaluation'] = score_predictions(probabilities[mask], test[1][mask], test[2][mask])
        metrics['unique_test_trials'] = len(unique_ids)
        np.save(OUT / 'test_probabilities.npy', probabilities)
        evaluation_file.write_text(json.dumps(metrics, indent=2))
    print('COMPLETED. Final held-out test:', evaluation_file.read_text(), flush=True)


if __name__ == '__main__':
    main()
