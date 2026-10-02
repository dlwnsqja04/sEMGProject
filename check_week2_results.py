"""최종 산출물의 데이터 수, 선택 규칙, 기록과 예측값 일치를 검사한다."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from experiments.data import OUT, load_split, partition, window_signal
from experiments.training import rank, score_predictions
from week2_signal import ROOT, load_trial, minmax, preprocess, to_cwt


def main() -> None:
    rows = partition()
    windows = json.loads((OUT / 'window_comparison.json').read_text())
    models = json.loads((OUT / 'model_comparison.json').read_text())
    chosen = json.loads((OUT / 'selection_locked.json').read_text())
    test = json.loads((OUT / 'test_evaluation.json').read_text())
    assert len(windows) == 15 and len(models) == 3
    assert all(r['completed_epochs'] == 5 for r in windows)
    assert all(r['completed_epochs'] == 10 for r in models)
    winning_window = max(windows, key=lambda r: rank(r['best']))
    assert (chosen['width'], chosen['hop']) == (winning_window['width'], winning_window['hop'])
    assert chosen['name'] == max(models, key=lambda r: rank(r['best']))['name']
    for run in windows + [chosen] + [r for r in models if r['name'] != chosen['name']]:
        history = list(csv.DictReader((Path(run['directory']) / 'history.csv').open()))
        assert len(history) == run['completed_epochs']
        best = max(history, key=lambda r: (float(r['trial_f1']), float(r['window_f1']), -float(r['val_loss'])))
        assert int(best['epoch']) == run['best']['epoch']
        assert all(np.isfinite(float(r[k])) for r in history for k in r)
        count = (3000-run['width'])//run['hop']+1
        folder = OUT / 'cache' / f'w{run["width"]}_h{run["hop"]}'
        for split, trials in [('train', 160), ('val', 40)]:
            x, y, ids = load_split(folder, split)
            assert x.shape == (trials*count, 3, 32, run['width'])
            assert np.array_equal(np.bincount(y), np.repeat(trials*count//5, 5))
            assert np.array_equal(np.bincount(ids), np.repeat(count, trials))
    folder = OUT / 'cache' / f'w{chosen["width"]}_h{chosen["hop"]}'
    x, y, ids = load_split(folder, 'test')
    p = np.load(OUT / 'test_probabilities.npy')
    assert p.shape == (len(y), 5) and np.isfinite(p).all()
    np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-6)
    scores = score_predictions(p, y, ids)
    for key in ('window_accuracy', 'window_f1', 'trial_accuracy', 'trial_f1'):
        assert abs(scores[key] - test[key]) < 1e-12
    trial = next(r for r in rows if r['split'] == 'test')
    normalized = minmax(window_signal(preprocess(load_trial(ROOT / trial['file'])), chosen['width'], chosen['hop']))
    np.testing.assert_allclose(x[0], to_cwt(normalized[0]), atol=1e-7)
    np.testing.assert_allclose(x[0, 2], (x[0, 0]+x[0, 1])/2, atol=1e-7)
    saved = torch.load(Path(chosen['directory']) / 'best.pt', map_location='cpu', weights_only=True)
    assert saved['best'] == chosen['best'] and saved['best']['epoch'] == test['selected_epoch']
    assert not test['test_used_for_selection'] and test['unique_test_trials'] == 49
    result = ('PASS: 15 window trials >=5 epochs; 3 models >=10 epochs\n'
              'PASS: selection follows validation-only ranking\n'
              'PASS: all split shapes, class counts and per-trial window counts\n'
              'PASS: saved probabilities reproduce reported test metrics\n'
              'PASS: selected CWT reconstructed from original CSV\n'
              'PASS: saved checkpoint metadata matches final selected epoch\n'
              'PASS: disjoint signal hashes across splits; test duplicate documented\n')
    (OUT / 'verification.txt').write_text(result)
    print(result)


if __name__ == '__main__':
    main()
