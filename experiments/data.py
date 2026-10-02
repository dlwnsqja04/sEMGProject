"""시행을 먼저 분리하고 설정별 CWT 배열을 디스크에 보관한다."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from week2_signal import ROOT, load_trial, minmax, preprocess, to_cwt

OUT = ROOT / 'results/week2_experiments'


def partition() -> list[dict]:
    with (ROOT / 'results/week2/trial_split.csv').open(encoding='utf-8-sig') as stream:
        original = list(csv.DictReader(stream))
    pool = [r for r in original if r['split'] == 'train']
    train, val = train_test_split(pool, test_size=.2, random_state=43,
                                 stratify=[r['label'] for r in pool])
    groups = [('train', train), ('val', val), ('test', [r for r in original if r['split'] == 'test'])]
    rows = []
    for split, group in groups:
        for r in group:
            digest = hashlib.sha256((ROOT / r['file']).read_bytes()).hexdigest()
            if digest != r['sha256']:
                raise ValueError(f"원본 파일 변경: {r['file']}")
            rows.append({'split': split, 'subject': r['subject'], 'label': int(r['label']),
                         'file': r['file'], 'sha256': digest})
    hashes = {split: {r['sha256'] for r in rows if r['split'] == split}
              for split in ('train', 'val', 'test')}
    if hashes['train'] & hashes['val'] or hashes['train'] & hashes['test'] or hashes['val'] & hashes['test']:
        raise ValueError('서로 다른 분할에 같은 신호가 있습니다.')
    return rows


def window_signal(x: np.ndarray, width: int, hop: int) -> np.ndarray:
    if not 32 <= width <= len(x) or not 1 <= hop <= width:
        raise ValueError(f'잘못된 윈도우: width={width}, hop={hop}')
    return np.stack([x[start:start + width] for start in range(0, len(x) - width + 1, hop)])


def prepare(width: int, hop: int, rows: list[dict], splits: tuple = ('train', 'val')) -> Path:
    folder = OUT / 'cache' / f'w{width}_h{hop}'
    folder.mkdir(parents=True, exist_ok=True)
    count = (3000 - width) // hop + 1
    for split in splits:
        marker = folder / f'{split}.done'
        if marker.exists():
            continue
        trials = [r for r in rows if r['split'] == split]
        shape = (len(trials) * count, 3, 32, width)
        x = np.lib.format.open_memmap(folder / f'X_{split}.npy', mode='w+', dtype='float32', shape=shape)
        labels, ids = [], []
        for index, row in enumerate(trials):
            windows = minmax(window_signal(preprocess(load_trial(ROOT / row['file'])), width, hop))
            for j, window in enumerate(windows):
                x[index * count + j] = to_cwt(window)
            labels.extend([row['label']] * count)
            ids.extend([index] * count)
        x.flush()
        if not np.isfinite(x).all():
            raise FloatingPointError('CWT 변환에 비정상 값이 있습니다.')
        np.save(folder / f'y_{split}.npy', np.array(labels, dtype=np.int64))
        np.save(folder / f'ids_{split}.npy', np.array(ids, dtype=np.int64))
        marker.write_text(json.dumps({'shape': shape, 'trials': len(trials)}))
        print(f'Prepared {width}/{hop} {split}: {shape}', flush=True)
    return folder


def load_split(folder: Path, split: str) -> tuple:
    return tuple(np.load(folder / f'{prefix}_{split}.npy', mmap_mode='c', allow_pickle=False)
                 for prefix in ('X', 'y', 'ids'))
