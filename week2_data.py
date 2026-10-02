"""시행 단위 분할, 중간 배열과 학습용 CWT 데이터셋 저장."""
from __future__ import annotations

import ast
import csv
import hashlib
import inspect
import json
from collections import Counter
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap
from sklearn.model_selection import train_test_split

from week2_signal import OUT, ROOT, SUBJECTS, load_trial, make_windows, minmax, preprocess, to_cwt


def code_signature(path: Path) -> bytes:
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                node.body.pop(0)
    return ast.dump(tree, include_attributes=False).encode('utf-8')


def split_trials(files: list[Path]) -> tuple[list[Path], list[Path]]:
    """반드시 시행 파일부터 나눈다. 윈도우 생성은 이 함수 호출 뒤에만 한다."""
    labels = [path.parent.name for path in files]
    train_files, test_files = train_test_split(
        files, test_size=0.2, stratify=labels, random_state=42,
    )
    if set(train_files) & set(test_files):
        message = '학습과 테스트에 같은 시행 파일이 있습니다.'
        raise RuntimeError(message)
    return train_files, test_files


def build_split(files: list[Path], split: str) -> None:
    """시행 순서를 보존하며 각 단계의 결과를 디스크에 바로 저장한다."""
    directory = OUT / 'dataset'
    directory.mkdir(parents=True, exist_ok=True)
    trial_count = len(files)
    filtered = open_memmap(directory / f'{split}_filtered.npy', mode='w+', dtype='float64', shape=(trial_count, 3000, 2))
    windows = open_memmap(directory / f'{split}_windows.npy', mode='w+', dtype='float64', shape=(trial_count, 19, 300, 2))
    normalized = open_memmap(directory / f'{split}_normalized.npy', mode='w+', dtype='float64', shape=windows.shape)
    features = open_memmap(directory / f'X_{split}.npy', mode='w+', dtype='float32', shape=(trial_count * 19, 3, 32, 300))
    targets = np.empty(trial_count * 19, dtype=np.int64)
    trial_ids: list[str] = []
    for i, path in enumerate(files):
        filtered[i] = preprocess(load_trial(path))
        windows[i] = make_windows(filtered[i])
        normalized[i] = minmax(windows[i])
        for j, window in enumerate(normalized[i]):
            features[i * 19 + j] = to_cwt(window)
        targets[i * 19:(i + 1) * 19] = SUBJECTS.index(path.parent.name)
        trial_ids.extend([str(path.relative_to(ROOT))] * 19)
        if (i + 1) % 25 == 0 or i + 1 == trial_count:
            print(f'CWT {split}: {i + 1}/{trial_count} trials', flush=True)
    for array in (filtered, windows, normalized, features):
        array.flush()
    np.save(directory / f'y_{split}.npy', targets, allow_pickle=False)
    np.save(directory / f'trial_ids_{split}.npy', np.array(trial_ids), allow_pickle=False)


def prepare_dataset() -> str:
    """원본·전처리 코드가 같은 경우에만 완성된 데이터셋을 재사용한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted((ROOT / 'data').rglob('*.csv'))
    counts = Counter(p.parent.name for p in files)
    if counts != Counter({subject: 50 for subject in SUBJECTS}):
        raise ValueError(f'피험자별 50개, 전체 250개 파일을 예상했습니다: {counts}')
    hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(hashes[path].encode())
    digest.update(code_signature(ROOT / 'week2_signal.py'))
    digest.update(code_signature(Path(__file__)))
    fingerprint = digest.hexdigest()
    train_files, test_files = split_trials(files)
    duplicate_hashes = {hashes[p] for p in train_files} & {hashes[p] for p in test_files}
    if duplicate_hashes:
        message = '서로 다른 파일명으로 동일한 신호가 학습과 테스트에 들어 있습니다.'
        raise RuntimeError(message)
    with (OUT / 'trial_split.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.writer(stream)
        writer.writerow(['split', 'subject', 'label', 'file', 'sha256', 'windows'])
        for split, group in [('train', train_files), ('test', test_files)]:
            for path in group:
                writer.writerow([split, path.parent.name, SUBJECTS.index(path.parent.name), str(path.relative_to(ROOT)), hashes[path], 19])
    with (OUT / 'dataset_summary.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.writer(stream)
        writer.writerow(['subject', 'train_trials', 'test_trials', 'train_windows', 'test_windows'])
        for subject in SUBJECTS:
            tr = sum(p.parent.name == subject for p in train_files)
            te = sum(p.parent.name == subject for p in test_files)
            writer.writerow([subject, tr, te, tr * 19, te * 19])
        writer.writerow(['TOTAL', len(train_files), len(test_files), len(train_files) * 19, len(test_files) * 19])
    (OUT / 'split_code.txt').write_text(inspect.getsource(split_trials) + '\n# 분할 후 집합별로 처리\ntrain_files, test_files = split_trials(files)\nbuild_split(train_files, "train")\nbuild_split(test_files, "test")\n', encoding='utf-8')
    for split, group in [('train', train_files), ('test', test_files)]:
        marker = OUT / f'{split}_complete.txt'
        required = [OUT / 'dataset' / f'{stem}_{split}.npy' for stem in ['X', 'y', 'trial_ids']]
        required += [OUT / 'dataset' / f'{split}_{stage}.npy' for stage in ['filtered', 'windows', 'normalized']]
        if marker.exists() and marker.read_text() == fingerprint and all(p.exists() for p in required):
            print(f'{split}: saved dataset reused', flush=True)
        else:
            build_split(group, split)
            marker.write_text(fingerprint)
    config = {'fingerprint': fingerprint, 'fs': 1000, 'notch_hz': 60, 'notch_q': 30,
              'bandpass_hz': [20, 499], 'bandpass_order': 4, 'window': 300, 'hop': 150,
              'normalization_axes': [1, 2], 'epsilon': 1e-8, 'wavelet': 'morl', 'scales': [1, 32],
              'split_seed': 42, 'train_trials': 200, 'test_trials': 50,
              'X_train_shape': [3800, 3, 32, 300], 'X_test_shape': [950, 3, 32, 300],
              'shared_trials': 0, 'shared_signal_hashes': 0, 'input_already_filtered': True}
    (OUT / 'preparation.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    print('Split checked: 200 train / 50 test trials; 3800 / 950 windows; shared trials = 0', flush=True)
    return fingerprint


if __name__ == '__main__':
    prepare_dataset()
