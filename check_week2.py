"""실행: .venv/bin/python check_week2.py — 신호 처리의 핵심 동작 확인."""
from __future__ import annotations

import numpy as np
import week2_signal as signal
from week2_data import split_trials


def check_filter() -> None:
    """10/60 Hz는 줄이고 100 Hz는 보존하는지 독립 합성 신호로 확인한다."""
    time = np.arange(5000) / 1000
    waves = np.sin(2 * np.pi * time[:, None] * np.array([10, 60, 100]))
    filtered = signal.preprocess(waves)
    ratios = np.std(filtered[1000:-1000], axis=0) / np.std(waves[1000:-1000], axis=0)
    assert ratios[0] < 0.02, ratios
    assert ratios[1] < 0.02, ratios
    assert ratios[2] > 0.95, ratios


def check_windows() -> None:
    """윈도우 수, 처음과 마지막 위치, 겹치는 부분을 원래 배열과 비교한다."""
    original = np.arange(6000, dtype=np.float64).reshape(3000, 2)
    windows = signal.make_windows(original)
    assert windows.shape == (19, 300, 2)
    np.testing.assert_array_equal(windows[0], original[:300])
    np.testing.assert_array_equal(windows[-1], original[2700:3000])
    np.testing.assert_array_equal(windows[0, 150:], windows[1, :150])


def check_minmax() -> None:
    """각 윈도우의 두 채널을 함께 정규화하고 상수 입력도 안전하게 처리한다."""
    source = np.array([[[0., 2.], [4., 6.]], [[10., 10.], [10., 10.]]])
    result = signal.minmax(source)
    np.testing.assert_allclose(result[0], [[0, 1 / 3], [2 / 3, 1]], atol=1e-8)
    np.testing.assert_array_equal(result[1], np.zeros((2, 2)))
    assert np.isfinite(result).all()


def check_cwt() -> None:
    """CWT의 세 번째 맵이 두 원래 채널 맵의 평균인지 확인한다."""
    original = np.zeros((300, 2))
    original[150, 0], original[200, 1] = 1., 0.5
    result = signal.to_cwt(original)
    assert result.shape == (3, 32, 300)
    assert result.dtype == np.float32
    assert np.isfinite(result).all() and (result >= 0).all()
    np.testing.assert_allclose(result[2], (result[0] + result[1]) / 2, atol=1e-7)


def check_split() -> None:
    """원본 시행 단위의 재현 가능한 8:2 분할과 클래스 분포를 확인한다."""
    files = sorted((signal.ROOT / 'data').rglob('*.csv'))
    train, test = split_trials(files)
    assert len(train) == 200 and len(test) == 50
    assert set(train).isdisjoint(test)
    assert set(train) | set(test) == set(files)
    assert (train, test) == split_trials(files)
    for subject in signal.SUBJECTS:
        assert sum(path.parent.name == subject for path in train) == 40
        assert sum(path.parent.name == subject for path in test) == 10


def check_saved_dataset() -> None:
    """저장된 학습 입력과 라벨을 시행 목록 및 실제 재계산 결과와 비교한다."""
    identities = []
    for split, count in [('train', 3800), ('test', 950)]:
        directory = signal.OUT / 'dataset'
        x = np.load(directory / f'X_{split}.npy', mmap_mode='r', allow_pickle=False)
        y = np.load(directory / f'y_{split}.npy', allow_pickle=False)
        ids = np.load(directory / f'trial_ids_{split}.npy', allow_pickle=False)
        assert x.shape == (count, 3, 32, 300) and y.shape == (count,)
        assert len(ids) == count and len(set(ids)) == count // 19
        assert np.isfinite(x).all()
        assert np.array_equal(np.bincount(y), np.full(5, count // 5))
        for start in range(0, count, 19):
            assert np.all(ids[start:start + 19] == ids[start])
            label = signal.SUBJECTS.index((signal.ROOT / ids[start]).parent.name)
            assert np.all(y[start:start + 19] == label)
        for index in [0, count - 1]:
            raw = signal.load_trial(signal.ROOT / ids[index])
            expected = signal.to_cwt(signal.minmax(signal.make_windows(signal.preprocess(raw)))[index % 19])
            np.testing.assert_allclose(x[index], expected, rtol=1e-6, atol=1e-7)
        normalized = np.load(directory / f'{split}_normalized.npy', mmap_mode='r', allow_pickle=False)
        assert normalized.min() == 0 and normalized.max() <= 1
        identities.append(set(ids))
    assert identities[0].isdisjoint(identities[1])


if __name__ == '__main__':
    checks = [check_filter, check_windows, check_minmax, check_cwt, check_split]
    if (signal.OUT / 'test_complete.txt').exists():
        checks.append(check_saved_dataset)
    for check in checks:
        check()
        print(f'{check.__name__}: PASS')
