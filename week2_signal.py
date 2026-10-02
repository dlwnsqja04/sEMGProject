from __future__ import annotations

import csv
from pathlib import Path
from typing import Final

import numpy as np
import pywt
from numpy.typing import NDArray
from scipy.signal import butter, filtfilt, iirnotch

ROOT: Final = Path(__file__).resolve().parent
OUT: Final = ROOT / 'results' / 'week2'
FS: Final = 1000
WIN: Final = 300
HOP: Final = 150
SCALES: Final = np.arange(1, 33)
SUBJECTS: Final = ('A', 'B', 'C', 'D', 'E')


def load_trial(path: Path) -> NDArray[np.float64]:
    with path.open(encoding='utf-8-sig') as stream:
        header = next(csv.reader(stream))
    values = np.loadtxt(path, delimiter=',', skiprows=1, ndmin=2)
    if header != ['Comp Ch 3', 'Comp Ch 4'] or values.shape != (3000, 2):
        raise ValueError(f'예상한 3000×2 CSV 형식과 다릅니다: {path}')
    if not np.isfinite(values).all():
        raise ValueError(f'유한하지 않은 값이 있습니다: {path}')
    return values


def preprocess(x: NDArray[np.float64]) -> NDArray[np.float64]:
    bn, an = iirnotch(60, 30, fs=FS)
    notched = filtfilt(bn, an, x, axis=0)
    b, a = butter(4, [20 / (FS / 2), 499 / (FS / 2)], btype='band')
    return np.asarray(filtfilt(b, a, notched, axis=0), dtype=np.float64)


def make_windows(x: NDArray[np.float64]) -> NDArray[np.float64]:
    if len(x) < WIN:
        raise ValueError(f'최소 {WIN}샘플이 필요합니다. 현재: {len(x)}')
    count = (len(x) - WIN) // HOP + 1
    return np.stack([x[i * HOP:i * HOP + WIN] for i in range(count)])


def minmax(windows: NDArray[np.float64]) -> NDArray[np.float64]:
    low = windows.min(axis=(1, 2), keepdims=True)
    high = windows.max(axis=(1, 2), keepdims=True)
    return (windows - low) / (high - low + 1e-8)


def to_cwt(window: NDArray[np.float64]) -> NDArray[np.float32]:
    maps = [np.abs(pywt.cwt(window[:, ch], SCALES, 'morl')[0]) for ch in range(2)]
    maps.append((maps[0] + maps[1]) / 2)
    return np.stack(maps).astype(np.float32)
