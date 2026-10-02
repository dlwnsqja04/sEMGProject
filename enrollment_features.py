"""재학습 없는 사용자 등록에 사용할 고정 길이 sEMG 특징을 추출한다."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

from week2_signal import load_trial, preprocess


FEATURE_VERSION: Final = 'phase-spectrum-54-v1'
FEATURE_COUNT: Final = 54
BANDS: Final = ((20, 60), (60, 120), (120, 250), (250, 450))


@dataclass(frozen=True)
class Sample:
    digest: str
    features: tuple[float, ...]


def extract_sample(path: Path) -> Sample:
    """각 동작 구간과 채널의 진폭·변화량·주파수 비율을 요약한다."""
    signal = preprocess(load_trial(path))
    vector: list[float] = []
    for phase in range(3):
        for channel in range(2):
            values = signal[phase * 1000:(phase + 1) * 1000, channel]
            magnitude = np.abs(values)
            frequencies = np.fft.rfftfreq(len(values), d=1 / 1000)
            power = np.abs(np.fft.rfft(values * np.hanning(len(values)))) ** 2
            band_power = [float(power[(frequencies >= low) & (frequencies < high)].sum())
                          for low, high in BANDS]
            total_power = sum(band_power) + 1e-12
            vector.extend((
                float(np.log1p(np.sqrt(np.mean(values ** 2)))),
                float(np.log1p(np.quantile(magnitude, .9))),
                float(np.log1p(np.mean(np.abs(np.diff(values))))),
                float(np.mean(values[:-1] * values[1:] < 0)),
                float(np.log1p(total_power)),
                *(value / total_power for value in band_power),
            ))
    if len(vector) != FEATURE_COUNT or not np.isfinite(vector).all():
        raise RuntimeError(f'특징 추출 실패: {path}')
    return Sample(hashlib.sha256(path.read_bytes()).hexdigest(), tuple(vector))
