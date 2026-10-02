"""필터 전후 스펙트럼, CWT 예시, 실제 학습 loss 그림."""
from __future__ import annotations

import csv
import json
import os

import numpy as np
from scipy.signal import welch

from week2_signal import FS, OUT, ROOT, load_trial, make_windows, minmax, preprocess, to_cwt

os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'work/matplotlib'))
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def preparation_plots() -> None:
    """첫 학습 시행을 사용하고 공개 데이터에 대한 추가 필터임을 명시한다."""
    with (OUT / 'trial_split.csv').open(encoding='utf-8-sig') as stream:
        trial = next(row for row in csv.DictReader(stream) if row['split'] == 'train')
    path = ROOT / trial['file']
    source = load_trial(path)
    filtered = preprocess(source)
    f0, before = welch(source[:, 0], fs=FS, nperseg=512)
    _, after = welch(filtered[:, 0], fs=FS, nperseg=512)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={'width_ratios': [2, 1]})
    for ax in axes:
        ax.semilogy(f0, before, label='Released data', lw=1)
        ax.semilogy(f0, after, label='After additional filters', lw=1)
        ax.axvline(60, color='red', ls='--', lw=0.8)
        ax.set(xlabel='Frequency (Hz)', ylabel='PSD (source units squared / Hz)')
        ax.grid(alpha=0.2)
    axes[0].set_xlim(0, 300)
    axes[0].axvspan(0, 20, alpha=0.10, color='grey')
    axes[0].legend(fontsize=8)
    axes[0].set_title(f"Subject {trial['subject']} / {path.name} / Ch 1")
    axes[1].set_xlim(40, 80)
    zoom = (f0 >= 40) & (f0 <= 80)
    axes[1].set_ylim(float(after[zoom].min()) * 0.6, float(before[zoom].max()) * 1.5)
    axes[1].set_title('Zoom around 60 Hz')
    fig.tight_layout()
    fig.savefig(OUT / 'filter_check.png', dpi=180)
    plt.close(fig)
    windows = make_windows(filtered)
    normalized = minmax(windows)
    tensor = to_cwt(normalized[0])
    np.save(OUT / 'cwt_example.npy', tensor, allow_pickle=False)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.3), sharey=True)
    for ch, ax in enumerate(axes):
        image = ax.imshow(tensor[ch], origin='lower', aspect='auto', cmap='viridis',
                          extent=[0, 300, 0.5, 32.5], vmin=0, vmax=float(tensor.max()))
        ax.set(xlabel='Time within window (ms)', title=['Ch 1', 'Ch 2', 'Mean of Ch 1 and Ch 2'][ch])
        ax.set_yticks([1, 8, 16, 24, 32])
    axes[0].set_ylabel('Scale')
    fig.colorbar(image, ax=axes, label='Absolute CWT coefficient', fraction=0.025, pad=0.025)
    fig.suptitle(f"Morlet CWT | {path.name} | first 300 ms | tensor (3, 32, 300)", fontsize=10)
    fig.subplots_adjust(left=0.07, bottom=0.18, top=0.80, right=0.86, wspace=0.18)
    fig.savefig(OUT / 'cwt_example.png', dpi=180)
    plt.close(fig)
    nearest = int(np.argmin(np.abs(f0 - 60)))
    info = {'example_file': trial['file'], 'source_std': float(source.std()),
            'filtered_std': float(filtered.std()), 'windows_shape': list(windows.shape),
            'normalized_min': float(normalized.min()), 'normalized_max': float(normalized.max()),
            'cwt_shape': list(tensor.shape), 'welch_nperseg': 512,
            'nearest_60hz_bin': float(f0[nearest]),
            'psd_before_at_bin': float(before[nearest]), 'psd_after_at_bin': float(after[nearest]),
            'low_band_before': float(before[f0 < 20].sum()), 'low_band_after': float(after[f0 < 20].sum())}
    (OUT / 'example_checks.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    print(f'Windows: {windows.shape}; normalized: {normalized.min():.8f} ~ {normalized.max():.8f}; CWT: {tensor.shape}', flush=True)


def learning_plot() -> None:
    """저장한 실제 학습 로그의 loss 변화를 그린다."""
    with (OUT / 'training_history.csv').open() as stream:
        history = list(csv.DictReader(stream))
    epochs = [int(row['epoch']) for row in history]
    losses = [float(row['loss']) for row in history]
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(epochs, losses, marker='o')
    ax.set(xlabel='Epoch', ylabel='Training cross-entropy loss', xticks=epochs)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / 'training_loss.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    preparation_plots()
    if (OUT / 'training_history.csv').exists():
        learning_plot()
