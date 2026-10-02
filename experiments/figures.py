"""실제 실험 기록으로 그림을 만들고 OpenCV CWT 예시를 저장한다."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import cv2
import numpy as np

from week2_signal import ROOT

os.environ['MPLCONFIGDIR'] = str(ROOT / 'work/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiments.data import OUT, load_split


def cwt_panel(tensor: np.ndarray, width: int, destination: Path) -> None:
    high = float(tensor.max())
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.1), constrained_layout=True)
    for channel, ax in enumerate(axes):
        values = np.round(tensor[channel] / max(high, 1e-8) * 255).astype(np.uint8)
        colored = cv2.applyColorMap(values, cv2.COLORMAP_VIRIDIS)
        cv2.imwrite(str(destination.with_name(f'{destination.stem}_ch{channel+1}.png')), colored)
        ax.imshow(cv2.cvtColor(colored, cv2.COLOR_BGR2RGB), origin='lower', aspect='auto',
                  extent=(0, width, .5, 32.5), interpolation='nearest')
        ax.set(title=('Physical channel 1', 'Physical channel 2', 'Mean of channels')[channel],
               xlabel='Time (ms)', ylabel='Scale index')
    mapper = plt.cm.ScalarMappable(norm=plt.Normalize(0, high), cmap='viridis')
    fig.colorbar(mapper, ax=axes, label='Absolute CWT', shrink=.8)
    fig.savefig(destination, dpi=170)
    plt.close(fig)


def main() -> None:
    selected = json.loads((OUT / 'selection_locked.json').read_text())
    test = json.loads((OUT / 'test_evaluation.json').read_text())
    base = np.load(ROOT / 'results/week2/cwt_example.npy')
    cwt_panel(base, 300, OUT / 'cwt_opencv_300ms.png')
    folder = OUT / 'cache' / f'w{selected["width"]}_h{selected["hop"]}'
    x, _, _ = load_split(folder, 'train')
    np.save(OUT / 'selected_cwt_example.npy', x[0])
    cwt_panel(x[0], selected['width'], OUT / 'cwt_opencv_selected.png')
    history = list(csv.DictReader((Path(selected['directory']) / 'history.csv').open()))
    epochs = [int(r['epoch']) for r in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), constrained_layout=True)
    for column, label in [('train_loss', 'Training loss'), ('val_loss', 'Validation loss')]:
        axes[0].plot(epochs, [float(r[column]) for r in history], label=label)
    axes[0].set(xlabel='Epoch', ylabel='Cross-entropy loss', title='Training and validation loss')
    for column, label in [('trial_f1', 'Trial macro F1'), ('window_f1', 'Window macro F1')]:
        axes[1].plot(epochs, [float(r[column]) for r in history], label=label)
    axes[1].set(xlabel='Epoch', ylabel='Validation macro F1', ylim=(0, 1.03), title='Validation score for selection')
    for ax in axes:
        ax.axvline(selected['best']['epoch'], color='gray', linestyle='--', label='Selected epoch')
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.savefig(OUT / 'learning_curves.png', dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.7), constrained_layout=True)
    for ax, key, title in zip(axes, ['trial_confusion', 'window_confusion'], ['50 held-out trial files', 'Held-out windows']):
        matrix = np.array(test[key])
        ax.imshow(matrix, cmap='Blues', vmin=0)
        ax.set(xticks=range(5), yticks=range(5), xticklabels=list('ABCDE'), yticklabels=list('ABCDE'),
               xlabel='Predicted subject', ylabel='Actual subject', title=title)
        for (row, col), value in np.ndenumerate(matrix):
            ax.text(col, row, str(value), ha='center', va='center', color='white' if value > matrix.max()/2 else 'black')
    fig.savefig(OUT / 'test_confusion.png', dpi=170)
    plt.close(fig)
    print('Saved OpenCV CWT panels, actual learning curves and confusion matrices.')


if __name__ == '__main__':
    main()
