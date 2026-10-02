"""동일한 테스트 집합에서 세 모델의 점수와 혼동행렬을 저장한다."""
from __future__ import annotations

import csv
import json
import os
from typing import Final

from experiments.data import OUT, load_split

FONT_CACHE: Final = OUT.parent / '.matplotlib'
FONT_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(FONT_CACHE))

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from experiments.training import create_model, evaluate


LABELS: Final = ('A', 'B', 'C', 'D', 'E')
DESTINATION: Final = OUT.parent / 'submission'


def trial_predictions(probabilities: np.ndarray, labels: np.ndarray,
                      trial_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """같은 시행의 윈도우 확률을 평균하여 시행별 예측을 만든다."""
    actual, predicted = [], []
    for trial_id in np.unique(trial_ids):
        mask = trial_ids == trial_id
        unique_labels = np.unique(labels[mask])
        if len(unique_labels) != 1:
            raise RuntimeError(f'시행 {trial_id}에 다른 정답 라벨이 섞여 있습니다.')
        actual.append(int(unique_labels[0]))
        predicted.append(int(probabilities[mask].mean(axis=0).argmax()))
    return np.array(actual), np.array(predicted)


def classification_scores(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        actual, predicted, labels=range(len(LABELS)), average='macro', zero_division=0)
    return {'accuracy': float(accuracy_score(actual, predicted)),
            'precision': float(precision), 'recall': float(recall), 'f1': float(f1)}


def save_matrix(matrix: np.ndarray, name: str, level: str) -> None:
    fig, axis = plt.subplots(figsize=(5.4, 4.8))
    axis.imshow(matrix, cmap='Blues')
    axis.set(xticks=range(5), yticks=range(5), xticklabels=LABELS,
             yticklabels=LABELS, xlabel='Predicted subject',
             ylabel='True subject', title=f'{name} - {level} confusion matrix')
    threshold = matrix.max() / 2
    for row in range(5):
        for column in range(5):
            axis.text(column, row, str(matrix[row, column]), ha='center', va='center',
                      color='white' if matrix[row, column] > threshold else 'black')
    fig.tight_layout()
    fig.savefig(DESTINATION / f'confusion_matrix_{name}_{level}.png', dpi=180)
    plt.close(fig)


def main() -> None:
    """저장된 검증 최적 체크포인트를 고정된 테스트 집합에서 평가한다."""
    DESTINATION.mkdir(parents=True, exist_ok=True)
    comparison = json.loads((OUT / 'model_comparison.json').read_text())
    selected = json.loads((OUT / 'selection_locked.json').read_text())
    width, hop = int(selected['width']), int(selected['hop'])
    test = load_split(OUT / 'cache' / f'w{width}_h{hop}', 'test')
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    torch.set_num_threads(4)
    rows = []
    details = {}
    for entry in comparison:
        name = entry['name']
        checkpoint = OUT / 'runs' / f'model_{name}_w{width}_h{hop}' / 'best.pt'
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        model = create_model(name).to(device)
        model.load_state_dict(saved['model'])
        _, probabilities = evaluate(model, test, device)
        window_actual, trial_ids = test[1], test[2]
        window_predicted = probabilities.argmax(axis=1)
        trial_actual, trial_predicted = trial_predictions(probabilities, window_actual, trial_ids)
        scores = {f'window_{key}': value for key, value in
                  classification_scores(window_actual, window_predicted).items()}
        scores.update({f'trial_{key}': value for key, value in
                       classification_scores(trial_actual, trial_predicted).items()})
        window_matrix = confusion_matrix(window_actual, window_predicted, labels=range(5))
        trial_matrix = confusion_matrix(trial_actual, trial_predicted, labels=range(5))
        save_matrix(window_matrix, name, 'window')
        save_matrix(trial_matrix, name, 'trial')
        row = {'model': name, 'checkpoint_epoch': int(saved['best']['epoch']),
               'validation_window_f1_at_10_epochs': float(entry['best']['window_f1']),
               'parameters': int(entry['parameters']), **scores}
        rows.append(row)
        details[name] = {'scores': scores, 'window_confusion': window_matrix.tolist(),
                         'trial_confusion': trial_matrix.tolist(),
                         'checkpoint_epoch': row['checkpoint_epoch']}
        print(f'{name}: window accuracy={scores["window_accuracy"]:.4f}, '
              f'macro F1={scores["window_f1"]:.4f}', flush=True)
        del model
        if device.type == 'mps':
            torch.mps.empty_cache()
    with (DESTINATION / 'model_metrics.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (DESTINATION / 'model_results.json').write_text(
        json.dumps({'test_windows': len(test[1]), 'test_trials': len(np.unique(test[2])),
                    'selected_by_validation': selected['name'], 'models': details}, indent=2),
        encoding='utf-8')


if __name__ == '__main__':
    main()
