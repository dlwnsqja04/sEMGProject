"""윈도우와 모델 후보 전체를 숫자로 비교해 기록한다."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from experiments.data import OUT


def export(records: list[dict], filename: str) -> list[dict]:
    result = []
    for r in records:
        best = r['best']
        history = list(csv.DictReader((Path(r['directory']) / 'history.csv').open()))
        elapsed = sum(float(h['seconds']) for h in history if int(h['epoch']) <= r['completed_epochs'])
        n = (3000-r['width'])//r['hop']+1
        result.append({'model': r['name'], 'window_ms': r['width'], 'hop_ms': r['hop'],
                       'overlap_percent': (1-r['hop']/r['width'])*100, 'windows_per_trial': n,
                       'train_windows': 160*n, 'validation_windows': 40*n,
                       'comparison_epochs': r['completed_epochs'], 'selected_epoch': best['epoch'],
                       'train_loss': best['train_loss'], 'val_loss': best['val_loss'],
                       'train_accuracy_percent': best['train_accuracy']*100,
                       'val_window_accuracy_percent': best['window_accuracy']*100,
                       'val_window_macro_f1': best['window_f1'],
                       'val_trial_accuracy_percent': best['trial_accuracy']*100,
                       'val_trial_macro_f1': best['trial_f1'], 'parameters': r['parameters'],
                       'training_validation_seconds': elapsed, 'average_epoch_seconds': elapsed/r['completed_epochs']})
    with (OUT / filename).open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    return result


def main() -> None:
    windows = export(json.loads((OUT / 'window_comparison.json').read_text()), 'window_comparison.csv')
    models = export(json.loads((OUT / 'model_comparison.json').read_text()), 'model_comparison.csv')
    lines = ['# 비교 결과 메모', '',
             '아래는 최종 테스트 결과가 아니라 설정을 고를 때 사용한 검증 결과다. 모든 후보의 시행 분할은 학습 160개, 검증 40개로 같다.',
             '정확도는 %, F1은 0~1 값이다. 각 후보의 검증 시행 macro F1 → 검증 윈도우 macro F1 → 낮은 검증 loss 순으로 에포크를 골랐다.', '',
             '## 윈도우 15가지: ResNet18, 조건별 5에포크',
             '|길이/간격 ms|겹침 %|시행당 조각|학습/검증 조각|선택 에포크|검증 loss|조각 정확도 %|조각 F1|시행 정확도 %|시행 F1|',
             '|---|---:|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in windows:
        lines.append(f'|{r["window_ms"]}/{r["hop_ms"]}|{r["overlap_percent"]:.0f}|{r["windows_per_trial"]}|{r["train_windows"]}/{r["validation_windows"]}|{r["selected_epoch"]}|{r["val_loss"]:.4f}|{r["val_window_accuracy_percent"]:.2f}|{r["val_window_macro_f1"]:.4f}|{r["val_trial_accuracy_percent"]:.2f}|{r["val_trial_macro_f1"]:.4f}|')
    best = max(windows, key=lambda r: (r['val_trial_macro_f1'],r['val_window_macro_f1'],-r['val_loss']))
    baseline = next(r for r in windows if r['window_ms'] == 300 and r['hop_ms'] == 150)
    lines += ['', f'강의 기본 300/150 ms: 검증 조각 정확도 {baseline["val_window_accuracy_percent"]:.2f}%, 조각 F1 {baseline["val_window_macro_f1"]:.4f}, 시행 F1 {baseline["val_trial_macro_f1"]:.4f}.',
              f'예비 비교 선택 {best["window_ms"]}/{best["hop_ms"]} ms: 검증 조각 정확도 {best["val_window_accuracy_percent"]:.2f}%, 조각 F1 {best["val_window_macro_f1"]:.4f}, 시행 F1 {best["val_trial_macro_f1"]:.4f}.',
              f'기본 설정 대비 조각 정확도 차이는 {best["val_window_accuracy_percent"]-baseline["val_window_accuracy_percent"]:+.2f}%p, 조각 F1 차이는 {best["val_window_macro_f1"]-baseline["val_window_macro_f1"]:+.4f}이다. 이 차이는 같은 분할의 5에포크 예비 비교에서 나온 값이다.',
              '', '## 모델 3가지: 같은 윈도우, 모델별 10에포크',
              '|모델|선택 에포크|검증 loss|조각 정확도 %|조각 F1|시행 정확도 %|시행 F1|파라미터|평균 초/에포크|',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in models:
        lines.append(f'|{r["model"]}|{r["selected_epoch"]}|{r["val_loss"]:.4f}|{r["val_window_accuracy_percent"]:.2f}|{r["val_window_macro_f1"]:.4f}|{r["val_trial_accuracy_percent"]:.2f}|{r["val_trial_macro_f1"]:.4f}|{r["parameters"]:,}|{r["average_epoch_seconds"]:.1f}|')
    lines += ['', '모델의 사전학습 가중치는 쓰지 않았다. 시간에는 학습과 검증이 포함되며 전처리 및 체크포인트 저장 시간은 제외했다.',
              '윈도우 예비 비교(5에포크)와 모델 비교(10에포크)는 서로 다른 단계다. 연장 학습 결과는 실험_메모.md와 최종 PDF에 별도로 적었다.',
              '윈도우 수가 다르면 에포크당 업데이트 횟수도 다르다. 모든 길이×모델×에포크 조합을 비교한 것은 아니며 한 번의 분할에서 나온 결과다.']
    (OUT / '비교_결과.md').write_text('\n'.join(lines), encoding='utf-8')
    print('Saved all 15 window comparisons and all 3 model comparisons, including accuracy, F1, loss and time.')


if __name__ == '__main__':
    main()
