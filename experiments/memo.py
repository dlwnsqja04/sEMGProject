"""에포크별 전체 기록과 제출용 핵심 메모를 만든다."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report

from experiments.data import OUT, load_split


def main() -> None:
    selected = json.loads((OUT / 'selection_locked.json').read_text())
    test = json.loads((OUT / 'test_evaluation.json').read_text())
    all_rows = []
    for path in sorted((OUT / 'runs').glob('*/history.csv')):
        metadata = json.loads((path.parent / 'complete.json').read_text())
        for row in csv.DictReader(path.open()):
            all_rows.append({'run': path.parent.name, 'stage': metadata['stage'],
                             'model': metadata['name'], 'width_ms': metadata['width'],
                             'hop_ms': metadata['hop'], **row})
    with (OUT / 'training_metrics_all.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    history = list(csv.DictReader((Path(selected['directory']) / 'history.csv').open()))
    folder = OUT / 'cache' / f'w{selected["width"]}_h{selected["hop"]}'
    _, y, ids = load_split(folder, 'test')
    p = np.load(OUT / 'test_probabilities.npy')
    trial_y = [int(y[ids == i][0]) for i in np.unique(ids)]
    trial_p = [int(p[ids == i].mean(axis=0).argmax()) for i in np.unique(ids)]
    per_class = classification_report(trial_y, trial_p, target_names=list('ABCDE'), labels=list(range(5)),
                                      output_dict=True, zero_division=0)
    (OUT / 'per_class_test_metrics.json').write_text(json.dumps(per_class, indent=2))
    b = selected['best']
    lowest_train = min(history, key=lambda r: float(r['train_loss']))
    lowest_val = min(history, key=lambda r: float(r['val_loss']))
    n = (3000-selected['width']) // selected['hop'] + 1
    lines = ['# 2주차 실험 메모', '',
             'loss는 퍼센트가 아니라 교차 엔트로피 손실값이다. 정확도는 맞힌 비율이고 F1은 정밀도와 재현율을 함께 본 점수다.',
             'CSV의 accuracy와 F1은 0~1 값이다. 100을 곱하면 %로 볼 수 있다. 검증 결과로만 설정을 골랐다.',
             'CSV의 window_accuracy, trial_accuracy, window_f1, trial_f1은 모두 검증셋 지표다. train_accuracy만 학습셋 정확도이며, 테스트 결과는 test_evaluation.json에 따로 있다.', '',
             '## 지표를 읽는 법',
             '- 학습 loss: 모델이 학습 데이터에서 낸 오차. 낮을수록 좋지만 이것만으로 모델을 고르지 않았다.',
             '- 검증 loss: 가중치 학습에 넣지 않은 검증 데이터의 오차. 학습 loss와 반대로 움직이면 과적합을 의심할 수 있다.',
             '- 정확도: 전체 예측 중 맞힌 비율. 윈도우 하나씩 맞혔는지와 시행 단위로 맞혔는지를 구분했다.',
             '- 정밀도: 어떤 사람이라고 예측한 것 중 실제 그 사람이었던 비율.',
             '- 재현율: 실제 그 사람의 데이터 중 그 사람으로 맞힌 비율.',
             '- macro F1: 사람별 정밀도와 재현율을 합친 F1을 구한 다음 다섯 사람을 같은 비중으로 평균한 값.',
             '- 학습률: 가중치를 얼마나 크게 바꿀지 정하는 설정값. 인식률과는 다르다.',
             '- 에포크: 학습셋 전체를 한 번 사용한 횟수. 최적 에포크는 마지막 에포크와 다를 수 있다.', '',
             '## 최종 설정',
             f'- 모델: {selected["name"]}, ImageNet 사전학습 가중치 없음(weights=None), 파라미터 {selected["parameters"]:,}개',
             f'- 분류 대상: 피험자 A~E 5명. 동작 3종을 분류한 결과가 아니다.',
             f'- 윈도우: {selected["width"]}ms, 간격 {selected["hop"]}ms, 시행당 {n}개',
             '- 입력: 1000 Hz, 물리 채널 2개, morl CWT 스케일 1~32, 두 채널 맵과 평균 맵',
             '- 필터: 60 Hz notch(Q=30), 4차 20~499 Hz bandpass, 시간축 filtfilt. 배포 데이터에도 필터가 이미 적용되어 있다.',
             '- 정규화: 각 윈도우 시간·채널 전체 min-max, epsilon=1e-8. 색칠한 PNG가 아닌 float32 CWT 배열을 학습한다.',
             f'- 시행 분할: train 160 / validation 40 / test 50. 윈도우 수: {160*n} / {40*n} / {50*n}',
             '- 클래스별 시행: 학습 32, 검증 8, 테스트 10. 시행별 신호 hash가 서로 다른 분할에 겹치지 않는다.',
             '- seed: 테스트 분할 42, 검증 분할 43, 모델 초기화 42. 모든 모델 새로 학습, 기존 1차 모델 재사용 안 함.',
             '- 옵티마이저 Adam, 초기 학습률 0.001, 배치 16, CrossEntropyLoss, 사용 가능하면 MPS(아니면 CPU), CPU 스레드 4',
             '- 검증 loss가 4회 연속 개선되지 않으면 학습률 0.5배, 최소 0.00001. 최종 연장 학습은 최대 40에폭, 선정 점수 10회 미개선 시 중단.',
             f'- 최종 학습 {selected["completed_epochs"]}에폭, 사용 체크포인트 {b["epoch"]}에폭',
             '- 선정 순서: 검증 시행 macro F1 → 검증 윈도우 macro F1 → 낮은 검증 loss',
             f'- 해당 체크포인트: train loss {b["train_loss"]:.6f}, val loss {b["val_loss"]:.6f}, train 정확도 {b["train_accuracy"]*100:.2f}%',
             f'- 최저 학습 loss: {float(lowest_train["train_loss"]):.6f} ({lowest_train["epoch"]}에포크). 최저 검증 loss: {float(lowest_val["val_loss"]):.6f} ({lowest_val["epoch"]}에포크). 선택 에포크와 다를 수 있다.',
             f'- 검증: 시행 정확도 {b["trial_accuracy"]*100:.2f}%, 시행 F1 {b["trial_f1"]:.4f}, 윈도우 정확도 {b["window_accuracy"]*100:.2f}%, 윈도우 F1 {b["window_f1"]:.4f}',
             f'- 최종 테스트: 시행 정확도 {test["trial_accuracy"]*100:.2f}%, F1 {test["trial_f1"]:.4f}; 윈도우 정확도 {test["window_accuracy"]*100:.2f}%, F1 {test["window_f1"]:.4f}; loss {test["test_loss"]:.6f}',
             f'- 중복 제외 49개 신호: 시행 정확도 {test["unique_signal_evaluation"]["trial_accuracy"]*100:.2f}%, F1 {test["unique_signal_evaluation"]["trial_f1"]:.4f}',
             f'- 전체 기록: {len(all_rows)}에폭 행. 학습+검증 시간 합계 {sum(float(r["seconds"]) for r in all_rows)/60:.1f}분(전처리·파일 저장 제외)',
             '', '## 최종 모델의 에포크별 기록',
             '|에포크|학습 loss|검증 loss|학습 정확도 %|검증 윈도우 정확도 %|검증 시행 정확도 %|검증 시행 F1|학습률|초|',
             '|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in history:
        lines.append(f'|{r["epoch"]}|{float(r["train_loss"]):.6f}|{float(r["val_loss"]):.6f}|{float(r["train_accuracy"])*100:.2f}|{float(r["window_accuracy"])*100:.2f}|{float(r["trial_accuracy"])*100:.2f}|{float(r["trial_f1"]):.4f}|{float(r["lr"]):.6f}|{float(r["seconds"]):.1f}|')
    lines += ['', '## 피험자별 최종 테스트 성능(파일 50개)', '|피험자|정밀도|재현율|F1|파일 수|', '|---|---:|---:|---:|---:|']
    for label in 'ABCDE':
        r = per_class[label]
        lines.append(f'|{label}|{r["precision"]:.4f}|{r["recall"]:.4f}|{r["f1-score"]:.4f}|{int(r["support"])}|')
    lines += ['', '## 모델별 10에포크 비교',
              '|모델|파라미터 수|선택 에포크|검증 시행 F1|검증 윈도우 F1|10에포크 시간(분)|',
              '|---|---:|---:|---:|---:|---:|']
    for model in json.loads((OUT / 'model_comparison.json').read_text()):
        candidate_history = list(csv.DictReader((Path(model['directory']) / 'history.csv').open()))
        seconds = sum(float(r['seconds']) for r in candidate_history if int(r['epoch']) <= 10)
        best = model['best']
        lines.append(f'|{model["name"]}|{model["parameters"]:,}|{best["epoch"]}|{best["trial_f1"]:.4f}|{best["window_f1"]:.4f}|{seconds/60:.2f}|')
    lines += ['', '시간은 이 컴퓨터에서 기록한 학습+검증 시간이며 전처리와 가중치 파일 저장 시간은 제외했다. 엄밀한 속도 벤치마크 수치는 아니다.']
    lines += ['', '## 주의해서 해석할 점',
              '- E/e (38).csv와 E/e (50).csv는 완전히 같은 데이터다. 둘 다 테스트에 있으며 중복을 뺀 결과도 따로 계산했다.',
              '- 한 시행의 윈도우 확률을 평균 낸 뒤 피험자를 정하는 방식이다. 윈도우 인식률과 시행 인식률은 같은 수치가 아니다.',
              '- 15가지 윈도우는 ResNet18 5에폭으로 예비 비교했고, 고른 윈도우에서 3개 모델을 10에폭씩 비교했다. 모델과 윈도우의 모든 조합을 비교한 것은 아니다.',
              '- 최종 모델만 연장 학습했으므로 다른 모델을 40에폭 학습한 결과보다 좋다고 결론 내릴 수 없다.',
              '- 같은 5명의 다른 시행을 나눈 실험이다. 새로운 사람이나 다른 날에도 같은 성능이 나온다고 확인한 것은 아니다.',
              '- 검증 시행은 40개이고 seed는 한 번이다. 여러 번 후보를 고른 영향과 우연의 변동이 있을 수 있다.',
              '', '## 기록 파일',
              '- training_metrics_all.csv: 모든 후보의 모든 에포크 원본 숫자',
              '- runs/*/training_log.txt와 history.csv: 모델별 실제 출력 및 기록',
              '- protocol.json: 실험 전 정한 범위와 선택 규칙',
              '- trial_split.csv: 파일명, 라벨, 분할, SHA-256',
              '- window_comparison.csv/json, model_comparison.json: 후보 비교',
              '- selection_locked.json: 테스트 전에 고른 모델과 에포크',
              '- test_probabilities.npy, test_evaluation.json: 최종 예측 확률과 점수',
              '- environment.json: 실제 Python, 운영체제, 라이브러리 버전 및 GPU 사용 가능 여부',
              '', '## 구현 참고 문서',
              '- PyTorch ResNet18: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html',
              '- 데이터 누수 방지: https://scikit-learn.org/stable/common_pitfalls.html',
              '- OpenCV 색상 맵: https://docs.opencv.org/4.x/d3/d50/group__imgproc__colormap.html']
    comparisons = (OUT / '비교_결과.md').read_text(encoding='utf-8')
    lines += ['', comparisons.replace('# 비교 결과 메모', '# 전체 후보 비교 수치', 1)]
    (OUT / '실험_메모.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'Saved memo and {len(all_rows)} epoch rows.')


if __name__ == '__main__':
    main()
