"""1주차: 실제 CSV 점검, 파형 비교, 결과 노트북 생성."""
from __future__ import annotations

import base64
import csv
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

ROOT: Final = Path(__file__).resolve().parent
MATPLOTLIB_CONFIG: Final = ROOT / 'work' / 'matplotlib'
MATPLOTLIB_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(MATPLOTLIB_CONFIG))

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

FS: Final = 1000


def load_signal(path: Path) -> NDArray[np.float64]:
    """공개 CSV의 헤더와 2채널 수치 데이터를 확인한다."""
    with path.open(encoding='utf-8-sig') as stream:
        header = next(csv.reader(stream))
    if header != ['Comp Ch 3', 'Comp Ch 4']:
        raise RuntimeError(f'예상하지 못한 헤더: {path}: {header}')
    signal = np.loadtxt(path, delimiter=',', skiprows=1, ndmin=2)
    if signal.shape[1] != 2 or len(signal) == 0 or not np.isfinite(signal).all():
        raise RuntimeError(f'유효하지 않은 2채널 데이터: {path}: {signal.shape}')
    return signal


def main() -> None:
    """실제 파일 전체를 분석하고 재현 가능한 제출 초안을 저장한다."""
    output = ROOT / 'results'
    output.mkdir(exist_ok=True)
    files = sorted((ROOT / 'data').rglob('*.csv'))
    if not files:
        raise FileNotFoundError(f'CSV 파일이 없습니다: {ROOT / "data"}')
    environment = []
    for module, package in [('numpy', 'numpy'), ('scipy', 'scipy'),
                            ('matplotlib', 'matplotlib'), ('pandas', 'pandas'),
                            ('sklearn', 'scikit-learn'), ('pywt', 'PyWavelets'),
                            ('torch', 'torch'), ('torchvision', 'torchvision')]:
        importlib.import_module(module)
        environment.append(f'{package} {importlib.metadata.version(package)}: OK')
    (output / 'environment_check.txt').write_text('\n'.join(environment), encoding='utf-8')
    subjects = sorted({p.parent.name for p in files})
    summary = ['| 피험자 | 시행 수 | 배열 모양 | 최솟값 | 최댓값 |',
               '|---|---:|---|---:|---:|']
    with (output / 'file_summary.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.writer(stream)
        writer.writerow(['subject', 'file', 'samples', 'channels', 'min', 'max'])
        for subject in subjects:
            paths = [p for p in files if p.parent.name == subject]
            signals = [load_signal(p) for p in paths]
            for path, x in zip(paths, signals):
                writer.writerow([subject, str(path.relative_to(ROOT)), *x.shape, x.min(), x.max()])
            shapes = sorted({x.shape for x in signals})
            summary.append(f'| {subject} | {len(paths)} | {shapes} | '
                           f'{min(float(x.min()) for x in signals):.6g} | '
                           f'{max(float(x.max()) for x in signals):.6g} |')
    selected = [next(p for p in files if p.parent.name == s) for s in subjects[:3]]
    if len(selected) < 3:
        raise RuntimeError('피험자 3명 이상이 필요합니다.')
    fig, axes = plt.subplots(2, 3, figsize=(14, 6), sharex=True, sharey='row')
    rms_notes = []
    for col, path in enumerate(selected):
        x = load_signal(path)
        t = np.arange(len(x)) / FS
        for ch in range(2):
            ax = axes[ch, col]
            ax.plot(t, x[:, ch], lw=0.5)
            ax.axvline(1, color='red', ls='--', lw=0.8)
            ax.axvline(2, color='red', ls='--', lw=0.8)
            ax.set(xlabel='Time (s)', ylabel=f'Ch {ch + 1} (source units)',
                   title=f'Subject {path.parent.name} / Ch {ch + 1}')
            for start, label in enumerate(['Grasp', 'Rotate', 'Stop']):
                ax.text(start + 0.5, 0.95, label, transform=ax.get_xaxis_transform(),
                        ha='center', va='top', fontsize=8)
        rms = [float(np.sqrt(np.mean(x[start * FS:(start + 1) * FS] ** 2)))
               for start in range(3)]
        rms_notes.append(f'- {path.parent.name} ({path.name}): 파지/회전/정지 RMS = '
                         + ' / '.join(f'{v:.5f}' for v in rms))
    fig.suptitle('Palm sEMG: first available trial per subject; common channel scales')
    fig.tight_layout()
    fig.savefig(output / 'subjects_comparison.png', dpi=160)
    plt.close(fig)
    x = load_signal(selected[0])
    fig, axes = plt.subplots(2, 1, figsize=(10, 4), sharex=True)
    for ch, ax in enumerate(axes):
        ax.plot(np.arange(len(x)) / FS, x[:, ch], lw=0.6)
        ax.axvline(1, color='red', ls='--')
        ax.axvline(2, color='red', ls='--')
        ax.set_ylabel(f'Ch {ch + 1}')
    axes[-1].set_xlabel('Time (s)')
    fig.tight_layout()
    fig.savefig(output / 'signal_example.png', dpi=160)
    plt.close(fig)
    report = '\n'.join([
        '# 1주차 sEMG 실습 결과', '',
        f'실제 CSV: {len(files)}개, 피험자: {len(subjects)}명. 샘플링 주파수는 저장소 설명의 1,000 Hz를 사용했다.',
        '헤더는 Comp Ch 3, Comp Ch 4이며 각각 APB, ADM 채널이다. 시간 열은 없다.',
        '', *summary, '',
        '## 파형과 관찰',
        '1초와 2초 경계는 실험 프로토콜에 따른 표시이며 신호로 추정한 경계가 아니다.',
        '원본을 자르거나 다시 필터링하거나 정규화하지 않았다. 단위는 CSV에 명시되지 않아 원본 단위로 표시했다.',
        *rms_notes, '',
        '관찰 메모 초안: 아래 수치와 그림을 보고 본인이 확인한 문장으로 수정한다.',
        '1. 음수와 양수가 함께 있으므로 제공 데이터는 0~1 min-max 정규화된 상태가 아니다.',
        '2. 동일한 축 범위에서 비교하면 피험자별 및 채널별 진폭과 시간 변화가 다르다. 한 시행만으로 개인의 고유 특성이라고 단정할 수는 없다.',
        '3. 선택한 A·B·C의 첫 시행에서는 두 채널을 합쳐 계산한 RMS가 모두 회전 구간에서 가장 컸다. 이는 선택한 시행의 관찰이며 전체 시행에 일반화하지 않는다.',
        '', '## 선행연구 비교표',
        '| 측정 부위 | 인원 | 채널 | 동작 | 특징 추출 + 모델 | 정확도(%) |',
        '|---|---:|---:|---|---|---|',
        '| 손목 | 50 | 8 | 박수치기 | GAN + DNN | 97.94 |',
        '| 전완 | 80 | 8 | 스마트폰 잠금해제 | Siamese CNN | 92.06 |',
        '| 전완 | 5 | 4 | 손동작 6종 | DWT/EWT/EMD + CNN | 최대 95.62 |',
        '| 이두·삼두 | 40 | 12 | 손동작 3종 | CQT + CNN | 97.50 |',
        '| 전완 | 21 | 4 | 손 펴기 | DWT/CWT + CNN | 최대 99.206 |',
        '| 손바닥 | 5 | 2 | 문손잡이 파지·회전·정지 | CWT + DenseNet161 | 시험 93.37; 5-fold 평균 91.66 ± 2.78 |',
        '', '기존 연구가 주로 손목·전완 등의 다채널 신호를 이용한 반면, 이 논문은 손바닥의 2채널 신호를 사용한다.',
        '별도 인증 제스처 대신 일상적인 문손잡이 회전을 이용하여 등록된 5명 중 사용자를 식별한다.',
        '연구별 데이터와 평가 조건이 달라 정확도 수치만으로 우열을 판단할 수 없다. 위 정확도는 논문 인용값이며 이번 실습의 모델 학습 결과가 아니다.',
        '', '## 출처',
        '- 강의 자료: Step1_1주차_강의와실습.pdf, 10–17페이지.',
        '- 참고논문: https://doi.org/10.1038/s41598-026-46294-3 (Table 1, Methods, Cross-validation results).',
        '- 데이터: https://github.com/sea3551/palm-sEMG-doorknob-filtered (CC BY 4.0, Yeonjung Shin).',
        '- 저장소에 포함된 README와 LICENSE는 data 폴더에 보존했다.',
        '- GitHub 제출 링크는 본인 저장소를 만든 뒤 추가할 것.',
    ])
    (output / 'week1_report.md').write_text(report, encoding='utf-8')
    image_data = base64.b64encode((output / 'subjects_comparison.png').read_bytes()).decode('ascii')
    notebook = {
        'cells': [
            {'cell_type': 'markdown', 'metadata': {}, 'source': [report]},
            {'cell_type': 'code', 'metadata': {}, 'execution_count': 1,
             'source': ['from week1 import main\nmain()'],
             'outputs': [{'output_type': 'stream', 'name': 'stdout',
                          'text': ['\n'.join(environment) + '\n\n' + '\n'.join(summary) + '\n']}]},
            {'cell_type': 'markdown', 'metadata': {},
             'source': ['## 피험자 3명 비교\n![comparison](attachment:comparison.png)'],
             'attachments': {'comparison.png': {'image/png': image_data}}},
        ],
        'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                     'language_info': {'name': 'python', 'version': '3.9.6'}},
        'nbformat': 4, 'nbformat_minor': 4,
    }
    (ROOT / 'week1_practice.ipynb').write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\n'.join(environment))
    print('\n'.join(summary))
    print(f'결과 저장: {output}\n노트북: {ROOT / "week1_practice.ipynb"}')


if __name__ == '__main__':
    main()
