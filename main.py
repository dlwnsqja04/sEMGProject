"""sEMG 실험과 새 사용자 등록 평가 결과를 만든다."""
from __future__ import annotations

from pathlib import Path
from shutil import copyfileobj
from zipfile import ZipFile

from evaluation import main as evaluate_models
from enrollment_experiment import main as evaluate_enrollment
from experiments.comparisons import main as export_comparisons
from experiments.memo import main as export_notes
from week1 import main as summarize_data
from week2_data import prepare_dataset
from week2_experiments import main as run_experiments


ROOT = Path(__file__).resolve().parent


def ensure_data() -> None:
    """동봉된 원본 압축 파일에서 피험자별 CSV를 필요할 때만 꺼낸다."""
    data_dir = ROOT / 'data' / 'data'
    if len(list(data_dir.glob('*/*.csv'))) == 250:
        return
    archive = ROOT / 'data-source.zip'
    if not archive.is_file():
        raise FileNotFoundError(f'원본 데이터 압축 파일이 없습니다: {archive}')
    with ZipFile(archive) as source:
        for member in source.infolist():
            parts = Path(member.filename).parts
            if len(parts) != 4 or parts[1] != 'data' or parts[2] not in ('A', 'B', 'C', 'D', 'E'):
                continue
            if not parts[3].lower().endswith('.csv'):
                continue
            target = data_dir / parts[2] / parts[3]
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open(member) as reader, target.open('wb') as writer:
                copyfileobj(reader, writer)
    if len(list(data_dir.glob('*/*.csv'))) != 250:
        raise RuntimeError('압축 파일에서 CSV 250개를 찾지 못했습니다.')


def main() -> None:
    ensure_data()
    week1_outputs = ('file_summary.csv', 'subjects_comparison.png', 'signal_example.png')
    if all((ROOT / 'results' / name).is_file() for name in week1_outputs):
        print('기존 1주차 데이터 요약을 재사용합니다.', flush=True)
    else:
        summarize_data()
    prepare_dataset()
    completed = ROOT / 'results' / 'week2_experiments' / 'test_evaluation.json'
    if completed.is_file():
        print('기존 학습 결과를 재사용합니다.', flush=True)
    else:
        run_experiments()
    export_comparisons()
    evaluate_models()
    export_notes()
    evaluate_enrollment()

if __name__ == '__main__':
    main()
