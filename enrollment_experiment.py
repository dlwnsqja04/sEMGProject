"""한 사람씩 미등록자로 빼고, 소수 시행 등록 전·후를 평가한다."""
from __future__ import annotations

import csv
import hashlib
import shlex
from pathlib import Path
from typing import Final

import numpy as np

from enrollment import Store, add_profile, calibrate, identify, save_store
from enrollment_features import Sample, extract_sample


ROOT: Final = Path(__file__).resolve().parent
OUTPUT: Final = ROOT / 'results' / 'enrollment'
SUBJECTS: Final = ('A', 'B', 'C', 'D', 'E')


def select_trials() -> tuple[dict[str, dict[str, list[Path]]], list[dict[str, str]]]:
    """원본 해시 중복을 제거하고 사용자별 등록 5·검증 10·테스트 나머지를 고정한다."""
    seen: set[str] = set()
    selected: dict[str, dict[str, list[Path]]] = {}
    rows: list[dict[str, str]] = []
    random = np.random.default_rng(42)
    for subject in SUBJECTS:
        files = []
        for path in sorted((ROOT / 'data' / 'data' / subject).glob('*.csv')):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            files.append(path)
        order = random.permutation(len(files))
        shuffled = [files[int(index)] for index in order]
        selected[subject] = {'enrollment': shuffled[:5],
                             'validation': shuffled[5:15],
                             'test': shuffled[15:]}
        for split, group in selected[subject].items():
            for path in group:
                rows.append({'subject': subject, 'split': split,
                             'file': str(path.relative_to(ROOT)),
                             'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    return selected, rows


def evaluate_fold(held_out: str, samples: dict[str, dict[str, list[Sample]]]) -> tuple[dict, Store, Store]:
    """미등록자 거부와 등록 직후의 식별·주장 ID 검증을 분리해 측정한다."""
    store = Store()
    for subject in SUBJECTS:
        if subject != held_out:
            store = add_profile(store, subject, samples[subject]['enrollment'])
    validation = {subject: samples[subject]['validation']
                  for subject in SUBJECTS if subject != held_out}
    before = calibrate(store, validation)
    unknown_decisions = [identify(before, sample) for sample in samples[held_out]['test']]
    after = add_profile(before, held_out, samples[held_out]['enrollment'])
    correct_new = 0
    correct_existing = 0
    false_accepts = 0
    false_rejects = 0
    total = 0
    for subject in SUBJECTS:
        for sample in samples[subject]['test']:
            decision = identify(after, sample)
            correct = decision.user_id == subject
            correct_new += int(correct and subject == held_out)
            correct_existing += int(correct and subject != held_out)
            false_rejects += int(not correct)
            false_accepts += int(decision.user_id is not None and decision.user_id != subject)
            total += 1
    new_count = len(samples[held_out]['test'])
    row = {'new_user': held_out, 'enrollment_trials': 5,
           'validation_trials_known_users': sum(len(group) for group in validation.values()),
           'new_user_test_trials': new_count, 'known_test_trials': total - new_count,
           'threshold': before.threshold,
           'unknown_rejection_before': sum(d.user_id is None for d in unknown_decisions) / new_count,
           'new_user_accuracy_after': correct_new / new_count,
           'existing_user_accuracy_after': correct_existing / (total - new_count),
           'genuine_false_reject_rate': false_rejects / total,
           'wrong_id_false_accept_rate': false_accepts / (total * (len(SUBJECTS) - 1))}
    return row, before, after


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    selected, split_rows = select_trials()
    with (OUTPUT / 'split.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(split_rows[0]))
        writer.writeheader()
        writer.writerows(split_rows)
    samples = {subject: {split: [extract_sample(path) for path in files]
                         for split, files in groups.items()}
               for subject, groups in selected.items()}
    results = []
    for subject in SUBJECTS:
        row, before, after = evaluate_fold(subject, samples)
        results.append(row)
        print(f'{subject} 신규 등록: 이전 미등록자 거부 {row["unknown_rejection_before"]:.1%}, '
              f'등록 후 본인 식별 {row["new_user_accuracy_after"]:.1%}', flush=True)
        if subject == 'E':
            save_store(OUTPUT / 'demo_initial.json', before)
            save_store(OUTPUT / 'demo_after_E.json', after)
    with (OUTPUT / 'evaluation.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    lines = ['# 새 사용자 즉석 등록 실험', '',
             '기존 DenseNet161의 고정 5분류 출력은 사용하지 않았다. 신호의 54개 고정 특징을 사용자별 템플릿으로 저장하고 거리를 비교했다. 새 사람 등록 시 모델 가중치 재학습은 없다.',
             '원본 250개 중 동일한 내용의 E 파일 1개를 제외한 249개를 사용했다. 사람마다 등록 5개, 검증 10개, 나머지를 테스트로 분리했다.',
             '매번 한 사람은 처음 네 사람의 등록·임계값 보정에서 완전히 빼고, 그 사람을 추가한 뒤 같은 테스트 파일로 전후를 비교했다. 사용자별 거리 기준은 기존 사용자들의 검증 신호 95백분위수에 5% 여유를 둔 값이다. 신규 사용자 반경은 5개 등록 신호의 하나씩 뺀 거리와 기존 기준 중앙값을 이용하며, 기존 템플릿·정규화·기준은 재학습하지 않는다.',
             '개발 중 시험 결과를 확인한 뒤 검증 데이터로 승인 기준을 조정했다. 아래 시험셋 수치는 최종 구현의 관찰 결과지만, 완전히 손대지 않은 독립 평가로 보기는 어렵다.', '',
             '|새 사용자|등록 전 미등록자 거부|등록 후 신규 사용자 식별|기존 사용자 식별|본인 거부율|다른 ID 잘못 승인율|',
             '|---|---:|---:|---:|---:|---:|']
    for row in results:
        lines.append(f'|{row["new_user"]}|{row["unknown_rejection_before"]:.1%}|'
                     f'{row["new_user_accuracy_after"]:.1%}|{row["existing_user_accuracy_after"]:.1%}|'
                     f'{row["genuine_false_reject_rate"]:.1%}|{row["wrong_id_false_accept_rate"]:.1%}|')
    lines += ['', '잘못 승인율은 각 시험 파일에 대해 틀린 사용자 ID 네 개를 주장했을 때의 비율이다. 보류와 다른 사람으로 잘못 식별하는 경우 모두 본인 인증 실패로 셌다.',
              '평균은 사람별 다섯 회의 단순 평균이다. 같은 사람의 테스트 파일이 여러 회차에 등장하므로 다섯 회를 독립 표본으로 취급하지 않는다.',
              *[f'- {label}: {np.mean([row[key] for row in results]):.1%} ± {np.std([row[key] for row in results]):.1%}'
                for label, key in (('등록 전 미등록자 거부', 'unknown_rejection_before'),
                                   ('등록 후 신규 사용자 식별', 'new_user_accuracy_after'),
                                   ('기존 사용자 식별', 'existing_user_accuracy_after'))],
              '', '이 결과는 같은 수집 환경의 다섯 사람으로 새 사용자 상황을 흉내 낸 것이다. 실제 처음 보는 사람, 다른 날 측정, 센서 위치 변화, 공격자의 위조 신호에 대한 성능은 확인하지 못했다.',
              '특징 템플릿 JSON에는 생체신호에서 파생된 값이 들어 있으므로 GitHub에 올리지 않는다.']
    (OUTPUT / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    enrollment_paths = selected['E']['enrollment']
    command = ' '.join(['python', 'enroll.py', '--db', 'results/enrollment/demo_initial.json',
                        'register', 'E', *(shlex.quote(str(path.relative_to(ROOT))) for path in enrollment_paths)])
    print('데모에서 E를 직접 등록하려면:', command)


if __name__ == '__main__':
    main()
