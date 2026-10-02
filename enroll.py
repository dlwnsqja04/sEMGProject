"""CSV 신호로 새 사용자를 등록하거나 등록된 사람을 확인한다."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from enrollment import EnrollmentError, Store, add_profile, calibrate, identify, load_store, save_store
from enrollment_features import Sample, extract_sample


ROOT = Path(__file__).resolve().parent
DEFAULT_STORE = ROOT / 'results' / 'enrollment' / 'profiles.json'


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description='손바닥 sEMG 신규 사용자 등록 및 확인')
    command.add_argument('--db', type=Path, default=DEFAULT_STORE, help='로컬 사용자 특징 저장소')
    actions = command.add_subparsers(dest='action', required=True)
    register = actions.add_parser('register', help='3개 이상 신호로 새 사람 등록')
    register.add_argument('user_id')
    register.add_argument('files', nargs='+', type=Path)
    calibration = actions.add_parser('calibrate', help='별도 검증 집합으로 판정 기준 보정')
    calibration.add_argument('split_file', type=Path, help='subject,file,split 열을 가진 CSV')
    identification = actions.add_parser('identify', help='등록된 사람 중 신호 주인 찾기')
    identification.add_argument('file', type=Path)
    verification = actions.add_parser('verify', help='주장한 사용자와 신호 비교')
    verification.add_argument('user_id')
    verification.add_argument('file', type=Path)
    actions.add_parser('show', help='등록된 ID와 보정 상태 보기')
    return command


def main() -> int:
    args = parser().parse_args()
    store = load_store(args.db) if args.db.exists() else Store()
    if args.action == 'register':
        samples = [extract_sample(path) for path in args.files]
        updated = add_profile(store, args.user_id, samples)
        save_store(args.db, updated)
        print(f'{args.user_id} 등록 완료: 시행 {len(samples)}개. 모델 재학습 없음.')
        return 0
    if args.action == 'calibrate':
        names = {profile.user_id for profile in store.profiles}
        validation: dict[str, list[Sample]] = {name: [] for name in names}
        with args.split_file.open(newline='', encoding='utf-8-sig') as stream:
            for row in csv.DictReader(stream):
                if row['split'] == 'validation' and row['subject'] in names:
                    validation[row['subject']].append(extract_sample(ROOT / row['file']))
        updated = calibrate(store, validation)
        save_store(args.db, updated)
        print(f'검증 신호로 거리 기준을 보정했습니다: {updated.threshold:.3f}')
        return 0
    if args.action == 'show':
        print('등록 사용자:', ', '.join(profile.user_id for profile in store.profiles) or '(없음)')
        print('판정 기준:', f'{store.threshold:.3f}' if store.threshold is not None else '보정 전')
        return 0
    if args.action == 'identify':
        decision = identify(store, extract_sample(args.file))
        print(f'식별 결과: {decision.user_id or "판정 보류"} '
              f'(거리 {decision.distance:.3f}, 기준 {decision.threshold:.3f})')
        return 0 if decision.user_id is not None else 1
    if args.action == 'verify':
        if args.user_id not in {profile.user_id for profile in store.profiles}:
            raise EnrollmentError(f'등록되지 않은 사용자입니다: {args.user_id}')
        decision = identify(store, extract_sample(args.file))
        accepted = decision.user_id == args.user_id
        print(f'인증 {"성공" if accepted else "실패"}: 주장 ID={args.user_id}, '
              f'식별={decision.user_id or "판정 보류"}, 거리={decision.distance:.3f}')
        return 0 if accepted else 1
    raise EnrollmentError(f'지원하지 않는 명령입니다: {args.action}')


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (EnrollmentError, FileNotFoundError) as error:
        print(f'오류: {error}')
        raise SystemExit(2) from error
