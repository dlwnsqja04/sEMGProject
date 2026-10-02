"""사용자별 sEMG 템플릿 저장, 등록, 거리 보정과 식별."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Final

import numpy as np
from sklearn.preprocessing import StandardScaler

from enrollment_features import FEATURE_COUNT, FEATURE_VERSION, Sample


STORE_VERSION: Final = 'enrollment-store-v1'


@dataclass(frozen=True)
class EnrollmentError(Exception):
    detail: str

    def __str__(self) -> str:
        return self.detail


@dataclass(frozen=True)
class Profile:
    user_id: str
    samples: tuple[Sample, ...]
    radius: float | None = None


@dataclass(frozen=True)
class Store:
    profiles: tuple[Profile, ...] = ()
    center: tuple[float, ...] = ()
    scale: tuple[float, ...] = ()
    threshold: float | None = None


@dataclass(frozen=True)
class Decision:
    user_id: str | None
    nearest_user: str
    distance: float
    threshold: float


def add_profile(store: Store, user_id: str, samples: list[Sample]) -> Store:
    """새 사람의 등록 신호만 추가한다. 기존 정규화와 모델은 갱신하지 않는다."""
    name = user_id.strip()
    if not name or len(name) > 64:
        raise EnrollmentError('사용자 ID는 1~64자여야 합니다.')
    if any(profile.user_id == name for profile in store.profiles):
        raise EnrollmentError(f'이미 등록된 사용자입니다: {name}')
    if len(samples) < 3:
        raise EnrollmentError('서로 다른 시행 파일이 최소 3개 필요합니다.')
    existing = {item.digest for profile in store.profiles for item in profile.samples}
    digests = [item.digest for item in samples]
    if len(set(digests)) != len(digests) or existing.intersection(digests):
        raise EnrollmentError('등록 파일에 중복 신호가 있거나 다른 사용자에게 이미 등록됐습니다.')
    for item in samples:
        if len(item.features) != FEATURE_COUNT or not np.isfinite(item.features).all():
            raise EnrollmentError('등록 신호의 특징 크기 또는 값이 올바르지 않습니다.')
    radius = None
    if store.threshold is not None:
        enrolled = (np.asarray([item.features for item in samples]) - np.asarray(store.center)) / np.asarray(store.scale)
        leave_one_out = [float(np.linalg.norm(enrolled[index] -
                          np.delete(enrolled, index, axis=0).mean(axis=0)))
                         for index in range(len(samples))]
        radius = max(store.threshold, max(leave_one_out) * 1.5)
    profile = Profile(name, tuple(samples), radius)
    return replace(store, profiles=(*store.profiles, profile))


def _templates(store: Store) -> dict[str, np.ndarray]:
    center = np.asarray(store.center)
    scale = np.asarray(store.scale)
    return {profile.user_id:
            ((np.asarray([item.features for item in profile.samples]) - center) / scale).mean(axis=0)
            for profile in store.profiles}


def calibrate(store: Store, validation: dict[str, list[Sample]]) -> Store:
    """등록과 겹치지 않는 검증 신호로 거리 임계값을 정한다."""
    if len(store.profiles) < 2:
        raise EnrollmentError('보정에는 등록된 사용자 2명 이상이 필요합니다.')
    names = {profile.user_id for profile in store.profiles}
    if set(validation) != names or any(not samples for samples in validation.values()):
        raise EnrollmentError('등록된 사용자마다 별도의 검증 신호가 필요합니다.')
    used = {item.digest for profile in store.profiles for item in profile.samples}
    if any(item.digest in used for samples in validation.values() for item in samples):
        raise EnrollmentError('등록 신호와 검증 신호가 겹칩니다.')
    training = np.asarray([item.features for profile in store.profiles for item in profile.samples])
    scaler = StandardScaler().fit(training)
    calibrated = replace(store, center=tuple(float(value) for value in scaler.mean_),
                         scale=tuple(float(value) for value in scaler.scale_))
    templates = _templates(calibrated)
    genuine: dict[str, list[float]] = {name: [] for name in names}
    for name, samples in validation.items():
        for sample in samples:
            feature = (np.asarray(sample.features) - scaler.mean_) / scaler.scale_
            genuine[name].append(float(np.linalg.norm(feature - templates[name])))
    radii = {name: float(np.quantile(distances, 0.95) * 1.05)
             for name, distances in genuine.items()}
    profiles = tuple(replace(profile, radius=radii[profile.user_id])
                     for profile in calibrated.profiles)
    return replace(calibrated, profiles=profiles,
                   threshold=float(np.median(list(radii.values()))))


def identify(store: Store, sample: Sample) -> Decision:
    """임계값 이내의 가장 가까운 등록자를 반환하고, 아니면 보류한다."""
    if store.threshold is None or len(store.center) != FEATURE_COUNT:
        raise EnrollmentError('먼저 별도 검증 신호로 등록 저장소를 보정해야 합니다.')
    feature = (np.asarray(sample.features) - np.asarray(store.center)) / np.asarray(store.scale)
    distances = {name: float(np.linalg.norm(feature - template))
                 for name, template in _templates(store).items()}
    nearest = min(distances, key=distances.get)
    distance = distances[nearest]
    profile = next(item for item in store.profiles if item.user_id == nearest)
    radius = profile.radius if profile.radius is not None else store.threshold
    return Decision(nearest if distance <= radius else None,
                    nearest, distance, radius)


def save_store(path: Path, store: Store) -> None:
    """특징 템플릿을 로컬 JSON에 저장한다. 원본 CSV는 저장하지 않는다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {'version': STORE_VERSION, 'feature_version': FEATURE_VERSION, **asdict(store)}
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def load_store(path: Path) -> Store:
    """저장된 특징의 형식과 버전을 확인한 뒤 읽는다."""
    try:
        document = json.loads(path.read_text(encoding='utf-8'))
        if document['version'] != STORE_VERSION or document['feature_version'] != FEATURE_VERSION:
            raise EnrollmentError('등록 특징 버전이 현재 코드와 다릅니다.')
        profiles = tuple(Profile(str(profile['user_id']),
                                 tuple(Sample(str(item['digest']), tuple(float(v) for v in item['features']))
                                       for item in profile['samples']),
                                 None if profile.get('radius') is None else float(profile['radius']))
                         for profile in document['profiles'])
        center = tuple(float(value) for value in document['center'])
        scale = tuple(float(value) for value in document['scale'])
        threshold = None if document['threshold'] is None else float(document['threshold'])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise EnrollmentError(f'등록 저장소 형식이 잘못됐습니다: {path}') from error
    if any(len(item.features) != FEATURE_COUNT or not np.isfinite(item.features).all()
           for profile in profiles for item in profile.samples):
        raise EnrollmentError('저장된 특징 크기 또는 값이 올바르지 않습니다.')
    if center and (len(center) != FEATURE_COUNT or len(scale) != FEATURE_COUNT
                   or not np.isfinite(center).all() or not np.isfinite(scale).all() or min(scale) <= 0):
        raise EnrollmentError('저장된 정규화 값이 올바르지 않습니다.')
    if threshold is not None and (not np.isfinite(threshold) or threshold < 0 or not center):
        raise EnrollmentError('저장된 거리 임계값이 올바르지 않습니다.')
    if any(profile.radius is not None and (not np.isfinite(profile.radius) or profile.radius < 0)
           for profile in profiles):
        raise EnrollmentError('사용자별 거리 임계값이 올바르지 않습니다.')
    return Store(profiles, center, scale, threshold)
