"""가변 윈도우, 시행 분리, 시행별 확률 평균을 확인한다."""
import numpy as np

from experiments.data import partition, window_signal
from experiments.training import score_predictions


def main():
    x = np.arange(6000).reshape(3000, 2)
    for width in (100, 200, 300, 400, 600):
        for hop in (width // 4, width // 2, width):
            windows = window_signal(x, width, hop)
            assert windows.shape == ((3000 - width) // hop + 1, width, 2)
            np.testing.assert_array_equal(windows[-1], x[(len(windows)-1)*hop:(len(windows)-1)*hop+width])
    rows = partition()
    for split, expected in [('train', 160), ('val', 40), ('test', 50)]:
        subset = [r for r in rows if r['split'] == split]
        assert len(subset) == expected
        assert all(sum(r['label'] == c for r in subset) == expected // 5 for c in range(5))
    assert len({r['file'] for r in rows}) == 250
    hashes = {s: {r['sha256'] for r in rows if r['split'] == s} for s in ('train', 'val', 'test')}
    assert not hashes['train'] & hashes['val']
    assert not hashes['train'] & hashes['test']
    assert not hashes['val'] & hashes['test']
    assert len(hashes['test']) == 49  # E의 테스트 파일 두 개가 동일한 내용이다.
    probabilities = np.array([[.8, .2, 0, 0, 0], [.4, .6, 0, 0, 0], [.1, .9, 0, 0, 0], [.2, .8, 0, 0, 0]])
    scores = score_predictions(probabilities, np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1]))
    assert scores['trial_accuracy'] == 1
    assert scores['window_accuracy'] == .75
    print('PASS: 15 window configurations, balanced/disjoint trial split, probability aggregation')


if __name__ == '__main__':
    main()
