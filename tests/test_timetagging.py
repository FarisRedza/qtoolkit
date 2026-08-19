import qkit
import numpy as np

def test_twofold_at_window_boundary():
    tags_a = np.array([1000], dtype=np.int64)
    tags_b = np.array([1250], dtype=np.int64)

    result = qkit.get_twofold_coincidences(
        tags_a,
        tags_b,
        coincidence_window=250,
    )

    assert result == 1

def test_twofold_outside_window():
    tags_a = np.array([1000], dtype=np.int64)
    tags_b = np.array([1251], dtype=np.int64)

    result = qkit.get_twofold_coincidences(
        tags_a,
        tags_b,
        coincidence_window=250,
    )

    assert result == 0

def test_twofold_empty():
    tags_a = np.array([], dtype=np.int64)
    tags_b = np.array([1000, 2000], dtype=np.int64)

    result = qkit.get_twofold_coincidences(
        tags_a,
        tags_b,
        coincidence_window=250,
    )

    assert result == 0

def test_twofold_accepts_lists():
    result = qkit.get_twofold_coincidences(
        [1000, 2000],
        [1100, 2100],
        coincidence_window=250,
    )

    assert result == 2