import qtoolkit
import numpy as np

def test_twofold_at_window_boundary():
    tags_a = np.asarray([1000], dtype=np.int64)
    tags_b = np.asarray([1250], dtype=np.int64)

    result = qtoolkit.get_twofold_coincidences(
        tags_a,
        tags_b,
        coincidence_window=250,
    )

    assert result == 1

def test_twofold_outside_window():
    tags_a = np.asarray([1000], dtype=np.int64)
    tags_b = np.asarray([1251], dtype=np.int64)

    result = qtoolkit.get_twofold_coincidences(
        tags_a,
        tags_b,
        coincidence_window=250,
    )

    assert result == 0

def test_twofold_empty():
    tags_a = np.asarray([], dtype=np.int64)
    tags_b = np.asarray([1000, 2000], dtype=np.int64)

    result = qtoolkit.get_twofold_coincidences(
        tags_a,
        tags_b,
        coincidence_window=250,
    )

    assert result == 0

def test_twofold_accepts_lists():
    result = qtoolkit.get_twofold_coincidences(
        [1000, 2000],
        [1100, 2100],
        coincidence_window=250,
    )

    assert result == 2

def test_threefold_basic():
    tags_a = np.asarray([1000, 2000], dtype=np.int64)
    tags_b = np.asarray([1050, 2050], dtype=np.int64)
    tags_c = np.asarray([1100, 2100], dtype=np.int64)

    result = qtoolkit.get_threefold_coincidences(
        tags_a,
        tags_b,
        tags_c,
        coincidence_window=250,
    )

    assert result == 2

def test_threefold_requires_all_events_within_window():
    tags_a = np.asarray([1000], dtype=np.int64)
    tags_b = np.asarray([1200], dtype=np.int64)
    tags_c = np.asarray([1400], dtype=np.int64)

    result = qtoolkit.get_threefold_coincidences(
        tags_a,
        tags_b,
        tags_c,
        coincidence_window=250,
    )

    assert result == 0

def test_fourfold_basic():
    tags_a = np.asarray([1000], dtype=np.int64)
    tags_b = np.asarray([1050], dtype=np.int64)
    tags_c = np.asarray([1100], dtype=np.int64)
    tags_d = np.asarray([1200], dtype=np.int64)

    result = qtoolkit.get_fourfold_coincidences(
        tags_a,
        tags_b,
        tags_c,
        tags_d,
        coincidence_window=250,
    )

    assert result == 1

def test_fourfold_outside_window():
    tags_a = np.asarray([1000], dtype=np.int64)
    tags_b = np.asarray([1050], dtype=np.int64)
    tags_c = np.asarray([1100], dtype=np.int64)
    tags_d = np.asarray([1300], dtype=np.int64)

    result = qtoolkit.get_fourfold_coincidences(
        tags_a,
        tags_b,
        tags_c,
        tags_d,
        coincidence_window=250,
    )

    assert result == 0