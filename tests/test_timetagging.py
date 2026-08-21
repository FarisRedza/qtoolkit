import qtoolkit
import numpy as np
import pytest

# get_twofold_coincidences

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

# get_threefold_coincidences

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

# get_fourfold_coincidences

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

# get_coincidences

def test_get_coincidences_multiple_pairs() -> None:
    timetags = np.array([
        1000,
        1050,
        2000,
        2050,
        3000,
        3050,
    ])

    channels = np.array([
        0,
        1,
        0,
        1,
        0,
        2,
    ])

    result = qtoolkit.get_coincidences(
        timetags=timetags,
        channels=channels,
        pairs=[
            (0, 1),
            (0, 2),
        ],
        coincidence_window=100,
    )

    assert result == {
        (0, 1): 2,
        (0, 2): 1,
    }

def test_get_coincidences_rejects_non_1d_timetags() -> None:
    with pytest.raises(
        ValueError,
        match='Timetags must be a 1D array',
    ):
        qtoolkit.get_coincidences(
            timetags=[[1000, 2000]],
            channels=[0, 1],
            pairs=[(0, 1)],
            coincidence_window=100,
        )


def test_get_coincidences_rejects_non_1d_channels() -> None:
    with pytest.raises(
        ValueError,
        match='Channels must be a 1D array',
    ):
        qtoolkit.get_coincidences(
            timetags=[1000, 2000],
            channels=[[0, 1]],
            pairs=[(0, 1)],
            coincidence_window=100,
        )


def test_get_coincidences_rejects_different_lengths() -> None:
    with pytest.raises(
        ValueError,
        match='must have the same length',
    ):
        qtoolkit.get_coincidences(
            timetags=[1000, 2000],
            channels=[0],
            pairs=[(0, 1)],
            coincidence_window=100,
        )

def test_get_coincidences_missing_channel() -> None:
    result = qtoolkit.get_coincidences(
        timetags=[1000, 2000],
        channels=[0, 0],
        pairs=[(0, 7)],
        coincidence_window=100,
    )

    assert result == {
        (0, 7): 0,
    }