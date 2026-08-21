import numpy as np
import pytest

import qtoolkit

# get_twofold_coincidences

def test_twofold_exact_match() -> None:
    result = qtoolkit.get_twofold_coincidences(
        tags_a=[1000],
        tags_b=[1000],
        coincidence_window=250,
    )

    assert result == 1


def test_twofold_at_window_boundary() -> None:
    result = qtoolkit.get_twofold_coincidences(
        tags_a=[1000],
        tags_b=[1250],
        coincidence_window=250,
    )

    assert result == 1


def test_twofold_outside_window() -> None:
    result = qtoolkit.get_twofold_coincidences(
        tags_a=[1000],
        tags_b=[1251],
        coincidence_window=250,
    )

    assert result == 0


@pytest.mark.parametrize(
    ('tags_a', 'tags_b'),
    [
        ([], [1000]),
        ([1000], []),
        ([], []),
    ],
)
def test_twofold_empty_input(
        tags_a,
        tags_b,
) -> None:
    result = qtoolkit.get_twofold_coincidences(
        tags_a=tags_a,
        tags_b=tags_b,
        coincidence_window=250,
    )

    assert result == 0


def test_twofold_multiple_coincidences() -> None:
    result = qtoolkit.get_twofold_coincidences(
        tags_a=[
            1000,
            2000,
            3000,
        ],
        tags_b=[
            1050,
            2050,
            3050,
        ],
        coincidence_window=100,
    )

    assert result == 3


def test_twofold_advances_first_channel() -> None:
    result = qtoolkit.get_twofold_coincidences(
        tags_a=[0, 1000],
        tags_b=[1000],
        coincidence_window=100,
    )

    assert result == 1


def test_twofold_advances_second_channel() -> None:
    result = qtoolkit.get_twofold_coincidences(
        tags_a=[1000],
        tags_b=[0, 1000],
        coincidence_window=100,
    )

    assert result == 1


def test_twofold_event_used_only_once() -> None:
    result = qtoolkit.get_twofold_coincidences(
        tags_a=[1000],
        tags_b=[950, 1050],
        coincidence_window=100,
    )

    assert result == 1


def test_twofold_accepts_numpy_arrays() -> None:
    tags_a = np.array(
        [1000, 2000],
        dtype=np.int64,
    )
    tags_b = np.array(
        [1050, 2050],
        dtype=np.int64,
    )

    result = qtoolkit.get_twofold_coincidences(
        tags_a=tags_a,
        tags_b=tags_b,
        coincidence_window=100,
    )

    assert result == 2

# get_threefold_coincidences

def test_threefold_exact_match() -> None:
    result = qtoolkit.get_threefold_coincidences(
        tags_a=[1000],
        tags_b=[1000],
        tags_c=[1000],
        coincidence_window=250,
    )

    assert result == 1


def test_threefold_at_window_boundary() -> None:
    result = qtoolkit.get_threefold_coincidences(
        tags_a=[1000],
        tags_b=[1100],
        tags_c=[1250],
        coincidence_window=250,
    )

    assert result == 1


def test_threefold_outside_window() -> None:
    result = qtoolkit.get_threefold_coincidences(
        tags_a=[1000],
        tags_b=[1100],
        tags_c=[1251],
        coincidence_window=250,
    )

    assert result == 0


def test_threefold_multiple_coincidences() -> None:
    result = qtoolkit.get_threefold_coincidences(
        tags_a=[1000, 2000],
        tags_b=[1050, 2050],
        tags_c=[1100, 2100],
        coincidence_window=250,
    )

    assert result == 2


@pytest.mark.parametrize(
    'early_index',
    [0, 1, 2],
)
def test_threefold_skips_early_event(
        early_index: int,
) -> None:
    timetags = [
        [1000],
        [1000],
        [1000],
    ]

    timetags[early_index] = [0, 1000]

    result = qtoolkit.get_threefold_coincidences(
        *timetags,
        coincidence_window=100,
    )

    assert result == 1

# get_fourfold_coincidences

def test_fourfold_exact_match() -> None:
    result = qtoolkit.get_fourfold_coincidences(
        tags_a=[1000],
        tags_b=[1000],
        tags_c=[1000],
        tags_d=[1000],
        coincidence_window=250,
    )

    assert result == 1


def test_fourfold_at_window_boundary() -> None:
    result = qtoolkit.get_fourfold_coincidences(
        tags_a=[1000],
        tags_b=[1050],
        tags_c=[1100],
        tags_d=[1250],
        coincidence_window=250,
    )

    assert result == 1


def test_fourfold_outside_window() -> None:
    result = qtoolkit.get_fourfold_coincidences(
        tags_a=[1000],
        tags_b=[1050],
        tags_c=[1100],
        tags_d=[1251],
        coincidence_window=250,
    )

    assert result == 0


def test_fourfold_multiple_coincidences() -> None:
    result = qtoolkit.get_fourfold_coincidences(
        tags_a=[1000, 2000],
        tags_b=[1050, 2050],
        tags_c=[1100, 2100],
        tags_d=[1150, 2150],
        coincidence_window=250,
    )

    assert result == 2


@pytest.mark.parametrize(
    'early_index',
    [0, 1, 2, 3],
)
def test_fourfold_skips_early_event(
        early_index: int,
) -> None:
    timetags = [
        [1000],
        [1000],
        [1000],
        [1000],
    ]

    timetags[early_index] = [0, 1000]

    result = qtoolkit.get_fourfold_coincidences(
        *timetags,
        coincidence_window=100,
    )

    assert result == 1

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


def test_get_coincidences_empty_pairs() -> None:
    result = qtoolkit.get_coincidences(
        timetags=[1000, 2000],
        channels=[0, 1],
        pairs=[],
        coincidence_window=100,
    )

    assert result == {}


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