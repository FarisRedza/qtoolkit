import pathlib

import numpy as np
import pytest

import qtoolkit

# TimetagData

@pytest.fixture
def timetag_data() -> qtoolkit.TimetagData:
    return qtoolkit.TimetagData(
        timetags=np.array(
            [
                1000,
                1050,
                2000,
                2050,
                3000,
                3050,
            ],
            dtype=np.int64,
        ),
        channels=np.array(
            [
                0,
                4,
                0,
                5,
                1,
                4,
            ],
            dtype=np.int8,
        ),
    )


# BBM92ChannelMap

@pytest.fixture
def bbm92_channel_map() -> qtoolkit.BBM92ChannelMap:
    return qtoolkit.BBM92ChannelMap(
        first=qtoolkit.PolarisationChannelMap(
            h=0,
            v=1,
            d=2,
            a=3,
            r=8,
            l=9,
        ),
        second=qtoolkit.PolarisationChannelMap(
            h=4,
            v=5,
            d=6,
            a=7,
            r=10,
            l=11,
        ),
    )

# ChannelPair

def test_channel_pair_as_tuple() -> None:
    pair = qtoolkit.ChannelPair(
        first=1,
        second=4,
        name='test',
    )

    assert pair.as_tuple() == (1, 4)

# ChannelMap

def test_channel_map_getitem() -> None:
    channel_map = qtoolkit.ChannelMap({
        'signal': 1,
        'idler': 5,
    })

    assert channel_map['signal'] == 1


def test_channel_map_get() -> None:
    channel_map = qtoolkit.ChannelMap({
        'signal': 1,
        'idler': 5,
    })

    assert channel_map.get('idler') == 5


def test_channel_map_get_missing() -> None:
    channel_map = qtoolkit.ChannelMap({
        'signal': 1,
    })

    assert channel_map.get('missing') is None


def test_channel_map_names() -> None:
    channel_map = qtoolkit.ChannelMap({
        'signal': 1,
        'idler': 5,
    })

    assert channel_map.names == (
        'signal',
        'idler',
    )


def test_channel_map_numbers() -> None:
    channel_map = qtoolkit.ChannelMap({
        'signal': 1,
        'idler': 5,
    })

    assert channel_map.numbers == (
        1,
        5,
    )

# PolarisationChannelMap

def test_polarisation_channel_map_channels() -> None:
    channel_map = qtoolkit.PolarisationChannelMap(
        h=0,
        v=1,
        r=4,
        l=5,
    )

    assert channel_map.channels == (
        0,
        1,
        4,
        5,
    )


def test_polarisation_channel_map_ignores_none() -> None:
    channel_map = qtoolkit.PolarisationChannelMap(
        h=0,
        v=1,
    )

    assert channel_map.channels == (
        0,
        1,
    )


def test_polarisation_channel_map_as_channel_map() -> None:
    channel_map = qtoolkit.PolarisationChannelMap(
        h=0,
        v=1,
        d=2,
        a=3,
    )

    result = channel_map.as_channel_map()

    assert result.channels == {
        'H': 0,
        'V': 1,
        'D': 2,
        'A': 3,
    }

# TimetagData

def test_timetag_data_length(
        timetag_data: qtoolkit.TimetagData,
) -> None:
    assert len(timetag_data) == 6


def test_timetag_data_duration(
        timetag_data: qtoolkit.TimetagData,
) -> None:
    assert timetag_data.duration_ps == 2050


@pytest.mark.parametrize(
    'number_of_timetags',
    [
        0,
        1,
    ],
)
def test_timetag_data_short_duration_is_zero(
        number_of_timetags: int,
) -> None:
    data = qtoolkit.TimetagData(
        timetags=np.arange(
            number_of_timetags,
            dtype=np.int64,
        ),
        channels=np.zeros(
            number_of_timetags,
            dtype=np.int8,
        ),
    )

    assert data.duration_ps == 0


def test_timetag_data_get_channel_timetags(
        timetag_data: qtoolkit.TimetagData,
) -> None:
    result = timetag_data.get_channel_timetags(0)

    np.testing.assert_array_equal(
        result,
        np.array(
            [1000, 2000],
            dtype=np.int64,
        ),
    )


def test_timetag_data_get_missing_channel(
        timetag_data: qtoolkit.TimetagData,
) -> None:
    result = timetag_data.get_channel_timetags(100)

    assert len(result) == 0


def test_timetag_data_count_all(
        timetag_data: qtoolkit.TimetagData,
) -> None:
    assert timetag_data.count() == 6


def test_timetag_data_count_channel(
        timetag_data: qtoolkit.TimetagData,
) -> None:
    assert timetag_data.count(0) == 2


def test_timetag_data_count_missing_channel(
        timetag_data: qtoolkit.TimetagData,
) -> None:
    assert timetag_data.count(100) == 0


def test_timetag_data_select_channels(
        timetag_data: qtoolkit.TimetagData,
) -> None:
    result = timetag_data.select_channels(
        0,
        1,
    )

    np.testing.assert_array_equal(
        result.timetags,
        np.array(
            [1000, 2000, 3000],
            dtype=np.int64,
        ),
    )

    np.testing.assert_array_equal(
        result.channels,
        np.array(
            [0, 0, 1],
            dtype=np.int8,
        ),
    )


def test_timetag_data_rejects_non_1d_timetags() -> None:
    with pytest.raises(
        ValueError,
        match='timetags must be one-dimensional',
    ):
        qtoolkit.TimetagData(
            timetags=np.array([
                [1, 2],
            ]),
            channels=np.array([
                0,
                1,
            ]),
        )


def test_timetag_data_rejects_non_1d_channels() -> None:
    with pytest.raises(
        ValueError,
        match='channels must be one-dimensional',
    ):
        qtoolkit.TimetagData(
            timetags=np.array([
                1,
                2,
            ]),
            channels=np.array([
                [0, 1],
            ]),
        )


def test_timetag_data_rejects_different_lengths() -> None:
    with pytest.raises(
        ValueError,
        match='must have the same length',
    ):
        qtoolkit.TimetagData(
            timetags=np.array([
                1,
                2,
            ]),
            channels=np.array([
                0,
            ]),
        )


def test_timetag_data_from_file(
        tmp_path: pathlib.Path,
) -> None:
    file_path = tmp_path.joinpath('timetags.txt')

    file_path.write_text(
        '941575226770542 6\n'
        '941575227172420 0\n'
        '941575227234390 6\n'
        '941575227678840 6\n'
    )

    result = qtoolkit.TimetagData.from_file(
        file_path
    )

    np.testing.assert_array_equal(
        result.timetags,
        np.array(
            [
                941575226770542,
                941575227172420,
                941575227234390,
                941575227678840,
            ],
            dtype=np.int64,
        ),
    )

    np.testing.assert_array_equal(
        result.channels,
        np.array(
            [6, 0, 6, 6],
            dtype=np.int8,
        ),
    )

    assert result.file_path == file_path


def test_timetag_data_from_string_path(
        tmp_path: pathlib.Path,
) -> None:
    file_path = tmp_path.joinpath('timetags.txt')
    file_path.write_text('1000 0\n')

    result = qtoolkit.TimetagData.from_file(
        str(file_path)
    )

    assert result.file_path == file_path


def test_timetag_data_from_single_row(
        tmp_path: pathlib.Path,
) -> None:
    file_path = tmp_path.joinpath('timetags.txt')
    file_path.write_text('1000 3\n')

    result = qtoolkit.TimetagData.from_file(
        file_path
    )

    np.testing.assert_array_equal(
        result.timetags,
        np.array(
            [1000],
            dtype=np.int64,
        ),
    )

    np.testing.assert_array_equal(
        result.channels,
        np.array(
            [3],
            dtype=np.int8,
        ),
    )


def test_timetag_data_from_file_rejects_extra_columns(
        tmp_path: pathlib.Path,
) -> None:
    file_path = tmp_path / 'invalid.txt'
    file_path.write_text(
        '1000 0 123\n'
    )

    with pytest.raises(
        ValueError,
        match='exactly two columns',
    ):
        qtoolkit.TimetagData.from_file(
            file_path
        )

# ProcessedTimetagData

def test_processed_timetag_data_from_timetag_data() -> None:
    data = qtoolkit.TimetagData(
        timetags=np.array(
            [
                1000,
                1050,
                2000,
                2050,
            ],
            dtype=np.int64,
        ),
        channels=np.array(
            [
                0,
                1,
                0,
                1,
            ],
            dtype=np.int8,
        ),
    )

    result = (
        qtoolkit.ProcessedTimetagData
        .from_timetag_data(
            timetag_data=data,
            pairs=[
                qtoolkit.ChannelPair(
                    first=0,
                    second=1,
                ),
            ],
            coincidence_window=100,
        )
    )

    assert result.coincidences == {
        (0, 1): 2,
    }

    assert result.coincidence_window == 100


def test_processed_timetag_data_accepts_tuple_pairs() -> None:
    data = qtoolkit.TimetagData(
        timetags=np.array(
            [
                1000,
                1050,
            ],
            dtype=np.int64,
        ),
        channels=np.array(
            [
                0,
                1,
            ],
            dtype=np.int8,
        ),
    )

    result = (
        qtoolkit.ProcessedTimetagData
        .from_timetag_data(
            timetag_data=data,
            pairs=[
                (0, 1),
            ],
            coincidence_window=100,
        )
    )

    assert result.coincidences == {
        (0, 1): 1,
    }


def test_processed_timetag_data_from_file(
        tmp_path: pathlib.Path,
) -> None:
    file_path = tmp_path.joinpath('timetags.txt')

    file_path.write_text(
        '1000 0\n'
        '1050 1\n'
        '2000 0\n'
        '2050 1\n'
    )

    result = (
        qtoolkit.ProcessedTimetagData
        .from_file(
            file_path=file_path,
            pairs=[
                qtoolkit.ChannelPair(
                    first=0,
                    second=1,
                ),
            ],
            coincidence_window=100,
        )
    )

    assert result.coincidences == {
        (0, 1): 2,
    }

    assert result.coincidence_window == 100
    assert result.file_path == file_path

# BasisMetrics

def test_basis_metrics_from_coincidences() -> None:
    pairs = (
        qtoolkit.ChannelPair(0, 4),
        qtoolkit.ChannelPair(0, 5),
        qtoolkit.ChannelPair(1, 4),
        qtoolkit.ChannelPair(1, 5),
    )

    coincidences = {
        (0, 4): 450,
        (0, 5): 25,
        (1, 4): 25,
        (1, 5): 500,
    }

    result = qtoolkit.BasisMetrics.from_coincidences(
        coincidences=coincidences,
        pairs=pairs,
    )

    assert result == qtoolkit.BasisMetrics(
        c_00=450,
        c_01=25,
        c_10=25,
        c_11=500,
    )


def test_basis_metrics_counts() -> None:
    metrics = qtoolkit.BasisMetrics(
        c_00=450,
        c_01=25,
        c_10=25,
        c_11=500,
    )

    assert metrics.odd == 50
    assert metrics.even == 950
    assert metrics.total == 1000


def test_basis_metrics_probabilities() -> None:
    metrics = qtoolkit.BasisMetrics(
        c_00=450,
        c_01=25,
        c_10=25,
        c_11=500,
    )

    assert (
        metrics.even_probability
        == pytest.approx(0.95)
    )

    assert (
        metrics.odd_probability
        == pytest.approx(0.05)
    )


def test_basis_metrics_empty_probabilities() -> None:
    metrics = qtoolkit.BasisMetrics(
        c_00=0,
        c_01=0,
        c_10=0,
        c_11=0,
    )

    assert np.isnan(metrics.even_probability)
    assert np.isnan(metrics.odd_probability)


def test_basis_metrics_qber() -> None:
    metrics = qtoolkit.BasisMetrics(
        c_00=450,
        c_01=25,
        c_10=25,
        c_11=500,
    )

    assert metrics.qber == pytest.approx(0.05)


def test_basis_metrics_visibility() -> None:
    metrics = qtoolkit.BasisMetrics(
        c_00=450,
        c_01=25,
        c_10=25,
        c_11=500,
    )

    assert metrics.visibility == pytest.approx(0.9)


def test_basis_metrics_as_row() -> None:
    metrics = qtoolkit.BasisMetrics(
        c_00=450,
        c_01=25,
        c_10=25,
        c_11=500,
    )

    assert metrics.as_row() == pytest.approx([
        450,
        25,
        25,
        500,
        50,
        950,
        1000,
        0.95,
        0.05,
        0.9,
    ])

# BBM92ChannelMap

def test_bbm92_zz_pairs(
        bbm92_channel_map: qtoolkit.BBM92ChannelMap,
) -> None:
    assert tuple(
        pair.as_tuple()
        for pair in bbm92_channel_map.zz_pairs
    ) == (
        (0, 4),
        (0, 5),
        (1, 4),
        (1, 5),
    )


def test_bbm92_xx_pairs(
        bbm92_channel_map: qtoolkit.BBM92ChannelMap,
) -> None:
    assert tuple(
        pair.as_tuple()
        for pair in bbm92_channel_map.xx_pairs
    ) == (
        (2, 6),
        (2, 7),
        (3, 6),
        (3, 7),
    )


def test_bbm92_yy_pairs(
        bbm92_channel_map: qtoolkit.BBM92ChannelMap,
) -> None:
    assert tuple(
        pair.as_tuple()
        for pair in bbm92_channel_map.yy_pairs
    ) == (
        (8, 10),
        (8, 11),
        (9, 10),
        (9, 11),
    )


@pytest.mark.parametrize(
    ('attribute', 'basis'),
    [
        ('zz_pairs', 'Z'),
        ('xx_pairs', 'X'),
        ('yy_pairs', 'Y'),
    ],
)
def test_bbm92_missing_basis(
        attribute: str,
        basis: str,
) -> None:
    channel_map = qtoolkit.BBM92ChannelMap(
        first=qtoolkit.PolarisationChannelMap(),
        second=qtoolkit.PolarisationChannelMap(),
    )

    with pytest.raises(
        ValueError,
        match=f'define the {basis} basis',
    ):
        getattr(channel_map, attribute)


def test_processed_timetag_data_get_basis_metrics() -> None:
    processed = qtoolkit.ProcessedTimetagData(
        coincidences={
            (0, 4): 450,
            (0, 5): 25,
            (1, 4): 25,
            (1, 5): 500,
        },
        coincidence_window=250,
    )

    pairs = (
        qtoolkit.ChannelPair(0, 4),
        qtoolkit.ChannelPair(0, 5),
        qtoolkit.ChannelPair(1, 4),
        qtoolkit.ChannelPair(1, 5),
    )

    result = processed.get_basis_metrics(
        pairs
    )

    assert result == qtoolkit.BasisMetrics(
        c_00=450,
        c_01=25,
        c_10=25,
        c_11=500,
    )


def test_bbm92_metrics_from_processed_data(
        bbm92_channel_map: qtoolkit.BBM92ChannelMap,
) -> None:
    processed = qtoolkit.ProcessedTimetagData(
        coincidences={
            (0, 4): 450,
            (0, 5): 25,
            (1, 4): 25,
            (1, 5): 500,
            (2, 6): 425,
            (2, 7): 50,
            (3, 6): 50,
            (3, 7): 475,
        },
        coincidence_window=250,
    )

    result = qtoolkit.BBM92Metrics.from_processed_data(
        processed=processed,
        channel_map=bbm92_channel_map,
    )

    assert result.zz == qtoolkit.BasisMetrics(
        c_00=450,
        c_01=25,
        c_10=25,
        c_11=500,
    )

    assert result.xx == qtoolkit.BasisMetrics(
        c_00=425,
        c_01=50,
        c_10=50,
        c_11=475,
    )

    assert result.zz.qber == pytest.approx(0.05)
    assert result.xx.qber == pytest.approx(0.10)
    assert result.fidelity == pytest.approx(0.85)


def test_bbm92_metrics_fidelity() -> None:
    metrics = qtoolkit.BBM92Metrics(
        zz=qtoolkit.BasisMetrics(
            c_00=450,
            c_01=25,
            c_10=25,
            c_11=500,
        ),
        xx=qtoolkit.BasisMetrics(
            c_00=425,
            c_01=50,
            c_10=50,
            c_11=475,
        ),
    )

    assert metrics.fidelity == pytest.approx(0.85)


def test_bbm92_metrics_string() -> None:
    metrics = qtoolkit.BBM92Metrics(
        zz=qtoolkit.BasisMetrics(
            c_00=450,
            c_01=25,
            c_10=25,
            c_11=500,
        ),
        xx=qtoolkit.BasisMetrics(
            c_00=425,
            c_01=50,
            c_10=50,
            c_11=475,
        ),
    )

    result = str(metrics)

    assert 'Basis' in result
    assert 'ZZ' in result
    assert 'XX' in result
    assert '0.050000' in result
    assert '0.100000' in result
    assert '0.850000' in result