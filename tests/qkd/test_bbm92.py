import pytest

import qtoolkit

# BBM92ChannelMap

@pytest.fixture
def bbm92_channel_map() -> qtoolkit.qkd.BBM92ChannelMap:
    return qtoolkit.qkd.BBM92ChannelMap(
        first=qtoolkit.polarisation.PolarisationChannelMap(
            h=0,
            v=1,
            d=2,
            a=3,
            r=8,
            l=9,
        ),
        second=qtoolkit.polarisation.PolarisationChannelMap(
            h=4,
            v=5,
            d=6,
            a=7,
            r=10,
            l=11,
        ),
    )


def test_bbm92_zz_pairs(
        bbm92_channel_map: qtoolkit.qkd.BBM92ChannelMap,
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
        bbm92_channel_map: qtoolkit.qkd.BBM92ChannelMap,
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
        bbm92_channel_map: qtoolkit.qkd.BBM92ChannelMap,
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
    channel_map = qtoolkit.qkd.BBM92ChannelMap(
        first=qtoolkit.polarisation.PolarisationChannelMap(),
        second=qtoolkit.polarisation.PolarisationChannelMap(),
    )

    with pytest.raises(
        ValueError,
        match=f'define the {basis} basis',
    ):
        getattr(channel_map, attribute)


def test_processed_timetag_data_get_basis_metrics() -> None:
    processed = qtoolkit.timetags.ProcessedTimetagData(
        coincidences={
            (0, 4): 450,
            (0, 5): 25,
            (1, 4): 25,
            (1, 5): 500,
        },
        coincidence_window=250,
    )

    pairs = (
        qtoolkit.timetags.ChannelPair(0, 4),
        qtoolkit.timetags.ChannelPair(0, 5),
        qtoolkit.timetags.ChannelPair(1, 4),
        qtoolkit.timetags.ChannelPair(1, 5),
    )

    result = processed.get_basis_metrics(
        pairs
    )

    assert result == qtoolkit.qkd.BasisMetrics(
        c_00=450,
        c_01=25,
        c_10=25,
        c_11=500,
    )


def test_bbm92_metrics_from_processed_data(
        bbm92_channel_map: qtoolkit.qkd.BBM92ChannelMap,
) -> None:
    processed = qtoolkit.timetags.ProcessedTimetagData(
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

    result = qtoolkit.qkd.BBM92Metrics.from_processed_data(
        processed=processed,
        channel_map=bbm92_channel_map,
    )

    assert result.zz == qtoolkit.qkd.BasisMetrics(
        c_00=450,
        c_01=25,
        c_10=25,
        c_11=500,
    )

    assert result.xx == qtoolkit.qkd.BasisMetrics(
        c_00=425,
        c_01=50,
        c_10=50,
        c_11=475,
    )

    assert result.zz.qber == pytest.approx(0.05)
    assert result.xx.qber == pytest.approx(0.10)
    assert result.fidelity == pytest.approx(0.85)


def test_bbm92_metrics_fidelity() -> None:
    metrics = qtoolkit.qkd.BBM92Metrics(
        zz=qtoolkit.qkd.BasisMetrics(
            c_00=450,
            c_01=25,
            c_10=25,
            c_11=500,
        ),
        xx=qtoolkit.qkd.BasisMetrics(
            c_00=425,
            c_01=50,
            c_10=50,
            c_11=475,
        ),
    )

    assert metrics.fidelity == pytest.approx(0.85)


def test_bbm92_metrics_string() -> None:
    metrics = qtoolkit.qkd.BBM92Metrics(
        zz=qtoolkit.qkd.BasisMetrics(
            c_00=450,
            c_01=25,
            c_10=25,
            c_11=500,
        ),
        xx=qtoolkit.qkd.BasisMetrics(
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