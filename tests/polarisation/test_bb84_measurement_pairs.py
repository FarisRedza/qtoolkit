import pytest

from qtoolkit.polarisation.channels import PolarisationChannelMap
from qtoolkit.polarisation.measurement import BB84Measurement, BB84MeasurementPair
from qtoolkit.qkd.metrics import BasisMetrics


@pytest.fixture
def pair():
    return BB84MeasurementPair(
        BB84Measurement(PolarisationChannelMap(h=0, v=1, d=2, a=3)),
        BB84Measurement(PolarisationChannelMap(h=4, v=5, d=6, a=7)),
    )


@pytest.mark.parametrize('first,second,expected,names', [
    ('Z', 'Z', [(0, 4), (0, 5), (1, 4), (1, 5)], ['HH', 'HV', 'VH', 'VV']),
    ('Z', 'X', [(0, 6), (0, 7), (1, 6), (1, 7)], ['HD', 'HA', 'VD', 'VA']),
    ('X', 'Z', [(2, 4), (2, 5), (3, 4), (3, 5)], ['DH', 'DV', 'AH', 'AV']),
    ('X', 'X', [(2, 6), (2, 7), (3, 6), (3, 7)], ['DD', 'DA', 'AD', 'AA']),
])
def test_pairs(pair, first, second, expected, names):
    result = pair.pairs(first, second)
    assert [p.as_tuple() for p in result] == expected
    assert [p.name for p in result] == names
    assert pair.pairs(first.lower(), second.lower()) == result
    counts = dict(zip(expected, (20, 2, 3, 15)))
    assert BasisMetrics.from_coincidences(counts, result).correlation == pytest.approx(0.75)


def test_legacy_properties_and_all_pairs(pair):
    assert pair.z_pairs == pair.pairs('Z', 'Z')
    assert pair.x_pairs == pair.pairs('X', 'X')
    assert pair.coincidence_pairs == (*pair.z_pairs, *pair.x_pairs)
    assert pair.all_pairs == (
        *pair.pairs('Z', 'Z'), *pair.pairs('Z', 'X'),
        *pair.pairs('X', 'Z'), *pair.pairs('X', 'X'),
    )
    assert len({p.as_tuple() for p in pair.all_pairs}) == 16


@pytest.mark.parametrize('first,second', [('Y', 'Z'), ('Z', 'Y'), ('', 'X')])
def test_invalid_basis(pair, first, second):
    with pytest.raises(ValueError, match='Supported BB84 bases'):
        pair.pairs(first, second)


def test_missing_channel():
    pair = BB84MeasurementPair(
        BB84Measurement(PolarisationChannelMap(h=0, v=1)),
        BB84Measurement(PolarisationChannelMap(d=2, a=3)),
    )
    assert len(pair.pairs('Z', 'X')) == 4
    with pytest.raises(ValueError, match='X-basis'):
        pair.pairs('X', 'Z')
