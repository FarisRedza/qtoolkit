import dataclasses

import numpy as np
import pytest

from qtoolkit.timetags import ChannelPair, MeasurementCounts, TimetagData


def test_measurement_counts_from_timetags():
    raw = TimetagData(
        timetags=np.array([100, 120, 500], dtype=np.int64),
        channels=np.array([0, 2, 0], dtype=np.int8),
        start_ps=0,
        stop_ps=1000000000000,
    )
    pairs = [ChannelPair(0, 2), ChannelPair(1, 2)]
    result = MeasurementCounts.from_timetag_data(raw, pairs, 50)
    assert dict(result.singles) == {0: 2, 1: 0, 2: 1}
    assert dict(result.coincidences) == {(0, 2): 1, (1, 2): 0}
    assert result.duration_s == 1.0
    assert (9, 2) not in result.coincidences


def test_explicit_singles_channels_and_unknown_duration():
    raw = TimetagData(np.array([], dtype=np.int64), np.array([], dtype=np.int8))
    result = MeasurementCounts.from_timetag_data(raw, [], 10, channels=[3, 4])
    assert dict(result.singles) == {3: 0, 4: 0}
    assert result.duration_s is None


def test_basis_metrics_and_missing_outcome():
    pairs = tuple(ChannelPair(a, b) for a, b in [(0, 2), (0, 3), (1, 2), (1, 3)])
    result = MeasurementCounts({0: 4}, {p.as_tuple(): n for p, n in zip(pairs, (8, 1, 1, 0))}, 1.0, 50)
    assert result.get_basis_metrics(pairs).correlation == pytest.approx(0.6)
    with pytest.raises(KeyError):
        result.get_basis_metrics(pairs[:3] + (ChannelPair(1, 9),))


def test_counts_are_immutable_and_copied():
    singles = {0: 1}
    coincidences = {(0, 1): 2}
    result = MeasurementCounts(singles, coincidences, None, 10)
    singles[0] = 5
    coincidences[(0, 1)] = 9
    assert result.singles[0] == 1
    assert result.coincidences[(0, 1)] == 2
    with pytest.raises(TypeError):
        result.singles[0] = 3
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.duration_s = 2.0


@pytest.mark.parametrize('kwargs', [
    {'singles': {0: -1}},
    {'coincidences': {(0, 1): -1}},
    {'coincidences': {(0,): 1}},
    {'duration_s': -1},
    {'duration_s': float('nan')},
    {'coincidence_window_ps': -1},
])
def test_invalid_counts(kwargs):
    fields = dict(singles={}, coincidences={}, duration_s=None, coincidence_window_ps=10)
    fields.update(kwargs)
    with pytest.raises(ValueError):
        MeasurementCounts(**fields)
