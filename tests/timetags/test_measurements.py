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


from pathlib import Path
from qtoolkit.timetags import aggregate_measurements


def _result(correct=0, incorrect=0, *, duration=1.0, window=50, path=None):
    return MeasurementCounts(
        singles={0: correct + incorrect, 1: 0, 2: correct + incorrect},
        coincidences={(0, 2): correct, (0, 3): incorrect, (1, 2): 0, (1, 3): 0},
        duration_s=duration,
        coincidence_window_ps=window,
        file_path=path,
    )


def test_aggregation_sums_counts_not_qber():
    from qtoolkit.timetags import ChannelPair
    first = _result(900, 100)
    second = _result(1, 9)
    combined = aggregate_measurements([first, second])
    pairs = tuple(ChannelPair(*pair) for pair in combined.coincidences)
    assert combined.coincidences[(0, 2)] == 901
    assert combined.coincidences[(0, 3)] == 109
    assert combined.get_basis_metrics(pairs).qber == pytest.approx(109 / 1010)
    assert combined.duration_s == 2.0
    assert first.coincidences[(0, 2)] == 900


def test_aggregation_preserves_source_paths_and_missing_duration():
    first = _result(path=Path('first.csv'))
    second = _result(duration=None, path=Path('second.csv'))
    combined = aggregate_measurements((first, second))
    assert combined.duration_s is None
    assert combined.file_path is None
    assert combined.source_file_paths == (Path('first.csv'), Path('second.csv'))
    assert aggregate_measurements([combined, first]).source_file_paths == (
        Path('first.csv'), Path('second.csv'), Path('first.csv')
    )


@pytest.mark.parametrize('changed', [
    {'coincidence_window_ps': 51},
    {'singles': {0: 0, 2: 0}},
    {'coincidences': {(0, 2): 0, (0, 3): 0, (1, 2): 0}},
])
def test_aggregation_rejects_incompatible_measurements(changed):
    first = _result()
    second = dataclasses.replace(first, **changed)
    with pytest.raises(ValueError):
        aggregate_measurements([first, second])


def test_aggregation_accepts_measured_zero_and_rejects_empty():
    combined = aggregate_measurements([_result(), _result()])
    assert set(combined.coincidences) == {(0, 2), (0, 3), (1, 2), (1, 3)}
    assert all(value == 0 for value in combined.coincidences.values())
    with pytest.raises(ValueError):
        aggregate_measurements([])
    with pytest.raises(TypeError):
        aggregate_measurements([_result(), object()])
