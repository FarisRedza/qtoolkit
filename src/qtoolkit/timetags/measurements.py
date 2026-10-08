"""Protocol-independent, immutable summaries of measured detector counts."""

import dataclasses
import math
import pathlib
import typing
from types import MappingProxyType

from .channels import BasisPairs, ChannelPair
from .data import TimetagData

if typing.TYPE_CHECKING:
    from ..qkd.metrics import BasisMetrics


@dataclasses.dataclass(frozen=True)
class MeasurementCounts:
    """Counts from one acquisition, without a protocol-specific interpretation.

    Keys absent from ``singles`` or ``coincidences`` mean *not measured*;
    keys present with value zero mean *measured, with no events*.

    ``duration_s`` is the acquisition duration, not the span between the
    first and last detected photons. It may be unknown (``None``).
    The coincidence window is expressed in picoseconds, matching the
    timetag-processing API. ``file_path`` is optional provenance only.
    """

    singles: typing.Mapping[int, int]
    coincidences: typing.Mapping[tuple[int, int], int]
    duration_s: typing.Optional[float]
    coincidence_window_ps: int
    file_path: typing.Optional[pathlib.Path] = None
    source_file_paths: tuple[pathlib.Path, ...] = ()

    def __post_init__(self) -> None:
        singles = dict(self.singles)
        coincidences = dict(self.coincidences)
        for channel, count in singles.items():
            if not isinstance(channel, int) or not isinstance(count, int) or count < 0:
                raise ValueError('Singles require integer channel IDs and nonnegative integer counts.')
        for pair, count in coincidences.items():
            if (not isinstance(pair, tuple) or len(pair) != 2
                    or any(not isinstance(channel, int) for channel in pair)
                    or not isinstance(count, int) or count < 0):
                raise ValueError('Coincidences require two integer channel IDs and nonnegative integer counts.')
        if not isinstance(self.coincidence_window_ps, int) or self.coincidence_window_ps < 0:
            raise ValueError('coincidence_window_ps must be a nonnegative integer.')
        if self.duration_s is not None and (not math.isfinite(self.duration_s) or self.duration_s < 0):
            raise ValueError('duration_s must be finite and nonnegative, or None.')
        paths = tuple(pathlib.Path(path) for path in self.source_file_paths)
        if self.file_path is not None and not paths:
            paths = (pathlib.Path(self.file_path),)
        object.__setattr__(self, 'source_file_paths', paths)
        object.__setattr__(self, 'singles', MappingProxyType(singles))
        object.__setattr__(self, 'coincidences', MappingProxyType(coincidences))

    @classmethod
    def from_timetag_data(
        cls,
        timetag_data: TimetagData,
        pairs: typing.Iterable[typing.Union[ChannelPair, tuple[int, int]]],
        coincidence_window_ps: int,
        channels: typing.Optional[typing.Iterable[int]] = None,
    ) -> 'MeasurementCounts':
        """Count requested channel pairs and singles from one timetag acquisition.

        When ``channels`` is omitted, singles are counted for channels
        observed in the data and channels referenced by requested pairs.
        Supply ``channels`` explicitly to include zero-count detectors.
        """
        from .coincidences import count_coincidences

        pair_keys = tuple(
            pair.as_tuple() if isinstance(pair, ChannelPair) else pair
            for pair in pairs
        )
        if channels is None:
            requested_channels = set(int(c) for c in timetag_data.channels)
            requested_channels.update(c for pair in pair_keys for c in pair)
        else:
            requested_channels = set(channels)

        return cls(
            singles={channel: timetag_data.count(channel) for channel in requested_channels},
            coincidences=count_coincidences(timetag_data, pair_keys, coincidence_window_ps),
            duration_s=timetag_data.duration_s,
            coincidence_window_ps=coincidence_window_ps,
            file_path=timetag_data.file_path,
        )

    def get_basis_metrics(self, pairs: BasisPairs) -> 'BasisMetrics':
        """Interpret four requested pairs in 00, 01, 10, 11 order.

        Raises ``KeyError`` when any outcome was not measured.
        """
        from ..qkd.metrics import BasisMetrics

        return BasisMetrics.from_coincidences(dict(self.coincidences), pairs)


def aggregate_measurements(
    results: typing.Iterable[MeasurementCounts],
) -> MeasurementCounts:
    """Sum compatible measurement counts before calculating derived metrics.

    Requires identical measured singles channels, coincidence pair keys, and
    coincidence-window widths. A missing key is not equivalent to a zero count.
    Durations are summed only when *every* duration is known; otherwise the
    combined duration is unknown. Source file paths are retained in order,
    including duplicates. Physical detector mappings and acquisition settings
    beyond the coincidence window must be checked by the caller.

    The result's ``file_path`` is ``None`` because it represents multiple
    acquisitions; use ``source_file_paths`` for source provenance.
    """
    results = tuple(results)
    if not results:
        raise ValueError('Cannot aggregate an empty collection of measurements.')
    if any(not isinstance(result, MeasurementCounts) for result in results):
        raise TypeError('All results must be MeasurementCounts instances.')

    first = results[0]
    singles_keys = set(first.singles)
    coincidence_keys = set(first.coincidences)
    for index, result in enumerate(results[1:], start=1):
        if result.coincidence_window_ps != first.coincidence_window_ps:
            raise ValueError(f'Measurement {index} has a different coincidence window.')
        if set(result.singles) != singles_keys:
            raise ValueError(f'Measurement {index} has different measured singles channels.')
        if set(result.coincidences) != coincidence_keys:
            raise ValueError(f'Measurement {index} has different measured coincidence pairs.')

    duration = (
        sum(result.duration_s for result in results if result.duration_s is not None)
        if all(result.duration_s is not None for result in results)
        else None
    )
    return MeasurementCounts(
        singles={key: sum(result.singles[key] for result in results) for key in first.singles},
        coincidences={key: sum(result.coincidences[key] for result in results) for key in first.coincidences},
        duration_s=duration,
        coincidence_window_ps=first.coincidence_window_ps,
        source_file_paths=tuple(
            path for result in results for path in result.source_file_paths
        ),
    )
