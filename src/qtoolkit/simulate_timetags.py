import dataclasses
import time
import typing

import numpy as np

from .lab import TimetagData


@dataclasses.dataclass(frozen=True)
class CoincidencePair:
    channel_a: int
    channel_b: int
    rate_hz: float
    delay_ps: int = 0
    jitter_ps: float = 0.0


def _validate_duration_s(duration_s: float) -> None:
    """Validate an acquisition duration."""
    if not np.isfinite(duration_s):
        raise ValueError(
            'duration_s must be finite.'
        )

    if duration_s <= 0:
        raise ValueError(
            'duration_s must be positive.'
        )


def _calculate_independent_rates(
    channel_rates: typing.Mapping[int, float],
    coincidence_pairs: typing.Sequence[CoincidencePair],
) -> dict[int, float]:
    """
    Validate a simulation configuration and calculate independent singles rates.

    ``channel_rates`` gives the expected final singles rate on each channel.

    Genuine coincidence events already contribute to the singles rate of both
    participating channels, so their contributions are subtracted before
    independent singles are generated.
    """
    for channel, rate_hz in channel_rates.items():
        if not isinstance(channel, (int, np.integer)):
            raise TypeError(
                'Channel numbers must be integers.'
            )

        if not (
            np.iinfo(np.int8).min
            <= channel
            <= np.iinfo(np.int8).max
        ):
            raise ValueError(
                f'Channel {channel} cannot be represented as np.int8.'
            )

        if not np.isfinite(rate_hz):
            raise ValueError(
                f'Rate for channel {channel} must be finite.'
            )

        if rate_hz < 0:
            raise ValueError(
                f'Rate for channel {channel} must be non-negative.'
            )

    # Determine how much of each channel's singles rate comes from
    # genuine coincidence events.
    coincidence_rates = {
        channel: 0.0
        for channel in channel_rates
    }

    for pair in coincidence_pairs:
        if pair.channel_a not in channel_rates:
            raise ValueError(
                f'Channel {pair.channel_a} has no singles rate.'
            )

        if pair.channel_b not in channel_rates:
            raise ValueError(
                f'Channel {pair.channel_b} has no singles rate.'
            )

        if not np.isfinite(pair.rate_hz):
            raise ValueError(
                'Coincidence rates must be finite.'
            )

        if pair.rate_hz < 0:
            raise ValueError(
                'Coincidence rates must be non-negative.'
            )

        if not np.isfinite(pair.jitter_ps):
            raise ValueError(
                'jitter_ps must be finite.'
            )

        if pair.jitter_ps < 0:
            raise ValueError(
                'jitter_ps must be non-negative.'
            )

        coincidence_rates[pair.channel_a] += pair.rate_hz
        coincidence_rates[pair.channel_b] += pair.rate_hz

    independent_rates: dict[int, float] = {}

    for channel, rate_hz in channel_rates.items():
        independent_rate = (
            rate_hz
            - coincidence_rates[channel]
        )

        if independent_rate < 0:
            raise ValueError(
                f'Coincidence rate contribution on channel {channel} '
                f'({coincidence_rates[channel]} Hz) exceeds its '
                f'singles rate ({rate_hz} Hz).'
            )

        independent_rates[channel] = independent_rate

    return independent_rates


def _generate_independent_events(
    independent_rates: typing.Mapping[int, float],
    start_ps: float,
    end_ps: float,
    duration_s: float,
    rng: np.random.Generator,
) -> tuple[
    list[np.typing.NDArray[np.int64]],
    list[np.typing.NDArray[np.int8]],
]:
    """Generate independent singles events for one acquisition interval."""
    timetag_arrays: list[np.typing.NDArray[np.int64]] = []
    channel_arrays: list[np.typing.NDArray[np.int8]] = []

    for channel, rate_hz in independent_rates.items():
        number_of_events = rng.poisson(
            rate_hz * duration_s
        )

        if number_of_events == 0:
            continue

        timetags = rng.uniform(
            start_ps,
            end_ps,
            number_of_events,
        ).astype(np.int64)

        channels = np.full(
            number_of_events,
            channel,
            dtype=np.int8,
        )

        timetag_arrays.append(timetags)
        channel_arrays.append(channels)

    return timetag_arrays, channel_arrays


def _combine_and_sort_events(
    timetag_arrays: typing.Sequence[
        np.typing.NDArray[np.int64]
    ],
    channel_arrays: typing.Sequence[
        np.typing.NDArray[np.int8]
    ],
) -> tuple[
    np.typing.NDArray[np.int64],
    np.typing.NDArray[np.int8],
]:
    """Combine event arrays and sort them chronologically."""
    if not timetag_arrays:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int8),
        )

    timetags = np.concatenate(timetag_arrays)
    channels = np.concatenate(channel_arrays)

    order = np.argsort(
        timetags,
        kind='stable',
    )

    return (
        timetags[order],
        channels[order],
    )


def _generate_timetags(
    channel_rates: typing.Mapping[int, float],
    coincidence_pairs: typing.Sequence[CoincidencePair],
    duration_s: float,
    rng: typing.Optional[np.random.Generator] = None,
) -> tuple[
    np.typing.NDArray[np.int64],
    np.typing.NDArray[np.int8],
]:
    """
    Generate one finite simulated acquisition.
    """
    _validate_duration_s(duration_s)

    if rng is None:
        rng = np.random.default_rng()

    independent_rates = _calculate_independent_rates(
        channel_rates=channel_rates,
        coincidence_pairs=coincidence_pairs,
    )

    duration_ps = (
        duration_s
        * 1e12
    )

    timetag_arrays, channel_arrays = (
        _generate_independent_events(
            independent_rates=independent_rates,
            start_ps=0,
            end_ps=duration_ps,
            duration_s=duration_s,
            rng=rng,
        )
    )

    # Generate genuine coincidence events.
    #
    # This represents a finite acquisition. If the delayed member of a
    # coincidence falls outside the acquisition interval, the whole pair is
    # discarded.
    for pair in coincidence_pairs:
        number_of_pairs = rng.poisson(
            pair.rate_hz * duration_s
        )

        if number_of_pairs == 0:
            continue

        timetags_a = rng.uniform(
            0,
            duration_ps,
            number_of_pairs,
        )

        jitter = rng.normal(
            loc=0.0,
            scale=pair.jitter_ps,
            size=number_of_pairs,
        )

        timetags_b = (
            timetags_a
            + pair.delay_ps
            + jitter
        )

        # Remove pairs shifted outside the acquisition interval.
        valid = (
            (timetags_b >= 0)
            & (timetags_b < duration_ps)
        )

        timetags_a = timetags_a[valid].astype(np.int64)
        timetags_b = timetags_b[valid].astype(np.int64)

        timetag_arrays.extend(
            [
                timetags_a,
                timetags_b,
            ]
        )

        channel_arrays.extend(
            [
                np.full(
                    len(timetags_a),
                    pair.channel_a,
                    dtype=np.int8,
                ),
                np.full(
                    len(timetags_b),
                    pair.channel_b,
                    dtype=np.int8,
                ),
            ]
        )

    return _combine_and_sort_events(
        timetag_arrays=timetag_arrays,
        channel_arrays=channel_arrays,
    )


def generate_timetags(
    channel_rates: typing.Mapping[int, float],
    coincidence_pairs: typing.Sequence[CoincidencePair],
    duration_s: float,
    rng: typing.Optional[np.random.Generator] = None,
) -> TimetagData:
    """
    Generate simulated timetag data with specified singles and coincidence rates.

    This function:

    1. Calculates the contribution of genuine coincidences to each channel's
       requested singles rate.
    2. Generates the remaining independent events using Poisson statistics,
       with event times distributed uniformly over the acquisition period.
    3. Generates genuine coincidence pairs with optional relative delay and
       Gaussian timing jitter.
    4. Removes coincident pairs that fall outside the finite acquisition
       period.
    5. Combines all events and sorts them chronologically by timetag.

    Parameters
    ----------
    channel_rates : Mapping[int, float]
        Expected final singles rate for each channel, in counts per second.

    coincidence_pairs : Sequence[CoincidencePair]
        Genuine coincidence processes between pairs of channels.

    duration_s : float
        Acquisition duration in seconds.

    rng : numpy.random.Generator | None
        Optional NumPy random number generator.

    Returns
    -------
    TimetagData
        Simulated timetag data.
    """
    timetags, channels = _generate_timetags(
        channel_rates=channel_rates,
        coincidence_pairs=coincidence_pairs,
        duration_s=duration_s,
        rng=rng,
    )

    return TimetagData(
        timetags=timetags,
        channels=channels,
    )


class LiveTimetagSimulator:
    """
    Continuously generate simulated timetag data.

    The simulator maintains a simulated acquisition clock between calls to
    :meth:`read`.

    ``channel_rates`` has the same meaning as in :func:`generate_timetags`:
    each value is the expected final singles rate on that channel, including
    events produced by genuine coincidences.

    Unlike :func:`generate_timetags`, events belonging to a genuine
    coincidence are allowed to cross acquisition-block boundaries. Such
    events are retained internally and emitted by a later call to
    :meth:`read`.
    """

    def __init__(
        self,
        channel_rates: typing.Mapping[int, float],
        coincidence_pairs: typing.Sequence[CoincidencePair] = (),
        rng: typing.Optional[np.random.Generator] = None,
        start_time_ps: int = 0,
    ) -> None:
        if start_time_ps < 0:
            raise ValueError(
                'start_time_ps must be non-negative.'
            )

        if start_time_ps > np.iinfo(np.int64).max:
            raise ValueError(
                'start_time_ps exceeds the range of np.int64.'
            )

        self._channel_rates = dict(
            channel_rates
        )

        self._coincidence_pairs = tuple(
            coincidence_pairs
        )

        self._rng = (
            np.random.default_rng()
            if rng is None
            else rng
        )

        self._time_ps = int(
            start_time_ps
        )

        self._independent_rates = (
            _calculate_independent_rates(
                channel_rates=self._channel_rates,
                coincidence_pairs=self._coincidence_pairs,
            )
        )

        self._pending_timetags = np.empty(
            0,
            dtype=np.int64,
        )

        self._pending_channels = np.empty(
            0,
            dtype=np.int8,
        )

    @property
    def time_ps(self) -> int:
        """Current simulated acquisition time in picoseconds."""
        return self._time_ps

    @property
    def time_s(self) -> float:
        """Current simulated acquisition time in seconds."""
        return (
            self._time_ps * 1e-12
        )

    @property
    def channel_rates(
        self,
    ) -> dict[int, float]:
        """Configured final singles rates in Hz."""
        return dict(
            self._channel_rates
        )

    @property
    def coincidence_pairs(
        self,
    ) -> tuple[CoincidencePair, ...]:
        """Configured genuine coincidence processes."""
        return self._coincidence_pairs

    def _generate_coincidence_events(
        self,
        start_ps: int,
        end_ps: int,
        duration_s: float,
    ) -> tuple[
        list[np.typing.NDArray[np.int64]],
        list[np.typing.NDArray[np.int8]],
    ]:
        """
        Generate genuine coincidence events for one live block.

        ``pair_times`` represents the earliest detection belonging to a
        coincidence. The other event may fall either in the current block or
        in a future block.
        """
        timetag_arrays: list[
            np.typing.NDArray[np.int64]
        ] = []

        channel_arrays: list[
            np.typing.NDArray[np.int8]
        ] = []

        future_timetags: list[
            np.typing.NDArray[np.int64]
        ] = []

        future_channels: list[
            np.typing.NDArray[np.int8]
        ] = []

        for pair in self._coincidence_pairs:
            number_of_pairs = self._rng.poisson(
                pair.rate_hz * duration_s
            )

            if number_of_pairs == 0:
                continue

            pair_times = self._rng.uniform(
                start_ps,
                end_ps,
                number_of_pairs,
            )

            relative_delay = (
                pair.delay_ps
                + self._rng.normal(
                    loc=0.0,
                    scale=pair.jitter_ps,
                    size=number_of_pairs,
                )
            )

            positive_delay = (
                relative_delay >= 0
            )

            # pair_times represents whichever event occurs first.
            #
            # If B - A >= 0:
            #
            #     A = pair_time
            #     B = pair_time + relative_delay
            #
            # If B - A < 0:
            #
            #     B = pair_time
            #     A = pair_time - relative_delay
            #
            # This preserves:
            #
            #     B - A = relative_delay
            #
            # while ensuring that neither event ever has to be inserted
            # retrospectively into an acquisition block that has already
            # been returned.
            timetags_a = np.where(
                positive_delay,
                pair_times,
                pair_times - relative_delay,
            ).astype(np.int64)

            timetags_b = np.where(
                positive_delay,
                pair_times + relative_delay,
                pair_times,
            ).astype(np.int64)

            self._split_current_and_future(
                timetags=timetags_a,
                channel=pair.channel_a,
                end_ps=end_ps,
                current_timetags=timetag_arrays,
                current_channels=channel_arrays,
                future_timetags=future_timetags,
                future_channels=future_channels,
            )

            self._split_current_and_future(
                timetags=timetags_b,
                channel=pair.channel_b,
                end_ps=end_ps,
                current_timetags=timetag_arrays,
                current_channels=channel_arrays,
                future_timetags=future_timetags,
                future_channels=future_channels,
            )

        if future_timetags:
            self._append_pending(
                timetags=np.concatenate(
                    future_timetags
                ),
                channels=np.concatenate(
                    future_channels
                ),
            )

        return (
            timetag_arrays,
            channel_arrays,
        )

    @staticmethod
    def _split_current_and_future(
        timetags: np.typing.NDArray[np.int64],
        channel: int,
        end_ps: int,
        current_timetags: list[
            np.typing.NDArray[np.int64]
        ],
        current_channels: list[
            np.typing.NDArray[np.int8]
        ],
        future_timetags: list[
            np.typing.NDArray[np.int64]
        ],
        future_channels: list[
            np.typing.NDArray[np.int8]
        ],
    ) -> None:
        """Split generated events at the current read boundary."""
        current_mask = (
            timetags < end_ps
        )

        if np.any(current_mask):
            current_tags = (
                timetags[current_mask]
            )

            current_timetags.append(
                current_tags
            )

            current_channels.append(
                np.full(
                    len(current_tags),
                    channel,
                    dtype=np.int8,
                )
            )

        future_mask = (
            ~current_mask
        )

        if np.any(future_mask):
            future_tags = (
                timetags[future_mask]
            )

            future_timetags.append(
                future_tags
            )

            future_channels.append(
                np.full(
                    len(future_tags),
                    channel,
                    dtype=np.int8,
                )
            )

    def _append_pending(
        self,
        timetags: np.typing.NDArray[np.int64],
        channels: np.typing.NDArray[np.int8],
    ) -> None:
        """Add events to the future-event queue."""
        if len(timetags) == 0:
            return

        if len(self._pending_timetags) == 0:
            self._pending_timetags = timetags
            self._pending_channels = channels
            return

        self._pending_timetags = np.concatenate(
            (
                self._pending_timetags,
                timetags,
            )
        )

        self._pending_channels = np.concatenate(
            (
                self._pending_channels,
                channels,
            )
        )

    def _take_pending_events(
        self,
        end_ps: int,
    ) -> tuple[
        np.typing.NDArray[np.int64],
        np.typing.NDArray[np.int8],
    ]:
        """
        Remove and return pending events belonging to the current block.
        """
        if len(self._pending_timetags) == 0:
            return (
                np.empty(
                    0,
                    dtype=np.int64,
                ),
                np.empty(
                    0,
                    dtype=np.int8,
                ),
            )

        current_mask = (
            self._pending_timetags
            < end_ps
        )

        timetags = (
            self._pending_timetags[
                current_mask
            ]
        )

        channels = (
            self._pending_channels[
                current_mask
            ]
        )

        future_mask = (
            ~current_mask
        )

        self._pending_timetags = (
            self._pending_timetags[
                future_mask
            ]
        )

        self._pending_channels = (
            self._pending_channels[
                future_mask
            ]
        )

        return (
            timetags,
            channels,
        )

    def read(
        self,
        duration_s: float,
    ) -> TimetagData:
        """
        Generate the next interval of simulated live timetag data.

        Parameters
        ----------
        duration_s : float
            Simulated acquisition duration in seconds.

        Returns
        -------
        TimetagData
            Timetags falling within the next acquisition interval.
        """
        _validate_duration_s(
            duration_s
        )

        duration_ps = int(
            round(
                duration_s * 1e12
            )
        )

        if duration_ps <= 0:
            raise ValueError(
                'duration_s is too small to represent in picoseconds.'
            )

        start_ps = self._time_ps

        end_ps = (
            start_ps
            + duration_ps
        )

        if end_ps > np.iinfo(np.int64).max:
            raise OverflowError(
                'Simulated time exceeds the range of np.int64.'
            )

        # The live clock is integer picoseconds, so use the duration actually
        # represented by that clock when determining Poisson event counts.
        actual_duration_s = (
            duration_ps * 1e-12
        )

        timetag_arrays: list[
            np.typing.NDArray[np.int64]
        ] = []

        channel_arrays: list[
            np.typing.NDArray[np.int8]
        ] = []

        pending_timetags, pending_channels = (
            self._take_pending_events(
                end_ps=end_ps
            )
        )

        if len(pending_timetags) > 0:
            timetag_arrays.append(
                pending_timetags
            )

            channel_arrays.append(
                pending_channels
            )

        independent_timetags, independent_channels = (
            _generate_independent_events(
                independent_rates=self._independent_rates,
                start_ps=start_ps,
                end_ps=end_ps,
                duration_s=actual_duration_s,
                rng=self._rng,
            )
        )

        timetag_arrays.extend(
            independent_timetags
        )

        channel_arrays.extend(
            independent_channels
        )

        coincidence_timetags, coincidence_channels = (
            self._generate_coincidence_events(
                start_ps=start_ps,
                end_ps=end_ps,
                duration_s=actual_duration_s,
            )
        )

        timetag_arrays.extend(
            coincidence_timetags
        )

        channel_arrays.extend(
            coincidence_channels
        )

        timetags, channels = (
            _combine_and_sort_events(
                timetag_arrays=timetag_arrays,
                channel_arrays=channel_arrays,
            )
        )

        self._time_ps = end_ps

        return TimetagData(
            timetags=timetags,
            channels=channels,
        )

    def stream(
        self,
        interval_s: float,
        realtime: bool = True,
    ) -> typing.Iterator[TimetagData]:
        """
        Continuously yield simulated timetag blocks.

        Parameters
        ----------
        interval_s : float
            Simulated duration represented by each block.

        realtime : bool
            If True, pace blocks according to wall-clock time. If False,
            generate them as quickly as possible.

        Yields
        ------
        TimetagData
            Successive blocks of simulated timetag data.
        """
        _validate_duration_s(
            interval_s
        )

        if not realtime:
            while True:
                yield self.read(
                    interval_s
                )

        next_deadline = (
            time.monotonic()
            + interval_s
        )

        while True:
            data = self.read(
                interval_s
            )

            sleep_time = (
                next_deadline
                - time.monotonic()
            )

            if sleep_time > 0:
                time.sleep(
                    sleep_time
                )

            yield data

            next_deadline += interval_s

    def reset(
        self,
        start_time_ps: int = 0,
    ) -> None:
        self._time_ps = start_time_ps

        self._pending_timetags = np.empty(
            0,
            dtype=np.int64,
        )

        self._pending_channels = np.empty(
            0,
            dtype=np.int8,
        )