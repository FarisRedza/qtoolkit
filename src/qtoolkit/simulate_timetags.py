import dataclasses
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


def _generate_timetags(
        channel_rates: dict[int, float],
        coincidence_pairs: list[CoincidencePair],
        duration_s: float,
        rng: typing.Optional[np.random.Generator] = None,
) -> tuple[
    np.typing.NDArray[np.int64],
    np.typing.NDArray[np.int8]
]:
    if duration_s <= 0:
        raise ValueError(
            'duration_s must be positive.'
        )

    if rng is None:
        rng = np.random.default_rng()

    for channel, rate_hz in channel_rates.items():
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
        if pair.rate_hz < 0:
            raise ValueError(
                'Coincidence rates must be non-negative.'
            )

        if pair.jitter_ps < 0:
            raise ValueError(
                'jitter_ps must be non-negative.'
            )

        if pair.channel_a not in channel_rates:
            raise ValueError(
                f'Channel {pair.channel_a} has no singles rate.'
            )

        if pair.channel_b not in channel_rates:
            raise ValueError(
                f'Channel {pair.channel_b} has no singles rate.'
            )

        coincidence_rates[pair.channel_a] += pair.rate_hz
        coincidence_rates[pair.channel_b] += pair.rate_hz

    # The remaining rate on each channel is uncorrelated singles.
    uncorrelated_rates: dict[int, float] = {}

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

        uncorrelated_rates[channel] = independent_rate

    timetag_arrays: list[np.ndarray] = []
    channel_arrays: list[np.ndarray] = []

    duration_ps = duration_s * 1e12

    # Generate independent singles.
    for channel, rate_hz in uncorrelated_rates.items():
        number_of_events = rng.poisson(
            rate_hz * duration_s
        )

        timetags = rng.uniform(
            0,
            duration_ps,
            number_of_events,
        ).astype(np.int64)

        channels = np.full(
            number_of_events,
            channel,
            dtype=np.int8,
        )

        timetag_arrays.append(timetags)
        channel_arrays.append(channels)

    # Generate genuine coincidence events.
    for pair in coincidence_pairs:
        number_of_pairs = rng.poisson(
            pair.rate_hz * duration_s
        )

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

        timetag_arrays.extend([
            timetags_a,
            timetags_b,
        ])

        channel_arrays.extend([
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
        ])

    if not timetag_arrays:
        return (
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int8),
        )

    timetags = np.concatenate(timetag_arrays)
    channels = np.concatenate(channel_arrays)

    order = np.argsort(timetags)

    return (
        timetags[order],
        channels[order]
    )

def generate_timetags(
        channel_rates: dict[int, float],
        coincidence_pairs: list[CoincidencePair],
        duration_s: float,
        rng: typing.Optional[np.random.Generator] = None,
) -> TimetagData:
    """
    Generate simulated timetag data with specified singles and coincidence rates.

    This function:
    1. Calculates the contribution of genuine coincidences to each channel's
    requested singles rate.
    2. Generates the remaining uncorrelated events using Poisson statistics,
    with event times distributed uniformly over the acquisition period.
    3. Generates genuine coincidence pairs with optional relative delay and
    Gaussian timing jitter.
    4. Removes coincident events that fall outside the acquisition period.
    5. Combines all events and sorting them chronologically by timetag.

    Parameters
    ----------
    channel_rates : dict[int, float]
        Expected final singles rate for each channel, in counts per second.
    coincidence_pairs : list[CoincidencePair]
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
        rng=rng
    )
    return TimetagData(
        timetags=timetags,
        channels=channels,
    )