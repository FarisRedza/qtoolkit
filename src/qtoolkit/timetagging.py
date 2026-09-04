import typing

import numpy as np
import numba

@numba.njit(cache=True)
def _count_twofold_coincidences(
        tags_a: np.typing.NDArray[np.int64],
        tags_b: np.typing.NDArray[np.int64],
        coincidence_window: int
) -> int:
    idx_a = 0
    idx_b = 0
    counts = 0

    while idx_a < len(tags_a) and idx_b < len(tags_b):
        time_a = tags_a[idx_a]
        time_b = tags_b[idx_b]

        minimum = min(time_a, time_b)
        maximum = max(time_a, time_b)

        if maximum - minimum <= coincidence_window:
            counts += 1
            idx_a += 1
            idx_b += 1

        elif time_a == minimum:
            idx_a += 1

        else:
            idx_b += 1

    return counts

@numba.njit(cache=True)
def _find_twofold_coincidence_indices(
        tags_a: np.typing.NDArray[np.int64],
        tags_b: np.typing.NDArray[np.int64],
        coincidence_window: int
) -> tuple[
    np.typing.NDArray[np.int64],
    np.typing.NDArray[np.int64]
]:
    idx_a = 0
    idx_b = 0
    counts = 0

    # this method is faster but uses more memory
    # on linux this is fine due to how virtual memory works,
    # but unclear if this is the case on windows or macos
    max_coincidences = min(len(tags_a), len(tags_b))
    indices_a = np.empty(max_coincidences, dtype=np.int64)
    indices_b = np.empty(max_coincidences, dtype=np.int64)

    # this method is slower but potentially uses less memory
    # assuming that virtual memory is a problem on windows
    # coincidences = _count_twofold_coincidences(
    #     tags_a=tags_a,
    #     tags_b=tags_b,
    #     coincidence_window=coincidence_window
    # )
    # indices_a = np.empty(coincidences, dtype=np.int64)
    # indices_b = np.empty(coincidences, dtype=np.int64)

    while idx_a < len(tags_a) and idx_b < len(tags_b):
        time_a = tags_a[idx_a]
        time_b = tags_b[idx_b]

        minimum = min(time_a, time_b)
        maximum = max(time_a, time_b)

        if maximum - minimum <= coincidence_window:
            indices_a[counts] = idx_a
            indices_b[counts] = idx_b

            counts += 1
            idx_a += 1
            idx_b += 1

        elif time_a == minimum:
            idx_a += 1

        else:
            idx_b += 1

    # return indices_a, indices_b
    return indices_a[:counts], indices_b[:counts]

@numba.njit(cache=True)
def _count_threefold_coincidences(
        tags_a: np.typing.NDArray[np.int64],
        tags_b: np.typing.NDArray[np.int64],
        tags_c: np.typing.NDArray[np.int64],
        coincidence_window: int
) -> int:
    idx_a = 0
    idx_b = 0
    idx_c = 0
    counts = 0

    while (
        idx_a < len(tags_a)
        and idx_b < len(tags_b)
        and idx_c < len(tags_c)
    ):
        time_a = tags_a[idx_a]
        time_b = tags_b[idx_b]
        time_c = tags_c[idx_c]

        minimum = min(time_a, time_b, time_c)
        maximum = max(time_a, time_b, time_c)

        if maximum - minimum <= coincidence_window:
            counts += 1
            idx_a += 1
            idx_b += 1
            idx_c += 1

        elif time_a == minimum:
            idx_a += 1

        elif time_b == minimum:
            idx_b += 1

        else:
            idx_c += 1

    return counts

@numba.njit(cache=True)
def _count_fourfold_coincidences(
        tags_a: np.typing.NDArray[np.int64],
        tags_b: np.typing.NDArray[np.int64],
        tags_c: np.typing.NDArray[np.int64],
        tags_d: np.typing.NDArray[np.int64],
        coincidence_window: int
) -> int:
    idx_a = 0
    idx_b = 0
    idx_c = 0
    idx_d = 0
    counts = 0

    while (
        idx_a < len(tags_a)
        and idx_b < len(tags_b)
        and idx_c < len(tags_c)
        and idx_d < len(tags_d)
    ):
        time_a = tags_a[idx_a]
        time_b = tags_b[idx_b]
        time_c = tags_c[idx_c]
        time_d = tags_d[idx_d]

        minimum = min(time_a, time_b, time_c, time_d)
        maximum = max(time_a, time_b, time_c, time_d)

        if maximum - minimum <= coincidence_window:
            counts += 1
            idx_a += 1
            idx_b += 1
            idx_c += 1
            idx_d += 1

        elif time_a == minimum:
            idx_a += 1

        elif time_b == minimum:
            idx_b += 1

        elif time_c == minimum:
            idx_c += 1

        else:
            idx_d += 1

    return counts

def count_twofold_coincidences(
        tags_a: typing.Union[list[int], np.typing.NDArray[np.int64]],
        tags_b: typing.Union[list[int], np.typing.NDArray[np.int64]],
        coincidence_window: int
) -> int:
    """
    Count the number of entries in two arrays that are within the coincidence
    window of each other.

    Parameters
    ----------
        tags_a: list[int] | np.typing.NDArray[np.int64]
            List/array of timetags
        tags_b: list[int] | np.typing.NDArray[np.int64]
            List/array of timetags
        coincidence_window: int
            The coincidence window (ps)
    
    Returns
    -------
    int
        Number of coincidences
    """
    tags_a = np.asarray(tags_a, dtype=np.int64)
    tags_b = np.asarray(tags_b, dtype=np.int64)

    return _count_twofold_coincidences(
        tags_a=tags_a,
        tags_b=tags_b,
        coincidence_window=coincidence_window
    )

def find_twofold_coincidence_indices(
        tags_a: np.typing.NDArray[np.int64],
        tags_b: np.typing.NDArray[np.int64],
        coincidence_window: int
) -> tuple[
    np.typing.NDArray[np.int64],
    np.typing.NDArray[np.int64]
]:
    tags_a = np.asarray(tags_a, dtype=np.int64)
    tags_b = np.asarray(tags_b, dtype=np.int64)

    return _find_twofold_coincidence_indices(
        tags_a=tags_a,
        tags_b=tags_b,
        coincidence_window=coincidence_window
    )

def count_threefold_coincidences(
        tags_a: typing.Union[list[int], np.typing.NDArray[np.int64]],
        tags_b: typing.Union[list[int], np.typing.NDArray[np.int64]],
        tags_c: typing.Union[list[int], np.typing.NDArray[np.int64]],
        coincidence_window: int
) -> int:
    """
    Count the number of entries in three arrays that are within the coincidence
    window of each other.

    Parameters
    ----------
        tags_a: list[int] | np.typing.NDArray[np.int64]
            List/array of timetags
        tags_b: list[int] | np.typing.NDArray[np.int64]
            List/array of timetags
        tags_c: list[int] | np.typing.NDArray[np.int64]
            List/array of timetags
        coincidence_window: int
            The coincidence window (ps)
    
    Returns
    -------
    int
        Number of coincidences
    """
    tags_a = np.asarray(tags_a, dtype=np.int64)
    tags_b = np.asarray(tags_b, dtype=np.int64)
    tags_c = np.asarray(tags_c, dtype=np.int64)

    return _count_threefold_coincidences(
        tags_a=tags_a,
        tags_b=tags_b,
        tags_c=tags_c,
        coincidence_window=coincidence_window
    )

def count_fourfold_coincidences(
        tags_a: typing.Union[list[int], np.typing.NDArray[np.int64]],
        tags_b: typing.Union[list[int], np.typing.NDArray[np.int64]],
        tags_c: typing.Union[list[int], np.typing.NDArray[np.int64]],
        tags_d: typing.Union[list[int], np.typing.NDArray[np.int64]],
        coincidence_window: int
) -> int:
    """
    Count the number of entries in four arrays that are within the coincidence
    window of each other.

    Parameters
    ----------
        tags_a: list[int] | np.typing.NDArray[np.int64]
            List/array of timetags
        tags_b: list[int] | np.typing.NDArray[np.int64]
            List/array of timetags
        tags_c: list[int] | np.typing.NDArray[np.int64]
            List/array of timetags
        tags_d: list[int] | np.typing.NDArray[np.int64]
            List/array of timetags
        coincidence_window: int
            The coincidence window (ps)
    
    Returns
    -------
    int
        Number of coincidences
    """
    tags_a = np.asarray(tags_a, dtype=np.int64)
    tags_b = np.asarray(tags_b, dtype=np.int64)
    tags_c = np.asarray(tags_c, dtype=np.int64)
    tags_d = np.asarray(tags_d, dtype=np.int64)

    return _count_fourfold_coincidences(
        tags_a=tags_a,
        tags_b=tags_b,
        tags_c=tags_c,
        tags_d=tags_d,
        coincidence_window=coincidence_window
    )

def count_coincidences(
    timetags: np.typing.ArrayLike,
    channels: np.typing.ArrayLike,
    pairs: typing.Iterable[tuple[int, int]],
    coincidence_window: int,
) -> dict[tuple[int, int], int]:
    """
    Counts twofold coincidences for selected channel pairs.

    Parameters
    ----------
    timetags: np.typing.ArrayLike
        Chronologically ordered timetags.

    channels: np.typing.ArrayLike
        Channel corresponding to each timetag.

    pairs: list[tuple[int, int]]
        Channel pairs for which coincidences should be calculated.

    coincidence_window: int
        Maximum separation between coincident timetags (ps).

    Returns
    -------
    dict
        Mapping ``(channel_a, channel_b)`` to coincidence count.
    """

    timetags = np.asarray(timetags, dtype=np.int64)
    channels = np.asarray(channels, dtype=np.int64)

    if timetags.ndim != 1:
        raise ValueError('Timetags must be a 1D array.')

    if channels.ndim != 1:
        raise ValueError('Channels must be a 1D array.')

    if len(timetags) != len(channels):
        raise ValueError(
            'Timetags and channels must have the same length.'
        )

    required_channels = {
        channel
        for pair in pairs
        for channel in pair
    }

    channel_tags = {
        channel: timetags[channels == channel]
        for channel in required_channels
    }

    coincidences = {}

    for channel_a, channel_b in pairs:
        coincidences[channel_a, channel_b] = (
            _count_twofold_coincidences(
                tags_a=channel_tags[channel_a],
                tags_b=channel_tags[channel_b],
                coincidence_window=coincidence_window,
            )
        )

    return coincidences