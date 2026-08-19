import typing

import numpy as np
import numba

@numba.njit(cache=True)
def _get_twofold_coincidences(
        tags_a: np.typing.NDArray[np.int64],
        tags_b: np.typing.NDArray[np.int64],
        coincidence_window: int
) -> float:
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

def get_twofold_coincidences(
        tags_a: typing.Union[list[int], np.typing.NDArray[np.int64]],
        tags_b: typing.Union[list[int], np.typing.NDArray[np.int64]],
        coincidence_window: int
) -> float:
    tags_a = np.array(tags_a, dtype=np.int64)
    tags_b = np.array(tags_b, dtype=np.int64)

    return _get_twofold_coincidences(
        tags_a=tags_a,
        tags_b=tags_b,
        coincidence_window=coincidence_window
    )