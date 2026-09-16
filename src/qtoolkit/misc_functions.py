import typing

import numpy as np
import numpy.typing as npt

def binary_entropy(
    p: typing.Union[float, npt.ArrayLike],
) -> typing.Union[float, npt.NDArray[np.float64]]:
    """
    Calculate the binary entropy, which quantifies the uncertainty associated
    with a binary outcome with probability :math:`p`.

    .. math::
        \\text{H}_2(p) = -p\\log_2(p) - (1-p)\\log_2(1-p)

    Parameters
    ----------
    p : float or array-like
        Probability of one of the two outcomes, in the range [0, 1].

    Returns
    -------
    float or numpy.ndarray
        Binary entropy of ``p``.
    """
    p = np.asarray(p, dtype=float)

    if np.any((p < 0) | (p > 1)):
        raise ValueError("p must be between 0 and 1.")

    result = np.zeros_like(p)
    mask = (p > 0) & (p < 1)

    result[mask] = (
        -p[mask] * np.log2(p[mask])
        - (1 - p[mask]) * np.log2(1 - p[mask])
    )

    return result.item() if result.ndim == 0 else result

def fraction_to_dB(
        x: typing.Union[float, npt.ArrayLike]
    ) -> typing.Union[float, npt.NDArray[np.float64]]:
    """Convert a linear fraction to loss in dB."""
    result = -10 * np.log10(x)
    return result.item() if np.ndim(result) == 0 else result

def dB_to_fraction(
        x: typing.Union[float, npt.ArrayLike]
    ) -> typing.Union[float, npt.NDArray[np.float64]]:
    """Convert loss in dB to a linear fraction."""
    result = np.power(10.0, -np.asarray(x) / 10)
    return result.item() if np.ndim(result) == 0 else result
