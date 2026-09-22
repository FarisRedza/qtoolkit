import typing

import numpy as np
import numpy.typing as npt
import scipy

@typing.overload
def turbulence(
    v: float,
    A: float,
    z: float,
) -> float:
    ...


@typing.overload
def turbulence(
    v: float,
    A: float,
    z: npt.NDArray[np.number],
) -> npt.NDArray[np.float64]:
    ...


@typing.overload
def turbulence(
    v: float,
    A: float,
    z: typing.Sequence[float],
) -> npt.NDArray[np.float64]:
    ...


def turbulence(
    v: float,
    A: float,
    z: npt.ArrayLike,
) -> typing.Union[float,npt.NDArray[np.float64]]:
    r"""
    Calculate atmospheric turbulence using the Hufnagel-Valley model.

    .. math::
        C_n^2(z) =
        5.94 \times 10^{-53}
        \left(\frac{v}{27}\right)^2
        z^{10} e^{-z/1000}
        + 2.7 \times 10^{-16} e^{-z/1500}
        + A e^{-z/100}

    Parameters
    ----------
    v : float
        High-altitude RMS wind speed in metres per second.
    A : float
        Near-ground turbulence strength in metres to the power
        minus two-thirds.
    z : float or array-like
        Altitude above ground level in metres.

    Returns
    -------
    float or numpy.ndarray
        Refractive-index structure parameter in metres to the power
        minus two-thirds. A scalar input produces a ``float``; an array-like
        input produces a ``numpy.ndarray`` with dtype ``numpy.float64``.

    Examples
    --------
    Calculate the turbulence strength at ground level:

    >>> turbulence(21.0, 1.7e-14, 0.0)
    1.727e-14

    Calculate the turbulence strength at multiple altitudes:

    >>> turbulence(21.0, 1.7e-14, [0.0, 1_000.0, 10_000.0])
    array([...])
    """
    z_array = np.asarray(z, dtype=np.float64)

    result = (
        5.94e-53
        * (v / 27.0) ** 2
        * z_array**10
        * np.exp(-z_array / 1000.0)
        + 2.7e-16 * np.exp(-z_array / 1500.0)
        + A * np.exp(-z_array / 100.0)
    )

    if z_array.ndim == 0:
        return float(result)

    return np.asarray(result, dtype=np.float64)

def fried_parameter(
    wavelength: float,
    v: float,
    A: float,
    maximum_altitude: float = 100_000.0,
) -> float:
    r"""
    Calculate the Fried coherence length for a vertical atmospheric path.

    .. math::
        r_0 =
        \left[
        0.423 k^2
        \int_0^{h_{\max}} C_n^2(z)\,dz
        \right]^{-3/5},

    where

    .. math::
        k = \frac{2\pi}{\lambda}.

    Parameters
    ----------
    wavelength : float
        Optical wavelength in metres.
    v : float
        High-altitude RMS wind speed in metres per second.
    A : float
        Near-ground turbulence strength in m^(-2/3).
    maximum_altitude : float, default=100000
        Upper limit of the atmospheric integral in metres.

    Returns
    -------
    float
        Fried coherence length in metres.
    """
    if wavelength <= 0:
        raise ValueError("wavelength must be positive")

    if maximum_altitude <= 0:
        raise ValueError("maximum_altitude must be positive")

    integrated_turbulence, _ = scipy.integrate.quad(
        lambda z: turbulence(v, A, z),
        0.0,
        maximum_altitude,
        epsabs=1e-30,
        epsrel=1e-10,
    )

    wave_number = 2.0 * np.pi / wavelength

    return float(
        (0.423 * wave_number**2 * integrated_turbulence) ** (-3.0 / 5.0)
    )