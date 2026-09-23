import typing

import numpy as np
import numpy.typing as npt

from qtoolkit.constants import SPEED_OF_LIGHT

from .materials import (
    NonlinearMaterial,
    RefractiveIndexAxis,
)


def wavevector(
    wavelength: npt.ArrayLike,
    refractive_index: npt.ArrayLike,
) -> typing.Union[
    float,
    npt.NDArray[np.float64],
]:
    r"""
    Calculate the magnitude of the wavevector.

    .. math::

        k = \frac{2\pi n}{\lambda},

    where :math:`n` is the refractive index and :math:`\lambda` is
    the vacuum wavelength.

    Parameters
    ----------
    wavelength
        Vacuum wavelength in metres.
    refractive_index
        Refractive index at the specified wavelength.

    Returns
    -------
    float or numpy.ndarray
        Wavevector in radians per metre.
    """
    result = (
        2 * np.pi
        * np.asarray(refractive_index)
        / np.asarray(wavelength)
    )

    return (
        float(result)
        if result.ndim == 0
        else result
    )

def wavevector_mismatch(
    pump_wavelength: npt.ArrayLike,
    signal_wavelength: npt.ArrayLike,
    idler_wavelength: npt.ArrayLike,
    temperature: float,
    poling_period: float,
    material: NonlinearMaterial,
    pump_axis: RefractiveIndexAxis,
    signal_axis: RefractiveIndexAxis,
    idler_axis: RefractiveIndexAxis,
    qpm_order: int = 1,
) -> typing.Union[
    float,
    npt.NDArray[np.float64],
]:
    r"""
    Calculate the quasi-phase-matching wavevector mismatch.

    .. math::

        \Delta k
        =
        k_p - k_s - k_i
        - \frac{2\pi m}{\Lambda(T)},

    where :math:`m` is the quasi-phase-matching order and
    :math:`\Lambda(T)` is the physical poling period at the crystal
    temperature.

    Perfect plane-wave phase matching occurs when

    .. math::

        \Delta k = 0.
        
    """

    n_p = material.refractive_index(
        pump_wavelength,
        temperature,
        pump_axis,
    )

    n_s = material.refractive_index(
        signal_wavelength,
        temperature,
        signal_axis,
    )

    n_i = material.refractive_index(
        idler_wavelength,
        temperature,
        idler_axis,
    )

    k_p = wavevector(
        pump_wavelength,
        n_p,
    )

    k_s = wavevector(
        signal_wavelength,
        n_s,
    )

    k_i = wavevector(
        idler_wavelength,
        n_i,
    )

    period = poling_period_at_temperature(
        poling_period,
        temperature,
    )

    return (
        k_p
        - k_s
        - k_i
        - 2 * np.pi * qpm_order / period
    )

_POLING_REFERENCE_TEMPERATURE = 19.0
_THERMAL_EXPANSION_ALPHA = 1.53e-5
_THERMAL_EXPANSION_BETA = 5.3e-9

def _poling_thermal_expansion_factor(
    temperature: float,
) -> float:
    delta_temperature = (
        temperature
        - _POLING_REFERENCE_TEMPERATURE
    )

    return (
        1
        + _THERMAL_EXPANSION_ALPHA
        * delta_temperature
        + _THERMAL_EXPANSION_BETA
        * delta_temperature**2
    )

def poling_period_at_temperature(
    poling_period: npt.ArrayLike,
    temperature: float,
) -> typing.Union[
    float,
    npt.NDArray[np.float64],
]:
    r"""
    Calculate the physical poling period at a given temperature.

    .. math::

        \Lambda(T)
        =
        \Lambda_{19}
        \left[
            1
            + \alpha(T - 19)
            + \beta(T - 19)^2
        \right],

    where

    .. math::

        \alpha = 1.53 \times 10^{-5}

    and

    .. math::

        \beta = 5.3 \times 10^{-9}.

    This function may be moved to a method in `NonlinearMaterial`
    due to thermal expansion being material dependent.

    Parameters
    ----------
    poling_period
        Poling period at 19 degrees Celsius, in metres.
    temperature
        Crystal temperature in degrees Celsius.

    Returns
    -------
    float or numpy.ndarray
        Poling period at the specified temperature, in metres.
    """
    result = (
        np.asarray(poling_period)
        * _poling_thermal_expansion_factor(
            temperature
        )
    )

    return (
        float(result)
        if result.ndim == 0
        else result
    )

def poling_period_at_reference_temperature(
    poling_period: npt.ArrayLike,
    temperature: float,
) -> typing.Union[
    float,
    npt.NDArray[np.float64],
]:
    r"""
    Convert a physical poling period to its value at 19 degrees Celsius.

    .. math::

        \Lambda_{19}
        =
        \frac{\Lambda(T)}
        {
            1
            + \alpha(T - 19)
            + \beta(T - 19)^2
        },

    where

    .. math::

        \alpha = 1.53 \times 10^{-5}

    and

    .. math::

        \beta = 5.3 \times 10^{-9}.

    Parameters
    ----------
    poling_period
        Physical poling period at the specified temperature, in metres.
    temperature
        Crystal temperature in degrees Celsius.

    Returns
    -------
    float or numpy.ndarray
        Equivalent poling period at 19 degrees Celsius, in metres.
    """
    result = (
        np.asarray(poling_period)
        / _poling_thermal_expansion_factor(
            temperature
        )
    )

    return (
        float(result)
        if result.ndim == 0
        else result
    )

def find_poling_period(
    pump_wavelength: npt.ArrayLike,
    signal_wavelength: npt.ArrayLike,
    idler_wavelength: npt.ArrayLike,
    temperature: float,
    material: NonlinearMaterial,
    pump_axis: RefractiveIndexAxis,
    signal_axis: RefractiveIndexAxis,
    idler_axis: RefractiveIndexAxis,
    qpm_order: int = 1,
) -> typing.Union[
    float,
    npt.NDArray[np.float64],
]:
    r"""
    Calculate the poling period required for quasi-phase matching.

    The physical poling period at the specified temperature is found
    from the perfect phase-matching condition

    .. math::

        \Delta k
        =
        k_p - k_s - k_i
        - \frac{2\pi m}{\Lambda(T)}
        = 0,

    giving

    .. math::

        \Lambda(T)
        =
        \frac{2\pi m}
             {k_p - k_s - k_i},

    where :math:`m` is the quasi-phase-matching order.

    The returned value is the equivalent poling period at 19 degrees
    Celsius, consistent with :func:`poling_period_at_temperature`.

    Parameters
    ----------
    pump_wavelength
        Pump wavelength in metres.
    signal_wavelength
        Signal wavelength in metres.
    idler_wavelength
        Idler wavelength in metres.
    temperature
        Crystal temperature in degrees Celsius.
    material
        Nonlinear optical material.
    pump_axis
        Refractive-index axis of the pump.
    signal_axis
        Refractive-index axis of the signal.
    idler_axis
        Refractive-index axis of the idler.
    qpm_order
        Quasi-phase-matching order. Default is 1.

    Returns
    -------
    float or numpy.ndarray
        Required poling period at 19 degrees Celsius, in metres.
    """
    n_p = material.refractive_index(
        pump_wavelength,
        temperature,
        pump_axis,
    )

    n_s = material.refractive_index(
        signal_wavelength,
        temperature,
        signal_axis,
    )

    n_i = material.refractive_index(
        idler_wavelength,
        temperature,
        idler_axis,
    )

    k_p = wavevector(
        pump_wavelength,
        n_p,
    )

    k_s = wavevector(
        signal_wavelength,
        n_s,
    )

    k_i = wavevector(
        idler_wavelength,
        n_i,
    )

    poling_period = (
        2 * np.pi * qpm_order
        / (k_p - k_s - k_i)
    )

    return poling_period_at_reference_temperature(
        poling_period,
        temperature,
    )