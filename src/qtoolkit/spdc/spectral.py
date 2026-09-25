import typing

import numpy as np
import numpy.typing as npt

from qtoolkit.constants import SPEED_OF_LIGHT

from .materials import (
    NonlinearMaterial,
    RefractiveIndexAxis,
)
from .phasematching import wavevector_mismatch


def angular_frequency(
    wavelength: npt.ArrayLike,
) -> typing.Union[
    float,
    npt.NDArray[np.float64],
]:
    r"""
    Convert vacuum wavelength to angular frequency.

    The angular frequency is

    .. math::

        \omega
        =
        \frac{2\pi c}{\lambda},

    where :math:`c` is the speed of light in vacuum and
    :math:`\lambda` is the vacuum wavelength.

    Parameters
    ----------
    wavelength
        Vacuum wavelength in metres.

    Returns
    -------
    float or numpy.ndarray
        Angular frequency in radians per second.

    Raises
    ------
    ValueError
        If any wavelength is less than or equal to zero.
    """
    wavelength_array = np.asarray(
        wavelength,
        dtype=np.float64,
    )

    if np.any(wavelength_array <= 0):
        raise ValueError(
            'Wavelength must be greater than zero.'
        )

    result = (
        2
        * np.pi
        * SPEED_OF_LIGHT
        / wavelength_array
    )

    return (
        float(result)
        if result.ndim == 0
        else result
    )


def wavelength_from_angular_frequency(
    angular_frequency: npt.ArrayLike,
) -> typing.Union[
    float,
    npt.NDArray[np.float64],
]:
    r"""
    Convert angular frequency to vacuum wavelength.

    The vacuum wavelength is

    .. math::

        \lambda
        =
        \frac{2\pi c}{\omega},

    where :math:`c` is the speed of light in vacuum and
    :math:`\omega` is the angular frequency.

    Parameters
    ----------
    angular_frequency
        Angular frequency in radians per second.

    Returns
    -------
    float or numpy.ndarray
        Vacuum wavelength in metres.

    Raises
    ------
    ValueError
        If any angular frequency is less than or equal to zero.
    """
    angular_frequency_array = np.asarray(
        angular_frequency,
        dtype=np.float64,
    )

    if np.any(angular_frequency_array <= 0):
        raise ValueError(
            'Angular frequency must be greater than zero.'
        )

    result = (
        2
        * np.pi
        * SPEED_OF_LIGHT
        / angular_frequency_array
    )

    return (
        float(result)
        if result.ndim == 0
        else result
    )


def gaussian_pump_amplitude(
    pump_angular_frequency: npt.ArrayLike,
    central_angular_frequency: float,
    bandwidth: float,
) -> typing.Union[
    float,
    npt.NDArray[np.float64],
]:
    r"""
    Calculate a Gaussian pump spectral amplitude.

    The pump spectral amplitude is defined as

    .. math::

        \alpha(\omega)
        =
        \exp\left[
            -\frac{
                (\omega-\omega_{p,0})^2
            }{
                2\sigma_p^2
            }
        \right],

    where :math:`\omega_{p,0}` is the central pump angular frequency
    and :math:`\sigma_p` is the standard deviation of the pump
    spectral amplitude.

    Parameters
    ----------
    pump_angular_frequency
        Pump angular frequency in radians per second.
    central_angular_frequency
        Central pump angular frequency in radians per second.
    bandwidth
        Standard deviation of the Gaussian spectral amplitude in
        radians per second.

    Returns
    -------
    float or numpy.ndarray
        Gaussian pump spectral amplitude.

    Raises
    ------
    ValueError
        If the bandwidth is less than or equal to zero.

    Notes
    -----
    ``bandwidth`` is the standard deviation of the *amplitude*
    distribution, not its intensity distribution and not its full
    width at half maximum.
    """
    if bandwidth <= 0:
        raise ValueError(
            'Pump bandwidth must be greater than zero.'
        )

    pump_angular_frequency_array = np.asarray(
        pump_angular_frequency,
        dtype=np.float64,
    )

    result = np.exp(
        -(
            pump_angular_frequency_array
            - central_angular_frequency
        )**2
        / (2 * bandwidth**2)
    )

    return (
        float(result)
        if result.ndim == 0
        else result
    )


def phase_matching_amplitude(
    delta_k: npt.ArrayLike,
    crystal_length: float,
) -> typing.Union[
    complex,
    npt.NDArray[np.complex128],
]:
    r"""
    Calculate the longitudinal phase-matching amplitude.

    For a uniform nonlinear crystal of length :math:`L`, the
    phase-matching amplitude is

    .. math::

        \Phi
        =
        \operatorname{sinc}
        \left(
            \frac{\Delta k L}{2}
        \right)
        \exp
        \left(
            i\frac{\Delta k L}{2}
        \right),

    where :math:`\Delta k` is the wavevector mismatch.

    Parameters
    ----------
    delta_k
        Wavevector mismatch in radians per metre.
    crystal_length
        Crystal length in metres.

    Returns
    -------
    complex or numpy.ndarray
        Complex phase-matching amplitude.

    Raises
    ------
    ValueError
        If the crystal length is less than or equal to zero.

    Notes
    -----
    NumPy defines ``numpy.sinc(x)`` as

    .. math::

        \frac{\sin(\pi x)}{\pi x}.

    The argument is therefore divided by :math:`\pi` when evaluating
    the conventional unnormalised sinc function

    .. math::

        \frac{\sin(x)}{x}.
    """
    if crystal_length <= 0:
        raise ValueError(
            'Crystal length must be greater than zero.'
        )

    argument = (
        np.asarray(
            delta_k,
            dtype=np.float64,
        )
        * crystal_length
        / 2
    )

    result = (
        np.sinc(argument / np.pi)
        * np.exp(1j * argument)
    )

    return (
        complex(result)
        if result.ndim == 0
        else result.astype(
            np.complex128,
            copy=False,
        )
    )


def joint_spectral_amplitude(
    signal_wavelength: npt.ArrayLike,
    idler_wavelength: npt.ArrayLike,
    pump_wavelength: float,
    pump_bandwidth_std: float,
    crystal_length: float,
    temperature: float,
    poling_period: float,
    material: NonlinearMaterial,
    pump_axis: RefractiveIndexAxis,
    signal_axis: RefractiveIndexAxis,
    idler_axis: RefractiveIndexAxis,
    qpm_order: int = 1,
    pump_angle: typing.Optional[float] = None,
    signal_angle: typing.Optional[float] = None,
    idler_angle: typing.Optional[float] = None,
) -> typing.Union[
    complex,
    npt.NDArray[np.complex128],
]:
    r"""
    Calculate the joint spectral amplitude of an SPDC source.

    The joint spectral amplitude (JSA) is calculated as

    .. math::

        f(\omega_s,\omega_i)
        =
        \alpha(\omega_s+\omega_i)
        \Phi(\omega_s,\omega_i),

    where :math:`\alpha` is the pump spectral amplitude and
    :math:`\Phi` is the crystal phase-matching amplitude.

    The pump spectral amplitude is modelled as a Gaussian centred at
    the angular frequency corresponding to ``pump_wavelength``.

    Parameters
    ----------
    signal_wavelength
        Signal vacuum wavelength in metres. May be a scalar or array.
    idler_wavelength
        Idler vacuum wavelength in metres. May be a scalar or array.
        Signal and idler wavelength arrays must be broadcastable.
    pump_wavelength
        Central pump vacuum wavelength in metres.
    pump_bandwidth_std
        Standard deviation of the pump spectral *amplitude* in
        radians per second.
    crystal_length
        Nonlinear crystal length in metres.
    temperature
        Crystal temperature in degrees Celsius.
    poling_period
        Poling period at the 19 degrees Celsius reference temperature,
        in metres.
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
    pump_angle
        Optional pump propagation angle in radians.
    signal_angle
        Optional signal propagation angle in radians.
    idler_angle
        Optional idler propagation angle in radians.
        
    Returns
    -------
    complex or numpy.ndarray
        Complex joint spectral amplitude.

    Notes
    -----
    Although signal and idler coordinates are supplied as wavelengths,
    the pump envelope is evaluated in angular-frequency space. For
    every signal-idler pair,

    .. math::

        \omega
        =
        \omega_s+\omega_i.

    This avoids applying a Gaussian pump envelope directly in
    wavelength space, where the energy-conservation relationship is
    nonlinear.

    The returned JSA is an unnormalised spectral amplitude.
    """
    signal_angular_frequency = (
        angular_frequency(
            signal_wavelength
        )
    )

    idler_angular_frequency = (
        angular_frequency(
            idler_wavelength
        )
    )

    central_pump_angular_frequency = (
        angular_frequency(
            pump_wavelength
        )
    )

    generated_pump_angular_frequency = (
        np.asarray(signal_angular_frequency)
        + np.asarray(idler_angular_frequency)
    )

    pump_amplitude = gaussian_pump_amplitude(
        generated_pump_angular_frequency,
        central_pump_angular_frequency,
        pump_bandwidth_std,
    )

    delta_k = wavevector_mismatch(
        pump_wavelength=(
            wavelength_from_angular_frequency(
                generated_pump_angular_frequency
            )
        ),
        signal_wavelength=signal_wavelength,
        idler_wavelength=idler_wavelength,
        temperature=temperature,
        poling_period=poling_period,
        material=material,
        pump_axis=pump_axis,
        signal_axis=signal_axis,
        idler_axis=idler_axis,
        qpm_order=qpm_order,
        pump_angle=pump_angle,
        signal_angle=signal_angle,
        idler_angle=idler_angle,
    )

    phase_matching = phase_matching_amplitude(
        delta_k,
        crystal_length,
    )

    result = (
        np.asarray(pump_amplitude)
        * np.asarray(phase_matching)
    )

    return (
        complex(result)
        if result.ndim == 0
        else result.astype(
            np.complex128,
            copy=False,
        )
    )


def joint_spectral_intensity(
    jsa: npt.ArrayLike,
) -> typing.Union[
    float,
    npt.NDArray[np.float64],
]:
    r"""
    Calculate joint spectral intensity from a joint spectral amplitude.

    The joint spectral intensity (JSI) is

    .. math::

        \mathrm{JSI}
        =
        |f|^2,

    where :math:`f` is the joint spectral amplitude.

    Parameters
    ----------
    jsa
        Complex joint spectral amplitude.

    Returns
    -------
    float or numpy.ndarray
        Joint spectral intensity.
    """
    result = (
        np.abs(
            np.asarray(jsa)
        )**2
    )

    return (
        float(result)
        if result.ndim == 0
        else result.astype(
            np.float64,
            copy=False,
        )
    )

def pump_wavelength_fwhm_to_angular_frequency_std(
    central_wavelength: float,
    wavelength_fwhm: float,
) -> float:
    r"""
    Convert pump wavelength FWHM to angular-frequency amplitude width.

    ``joint_spectral_amplitude`` describes the pump using a Gaussian
    spectral amplitude,

    .. math::

        \alpha(\omega)
        =
        \exp\left[
            -\frac{(\omega-\omega_0)^2}
            {2\sigma_\omega^2}
        \right].

    Experimental pump bandwidths are commonly specified as the FWHM
    of the spectral intensity in wavelength. This function converts
    that quantity to the standard deviation of the Gaussian spectral
    amplitude in angular frequency.

    Parameters
    ----------
    central_wavelength
        Central pump wavelength in metres.
    wavelength_fwhm
        Intensity FWHM in wavelength, in metres.

    Returns
    -------
    float
        Standard deviation of the Gaussian spectral amplitude in
        radians per second.

    Raises
    ------
    ValueError
        If either wavelength is not positive, or if the FWHM would
        extend to zero wavelength.

    Notes
    -----
    The conversion from the two wavelength half-maximum points to
    angular frequency is performed exactly rather than using the
    narrow-band approximation.
    """
    if central_wavelength <= 0:
        raise ValueError(
            'Central wavelength must be greater than zero.'
        )

    if wavelength_fwhm <= 0:
        raise ValueError(
            'Wavelength FWHM must be greater than zero.'
        )

    lower_wavelength = (
        central_wavelength
        - wavelength_fwhm / 2
    )

    upper_wavelength = (
        central_wavelength
        + wavelength_fwhm / 2
    )

    if lower_wavelength <= 0:
        raise ValueError(
            'Wavelength FWHM is too large for the '
            'specified central wavelength.'
        )

    upper_angular_frequency = angular_frequency(
        lower_wavelength
    )

    lower_angular_frequency = angular_frequency(
        upper_wavelength
    )

    intensity_fwhm = (
        upper_angular_frequency
        - lower_angular_frequency
    )

    return float(
        intensity_fwhm
        / (
            2
            * np.sqrt(np.log(2))
        )
    )