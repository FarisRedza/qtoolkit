import typing

import numpy as np
import numpy.typing as npt
import scipy

from .materials import (
    NonlinearMaterial,
    RefractiveIndexAxis,
)

def conjugate_wavelength(
    pump_wavelength: npt.ArrayLike,
    wavelength: npt.ArrayLike,
) -> typing.Union[
    float,
    npt.NDArray[np.float64],
]:
    r"""
    Calculate the conjugate wavelength from energy conservation.

    For spontaneous parametric down-conversion,

    .. math::

        \frac{1}{\lambda_p}
        =
        \frac{1}{\lambda_s}
        +
        \frac{1}{\lambda_i}.

    Therefore, given the pump wavelength and either the signal or
    idler wavelength,

    .. math::

        \lambda_c
        =
        \left(
            \frac{1}{\lambda_p}
            -
            \frac{1}{\lambda}
        \right)^{-1}.

    Parameters
    ----------
    pump_wavelength
        Pump wavelength in metres.
    wavelength
        Signal or idler wavelength in metres.

    Returns
    -------
    float or numpy.ndarray
        Conjugate wavelength in metres.
    """
    pump = np.asarray(pump_wavelength)
    wavelength = np.asarray(wavelength)

    if np.any(wavelength <= pump):
        raise ValueError(
            'Signal and idler wavelengths must be greater '
            'than the pump wavelength.'
        )

    result = 1 / (1 / pump - 1 / wavelength)

    return (
        float(result)
        if result.ndim == 0
        else result
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

_WAVELENGTH_ROOT_XTOL = 1e-15
_ROOT_RTOL = 1e-12


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

    The temperature-dependent poling period is calculated using

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

        \alpha = 1.53 \times 10^{-5}\,\mathrm{K}^{-1}

    and

    .. math::

        \beta = 5.3 \times 10^{-9}\,\mathrm{K}^{-2}.

    Here, :math:`\Lambda_{19}` is the grating period defined at the
    19 degrees Celsius clean-room temperature used during the poling
    process.

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

    Notes
    -----
    The expression and 19 degrees Celsius reference convention follow
    Paul et al., Applied Physics B 86, 111-115 (2007). The thermal
    expansion coefficients originate from Kim and Smith,
    J. Appl. Phys. 40, 4637 (1969).

    This function may be moved to a method in `NonlinearMaterial`
    due to thermal expansion being material dependent.
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


def _temperature_mismatch(
    temperature: float,
    pump_wavelength: float,
    signal_wavelength: float,
    idler_wavelength: float,
    poling_period: float,
    material: NonlinearMaterial,
    pump_axis: RefractiveIndexAxis,
    signal_axis: RefractiveIndexAxis,
    idler_axis: RefractiveIndexAxis,
    qpm_order: int,
) -> float:
    return float(
        wavevector_mismatch(
            pump_wavelength=pump_wavelength,
            signal_wavelength=signal_wavelength,
            idler_wavelength=idler_wavelength,
            temperature=temperature,
            poling_period=poling_period,
            material=material,
            pump_axis=pump_axis,
            signal_axis=signal_axis,
            idler_axis=idler_axis,
            qpm_order=qpm_order,
        )
    )


def find_phase_matching_temperature(
    pump_wavelength: float,
    signal_wavelength: float,
    idler_wavelength: float,
    poling_period: float,
    material: NonlinearMaterial,
    pump_axis: RefractiveIndexAxis,
    signal_axis: RefractiveIndexAxis,
    idler_axis: RefractiveIndexAxis,
    temperature_bounds: typing.Tuple[
        float,
        float,
    ] = (20.0, 200.0),
    qpm_order: int = 1,
) -> float:
    r"""
    Find the temperature required for quasi-phase matching.

    The phase-matching temperature is found numerically by solving

    .. math::

        \Delta k(T) = 0,

    where

    .. math::

        \Delta k(T)
        =
        k_p(T)
        - k_s(T)
        - k_i(T)
        - \frac{2\pi m}{\Lambda(T)}.

    Both the refractive indices and physical poling period are
    temperature dependent.

    Parameters
    ----------
    pump_wavelength
        Pump wavelength in metres.
    signal_wavelength
        Signal wavelength in metres.
    idler_wavelength
        Idler wavelength in metres.
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
    temperature_bounds
        Lower and upper bounds of the temperature interval to search,
        in degrees Celsius. Default is 20 to 200 degrees Celsius.
    qpm_order
        Quasi-phase-matching order. Default is 1.

    Returns
    -------
    float
        Phase-matching temperature in degrees Celsius.

    Raises
    ------
    ValueError
        If the phase-matching condition does not have a root within
        the specified temperature interval.
    """
    def mismatch(
        temperature: float,
    ) -> float:
        return _temperature_mismatch(
            temperature=temperature,
            pump_wavelength=pump_wavelength,
            signal_wavelength=signal_wavelength,
            idler_wavelength=idler_wavelength,
            poling_period=poling_period,
            material=material,
            pump_axis=pump_axis,
            signal_axis=signal_axis,
            idler_axis=idler_axis,
            qpm_order=qpm_order,
        )

    lower_temperature, upper_temperature = temperature_bounds

    if lower_temperature >= upper_temperature:
        raise ValueError(
            'The lower temperature bound must be less than '
            'the upper temperature bound.'
        )

    try:
        return float(
            scipy.optimize.brentq(
                mismatch,
                lower_temperature,
                upper_temperature,
            )
        )
    except ValueError as error:
        raise ValueError(
            'No phase-matching temperature was found '
            f'between {lower_temperature} and '
            f'{upper_temperature} degrees Celsius.'
        ) from error


def find_phase_matching_temperatures(
    pump_wavelength: float,
    signal_wavelength: float,
    idler_wavelength: float,
    poling_period: float,
    material: NonlinearMaterial,
    pump_axis: RefractiveIndexAxis,
    signal_axis: RefractiveIndexAxis,
    idler_axis: RefractiveIndexAxis,
    temperature_bounds: typing.Tuple[
        float,
        float,
    ] = (20.0, 200.0),
    qpm_order: int = 1,
    samples: int = 1000,
) -> npt.NDArray[np.float64]:
    r"""
    Find phase-matching temperatures within an interval.

    The temperature interval is sampled to identify brackets in which

    .. math::

        \Delta k(T) = 0

    has a solution. Each bracketed root is then refined using Brent's
    method.

    Parameters
    ----------
    pump_wavelength
        Pump wavelength in metres.
    signal_wavelength
        Signal wavelength in metres.
    idler_wavelength
        Idler wavelength in metres.
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
    temperature_bounds
        Lower and upper bounds of the temperature interval to search,
        in degrees Celsius.
    qpm_order
        Quasi-phase-matching order. Default is 1.
    samples
        Number of temperatures used to search for root brackets.
        Default is 1000.

    Returns
    -------
    numpy.ndarray
        Phase-matching temperatures in degrees Celsius. An empty
        array is returned if no roots are found.

    Notes
    -----
    Roots are detected by sampling the requested temperature interval
    and identifying changes in the sign of the wavevector mismatch.
    Consequently, roots may be missed if multiple roots occur between
    adjacent sample points or if a root touches zero without changing
    sign.
    """
    lower_temperature, upper_temperature = temperature_bounds

    if lower_temperature >= upper_temperature:
        raise ValueError(
            'The lower temperature bound must be less than '
            'the upper temperature bound.'
        )

    if samples < 2:
        raise ValueError(
            'samples must be at least 2.'
        )

    def mismatch(
        temperature: float,
    ) -> float:
        return _temperature_mismatch(
            temperature=temperature,
            pump_wavelength=pump_wavelength,
            signal_wavelength=signal_wavelength,
            idler_wavelength=idler_wavelength,
            poling_period=poling_period,
            material=material,
            pump_axis=pump_axis,
            signal_axis=signal_axis,
            idler_axis=idler_axis,
            qpm_order=qpm_order,
        )

    temperatures = np.linspace(
        lower_temperature,
        upper_temperature,
        samples,
    )

    mismatches = np.asarray([
        mismatch(temperature)
        for temperature in temperatures
    ])

    roots = []

    for index in range(len(temperatures) - 1):
        lower = temperatures[index]
        upper = temperatures[index + 1]

        lower_mismatch = mismatches[index]
        upper_mismatch = mismatches[index + 1]

        if lower_mismatch == 0.0:
            roots.append(lower)
            continue

        if lower_mismatch * upper_mismatch < 0:
            roots.append(
                scipy.optimize.brentq(
                    mismatch,
                    lower,
                    upper,
                )
            )

    if mismatches[-1] == 0.0:
        roots.append(temperatures[-1])

    return np.asarray(
        roots,
        dtype=np.float64,
    )

def find_phase_matching_wavelengths(
    pump_wavelength: float,
    temperature: float,
    poling_period: float,
    material: NonlinearMaterial,
    pump_axis: RefractiveIndexAxis,
    signal_axis: RefractiveIndexAxis,
    idler_axis: RefractiveIndexAxis,
    signal_wavelength_bounds: typing.Tuple[
        float,
        float,
    ],
    qpm_order: int = 1,
) -> typing.Tuple[float, float]:
    r"""
    Find signal and idler wavelengths satisfying energy conservation
    and quasi-phase matching.

    The idler wavelength is determined from energy conservation,

    .. math::

        \lambda_i
        =
        \left(
            \frac{1}{\lambda_p}
            -
            \frac{1}{\lambda_s}
        \right)^{-1},

    and the signal wavelength is found numerically by solving

    .. math::

        \Delta k(\lambda_s, \lambda_i) = 0.

    The degenerate SPDC wavelength,

    .. math::

        \lambda_s = \lambda_i = 2\lambda_p,

    is checked explicitly when it lies inside the requested search
    interval. This is necessary because the phase-mismatch function
    can touch zero at degeneracy without changing sign, in which case
    a bracketed root finder alone would not detect the solution.

    Parameters
    ----------
    pump_wavelength
        Pump wavelength in metres.
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
    signal_wavelength_bounds
        Lower and upper bounds of the signal wavelength interval to
        search, in metres. Both bounds must be greater than the pump
        wavelength.
    qpm_order
        Quasi-phase-matching order. Default is 1.

    Returns
    -------
    tuple of float
        Phase-matched signal and idler wavelengths, in metres.

    Raises
    ------
    ValueError
        If the wavelength bounds are invalid or no phase-matching
        solution is found within the specified interval.
    """
    lower_wavelength, upper_wavelength = signal_wavelength_bounds

    if lower_wavelength >= upper_wavelength:
        raise ValueError(
            'The lower signal wavelength bound must be less than '
            'the upper signal wavelength bound.'
        )

    if lower_wavelength <= pump_wavelength:
        raise ValueError(
            'Signal wavelength bounds must be greater than '
            'the pump wavelength.'
        )

    def mismatch(
        signal_wavelength: float,
    ) -> float:
        idler_wavelength = conjugate_wavelength(
            pump_wavelength,
            signal_wavelength,
        )

        return float(
            wavevector_mismatch(
                pump_wavelength=pump_wavelength,
                signal_wavelength=signal_wavelength,
                idler_wavelength=idler_wavelength,
                temperature=temperature,
                poling_period=poling_period,
                material=material,
                pump_axis=pump_axis,
                signal_axis=signal_axis,
                idler_axis=idler_axis,
                qpm_order=qpm_order,
            )
        )

    degenerate_wavelength = 2 * pump_wavelength

    if (
        lower_wavelength
        <= degenerate_wavelength
        <= upper_wavelength
    ):
        degenerate_period = find_poling_period(
            pump_wavelength=pump_wavelength,
            signal_wavelength=degenerate_wavelength,
            idler_wavelength=degenerate_wavelength,
            temperature=temperature,
            material=material,
            pump_axis=pump_axis,
            signal_axis=signal_axis,
            idler_axis=idler_axis,
            qpm_order=qpm_order,
        )

        if np.isclose(
            poling_period,
            degenerate_period,
            rtol=1e-10,
            atol=0.0,
        ):
            return (
                degenerate_wavelength,
                degenerate_wavelength,
            )

    try:
        signal_wavelength = float(
            scipy.optimize.brentq(
                mismatch,
                lower_wavelength,
                upper_wavelength,
                xtol=_WAVELENGTH_ROOT_XTOL,
                rtol=_ROOT_RTOL,
            )
        )
    except ValueError as error:
        raise ValueError(
            'No phase-matching wavelength was found '
            f'between {lower_wavelength} and '
            f'{upper_wavelength} metres.'
        ) from error

    idler_wavelength = float(
        conjugate_wavelength(
            pump_wavelength,
            signal_wavelength,
        )
    )

    return (
        signal_wavelength,
        idler_wavelength,
    )

def find_phase_matching_wavelength_pairs(
    pump_wavelength: float,
    temperature: float,
    poling_period: float,
    material: NonlinearMaterial,
    pump_axis: RefractiveIndexAxis,
    signal_axis: RefractiveIndexAxis,
    idler_axis: RefractiveIndexAxis,
    signal_wavelength_bounds: typing.Tuple[
        float,
        float,
    ],
    qpm_order: int = 1,
    samples: int = 1000,
) -> npt.NDArray[np.float64]:
    r"""
    Find phase-matched signal and idler wavelength pairs.

    The signal-wavelength interval is sampled to identify solutions
    satisfying both energy conservation,

    .. math::

        \frac{1}{\lambda_p}
        =
        \frac{1}{\lambda_s}
        +
        \frac{1}{\lambda_i},

    and quasi-phase matching,

    .. math::

        \Delta k(\lambda_s, \lambda_i) = 0.

    For each trial signal wavelength, the idler wavelength is
    determined from energy conservation. Sign changes in the
    wavevector mismatch are refined using Brent's method.

    The degenerate solution,

    .. math::

        \lambda_s = \lambda_i = 2\lambda_p,

    is checked explicitly because it may correspond to a root that
    touches zero without changing sign.

    Parameters
    ----------
    pump_wavelength
        Pump wavelength in metres.
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
    signal_wavelength_bounds
        Lower and upper bounds of the signal wavelength interval to
        search, in metres. Both bounds must be greater than the pump
        wavelength.
    qpm_order
        Quasi-phase-matching order. Default is 1.
    samples
        Number of signal wavelengths used to search for root
        brackets. Default is 1000.

    Returns
    -------
    numpy.ndarray
        Array with shape ``(N, 2)`` containing phase-matched
        ``(signal_wavelength, idler_wavelength)`` pairs in metres.
        An empty array with shape ``(0, 2)`` is returned if no
        solutions are found.

    Notes
    -----
    Non-degenerate roots are detected by identifying changes in the
    sign of the wavevector mismatch. Consequently, sufficiently
    closely spaced roots may be missed if multiple roots occur
    between adjacent sample points.

    Degenerate phase matching is checked separately and therefore
    does not require a sign change.
    """
    lower_wavelength, upper_wavelength = (
        signal_wavelength_bounds
    )

    if lower_wavelength >= upper_wavelength:
        raise ValueError(
            'The lower signal wavelength bound must be less than '
            'the upper signal wavelength bound.'
        )

    if lower_wavelength <= pump_wavelength:
        raise ValueError(
            'Signal wavelength bounds must be greater than '
            'the pump wavelength.'
        )

    if samples < 2:
        raise ValueError(
            'samples must be at least 2.'
        )

    def mismatch(
        signal_wavelength: float,
    ) -> float:
        idler_wavelength = conjugate_wavelength(
            pump_wavelength,
            signal_wavelength,
        )

        return float(
            wavevector_mismatch(
                pump_wavelength=pump_wavelength,
                signal_wavelength=signal_wavelength,
                idler_wavelength=idler_wavelength,
                temperature=temperature,
                poling_period=poling_period,
                material=material,
                pump_axis=pump_axis,
                signal_axis=signal_axis,
                idler_axis=idler_axis,
                qpm_order=qpm_order,
            )
        )

    signal_wavelengths = np.linspace(
        lower_wavelength,
        upper_wavelength,
        samples,
    )

    mismatches = np.asarray([
        mismatch(signal_wavelength)
        for signal_wavelength in signal_wavelengths
    ])

    signal_roots = []

    for index in range(
        len(signal_wavelengths) - 1
    ):
        lower = signal_wavelengths[index]
        upper = signal_wavelengths[index + 1]

        lower_mismatch = mismatches[index]
        upper_mismatch = mismatches[index + 1]

        if lower_mismatch == 0.0:
            signal_roots.append(lower)
            continue

        if lower_mismatch * upper_mismatch < 0:
            signal_roots.append(
                scipy.optimize.brentq(
                    mismatch,
                    lower,
                    upper,
                    xtol=_WAVELENGTH_ROOT_XTOL,
                    rtol=_ROOT_RTOL,
                )
            )

    if mismatches[-1] == 0.0:
        signal_roots.append(
            signal_wavelengths[-1]
        )

    degenerate_wavelength = (
        2 * pump_wavelength
    )

    if (
        lower_wavelength
        <= degenerate_wavelength
        <= upper_wavelength
    ):
        degenerate_period = find_poling_period(
            pump_wavelength=pump_wavelength,
            signal_wavelength=degenerate_wavelength,
            idler_wavelength=degenerate_wavelength,
            temperature=temperature,
            material=material,
            pump_axis=pump_axis,
            signal_axis=signal_axis,
            idler_axis=idler_axis,
            qpm_order=qpm_order,
        )

        if np.isclose(
            poling_period,
            degenerate_period,
            rtol=1e-10,
            atol=0.0,
        ):
            signal_roots.append(
                degenerate_wavelength
            )

    if not signal_roots:
        return np.empty(
            (0, 2),
            dtype=np.float64,
        )

    signal_roots = np.asarray(
        signal_roots,
        dtype=np.float64,
    )

    signal_roots = np.unique(
        signal_roots
    )

    idler_roots = np.asarray([
        conjugate_wavelength(
            pump_wavelength,
            signal_wavelength,
        )
        for signal_wavelength in signal_roots
    ])

    return np.column_stack((
        signal_roots,
        idler_roots,
    ))