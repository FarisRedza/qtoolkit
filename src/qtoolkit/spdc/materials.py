from abc import ABC, abstractmethod
import typing
from enum import Enum

import numpy as np
import numpy.typing as npt


class RefractiveIndexAxis(Enum):
    ORDINARY = 'ordinary'
    EXTRAORDINARY = 'extraordinary'


class NonlinearMaterial(ABC):
    """Base class for nonlinear optical material models."""

    @abstractmethod
    def refractive_index(
        self,
        wavelength: npt.ArrayLike,
        temperature: float,
        axis: RefractiveIndexAxis,
    ) -> typing.Union[float, npt.NDArray[np.float64]]:
        """Calculate the refractive index."""
        ...


class MgOLithiumNiobate(NonlinearMaterial):
    r"""
    5 mol% MgO-doped congruent lithium niobate.

    Implements the temperature-dependent Sellmeier equations from
    Gayer et al., Applied Physics B 91, 343-348 (2008).

    The refractive index is calculated using

    .. math::

        n^2(\lambda, T)
        =
        a_1 + b_1 f
        + \frac{a_2 + b_2 f}
               {\lambda^2 - (a_3 + b_3 f)^2}
        + \frac{a_4 + b_4 f}
               {\lambda^2 - a_5^2}
        - a_6 \lambda^2,

    where

    .. math::

        f = (T - 24.5)(T + 570.82).

    In the Sellmeier equation, :math:`\lambda` is expressed in
    micrometres and :math:`T` in degrees Celsius. Wavelengths supplied
    to this class are expressed in metres and converted internally.

    The extraordinary refractive-index equation is reported as valid
    over wavelengths from 0.5 to 4 um and temperatures from 20 to
    200 degrees C.

    The ordinary refractive-index equation was experimentally verified
    over a more limited range and is considered reliable up to 1.62 um
    and approximately 20 to 100 degrees C.

    References
    ----------
    O. Gayer, Z. Sacks, E. Galun, and A. Arie,
    "Temperature and wavelength dependent refractive index equations
    for MgO-doped congruent and stoichiometric LiNbO3",
    Applied Physics B 91, 343-348 (2008).
    """

    @staticmethod
    def _temperature_factor(
        temperature: float,
    ) -> float:
        return (
            (temperature - 24.5)
            * (temperature + 570.82)
        )

    def refractive_index(
        self,
        wavelength: npt.ArrayLike,
        temperature: float,
        axis: RefractiveIndexAxis,
    ) -> typing.Union[float, npt.NDArray[np.float64]]:
        if axis == 'ordinary':
            return self.ordinary_refractive_index(
                wavelength,
                temperature,
            )

        if axis == 'extraordinary':
            return self.extraordinary_refractive_index(
                wavelength,
                temperature,
            )

        raise ValueError(
            "polarisation must be 'ordinary' or 'extraordinary'."
        )

    def extraordinary_refractive_index(
        self,
        wavelength: npt.ArrayLike,
        temperature: float,
    ) -> typing.Union[float, npt.NDArray[np.float64]]:
        wavelength_um = (
            np.asarray(wavelength)
            * 1e6
        )

        f = self._temperature_factor(
            temperature
        )

        n_squared = (
            5.756
            + 2.860e-6 * f
            + (
                0.0983 + 4.700e-8 * f
            ) / (
                wavelength_um**2
                - (
                    0.2020
                    + 6.113e-8 * f
                )**2
            )
            + (
                189.32
                + 1.516e-4 * f
            ) / (
                wavelength_um**2
                - 12.52**2
            )
            - 1.32e-2 * wavelength_um**2
        )

        result = np.sqrt(n_squared)

        return (
            float(result)
            if result.ndim == 0
            else result
        )

    def ordinary_refractive_index(
        self,
        wavelength: npt.ArrayLike,
        temperature: float,
    ) -> typing.Union[float, npt.NDArray[np.float64]]:
        wavelength_um = (
            np.asarray(wavelength)
            * 1e6
        )

        f = self._temperature_factor(
            temperature
        )

        n_squared = (
            5.653
            + 7.941e-7 * f
            + (
                0.1185 + 3.134e-8 * f
            ) / (
                wavelength_um**2
                - (
                    0.2091
                    - 4.641e-9 * f
                )**2
            )
            + (
                89.61
                - 2.188e-6 * f
            ) / (
                wavelength_um**2
                - 10.85**2
            )
            - 1.97e-2 * wavelength_um**2
        )

        result = np.sqrt(n_squared)

        return (
            float(result)
            if result.ndim == 0
            else result
        )