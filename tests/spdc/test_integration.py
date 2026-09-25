import numpy as np
import pytest

from qtoolkit.spdc import (
    MgOLithiumNiobate,
    RefractiveIndexAxis,
    conjugate_wavelength,
    find_phase_matching_angle,
    joint_spectral_amplitude,
    joint_spectral_intensity,
    pump_wavelength_fwhm_to_angular_frequency_std,
)


def test_non_degenerate_ppln_jsi_regression() -> None:
    material = MgOLithiumNiobate()
    axis = RefractiveIndexAxis.EXTRAORDINARY

    pump_wavelength = 523.5e-9
    pump_fwhm = 92e-12
    crystal_length = 20e-3
    temperature = 84.0
    poling_period = 7.15e-6

    idler_centre = 785e-9
    signal_centre = conjugate_wavelength(
        pump_wavelength,
        idler_centre,
    )

    angle = find_phase_matching_angle(
        pump_wavelength=pump_wavelength,
        signal_wavelength=signal_centre,
        idler_wavelength=idler_centre,
        temperature=temperature,
        poling_period=poling_period,
        material=material,
        pump_axis=axis,
        signal_axis=axis,
        idler_axis=axis,
    )

    assert np.rad2deg(angle) == pytest.approx(
        80.096284520,
        abs=1e-6,
    )

    pump_bandwidth = (
        pump_wavelength_fwhm_to_angular_frequency_std(
            pump_wavelength,
            pump_fwhm,
        )
    )

    idler_wavelengths = np.linspace(
        783e-9,
        787e-9,
        301,
    )
    signal_wavelengths = np.linspace(
        1566e-9,
        1578e-9,
        301,
    )

    idler_grid, signal_grid = np.meshgrid(
        idler_wavelengths,
        signal_wavelengths,
        indexing='xy',
    )

    jsa = joint_spectral_amplitude(
        signal_wavelength=signal_grid,
        idler_wavelength=idler_grid,
        pump_wavelength=pump_wavelength,
        pump_bandwidth_std=pump_bandwidth,
        crystal_length=crystal_length,
        temperature=temperature,
        poling_period=poling_period,
        material=material,
        pump_axis=axis,
        signal_axis=axis,
        idler_axis=axis,
        pump_angle=angle,
        signal_angle=angle,
        idler_angle=angle,
    )

    jsi = joint_spectral_intensity(jsa)
    maximum_index = np.unravel_index(
        np.argmax(jsi),
        jsi.shape,
    )

    peak_idler = idler_grid[maximum_index]
    peak_signal = signal_grid[maximum_index]

    assert peak_idler == pytest.approx(
        idler_centre,
        abs=0.03e-9,
    )
    assert peak_signal == pytest.approx(
        signal_centre,
        abs=0.08e-9,
    )
