"""
Calculate the joint spectral intensity of an SPDC source.

This example models a non-degenerate type-0 SPDC source using a
20 mm, 5% MgO-doped periodically poled lithium niobate crystal.

The source is pumped at 523.5 nm and is phase matched for photon pairs
near 785 nm and 1572 nm. The crystal propagation angle required for
phase matching is calculated before evaluating the joint spectral
amplitude.

The joint spectral intensity (JSI) is then plotted as a function of
signal and idler wavelength.
"""

import pathlib

import matplotlib.pyplot as plt
import numpy as np

from qtoolkit.spdc import (
    MgOLithiumNiobate,
    RefractiveIndexAxis,
    conjugate_wavelength,
    find_phase_matching_angle,
    joint_spectral_amplitude,
    joint_spectral_intensity,
    pump_wavelength_fwhm_to_angular_frequency_std,
)

def fwhm(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    """Calculate the FWHM of a sampled distribution."""
    y = y / np.max(y)

    above_half_maximum = np.flatnonzero(
        y >= 0.5
    )

    if len(above_half_maximum) < 2:
        raise ValueError(
            'Insufficient samples to determine FWHM.'
        )

    left = above_half_maximum[0]
    right = above_half_maximum[-1]

    if left == 0 or right == len(y) - 1:
        raise ValueError(
            'Distribution does not fall below half maximum '
            'within the sampled range.'
        )

    left_crossing = np.interp(
        0.5,
        [
            y[left - 1],
            y[left],
        ],
        [
            x[left - 1],
            x[left],
        ],
    )

    right_crossing = np.interp(
        0.5,
        [
            y[right + 1],
            y[right],
        ],
        [
            x[right + 1],
            x[right],
        ],
    )

    return float(
        right_crossing
        - left_crossing
    )

def main() -> None:
    material = MgOLithiumNiobate()
    axis = RefractiveIndexAxis.EXTRAORDINARY

    # Source parameters.
    pump_wavelength = 523.5e-9
    pump_wavelength_fwhm = 92e-12

    crystal_length = 20e-3
    temperature = 84.0

    # Poling period at the 19 degrees Celsius reference temperature
    # used by qtoolkit's thermal-expansion model.
    poling_period = 7.15e-6

    # Type-0 e -> e + e SPDC is considered.
    #
    # Specify the short-wavelength photon and determine its
    # energy-conserving conjugate.
    idler_centre = 785e-9

    signal_centre = conjugate_wavelength(
        pump_wavelength,
        idler_centre,
    )

    # Determine the common propagation angle that satisfies the
    # quasi-phase-matching condition at the central wavelengths.
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

    print(
        'Central wavelengths: '
        f'{idler_centre * 1e9:.3f} nm, '
        f'{signal_centre * 1e9:.3f} nm'
    )

    print(
        'Phase-matching angle: '
        f'{np.rad2deg(angle):.6f} degrees'
    )

    # Convert the measured pump intensity FWHM in wavelength to the
    # angular-frequency amplitude standard deviation expected by the
    # JSA calculation.
    pump_bandwidth_std = (
        pump_wavelength_fwhm_to_angular_frequency_std(
            pump_wavelength,
            pump_wavelength_fwhm,
        )
    )

    # Spectral ranges surrounding the phase-matched wavelengths.
    idler_wavelengths = np.linspace(
        783e-9,
        787e-9,
        1000,
    )

    signal_wavelengths = np.linspace(
        1566e-9,
        1578e-9,
        1000,
    )

    idler_grid, signal_grid = np.meshgrid(
        idler_wavelengths,
        signal_wavelengths,
        indexing='xy',
    )

    # Calculate the joint spectral amplitude.
    jsa = joint_spectral_amplitude(
        signal_wavelength=signal_grid,
        idler_wavelength=idler_grid,
        pump_wavelength=pump_wavelength,
        pump_bandwidth_std=pump_bandwidth_std,
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

    # The joint spectral intensity is |JSA|^2.
    jsi = joint_spectral_intensity(
        jsa
    )

    # Normalise for plotting.
    jsi /= np.max(jsi)
    # Calculate wavelength-domain marginal spectra.
    idler_spectrum = np.trapezoid(
        jsi,
        signal_wavelengths,
        axis=0,
    )

    signal_spectrum = np.trapezoid(
        jsi,
        idler_wavelengths,
        axis=1,
    )

    idler_spectrum /= np.max(
        idler_spectrum
    )

    signal_spectrum /= np.max(
        signal_spectrum
    )

    idler_fwhm = fwhm(
        idler_wavelengths,
        idler_spectrum,
    )

    signal_fwhm = fwhm(
        signal_wavelengths,
        signal_spectrum,
    )

    print(
        'Idler FWHM: '
        f'{idler_fwhm * 1e9:.3f} nm'
    )

    print(
        'Signal FWHM: '
        f'{signal_fwhm * 1e9:.3f} nm'
    )

    # Find the location of the maximum as a simple numerical check.
    maximum_index = np.unravel_index(
        np.argmax(jsi),
        jsi.shape,
    )

    peak_idler = (
        idler_grid[maximum_index]
    )

    peak_signal = (
        signal_grid[maximum_index]
    )

    print(
        'JSI maximum: '
        f'{peak_idler * 1e9:.3f} nm, '
        f'{peak_signal * 1e9:.3f} nm'
    )

    # Plot the JSI.
    fig, ax = plt.subplots(
        constrained_layout=True,
    )

    image = ax.pcolormesh(
        idler_wavelengths * 1e9,
        signal_wavelengths * 1e9,
        jsi,
        shading='auto',
    )

    ax.plot(
        idler_centre * 1e9,
        signal_centre * 1e9,
        marker='x',
        linestyle='none',
        label='Phase-matched centre',
    )

    ax.set(
        xlabel='Idler wavelength (nm)',
        ylabel='Signal wavelength (nm)',
        title='Joint spectral intensity',
    )

    fig.colorbar(
        image,
        ax=ax,
        label='Normalised JSI',
    )

    ax.legend()

    plt.show()
    fig.savefig(
        pathlib.Path(__file__).with_suffix('.png'),
        dpi='figure',
        bbox_inches='tight',
    )

if __name__ == '__main__':
    main()