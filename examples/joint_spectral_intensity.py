"""
Calculate and plot the joint spectral intensity of an SPDC source.

Unlike the one-dimensional phase-matching spectrum, the JSI includes
both the finite pump spectrum and the crystal phase-matching function.
The result shows the spectral correlations between the generated
signal and idler photons.
"""

import pathlib

import matplotlib.pyplot as plt
import numpy as np

from qtoolkit.spdc.materials import (
    MgOLithiumNiobate,
    RefractiveIndexAxis,
)
from qtoolkit.spdc.phasematching import (
    find_phase_matching_wavelength_pairs,
)
from qtoolkit.spdc.spectral import (
    joint_spectral_amplitude,
    joint_spectral_intensity,
)


def main() -> None:
    material = MgOLithiumNiobate()

    pump_wavelength = 523.5e-9
    temperature = 84.0
    poling_period = 7.15e-6
    crystal_length = 20e-3

    # Gaussian pump amplitude bandwidth in rad/s.
    pump_bandwidth_std = 2.5e11

    pairs = find_phase_matching_wavelength_pairs(
        pump_wavelength=pump_wavelength,
        temperature=temperature,
        poling_period=poling_period,
        material=material,
        pump_axis=RefractiveIndexAxis.EXTRAORDINARY,
        signal_axis=RefractiveIndexAxis.EXTRAORDINARY,
        idler_axis=RefractiveIndexAxis.EXTRAORDINARY,
        signal_wavelength_bounds=(700e-9,2e-6)
    )

    signal_centre, idler_centre = pairs[0]

    signal_wavelengths = np.linspace(
        signal_centre - 4e-9,
        signal_centre + 4e-9,
        500,
    )

    idler_wavelengths = np.linspace(
        idler_centre - 15e-9,
        idler_centre + 15e-9,
        500,
    )

    signal_grid, idler_grid = np.meshgrid(
        signal_wavelengths,
        idler_wavelengths,
    )

    jsa = joint_spectral_amplitude(
        signal_wavelength=signal_grid,
        idler_wavelength=idler_grid,
        pump_wavelength=pump_wavelength,
        pump_bandwidth_std=pump_bandwidth_std,
        crystal_length=crystal_length,
        temperature=temperature,
        poling_period=poling_period,
        material=material,
        pump_axis=RefractiveIndexAxis.EXTRAORDINARY,
        signal_axis=RefractiveIndexAxis.EXTRAORDINARY,
        idler_axis=RefractiveIndexAxis.EXTRAORDINARY,
    )

    jsi = joint_spectral_intensity(
        jsa
    )

    jsi /= np.max(jsi)

    fig, ax = plt.subplots(
        constrained_layout=True,
    )

    image = ax.pcolormesh(
        signal_wavelengths * 1e9,
        idler_wavelengths * 1e9,
        jsi,
        shading="auto",
    )

    ax.set(
        xlabel="Signal wavelength (nm)",
        ylabel="Idler wavelength (nm)",
        title="Joint spectral intensity",
    )

    fig.colorbar(
        image,
        ax=ax,
        label="Normalised JSI",
    )

    plt.show()
    fig.savefig(
        pathlib.Path(__file__).with_suffix('.png'),
        dpi='figure',
        bbox_inches='tight',
    )

if __name__ == "__main__":
    main()