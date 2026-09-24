"""
Plot the phase-matching spectrum of a periodically poled crystal.

The idler wavelength is calculated from energy conservation for each
signal wavelength. The resulting curve therefore represents a
one-dimensional slice through the joint spectrum for a monochromatic
pump.
"""

import pathlib

import matplotlib.pyplot as plt
import numpy as np

from qtoolkit.spdc.materials import (
    MgOLithiumNiobate,
    RefractiveIndexAxis,
)
from qtoolkit.spdc.phasematching import (
    conjugate_wavelength,
    wavevector_mismatch,
)
from qtoolkit.spdc.spectral import (
    phase_matching_amplitude,
)


def main() -> None:
    material = MgOLithiumNiobate()

    pump_wavelength = 523.5e-9
    temperature = 84.0
    poling_period = 7.15e-6
    crystal_length = 20e-3

    signal_wavelengths = np.linspace(
        785e-9,
        795e-9,
        2000,
    )

    idler_wavelengths = conjugate_wavelength(
        pump_wavelength,
        signal_wavelengths,
    )

    delta_k = wavevector_mismatch(
        pump_wavelength=pump_wavelength,
        signal_wavelength=signal_wavelengths,
        idler_wavelength=idler_wavelengths,
        temperature=temperature,
        poling_period=poling_period,
        material=material,
        pump_axis=RefractiveIndexAxis.EXTRAORDINARY,
        signal_axis=RefractiveIndexAxis.EXTRAORDINARY,
        idler_axis=RefractiveIndexAxis.EXTRAORDINARY,
    )

    amplitude = phase_matching_amplitude(
        delta_k,
        crystal_length,
    )

    intensity = np.abs(amplitude) ** 2
    intensity /= np.max(intensity)

    peak = np.argmax(intensity)

    print(
        "Peak wavelengths: "
        f"{signal_wavelengths[peak] * 1e9:.3f} nm, "
        f"{idler_wavelengths[peak] * 1e9:.3f} nm"
    )

    fig, ax = plt.subplots(
        constrained_layout=True,
    )

    ax.plot(
        signal_wavelengths * 1e9,
        intensity,
    )

    ax.axvline(
        signal_wavelengths[peak] * 1e9,
        linestyle="--",
        label="Phase-matching maximum",
    )

    ax.set(
        xlabel="Signal wavelength (nm)",
        ylabel="Normalised intensity",
        title="SPDC phase-matching spectrum",
    )

    ax.legend()

    plt.show()
    fig.savefig(
        pathlib.Path(__file__).with_suffix('.png'),
        dpi='figure',
        bbox_inches='tight',
    )

if __name__ == "__main__":
    main()