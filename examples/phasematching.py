"""
Find phase-matched wavelengths for periodically poled MgO:LiNbO3.

This example calculates the signal and idler wavelengths that satisfy
energy conservation and quasi-phase matching for collinear, type-0 SPDC.

The poling period supplied to qtoolkit is defined at the reference
temperature used by the thermal-expansion model.
"""

from qtoolkit.spdc.materials import (
    MgOLithiumNiobate,
    RefractiveIndexAxis,
)
from qtoolkit.spdc.phasematching import (
    find_phase_matching_wavelength_pairs,
)


def main() -> None:
    material = MgOLithiumNiobate()

    pump_wavelength = 523.5e-9
    temperature = 84.0
    poling_period = 7.15e-6

    wavelength_pairs = find_phase_matching_wavelength_pairs(
        pump_wavelength=pump_wavelength,
        temperature=temperature,
        poling_period=poling_period,
        material=material,
        signal_axis=RefractiveIndexAxis.EXTRAORDINARY,
        idler_axis=RefractiveIndexAxis.EXTRAORDINARY,
        pump_axis=RefractiveIndexAxis.EXTRAORDINARY,
        signal_wavelength_bounds=(700e-9,2e-6)
    )

    print("Phase-matched wavelength pairs:")

    for signal, idler in wavelength_pairs:
        print(
            f"  {signal * 1e9:.3f} nm, "
            f"{idler * 1e9:.3f} nm"
        )


if __name__ == "__main__":
    main()