import pathlib

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize

from qtoolkit.misc_functions import dB_to_fraction
from qtoolkit.qkd import cw


def secure_key_rate(
    total_attenuation_db: float,
    brightness: float,
    coincidence_window: float,
    timing_imprecision: float,
    *,
    dark_count_rate: float = 1000.0,
    polarisation_error: float = 0.01,
) -> float:
    '''Calculate the secure key rate for a symmetric link.'''

    efficiency = float(dB_to_fraction(total_attenuation_db / 2))

    true_singles = cw.true_singles_rate(
        brightness=float(brightness),
        efficiency=efficiency,
    )

    measured_singles = cw.measured_singles_rate(
        true_singles_rate=true_singles,
        dark_count_rate=dark_count_rate,
    )

    true_coincidences = cw.true_coincidence_rate(
        brightness=brightness,
        efficiency_a=efficiency,
        efficiency_b=efficiency,
    )

    accidental_coincidences = cw.accidental_coincidence_rate_approx(
        measured_singles_a=measured_singles,
        measured_singles_b=measured_singles,
        coincidence_window=coincidence_window,
    )

    window_efficiency = cw.coincidence_window_efficiency(
        coincidence_window=coincidence_window,
        timing_imprecision=timing_imprecision,
    )

    measured_coincidences = cw.measured_coincidence_rate(
        coincidence_window_efficiency=window_efficiency,
        true_coincidence_rate=true_coincidences,
        accidental_coincidence_rate=accidental_coincidences,
    )

    erroneous_coincidences = cw.erroneous_coincidence_rate(
        coincidence_window_efficiency=window_efficiency,
        true_coincidence_rate=true_coincidences,
        polarisation_error=polarisation_error,
        accidental_coincidence_rate=accidental_coincidences,
    )

    qber = cw.qber_from_rate(
        erroneous_coincidence_rate=erroneous_coincidences,
        measured_coincidence_rate=measured_coincidences,
    )

    return cw.secure_key_rate_symmetric(
        qber=qber,
        measured_coincidence_rate=measured_coincidences,
    )


def optimise_secure_key_rate(
    total_attenuation_db: float,
    timing_imprecision: float,
    *,
    dark_count_rate: float = 1000.0,
    polarisation_error: float = 0.01,
    x0: tuple[float, float] = (7.0, 1.0),
):
    def objective(x: npt.NDArray) -> float:
        brightness = 10**x[0]
        coincidence_window = x[1] * timing_imprecision

        return -secure_key_rate(
            total_attenuation_db=total_attenuation_db,
            brightness=brightness,
            coincidence_window=coincidence_window,
            timing_imprecision=timing_imprecision,
            dark_count_rate=dark_count_rate,
            polarisation_error=polarisation_error,
        )

    result = minimize(
        objective,
        x0=x0,
        bounds=[
            (2.0, 14.0),
            (0.001, 10.0),
        ],
        method='Nelder-Mead',
    )

    return -result.fun, result.x


if __name__ == '__main__':
    fig, ax = plt.subplots()
    timing_imprecision = [
        1e-9,1e-10,4e-11,1e-11,1e-12
    ]

    attenuations = np.linspace(20, 156, 500)

    key_rates = np.empty_like(attenuations)

    for t_delta in timing_imprecision:
        key_rates = np.empty_like(attenuations)

        x0 = (7.0, 1.0)

        for i, attenuation in enumerate(attenuations):
            key_rates[i], x0 = optimise_secure_key_rate(
                total_attenuation_db=attenuation,
                timing_imprecision=t_delta,
                dark_count_rate=1000,
                polarisation_error=0.01,
                x0=x0,
            )

        ax.plot(
            attenuations,
            key_rates,
            label=f'{t_delta=}'
        )

    ax.set_xlabel('Total link attenuation (dB)')
    ax.set_ylabel('Secure key rate (bits/s)')
    ax.set_xlim(20,155)
    ax.set_ylim(1e-6,1e9)
    ax.set_yscale('log')
    ax.grid()
    ax.legend()


    fig.savefig(
        pathlib.Path(__file__).with_suffix('.png'),
        dpi='figure',
        bbox_inches='tight',
    )
    plt.show()
