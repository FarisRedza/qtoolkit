import pathlib
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from qtoolkit.misc_functions import dB_to_fraction
from qtoolkit.qkd import cw


@dataclass
class Link:
    loss_a: float
    loss_b: float
    polarisation_error: float
    coincidence_window: float

    timing_imprecision_intercept: float
    timing_imprecision_slope: float

    @property
    def label(self) -> str:
        return (
            rf'{self.loss_a:.0f} + {self.loss_b:.0f} dB, '
            rf'$e^{{\mathrm{{pol}}}}='
            rf'{100 * self.polarisation_error:.2f}\%$'
        )

    def timing_imprecision(
        self,
        brightness: float,
    ) -> float:
        """
        Estimate timing imprecision as a linear function of brightness.
        """
        return (
            self.timing_imprecision_intercept
            + self.timing_imprecision_slope * brightness
        )


def secure_key_rate(
    brightness: float,
    link: Link,
    *,
    dark_count_rate_a: float = 500.0,
    dark_count_rate_b: float = 500.0,
) -> float:
    efficiency_a = float(
        dB_to_fraction(link.loss_a)
    )
    efficiency_b = float(
        dB_to_fraction(link.loss_b)
    )

    true_singles_a = cw.true_single_rate(
        brightness=brightness,
        efficiency=efficiency_a,
    )

    true_singles_b = cw.true_single_rate(
        brightness=brightness,
        efficiency=efficiency_b,
    )

    measured_singles_a = cw.measured_single_rate(
        true_singles_rate=true_singles_a,
        dark_count_rate=dark_count_rate_a,
    )

    measured_singles_b = cw.measured_single_rate(
        true_singles_rate=true_singles_b,
        dark_count_rate=dark_count_rate_b,
    )

    true_coincidences = cw.true_coincidence_rate(
        brightness=brightness,
        efficiency_a=efficiency_a,
        efficiency_b=efficiency_b,
    )

    mean_clicks_a = cw.mean_clicks(
        measured_singles_rate=measured_singles_a,
        coincidence_window=link.coincidence_window,
    )

    mean_clicks_b = cw.mean_clicks(
        measured_singles_rate=measured_singles_b,
        coincidence_window=link.coincidence_window,
    )

    accidental_probability = cw.accidental_probability(
        mean_clicks_a=mean_clicks_a,
        mean_clicks_b=mean_clicks_b,
    )

    accidental_coincidences = cw.accidental_coincidence_rate(
        probability=accidental_probability,
        coincidence_window=link.coincidence_window,
    )

    timing_imprecision = link.timing_imprecision(
        brightness
    )

    window_efficiency = cw.coincidence_window_efficiency(
        coincidence_window=link.coincidence_window,
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
        polarisation_error=link.polarisation_error,
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


if __name__ == '__main__':
    fig, ax = plt.subplots()

    links = [
        Link(
            loss_a=20,
            loss_b=20,
            polarisation_error=0.0121,
            coincidence_window=134e-12,
            timing_imprecision_intercept=50e-12,
            timing_imprecision_slope=7.2e-20,
        ),
        Link(
            loss_a=30,
            loss_b=20,
            polarisation_error=0.0535,
            coincidence_window=93e-12,
            timing_imprecision_intercept=50e-12,
            timing_imprecision_slope=4.3e-20,
        ),
        Link(
            loss_a=30,
            loss_b=30,
            polarisation_error=0.0182,
            coincidence_window=76e-12,
            timing_imprecision_intercept=50e-12,
            timing_imprecision_slope=1.6e-20,
        ),
        Link(
            loss_a=40,
            loss_b=20,
            polarisation_error=0.0295,
            coincidence_window=86e-12,
            timing_imprecision_intercept=50e-12,
            timing_imprecision_slope=3.1e-20,
        ),
        Link(
            loss_a=40,
            loss_b=40,
            polarisation_error=0.0401,
            coincidence_window=46e-12,
            timing_imprecision_intercept=50e-12,
            timing_imprecision_slope=0.0,
        ),
    ]

    brightnesses = np.linspace(
        1e6,
        2.75e9,
        1500,
    )

    dark_count_rate_per_detector = 250.0
    number_of_detectors = 2

    dark_count_rate = (
        number_of_detectors
        * dark_count_rate_per_detector
    )

    for link in links:
        key_rates = np.array([
            secure_key_rate(
                brightness=brightness,
                link=link,
                dark_count_rate_a=dark_count_rate,
                dark_count_rate_b=dark_count_rate,
            )
            for brightness in brightnesses
        ])

        positive = key_rates > 0

        ax.plot(
            brightnesses[positive],
            key_rates[positive],
            label=link.label,
        )

    ax.set_xlabel('Brightness (cps)')
    ax.set_ylabel('Secure key rate (bits/s)')

    ax.set_xlim(0, 2.75e9)
    ax.set_ylim(1e-2, 1e5)

    ax.set_yscale('log')

    ax.grid()
    ax.legend()

    fig.savefig(
        pathlib.Path(__file__).with_suffix('.png'),
        dpi='figure',
        bbox_inches='tight',
    )

    plt.show()