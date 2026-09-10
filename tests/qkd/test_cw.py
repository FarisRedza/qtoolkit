import math

import numpy as np
import pytest

from qtoolkit.qkd import cw


def test_true_single_rate() -> None:
    assert cw.true_single_rate(
        brightness=1_000_000,
        efficiency=0.2,
    ) == pytest.approx(200_000)


def test_true_coincidence_rate() -> None:
    assert cw.true_coincidence_rate(
        brightness=1_000_000,
        efficiency_a=0.2,
        efficiency_b=0.3,
    ) == pytest.approx(60_000)


def test_heralding_efficiency() -> None:
    brightness = 1_000_000
    efficiency_a = 0.2
    efficiency_b = 0.3

    singles_a = cw.true_single_rate(
        brightness,
        efficiency_a,
    )

    singles_b = cw.true_single_rate(
        brightness,
        efficiency_b,
    )

    coincidences = cw.true_coincidence_rate(
        brightness,
        efficiency_a,
        efficiency_b,
    )

    # Equation 4 uses the opposite arm's singles.
    assert cw.heralding_efficiency(
        coincidences,
        singles_b,
    ) == pytest.approx(efficiency_a)

    assert cw.heralding_efficiency(
        coincidences,
        singles_a,
    ) == pytest.approx(efficiency_b)


def test_total_heralding_efficiency() -> None:
    result = cw.total_heralding_efficiency(
        heralding_efficiency_a=0.2,
        heralding_efficiency_b=0.8,
    )

    assert result == pytest.approx(
        math.sqrt(0.2 * 0.8)
    )


@pytest.mark.parametrize(
    ('error_a', 'error_b', 'expected'),
    [
        (0.0, 0.0, 0.0),
        (0.01, 0.0, 0.01),
        (0.0, 0.02, 0.02),
        (0.01, 0.02, 0.0296),
        (0.5, 0.5, 0.5),
    ],
)
def test_total_polarisation_error(
    error_a,
    error_b,
    expected,
) -> None:
    assert cw.total_polarisation_error(
        error_a,
        error_b,
    ) == pytest.approx(expected)


def test_measured_single_rate() -> None:
    assert cw.measured_single_rate(
        true_singles_rate=100_000,
        dark_count_rate=250,
    ) == pytest.approx(100_250)


def test_mean_clicks() -> None:
    assert cw.mean_clicks(
        measured_single_rate=100_000,
        coincidence_window=1e-9,
    ) == pytest.approx(1e-4)


def test_accidental_probability() -> None:
    mu_a = 0.1
    mu_b = 0.2

    expected = (
        (1 - math.exp(-mu_a))
        * (1 - math.exp(-mu_b))
    )

    assert cw.accidental_probability(
        mu_a,
        mu_b,
    ) == pytest.approx(expected)


def test_accidental_probability_approx_small_mu() -> None:
    mu_a = 1e-5
    mu_b = 2e-5

    exact = cw.accidental_probability(
        mu_a,
        mu_b,
    )

    approximate = cw.accidental_probability_approx(
        mu_a,
        mu_b,
    )

    assert approximate == pytest.approx(
        exact,
        rel=2e-5,
    )


def test_accidental_coincidence_rate() -> None:
    probability = 1e-6
    coincidence_window = 1e-9

    assert cw.accidental_coincidence_rate(
        probability,
        coincidence_window,
    ) == pytest.approx(1000)


def test_accidental_coincidence_rate_approx() -> None:
    assert cw.accidental_coincidence_rate_approx(
        measured_singles_a=100_000,
        measured_singles_b=200_000,
        coincidence_window=1e-9,
    ) == pytest.approx(20)


def test_temporal_correlation_density_peak() -> None:
    timing_imprecision = 100e-12
    delay = 20e-9

    expected_peak = (
        2
        / timing_imprecision
        * math.sqrt(
            math.log(2)
            / math.pi
        )
    )

    assert cw.temporal_correlation_density(
        time=delay,
        delay=delay,
        timing_imprecision=timing_imprecision,
    ) == pytest.approx(expected_peak)


def test_temporal_correlation_density_fwhm() -> None:
    """
    At +/- FWHM / 2 from the centre, the Gaussian should be
    exactly half its maximum value.
    """
    timing_imprecision = 100e-12
    delay = 0.0

    peak = cw.temporal_correlation_density(
        time=delay,
        delay=delay,
        timing_imprecision=timing_imprecision,
    )

    half_width = timing_imprecision / 2

    left = cw.temporal_correlation_density(
        time=delay - half_width,
        delay=delay,
        timing_imprecision=timing_imprecision,
    )

    right = cw.temporal_correlation_density(
        time=delay + half_width,
        delay=delay,
        timing_imprecision=timing_imprecision,
    )

    assert left == pytest.approx(peak / 2)
    assert right == pytest.approx(peak / 2)


def test_temporal_correlation_density_normalised() -> None:
    timing_imprecision = 100e-12

    times = np.linspace(
        -10 * timing_imprecision,
        10 * timing_imprecision,
        100_001,
    )

    density = cw.temporal_correlation_density(
        time=times,
        delay=0.0,
        timing_imprecision=timing_imprecision,
    )

    integral = np.trapezoid(
        density,
        times,
    )

    assert integral == pytest.approx(
        1.0,
        rel=1e-6,
    )


def test_coincidence_window_efficiency_matches_integration() -> None:
    timing_imprecision = 100e-12
    coincidence_window = 150e-12

    expected = cw.coincidence_window_efficiency(
        coincidence_window,
        timing_imprecision,
    )

    times = np.linspace(
        -coincidence_window / 2,
        coincidence_window / 2,
        100_001,
    )

    density = cw.temporal_correlation_density(
        time=times,
        delay=0.0,
        timing_imprecision=timing_imprecision,
    )

    numerical = np.trapezoid(
        density,
        times,
    )

    assert numerical == pytest.approx(
        expected,
        rel=1e-6,
    )


def test_coincidence_window_efficiency_limits() -> None:
    timing_imprecision = 100e-12

    assert cw.coincidence_window_efficiency(
        0.0,
        timing_imprecision,
    ) == pytest.approx(0.0)

    assert cw.coincidence_window_efficiency(
        100 * timing_imprecision,
        timing_imprecision,
    ) == pytest.approx(1.0)


def test_measured_coincidence_rate() -> None:
    assert cw.measured_coincidence_rate(
        coincidence_window_efficiency=0.9,
        true_coincidence_rate=1000,
        accidental_coincidence_rate=50,
    ) == pytest.approx(950)


def test_erroneous_coincidence_rate() -> None:
    result = cw.erroneous_coincidence_rate(
        coincidence_window_efficiency=0.9,
        true_coincidence_rate=1000,
        polarisation_error=0.01,
        accidental_coincidence_rate=50,
    )

    assert result == pytest.approx(
        0.9 * 1000 * 0.01
        + 0.5 * 50
    )


def test_qber_from_rate() -> None:
    assert cw.qber_from_rate(
        erroneous_coincidence_rate=50,
        measured_coincidence_rate=1000,
    ) == pytest.approx(0.05)


def test_visibility() -> None:
    assert cw.visibility(
        qber=0.05,
    ) == pytest.approx(0.9)


def test_secure_key_rate_zero_error() -> None:
    rate = cw.secure_key_rate(
        measured_coincidence_rate=1000,
        bit_error_rate=0.0,
        phase_error_rate=0.0,
    )

    # q = 0.5 by default.
    assert rate == pytest.approx(500)


def test_secure_key_rate_general_matches_symmetric() -> None:
    qber = 0.05
    coincidences = 10_000

    general = cw.secure_key_rate(
        measured_coincidence_rate=coincidences,
        bit_error_rate=qber,
        phase_error_rate=qber,
        sifting_probability=0.5,
        error_correction_efficiency=1.1,
    )

    symmetric = cw.secure_key_rate_symmetric(
        qber=qber,
        measured_coincidence_rate=coincidences,
    )

    assert general == pytest.approx(
        symmetric
    )


def test_secure_key_rate_threshold() -> None:
    """
    Equation 19 gives a maximum tolerable QBER of approximately
    10.2%, as stated in the paper.
    """
    coincidences = 1000

    assert cw.secure_key_rate_symmetric(
        qber=0.10,
        measured_coincidence_rate=coincidences,
    ) > 0

    assert cw.secure_key_rate_symmetric(
        qber=0.11,
        measured_coincidence_rate=coincidences,
    ) < 0

def test_find_key_rate_threshold() -> None:
    """
    Numerically find E where Equation 19 crosses zero.
    """
    low = 0.0
    high = 0.5

    for _ in range(100):
        midpoint = (
            low + high
        ) / 2

        rate = cw.secure_key_rate_symmetric(
            qber=midpoint,
            measured_coincidence_rate=1.0,
        )

        if rate > 0:
            low = midpoint
        else:
            high = midpoint

    max_qber = (low + high) / 2

    assert max_qber == pytest.approx(
        expected=0.102,
        abs=1e-3
    )

def test_coincidence_window_efficiency_three_fwhm() -> None:
    """
    The paper recommends t_CC = 3 t_delta as a rule of thumb,
    for which the coincidence-window efficiency is effectively 1.
    """
    timing_imprecision = 100e-12

    efficiency = cw.coincidence_window_efficiency(
        coincidence_window=3 * timing_imprecision,
        timing_imprecision=timing_imprecision,
    )

    assert efficiency == pytest.approx(
        0.999593,
        rel=1e-5,
    )

def test_coincidence_scaling_with_brightness() -> None:
    """
    In the source-dominated regime:

    - true coincidence rate scales linearly with brightness
    - approximate accidental coincidence rate scales quadratically
    """
    efficiency_a = 0.01
    efficiency_b = 0.01
    coincidence_window = 100e-12

    brightness_1 = 1e8
    brightness_2 = 2e8

    true_1 = cw.true_coincidence_rate(
        brightness=brightness_1,
        efficiency_a=efficiency_a,
        efficiency_b=efficiency_b,
    )

    true_2 = cw.true_coincidence_rate(
        brightness=brightness_2,
        efficiency_a=efficiency_a,
        efficiency_b=efficiency_b,
    )

    singles_a_1 = cw.true_single_rate(
        brightness=brightness_1,
        efficiency=efficiency_a,
    )

    singles_b_1 = cw.true_single_rate(
        brightness=brightness_1,
        efficiency=efficiency_b,
    )

    singles_a_2 = cw.true_single_rate(
        brightness=brightness_2,
        efficiency=efficiency_a,
    )

    singles_b_2 = cw.true_single_rate(
        brightness=brightness_2,
        efficiency=efficiency_b,
    )

    accidental_1 = cw.accidental_coincidence_rate_approx(
        measured_singles_a=singles_a_1,
        measured_singles_b=singles_b_1,
        coincidence_window=coincidence_window,
    )

    accidental_2 = cw.accidental_coincidence_rate_approx(
        measured_singles_a=singles_a_2,
        measured_singles_b=singles_b_2,
        coincidence_window=coincidence_window,
    )

    assert true_2 / true_1 == pytest.approx(2.0)

    assert accidental_2 / accidental_1 == pytest.approx(4.0)