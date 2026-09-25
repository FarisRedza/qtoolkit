import numpy as np
import pytest

from qtoolkit.constants import SPEED_OF_LIGHT
from qtoolkit.spdc.materials import (
    MgOLithiumNiobate,
    RefractiveIndexAxis,
)
from qtoolkit.spdc.phasematching import (
    conjugate_wavelength,
    find_poling_period,
)
from qtoolkit.spdc.spectral import (
    angular_frequency,
    wavelength_from_angular_frequency,
    pump_wavelength_fwhm_to_angular_frequency_std,
    gaussian_pump_amplitude,
    phase_matching_amplitude,
    joint_spectral_amplitude,
    joint_spectral_intensity,
)


def test_angular_frequency() -> None:
    wavelength = 1550e-9

    expected = (
        2
        * np.pi
        * SPEED_OF_LIGHT
        / wavelength
    )

    result = angular_frequency(
        wavelength
    )

    assert result == pytest.approx(
        expected
    )


def test_angular_frequency_array() -> None:
    wavelengths = np.array([
        775e-9,
        1550e-9,
        2000e-9,
    ])

    expected = (
        2
        * np.pi
        * SPEED_OF_LIGHT
        / wavelengths
    )

    result = angular_frequency(
        wavelengths
    )

    np.testing.assert_allclose(
        result,
        expected,
    )


def test_angular_frequency_rejects_non_positive_wavelength() -> None:
    with pytest.raises(
        ValueError,
        match='greater than zero',
    ):
        angular_frequency(
            0.0
        )


def test_angular_frequency_rejects_array_with_non_positive_wavelength() -> None:
    wavelengths = np.array([
        775e-9,
        0.0,
        1550e-9,
    ])

    with pytest.raises(
        ValueError,
        match='greater than zero',
    ):
        angular_frequency(
            wavelengths
        )


def test_wavelength_from_angular_frequency() -> None:
    expected_wavelength = 1550e-9

    omega = (
        2
        * np.pi
        * SPEED_OF_LIGHT
        / expected_wavelength
    )

    result = wavelength_from_angular_frequency(
        omega
    )

    assert result == pytest.approx(
        expected_wavelength
    )


def test_wavelength_from_angular_frequency_array() -> None:
    expected_wavelengths = np.array([
        775e-9,
        1550e-9,
        2000e-9,
    ])

    angular_frequencies = (
        2
        * np.pi
        * SPEED_OF_LIGHT
        / expected_wavelengths
    )

    result = wavelength_from_angular_frequency(
        angular_frequencies
    )

    np.testing.assert_allclose(
        result,
        expected_wavelengths,
    )


def test_wavelength_from_angular_frequency_rejects_non_positive_frequency(
) -> None:
    with pytest.raises(
        ValueError,
        match='greater than zero',
    ):
        wavelength_from_angular_frequency(
            0.0
        )


def test_wavelength_angular_frequency_round_trip() -> None:
    wavelengths = np.array([
        775e-9,
        1200e-9,
        1550e-9,
        2000e-9,
    ])

    result = wavelength_from_angular_frequency(
        angular_frequency(
            wavelengths
        )
    )

    np.testing.assert_allclose(
        result,
        wavelengths,
        rtol=1e-14,
        atol=0.0,
    )


def test_gaussian_pump_amplitude_at_centre() -> None:
    central_frequency = 2.0e15
    bandwidth = 1.0e12

    result = gaussian_pump_amplitude(
        central_frequency,
        central_frequency,
        bandwidth,
    )

    assert result == pytest.approx(
        1.0
    )


def test_gaussian_pump_amplitude_one_standard_deviation() -> None:
    central_frequency = 2.0e15
    bandwidth = 1.0e12

    result = gaussian_pump_amplitude(
        central_frequency + bandwidth,
        central_frequency,
        bandwidth,
    )

    expected = np.exp(-0.5)

    assert result == pytest.approx(
        expected
    )


def test_gaussian_pump_amplitude_is_symmetric() -> None:
    central_frequency = 2.0e15
    bandwidth = 1.0e12

    lower = gaussian_pump_amplitude(
        central_frequency - bandwidth,
        central_frequency,
        bandwidth,
    )

    upper = gaussian_pump_amplitude(
        central_frequency + bandwidth,
        central_frequency,
        bandwidth,
    )

    assert lower == pytest.approx(
        upper
    )


def test_gaussian_pump_amplitude_array() -> None:
    central_frequency = 2.0e15
    bandwidth = 1.0e12

    frequencies = np.array([
        central_frequency - bandwidth,
        central_frequency,
        central_frequency + bandwidth,
    ])

    expected = np.array([
        np.exp(-0.5),
        1.0,
        np.exp(-0.5),
    ])

    result = gaussian_pump_amplitude(
        frequencies,
        central_frequency,
        bandwidth,
    )

    np.testing.assert_allclose(
        result,
        expected,
    )


def test_gaussian_pump_amplitude_rejects_zero_bandwidth() -> None:
    with pytest.raises(
        ValueError,
        match='greater than zero',
    ):
        gaussian_pump_amplitude(
            2.0e15,
            2.0e15,
            0.0,
        )


def test_gaussian_pump_amplitude_rejects_negative_bandwidth() -> None:
    with pytest.raises(
        ValueError,
        match='greater than zero',
    ):
        gaussian_pump_amplitude(
            2.0e15,
            2.0e15,
            -1.0e12,
        )


def test_phase_matching_amplitude_at_zero_mismatch() -> None:
    result = phase_matching_amplitude(
        delta_k=0.0,
        crystal_length=20e-3,
    )

    assert result.real == pytest.approx(
        1.0
    )

    assert result.imag == pytest.approx(
        0.0
    )


def test_phase_matching_amplitude_first_zero() -> None:
    crystal_length = 20e-3

    delta_k = (
        2
        * np.pi
        / crystal_length
    )

    result = phase_matching_amplitude(
        delta_k=delta_k,
        crystal_length=crystal_length,
    )

    assert abs(result) == pytest.approx(
        0.0,
        abs=1e-15,
    )


def test_phase_matching_amplitude_magnitude_is_symmetric() -> None:
    crystal_length = 20e-3
    delta_k = 100.0

    positive = phase_matching_amplitude(
        delta_k=delta_k,
        crystal_length=crystal_length,
    )

    negative = phase_matching_amplitude(
        delta_k=-delta_k,
        crystal_length=crystal_length,
    )

    assert abs(positive) == pytest.approx(
        abs(negative)
    )


def test_phase_matching_amplitude_conjugate_symmetry() -> None:
    crystal_length = 20e-3
    delta_k = 100.0

    positive = phase_matching_amplitude(
        delta_k=delta_k,
        crystal_length=crystal_length,
    )

    negative = phase_matching_amplitude(
        delta_k=-delta_k,
        crystal_length=crystal_length,
    )

    assert negative == pytest.approx(
        np.conj(positive)
    )


def test_phase_matching_amplitude_array() -> None:
    crystal_length = 20e-3

    delta_k = np.array([
        -100.0,
        0.0,
        100.0,
    ])

    result = phase_matching_amplitude(
        delta_k=delta_k,
        crystal_length=crystal_length,
    )

    assert result.shape == delta_k.shape
    assert np.iscomplexobj(result)

    assert result[1] == pytest.approx(
        1.0 + 0.0j
    )


def test_phase_matching_amplitude_rejects_non_positive_length() -> None:
    with pytest.raises(
        ValueError,
        match='greater than zero',
    ):
        phase_matching_amplitude(
            delta_k=0.0,
            crystal_length=0.0,
        )


def test_joint_spectral_intensity() -> None:
    jsa = np.array([
        1.0 + 0.0j,
        1.0 + 1.0j,
        0.0 + 2.0j,
    ])

    expected = np.array([
        1.0,
        2.0,
        4.0,
    ])

    result = joint_spectral_intensity(
        jsa
    )

    np.testing.assert_allclose(
        result,
        expected,
    )


def test_joint_spectral_intensity_scalar() -> None:
    result = joint_spectral_intensity(
        3.0 + 4.0j
    )

    assert result == pytest.approx(
        25.0
    )


def test_joint_spectral_intensity_is_real() -> None:
    jsa = np.array([
        1.0 + 1.0j,
        2.0 - 3.0j,
    ])

    result = joint_spectral_intensity(
        jsa
    )

    assert np.isrealobj(result)


def test_joint_spectral_amplitude_at_phase_matching() -> None:
    material = MgOLithiumNiobate()
    axis = RefractiveIndexAxis.EXTRAORDINARY

    pump_wavelength = 775e-9
    signal_wavelength = 1550e-9
    idler_wavelength = 1550e-9

    temperature = 100.0
    crystal_length = 20e-3

    pump_bandwidth_std = 1.0e12

    poling_period = find_poling_period(
        pump_wavelength=pump_wavelength,
        signal_wavelength=signal_wavelength,
        idler_wavelength=idler_wavelength,
        temperature=temperature,
        material=material,
        pump_axis=axis,
        signal_axis=axis,
        idler_axis=axis,
    )

    result = joint_spectral_amplitude(
        signal_wavelength=signal_wavelength,
        idler_wavelength=idler_wavelength,
        pump_wavelength=pump_wavelength,
        pump_bandwidth_std=pump_bandwidth_std,
        crystal_length=crystal_length,
        temperature=temperature,
        poling_period=poling_period,
        material=material,
        pump_axis=axis,
        signal_axis=axis,
        idler_axis=axis,
    )

    assert abs(result) == pytest.approx(
        1.0,
        rel=1e-10,
        abs=1e-12,
    )


def test_joint_spectral_amplitude_non_degenerate_phase_matching() -> None:
    material = MgOLithiumNiobate()
    axis = RefractiveIndexAxis.EXTRAORDINARY

    pump_wavelength = 775e-9
    signal_wavelength = 1200e-9

    idler_wavelength = conjugate_wavelength(
        pump_wavelength,
        signal_wavelength,
    )

    temperature = 100.0
    crystal_length = 20e-3
    pump_bandwidth_std = 1.0e12

    poling_period = find_poling_period(
        pump_wavelength=pump_wavelength,
        signal_wavelength=signal_wavelength,
        idler_wavelength=idler_wavelength,
        temperature=temperature,
        material=material,
        pump_axis=axis,
        signal_axis=axis,
        idler_axis=axis,
    )

    result = joint_spectral_amplitude(
        signal_wavelength=signal_wavelength,
        idler_wavelength=idler_wavelength,
        pump_wavelength=pump_wavelength,
        pump_bandwidth_std=pump_bandwidth_std,
        crystal_length=crystal_length,
        temperature=temperature,
        poling_period=poling_period,
        material=material,
        pump_axis=axis,
        signal_axis=axis,
        idler_axis=axis,
    )

    assert abs(result) == pytest.approx(
        1.0,
        rel=1e-10,
        abs=1e-12,
    )


def test_joint_spectral_amplitude_off_energy_conservation_is_suppressed(
) -> None:
    material = MgOLithiumNiobate()
    axis = RefractiveIndexAxis.EXTRAORDINARY

    pump_wavelength = 775e-9
    matched_wavelength = 1550e-9

    temperature = 100.0
    crystal_length = 20e-3

    pump_bandwidth_std = 1.0e11

    poling_period = find_poling_period(
        pump_wavelength=pump_wavelength,
        signal_wavelength=matched_wavelength,
        idler_wavelength=matched_wavelength,
        temperature=temperature,
        material=material,
        pump_axis=axis,
        signal_axis=axis,
        idler_axis=axis,
    )

    centre = joint_spectral_amplitude(
        signal_wavelength=matched_wavelength,
        idler_wavelength=matched_wavelength,
        pump_wavelength=pump_wavelength,
        pump_bandwidth_std=pump_bandwidth_std,
        crystal_length=crystal_length,
        temperature=temperature,
        poling_period=poling_period,
        material=material,
        pump_axis=axis,
        signal_axis=axis,
        idler_axis=axis,
    )

    off_centre = joint_spectral_amplitude(
        signal_wavelength=1500e-9,
        idler_wavelength=1500e-9,
        pump_wavelength=pump_wavelength,
        pump_bandwidth_std=pump_bandwidth_std,
        crystal_length=crystal_length,
        temperature=temperature,
        poling_period=poling_period,
        material=material,
        pump_axis=axis,
        signal_axis=axis,
        idler_axis=axis,
    )

    assert abs(off_centre) < abs(centre)


def test_joint_spectral_amplitude_broadcasts_over_grid() -> None:
    material = MgOLithiumNiobate()
    axis = RefractiveIndexAxis.EXTRAORDINARY

    pump_wavelength = 775e-9
    central_wavelength = 1550e-9

    temperature = 100.0
    crystal_length = 20e-3
    pump_bandwidth_std = 1.0e12

    poling_period = find_poling_period(
        pump_wavelength=pump_wavelength,
        signal_wavelength=central_wavelength,
        idler_wavelength=central_wavelength,
        temperature=temperature,
        material=material,
        pump_axis=axis,
        signal_axis=axis,
        idler_axis=axis,
    )

    signal_wavelengths = np.linspace(
        1500e-9,
        1600e-9,
        11,
    )

    idler_wavelengths = np.linspace(
        1500e-9,
        1600e-9,
        13,
    )

    signal_grid, idler_grid = np.meshgrid(
        signal_wavelengths,
        idler_wavelengths,
        indexing='xy',
    )

    result = joint_spectral_amplitude(
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
    )

    assert result.shape == (
        len(idler_wavelengths),
        len(signal_wavelengths),
    )

    assert np.iscomplexobj(result)


def test_joint_spectral_intensity_grid_peaks_near_phase_matching() -> None:
    material = MgOLithiumNiobate()
    axis = RefractiveIndexAxis.EXTRAORDINARY

    pump_wavelength = 775e-9
    central_wavelength = 1550e-9

    temperature = 100.0
    crystal_length = 20e-3

    pump_bandwidth_std = 1.0e12

    poling_period = find_poling_period(
        pump_wavelength=pump_wavelength,
        signal_wavelength=central_wavelength,
        idler_wavelength=central_wavelength,
        temperature=temperature,
        material=material,
        pump_axis=axis,
        signal_axis=axis,
        idler_axis=axis,
    )

    signal_wavelengths = np.linspace(
        1500e-9,
        1600e-9,
        101,
    )

    idler_wavelengths = np.linspace(
        1500e-9,
        1600e-9,
        101,
    )

    signal_grid, idler_grid = np.meshgrid(
        signal_wavelengths,
        idler_wavelengths,
        indexing='xy',
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
        pump_axis=axis,
        signal_axis=axis,
        idler_axis=axis,
    )

    jsi = joint_spectral_intensity(
        jsa
    )

    maximum_index = np.unravel_index(
        np.argmax(jsi),
        jsi.shape,
    )

    peak_signal_wavelength = (
        signal_grid[maximum_index]
    )

    peak_idler_wavelength = (
        idler_grid[maximum_index]
    )

    grid_spacing = (
        signal_wavelengths[1]
        - signal_wavelengths[0]
    )

    assert peak_signal_wavelength == pytest.approx(
        central_wavelength,
        abs=grid_spacing,
    )

    assert peak_idler_wavelength == pytest.approx(
        central_wavelength,
        abs=grid_spacing,
    )

    assert jsi[maximum_index] == pytest.approx(
        1.0,
        rel=1e-10,
        abs=1e-12,
    )

def test_pump_wavelength_fwhm_to_angular_frequency_std() -> None:
    central_wavelength = 523.5e-9
    wavelength_fwhm = 92e-12

    result = (
        pump_wavelength_fwhm_to_angular_frequency_std(
            central_wavelength,
            wavelength_fwhm,
        )
    )

    assert result > 0


def test_pump_wavelength_fwhm_conversion_recovers_fwhm() -> None:
    central_wavelength = 523.5e-9
    wavelength_fwhm = 92e-12

    sigma = (
        pump_wavelength_fwhm_to_angular_frequency_std(
            central_wavelength,
            wavelength_fwhm,
        )
    )

    omega_low = angular_frequency(
        central_wavelength
        + wavelength_fwhm / 2
    )

    omega_high = angular_frequency(
        central_wavelength
        - wavelength_fwhm / 2
    )

    intensity_low = (
        gaussian_pump_amplitude(
            omega_low,
            angular_frequency(
                central_wavelength
            ),
            sigma,
        )**2
    )

    intensity_high = (
        gaussian_pump_amplitude(
            omega_high,
            angular_frequency(
                central_wavelength
            ),
            sigma,
        )**2
    )

    # Because wavelength is nonlinear in angular frequency, the two
    # wavelength half-maximum points are not exactly symmetric about
    # the central angular frequency. For a narrow pump bandwidth the
    # difference is negligible.
    assert intensity_low == pytest.approx(
        0.5,
        rel=5e-4,
    )

    assert intensity_high == pytest.approx(
        0.5,
        rel=5e-4,
    )


@pytest.mark.parametrize(
    'central_wavelength, wavelength_fwhm',
    [
        (0.0, 1e-12),
        (-500e-9, 1e-12),
        (500e-9, 0.0),
        (500e-9, -1e-12),
    ],
)
def test_pump_wavelength_fwhm_rejects_non_positive_values(
    central_wavelength,
    wavelength_fwhm,
) -> None:
    with pytest.raises(ValueError):
        pump_wavelength_fwhm_to_angular_frequency_std(
            central_wavelength,
            wavelength_fwhm,
        )


def test_joint_spectral_amplitude_angle_pi_over_two_matches_default(
) -> None:
    material = MgOLithiumNiobate()
    axis = RefractiveIndexAxis.EXTRAORDINARY

    pump_wavelength = 523.5e-9
    signal_wavelength = 790.6e-9

    idler_wavelength = conjugate_wavelength(
        pump_wavelength,
        signal_wavelength,
    )

    pump_bandwidth_std = (
        pump_wavelength_fwhm_to_angular_frequency_std(
            pump_wavelength,
            92e-12,
        )
    )

    kwargs = dict(
        signal_wavelength=signal_wavelength,
        idler_wavelength=idler_wavelength,
        pump_wavelength=pump_wavelength,
        pump_bandwidth_std=pump_bandwidth_std,
        crystal_length=20e-3,
        temperature=84.0,
        poling_period=7.15e-6,
        material=material,
        pump_axis=axis,
        signal_axis=axis,
        idler_axis=axis,
    )

    default = joint_spectral_amplitude(
        **kwargs,
    )

    angled = joint_spectral_amplitude(
        **kwargs,
        pump_angle=np.pi / 2,
        signal_angle=np.pi / 2,
        idler_angle=np.pi / 2,
    )

    assert angled == pytest.approx(
        default,
        rel=1e-12,
        abs=1e-12,
    )