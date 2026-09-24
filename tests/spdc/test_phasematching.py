import numpy as np
import pytest

from qtoolkit.spdc.materials import (
    MgOLithiumNiobate,
    RefractiveIndexAxis,
)
from qtoolkit.spdc.phasematching import (
    conjugate_wavelength,
    wavevector,
    wavevector_mismatch,
    poling_period_at_temperature,
    poling_period_at_reference_temperature,
    find_poling_period,
    find_phase_matching_temperature,
    find_phase_matching_temperatures,
    find_phase_matching_wavelengths,
    find_phase_matching_wavelength_pairs,
)

def test_conjugate_wavelength_degenerate() -> None:
    result = conjugate_wavelength(
        775e-9,
        1550e-9,
    )

    assert result == pytest.approx(
        1550e-9
    )


def test_conjugate_wavelength_energy_conservation() -> None:
    pump_wavelength = 775e-9
    signal_wavelength = 1200e-9

    idler_wavelength = conjugate_wavelength(
        pump_wavelength,
        signal_wavelength,
    )

    assert (
        1 / pump_wavelength
        == pytest.approx(
            1 / signal_wavelength
            + 1 / idler_wavelength
        )
    )


def test_conjugate_wavelength_array() -> None:
    pump_wavelength = 775e-9

    wavelengths = np.array([
        1200e-9,
        1550e-9,
        2000e-9,
    ])

    result = conjugate_wavelength(
        pump_wavelength,
        wavelengths,
    )

    expected = (
        1
        / (
            1 / pump_wavelength
            - 1 / wavelengths
        )
    )

    np.testing.assert_allclose(
        result,
        expected,
    )


def test_conjugate_wavelength_rejects_non_spdc_wavelength() -> None:
    with pytest.raises(
        ValueError,
        match='greater than the pump wavelength',
    ):
        conjugate_wavelength(
            775e-9,
            700e-9,
        )


def test_wavevector() -> None:
    wavelength = 1550e-9
    refractive_index = 2.1

    expected = (
        2 * np.pi
        * refractive_index
        / wavelength
    )

    result = wavevector(
        wavelength,
        refractive_index,
    )

    assert result == pytest.approx(expected)


def test_wavevector_array() -> None:
    wavelengths = np.array([
        780e-9,
        1550e-9,
    ])

    refractive_indices = np.array([
        2.2,
        2.1,
    ])

    expected = (
        2 * np.pi
        * refractive_indices
        / wavelengths
    )

    result = wavevector(
        wavelengths,
        refractive_indices,
    )

    np.testing.assert_allclose(
        result,
        expected,
    )


def test_poling_period_at_reference_temperature() -> None:
    poling_period = 19e-6

    result = poling_period_at_temperature(
        poling_period,
        19.0,
    )

    assert result == pytest.approx(
        poling_period
    )


def test_poling_period_increases_with_temperature() -> None:
    poling_period = 19e-6

    result = poling_period_at_temperature(
        poling_period,
        100.0,
    )

    assert result > poling_period


def test_poling_period_at_temperature() -> None:
    poling_period = 19e-6
    temperature = 100.0

    alpha = 1.53e-5
    beta = 5.3e-9

    expected = (
        poling_period
        * (
            1
            + alpha * (temperature - 19)
            + beta * (temperature - 19)**2
        )
    )

    result = poling_period_at_temperature(
        poling_period,
        temperature,
    )

    assert result == pytest.approx(expected)


def test_poling_period_array() -> None:
    poling_periods = np.array([
        18e-6,
        19e-6,
        20e-6,
    ])

    temperature = 100.0

    alpha = 1.53e-5
    beta = 5.3e-9

    expected = (
        poling_periods
        * (
            1
            + alpha * (temperature - 19)
            + beta * (temperature - 19)**2
        )
    )

    result = poling_period_at_temperature(
        poling_periods,
        temperature,
    )

    np.testing.assert_allclose(
        result,
        expected,
    )


def test_wavevector_mismatch_zero_at_phase_matching() -> None:
    material = MgOLithiumNiobate()

    pump_wavelength = 775e-9
    signal_wavelength = 1550e-9
    idler_wavelength = 1550e-9

    temperature = 19.0

    axis = RefractiveIndexAxis.EXTRAORDINARY

    n_p = material.refractive_index(
        pump_wavelength,
        temperature,
        axis,
    )

    n_s = material.refractive_index(
        signal_wavelength,
        temperature,
        axis,
    )

    n_i = material.refractive_index(
        idler_wavelength,
        temperature,
        axis,
    )

    k_p = wavevector(
        pump_wavelength,
        n_p,
    )

    k_s = wavevector(
        signal_wavelength,
        n_s,
    )

    k_i = wavevector(
        idler_wavelength,
        n_i,
    )

    poling_period = (
        2 * np.pi
        / (k_p - k_s - k_i)
    )

    result = wavevector_mismatch(
        pump_wavelength=pump_wavelength,
        signal_wavelength=signal_wavelength,
        idler_wavelength=idler_wavelength,
        temperature=temperature,
        poling_period=poling_period,
        material=material,
        pump_axis=axis,
        signal_axis=axis,
        idler_axis=axis,
    )

    assert result == pytest.approx(
        0.0,
        abs=1e-8,
    )


def test_wavevector_mismatch_nonzero_away_from_phase_matching() -> None:
    material = MgOLithiumNiobate()

    result = wavevector_mismatch(
        pump_wavelength=775e-9,
        signal_wavelength=1550e-9,
        idler_wavelength=1550e-9,
        temperature=19.0,
        poling_period=20e-6,
        material=material,
        pump_axis=RefractiveIndexAxis.EXTRAORDINARY,
        signal_axis=RefractiveIndexAxis.EXTRAORDINARY,
        idler_axis=RefractiveIndexAxis.EXTRAORDINARY,
    )

    assert not np.isclose(
        result,
        0.0,
        atol=1e-8,
    )


def test_wavevector_mismatch_qpm_order() -> None:
    material = MgOLithiumNiobate()

    kwargs = {
        'pump_wavelength': 775e-9,
        'signal_wavelength': 1550e-9,
        'idler_wavelength': 1550e-9,
        'temperature': 19.0,
        'poling_period': 20e-6,
        'material': material,
        'pump_axis': RefractiveIndexAxis.EXTRAORDINARY,
        'signal_axis': RefractiveIndexAxis.EXTRAORDINARY,
        'idler_axis': RefractiveIndexAxis.EXTRAORDINARY,
    }

    delta_k_1 = wavevector_mismatch(
        **kwargs,
        qpm_order=1,
    )

    delta_k_3 = wavevector_mismatch(
        **kwargs,
        qpm_order=3,
    )

    expected_difference = (
        -4 * np.pi / 20e-6
    )

    assert (
        delta_k_3 - delta_k_1
        == pytest.approx(expected_difference)
    )


def test_poling_period_temperature_round_trip() -> None:
    poling_period = 19e-6
    temperature = 100.0

    expanded = poling_period_at_temperature(
        poling_period,
        temperature,
    )

    result = poling_period_at_reference_temperature(
        expanded,
        temperature,
    )

    assert result == pytest.approx(
        poling_period
    )


def test_find_poling_period_phase_matches() -> None:
    material = MgOLithiumNiobate()

    pump_wavelength = 775e-9
    signal_wavelength = 1550e-9
    idler_wavelength = 1550e-9
    temperature = 50.0

    axis = RefractiveIndexAxis.EXTRAORDINARY

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

    delta_k = wavevector_mismatch(
        pump_wavelength=pump_wavelength,
        signal_wavelength=signal_wavelength,
        idler_wavelength=idler_wavelength,
        temperature=temperature,
        poling_period=poling_period,
        material=material,
        pump_axis=axis,
        signal_axis=axis,
        idler_axis=axis,
    )

    assert delta_k == pytest.approx(
        0.0,
        abs=1e-8,
    )


def test_find_poling_period_scales_with_qpm_order() -> None:
    material = MgOLithiumNiobate()

    kwargs = {
        'pump_wavelength': 775e-9,
        'signal_wavelength': 1550e-9,
        'idler_wavelength': 1550e-9,
        'temperature': 50.0,
        'material': material,
        'pump_axis': RefractiveIndexAxis.EXTRAORDINARY,
        'signal_axis': RefractiveIndexAxis.EXTRAORDINARY,
        'idler_axis': RefractiveIndexAxis.EXTRAORDINARY,
    }

    first_order = find_poling_period(
        **kwargs,
        qpm_order=1,
    )

    third_order = find_poling_period(
        **kwargs,
        qpm_order=3,
    )

    assert third_order == pytest.approx(
        3 * first_order
    )


def test_find_phase_matching_temperature() -> None:
    material = MgOLithiumNiobate()

    pump_wavelength = 775e-9
    signal_wavelength = 1550e-9
    idler_wavelength = 1550e-9

    expected_temperature = 100.0

    axis = RefractiveIndexAxis.EXTRAORDINARY

    poling_period = find_poling_period(
        pump_wavelength=pump_wavelength,
        signal_wavelength=signal_wavelength,
        idler_wavelength=idler_wavelength,
        temperature=expected_temperature,
        material=material,
        pump_axis=axis,
        signal_axis=axis,
        idler_axis=axis,
    )

    result = find_phase_matching_temperature(
        pump_wavelength=pump_wavelength,
        signal_wavelength=signal_wavelength,
        idler_wavelength=idler_wavelength,
        poling_period=poling_period,
        material=material,
        pump_axis=axis,
        signal_axis=axis,
        idler_axis=axis,
        temperature_bounds=(
            20.0,
            200.0,
        ),
    )

    assert result == pytest.approx(
        expected_temperature
    )


def test_found_temperature_phase_matches() -> None:
    material = MgOLithiumNiobate()

    axis = RefractiveIndexAxis.EXTRAORDINARY

    pump_wavelength = 775e-9
    signal_wavelength = 1550e-9
    idler_wavelength = 1550e-9

    poling_period = find_poling_period(
        pump_wavelength=pump_wavelength,
        signal_wavelength=signal_wavelength,
        idler_wavelength=idler_wavelength,
        temperature=100.0,
        material=material,
        pump_axis=axis,
        signal_axis=axis,
        idler_axis=axis,
    )

    temperature = find_phase_matching_temperature(
        pump_wavelength=pump_wavelength,
        signal_wavelength=signal_wavelength,
        idler_wavelength=idler_wavelength,
        poling_period=poling_period,
        material=material,
        pump_axis=axis,
        signal_axis=axis,
        idler_axis=axis,
    )

    delta_k = wavevector_mismatch(
        pump_wavelength=pump_wavelength,
        signal_wavelength=signal_wavelength,
        idler_wavelength=idler_wavelength,
        temperature=temperature,
        poling_period=poling_period,
        material=material,
        pump_axis=axis,
        signal_axis=axis,
        idler_axis=axis,
    )

    assert delta_k == pytest.approx(
        0.0,
        abs=1e-8,
    )


def test_find_phase_matching_temperature_no_root() -> None:
    material = MgOLithiumNiobate()

    with pytest.raises(
        ValueError,
        match='No phase-matching temperature',
    ):
        find_phase_matching_temperature(
            pump_wavelength=775e-9,
            signal_wavelength=1550e-9,
            idler_wavelength=1550e-9,
            poling_period=100e-6,
            material=material,
            pump_axis=RefractiveIndexAxis.EXTRAORDINARY,
            signal_axis=RefractiveIndexAxis.EXTRAORDINARY,
            idler_axis=RefractiveIndexAxis.EXTRAORDINARY,
            temperature_bounds=(
                20.0,
                200.0,
            ),
        )


def test_all_found_temperatures_phase_match() -> None:
    material = MgOLithiumNiobate()
    axis = RefractiveIndexAxis.EXTRAORDINARY

    pump_wavelength = 775e-9
    signal_wavelength = 1550e-9
    idler_wavelength = 1550e-9

    poling_period = find_poling_period(
        pump_wavelength=pump_wavelength,
        signal_wavelength=signal_wavelength,
        idler_wavelength=idler_wavelength,
        temperature=100.0,
        material=material,
        pump_axis=axis,
        signal_axis=axis,
        idler_axis=axis,
    )

    temperatures = find_phase_matching_temperatures(
        pump_wavelength=pump_wavelength,
        signal_wavelength=signal_wavelength,
        idler_wavelength=idler_wavelength,
        poling_period=poling_period,
        material=material,
        pump_axis=axis,
        signal_axis=axis,
        idler_axis=axis,
    )

    for temperature in temperatures:
        delta_k = wavevector_mismatch(
            pump_wavelength=pump_wavelength,
            signal_wavelength=signal_wavelength,
            idler_wavelength=idler_wavelength,
            temperature=temperature,
            poling_period=poling_period,
            material=material,
            pump_axis=axis,
            signal_axis=axis,
            idler_axis=axis,
        )

        assert delta_k == pytest.approx(
            0.0,
            abs=1e-8,
        )


def test_find_phase_matching_wavelengths_non_degenerate() -> None:
    material = MgOLithiumNiobate()
    axis = RefractiveIndexAxis.EXTRAORDINARY

    pump_wavelength = 775e-9
    expected_signal_wavelength = 1200e-9
    expected_idler_wavelength = conjugate_wavelength(
        pump_wavelength,
        expected_signal_wavelength,
    )
    temperature = 100.0

    poling_period = find_poling_period(
        pump_wavelength=pump_wavelength,
        signal_wavelength=expected_signal_wavelength,
        idler_wavelength=expected_idler_wavelength,
        temperature=temperature,
        material=material,
        pump_axis=axis,
        signal_axis=axis,
        idler_axis=axis,
    )

    signal_wavelength, idler_wavelength = (
        find_phase_matching_wavelengths(
            pump_wavelength=pump_wavelength,
            temperature=temperature,
            poling_period=poling_period,
            material=material,
            pump_axis=axis,
            signal_axis=axis,
            idler_axis=axis,
            signal_wavelength_bounds=(
                1100e-9,
                1300e-9,
            ),
        )
    )

    assert signal_wavelength == pytest.approx(
        expected_signal_wavelength,
        rel=1e-9,
        abs=1e-15,
    )

    assert idler_wavelength == pytest.approx(
        expected_idler_wavelength,
        rel=1e-9,
        abs=1e-15,
    )


def test_find_phase_matching_wavelengths_degenerate() -> None:
    material = MgOLithiumNiobate()
    axis = RefractiveIndexAxis.EXTRAORDINARY

    pump_wavelength = 775e-9
    expected_wavelength = 2 * pump_wavelength
    temperature = 100.0

    poling_period = find_poling_period(
        pump_wavelength=pump_wavelength,
        signal_wavelength=expected_wavelength,
        idler_wavelength=expected_wavelength,
        temperature=temperature,
        material=material,
        pump_axis=axis,
        signal_axis=axis,
        idler_axis=axis,
    )

    signal_wavelength, idler_wavelength = (
        find_phase_matching_wavelengths(
            pump_wavelength=pump_wavelength,
            temperature=temperature,
            poling_period=poling_period,
            material=material,
            pump_axis=axis,
            signal_axis=axis,
            idler_axis=axis,
            signal_wavelength_bounds=(
                1400e-9,
                1700e-9,
            ),
        )
    )

    assert signal_wavelength == pytest.approx(
        expected_wavelength
    )

    assert idler_wavelength == pytest.approx(
        expected_wavelength
    )


def test_find_phase_matching_wavelengths_rejects_invalid_bounds() -> None:
    material = MgOLithiumNiobate()

    with pytest.raises(
        ValueError,
        match='greater than the pump wavelength',
    ):
        find_phase_matching_wavelengths(
            pump_wavelength=775e-9,
            temperature=100.0,
            poling_period=19e-6,
            material=material,
            pump_axis=RefractiveIndexAxis.EXTRAORDINARY,
            signal_axis=RefractiveIndexAxis.EXTRAORDINARY,
            idler_axis=RefractiveIndexAxis.EXTRAORDINARY,
            signal_wavelength_bounds=(
                700e-9,
                1700e-9,
            ),
        )

def test_find_phase_matching_wavelength_pairs() -> None:
    material = MgOLithiumNiobate()
    axis = RefractiveIndexAxis.EXTRAORDINARY

    pump_wavelength = 775e-9
    expected_signal_wavelength = 1200e-9

    expected_idler_wavelength = (
        conjugate_wavelength(
            pump_wavelength,
            expected_signal_wavelength,
        )
    )

    temperature = 100.0

    poling_period = find_poling_period(
        pump_wavelength=pump_wavelength,
        signal_wavelength=expected_signal_wavelength,
        idler_wavelength=expected_idler_wavelength,
        temperature=temperature,
        material=material,
        pump_axis=axis,
        signal_axis=axis,
        idler_axis=axis,
    )

    result = find_phase_matching_wavelength_pairs(
        pump_wavelength=pump_wavelength,
        temperature=temperature,
        poling_period=poling_period,
        material=material,
        pump_axis=axis,
        signal_axis=axis,
        idler_axis=axis,
        signal_wavelength_bounds=(
            1000e-9,
            3000e-9,
        ),
    )

    assert result.shape[1] == 2

    assert np.any(
        np.isclose(
            result[:, 0],
            expected_signal_wavelength,
            rtol=1e-9,
            atol=1e-15,
        )
    )

def test_all_wavelength_pairs_conserve_energy() -> None:
    material = MgOLithiumNiobate()
    axis = RefractiveIndexAxis.EXTRAORDINARY

    pump_wavelength = 775e-9
    signal_wavelength = 1200e-9

    idler_wavelength = conjugate_wavelength(
        pump_wavelength,
        signal_wavelength,
    )

    temperature = 100.0

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

    pairs = find_phase_matching_wavelength_pairs(
        pump_wavelength=pump_wavelength,
        temperature=temperature,
        poling_period=poling_period,
        material=material,
        pump_axis=axis,
        signal_axis=axis,
        idler_axis=axis,
        signal_wavelength_bounds=(
            1000e-9,
            3000e-9,
        ),
    )

    for signal, idler in pairs:
        assert (
            1 / pump_wavelength
            == pytest.approx(
                1 / signal
                + 1 / idler
            )
        )

def test_all_wavelength_pairs_phase_match() -> None:
    material = MgOLithiumNiobate()
    axis = RefractiveIndexAxis.EXTRAORDINARY

    pump_wavelength = 775e-9
    signal_wavelength = 1200e-9

    idler_wavelength = conjugate_wavelength(
        pump_wavelength,
        signal_wavelength,
    )

    temperature = 100.0

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

    pairs = find_phase_matching_wavelength_pairs(
        pump_wavelength=pump_wavelength,
        temperature=temperature,
        poling_period=poling_period,
        material=material,
        pump_axis=axis,
        signal_axis=axis,
        idler_axis=axis,
        signal_wavelength_bounds=(
            1000e-9,
            3000e-9,
        ),
    )

    for signal, idler in pairs:
        delta_k = wavevector_mismatch(
            pump_wavelength=pump_wavelength,
            signal_wavelength=signal,
            idler_wavelength=idler,
            temperature=temperature,
            poling_period=poling_period,
            material=material,
            pump_axis=axis,
            signal_axis=axis,
            idler_axis=axis,
        )

        assert delta_k == pytest.approx(
            0.0,
            abs=1e-3,
        )

def test_find_phase_matching_wavelength_pairs_degenerate() -> None:
    material = MgOLithiumNiobate()
    axis = RefractiveIndexAxis.EXTRAORDINARY

    pump_wavelength = 775e-9
    degenerate_wavelength = (
        2 * pump_wavelength
    )

    temperature = 100.0

    poling_period = find_poling_period(
        pump_wavelength=pump_wavelength,
        signal_wavelength=degenerate_wavelength,
        idler_wavelength=degenerate_wavelength,
        temperature=temperature,
        material=material,
        pump_axis=axis,
        signal_axis=axis,
        idler_axis=axis,
    )

    pairs = find_phase_matching_wavelength_pairs(
        pump_wavelength=pump_wavelength,
        temperature=temperature,
        poling_period=poling_period,
        material=material,
        pump_axis=axis,
        signal_axis=axis,
        idler_axis=axis,
        signal_wavelength_bounds=(
            1200e-9,
            2000e-9,
        ),
    )

    matches = (
        np.isclose(
            pairs[:, 0],
            degenerate_wavelength,
            rtol=1e-9,
            atol=1e-15,
        )
        & np.isclose(
            pairs[:, 1],
            degenerate_wavelength,
            rtol=1e-9,
            atol=1e-15,
        )
    )

    assert np.any(matches)

def test_find_phase_matching_wavelength_pairs_no_roots() -> None:
    material = MgOLithiumNiobate()

    result = find_phase_matching_wavelength_pairs(
        pump_wavelength=775e-9,
        temperature=100.0,
        poling_period=100e-6,
        material=material,
        pump_axis=RefractiveIndexAxis.EXTRAORDINARY,
        signal_axis=RefractiveIndexAxis.EXTRAORDINARY,
        idler_axis=RefractiveIndexAxis.EXTRAORDINARY,
        signal_wavelength_bounds=(
            1000e-9,
            3000e-9,
        ),
    )

    assert result.shape == (0, 2)