import numpy as np
import pytest

from qtoolkit.spdc.materials import (
    MgOLithiumNiobate,
    RefractiveIndexAxis,
)
from qtoolkit.spdc.phasematching import (
    wavevector,
    wavevector_mismatch,
    poling_period_at_temperature,
    poling_period_at_reference_temperature,
    find_poling_period,
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