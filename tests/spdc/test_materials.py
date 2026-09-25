import numpy as np
import pytest

from qtoolkit.spdc.materials import (
    MgOLithiumNiobate,
    RefractiveIndexAxis,
)


@pytest.fixture
def material() -> MgOLithiumNiobate:
    return MgOLithiumNiobate()


@pytest.mark.parametrize(
    'wavelength, temperature',
    [
        (405e-9, 25.0),
        (780e-9, 25.0),
        (942.33e-9, 25.0),
        (1550e-9, 25.0),
        (1551.51e-9, 100.0),
    ],
)
def test_refractive_indices_are_physical(
    material,
    wavelength,
    temperature,
) -> None:
    n_o = material.ordinary_refractive_index(
        wavelength,
        temperature,
    )

    n_e = material.extraordinary_refractive_index(
        wavelength,
        temperature,
    )

    assert n_o > 1
    assert n_e > 1


def test_effective_extraordinary_index_at_zero(
    material,
) -> None:
    wavelength = 785e-9
    temperature = 84.0

    result = material.refractive_index_at_angle(
        wavelength=wavelength,
        temperature=temperature,
        axis=RefractiveIndexAxis.EXTRAORDINARY,
        angle=0.0,
    )

    expected = material.ordinary_refractive_index(
        wavelength,
        temperature,
    )

    assert result == pytest.approx(
        expected
    )


def test_effective_extraordinary_index_at_pi_over_two(
    material,
) -> None:
    wavelength = 785e-9
    temperature = 84.0

    result = material.refractive_index_at_angle(
        wavelength=wavelength,
        temperature=temperature,
        axis=RefractiveIndexAxis.EXTRAORDINARY,
        angle=np.pi / 2,
    )

    expected = material.extraordinary_refractive_index(
        wavelength,
        temperature,
    )

    assert result == pytest.approx(
        expected
    )


def test_effective_extraordinary_index_between_principal_indices(
    material,
) -> None:
    wavelength = 785e-9
    temperature = 84.0

    n_o = material.ordinary_refractive_index(
        wavelength,
        temperature,
    )

    n_e = material.extraordinary_refractive_index(
        wavelength,
        temperature,
    )

    result = material.refractive_index_at_angle(
        wavelength=wavelength,
        temperature=temperature,
        axis=RefractiveIndexAxis.EXTRAORDINARY,
        angle=np.deg2rad(45),
    )

    assert n_e < result < n_o


def test_ordinary_index_is_independent_of_angle(
    material,
) -> None:
    wavelength = 785e-9
    temperature = 84.0

    expected = material.ordinary_refractive_index(
        wavelength,
        temperature,
    )

    for angle in np.linspace(
        0,
        np.pi / 2,
        11,
    ):
        result = material.refractive_index_at_angle(
            wavelength=wavelength,
            temperature=temperature,
            axis=RefractiveIndexAxis.ORDINARY,
            angle=angle,
        )

        assert result == pytest.approx(
            expected
        )


def test_effective_extraordinary_index_array(
    material,
) -> None:
    wavelengths = np.array([
        523.5e-9,
        785e-9,
        1572e-9,
    ])

    result = material.refractive_index_at_angle(
        wavelength=wavelengths,
        temperature=84.0,
        axis=RefractiveIndexAxis.EXTRAORDINARY,
        angle=np.deg2rad(79),
    )

    assert result.shape == wavelengths.shape

    assert np.all(
        result > 1
    )


def test_effective_extraordinary_index_angle_array(
    material,
) -> None:
    angles = np.linspace(
        0,
        np.pi / 2,
        11,
    )

    result = material.refractive_index_at_angle(
        wavelength=785e-9,
        temperature=84.0,
        axis=RefractiveIndexAxis.EXTRAORDINARY,
        angle=angles,
    )

    assert result.shape == angles.shape

@pytest.mark.parametrize(
    'wavelength, expected',
    [
        (523.5e-9, 2.331502701758576),
        (785e-9, 2.261949791474799),
        (1571.500956022945e-9, 2.215371291126802),
    ],
)
def test_ordinary_refractive_index_regression(
    material,
    wavelength,
    expected,
) -> None:
    """Regression values independently evaluated from the Gayer model."""
    result = material.ordinary_refractive_index(
        wavelength,
        84.0,
    )

    assert result == pytest.approx(
        expected,
        rel=1e-12,
    )


@pytest.mark.parametrize(
    'wavelength, expected',
    [
        (523.5e-9, 2.246608420968678),
        (785e-9, 2.187058845556199),
        (1571.500956022945e-9, 2.147248608670988),
    ],
)
def test_extraordinary_refractive_index_regression(
    material,
    wavelength,
    expected,
) -> None:
    """Regression values independently evaluated from the Gayer model."""
    result = material.extraordinary_refractive_index(
        wavelength,
        84.0,
    )

    assert result == pytest.approx(
        expected,
        rel=1e-12,
    )

def test_refractive_index_array_matches_scalar_evaluation(
    material,
) -> None:
    wavelengths = np.array([
        523.5e-9,
        785e-9,
        1571.500956022945e-9,
    ])

    result = material.refractive_index(
        wavelengths,
        84.0,
        RefractiveIndexAxis.EXTRAORDINARY,
    )

    expected = np.array([
        material.refractive_index(
            wavelength,
            84.0,
            RefractiveIndexAxis.EXTRAORDINARY,
        )
        for wavelength in wavelengths
    ])

    np.testing.assert_allclose(
        result,
        expected,
        rtol=1e-14,
    )