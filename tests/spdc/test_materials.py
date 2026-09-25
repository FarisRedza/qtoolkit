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