import pytest

from qtoolkit.spdc.materials import (
    MgOLithiumNiobate,
)


@pytest.fixture
def material() -> None:
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