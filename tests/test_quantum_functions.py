import numpy as np
import pytest

import qtoolkit

# qber

@pytest.mark.parametrize(
    ('correct', 'incorrect', 'expected'),
    [
        (950, 50, 0.05),
        (1000, 0, 0.0),
        (0, 1000, 1.0),
        (50, 50, 0.5),
        (75, 25, 0.25),
    ],
)
def test_qber(
        correct: float,
        incorrect: float,
        expected: float,
) -> None:
    assert (
        qtoolkit.qber(
            correct=correct,
            incorrect=incorrect,
        )
        == pytest.approx(expected)
    )


def test_qber_from_coincidences_correlated() -> None:
    result = qtoolkit.qber_from_coincidences(
        c_00=450,
        c_01=25,
        c_10=25,
        c_11=500,
    )

    assert result == pytest.approx(0.05)


def test_qber_from_coincidences_anticorrelated() -> None:
    result = qtoolkit.qber_from_coincidences(
        c_00=25,
        c_01=450,
        c_10=500,
        c_11=25,
        correlated=False,
    )

    assert result == pytest.approx(0.05)


def test_qz_correlated() -> None:
    result = qtoolkit.qz(
        c_hh=450,
        c_hv=25,
        c_vh=25,
        c_vv=500,
    )

    assert result == pytest.approx(0.05)


def test_qz_anticorrelated() -> None:
    result = qtoolkit.qz(
        c_hh=25,
        c_hv=450,
        c_vh=500,
        c_vv=25,
        correlated=False,
    )

    assert result == pytest.approx(0.05)


def test_qx_correlated() -> None:
    result = qtoolkit.qx(
        c_dd=450,
        c_da=25,
        c_ad=25,
        c_aa=500,
    )

    assert result == pytest.approx(0.05)


def test_qx_anticorrelated() -> None:
    result = qtoolkit.qx(
        c_dd=25,
        c_da=450,
        c_ad=500,
        c_aa=25,
        correlated=False,
    )

    assert result == pytest.approx(0.05)


def test_qy_correlated() -> None:
    result = qtoolkit.qy(
        c_rr=450,
        c_rl=25,
        c_lr=25,
        c_ll=500,
    )

    assert result == pytest.approx(0.05)


def test_qy_anticorrelated() -> None:
    result = qtoolkit.qy(
        c_rr=25,
        c_rl=450,
        c_lr=500,
        c_ll=25,
        correlated=False,
    )

    assert result == pytest.approx(0.05)

# visibility

@pytest.mark.parametrize(
    ('visibility', 'expected'),
    [
        (1.0, 0.0),
        (0.9, 0.05),
        (0.5, 0.25),
        (0.0, 0.5),
    ],
)
def test_qber_from_visibility(
        visibility: float,
        expected: float,
) -> None:
    assert (
        qtoolkit.qber_from_visibility(visibility)
        == pytest.approx(expected)
    )


@pytest.mark.parametrize(
    ('maximum', 'minimum', 'expected'),
    [
        (100, 0, 1.0),
        (100, 100, 0.0),
        (75, 25, 0.5),
        (95, 5, 0.9),
    ],
)
def test_visibility(
        maximum: float,
        minimum: float,
        expected: float,
) -> None:
    assert (
        qtoolkit.visibility(
            max=maximum,
            min=minimum,
        )
        == pytest.approx(expected)
    )


@pytest.mark.parametrize(
    ('qber', 'expected'),
    [
        (0.0, 1.0),
        (0.05, 0.9),
        (0.25, 0.5),
        (0.5, 0.0),
    ],
)
def test_visibility_from_qber(
        qber: float,
        expected: float,
) -> None:
    assert (
        qtoolkit.visibility_from_qber(qber)
        == pytest.approx(expected)
    )


@pytest.mark.parametrize(
    'qber',
    [
        0.0,
        0.05,
        0.1,
        0.25,
        0.5,
    ],
)
def test_qber_visibility_round_trip(
        qber: float,
) -> None:
    visibility = qtoolkit.visibility_from_qber(qber)
    result = qtoolkit.qber_from_visibility(visibility)

    assert result == pytest.approx(qber)

# heralding

def test_symmetric_heralding_efficiency() -> None:
    result = qtoolkit.symmetric_heralding_efficiency(
        coincidences=500,
        singles_a=1000,
        singles_b=1000,
    )

    assert result == pytest.approx(0.5)

# fidelity

def test_fidelity_from_two_visibilities() -> None:
    result = qtoolkit.fidelity_from_visibility(
        visibility_z=0.9,
        visibility_x=0.8,
    )

    assert result == pytest.approx(0.85)


def test_fidelity_from_three_visibilities() -> None:
    result = qtoolkit.fidelity_from_visibility(
        visibility_z=0.9,
        visibility_x=0.8,
        visibility_y=0.7,
    )

    assert result == pytest.approx(0.85)


def test_fidelity_from_qber() -> None:
    result = qtoolkit.fidelity_from_qber(
        qx=0.05,
        qz=0.10,
    )

    assert result == pytest.approx(0.85)


def test_fidelity_qber_visibility_equivalence() -> None:
    qx = 0.05
    qz = 0.10

    from_qber = qtoolkit.fidelity_from_qber(
        qx=qx,
        qz=qz,
    )

    from_visibility = qtoolkit.fidelity_from_visibility(
        visibility_x=qtoolkit.visibility_from_qber(qx),
        visibility_z=qtoolkit.visibility_from_qber(qz),
    )

    assert from_qber == pytest.approx(from_visibility)

# purity

def test_purity_pure_state() -> None:
    density_matrix = np.array([
        [1, 0],
        [0, 0],
    ])

    assert (
        qtoolkit.purity(density_matrix)
        == pytest.approx(1.0)
    )


def test_purity_maximally_mixed_state() -> None:
    density_matrix = np.array([
        [0.5, 0],
        [0, 0.5],
    ])

    assert (
        qtoolkit.purity(density_matrix)
        == pytest.approx(0.5)
    )


def test_purity_complex_pure_state() -> None:
    density_matrix = np.array([
        [0.5, 0.5j],
        [-0.5j, 0.5],
    ])

    assert (
        qtoolkit.purity(density_matrix)
        == pytest.approx(1.0)
    )


@pytest.mark.parametrize(
    'density_matrix',
    [
        [1, 0],
        [
            [1, 0, 0],
            [0, 1, 0],
        ],
    ],
)
def test_purity_requires_square_matrix(
        density_matrix,
) -> None:
    with pytest.raises(
        ValueError,
        match='Density matrix must be square',
    ):
        qtoolkit.purity(density_matrix)