import numpy as np
import pytest

import qtoolkit


def test_polarisation_basis_states_are_orthonormal() -> None:
    for first, second in [
        (qtoolkit.polarisation.H, qtoolkit.polarisation.V),
        (qtoolkit.polarisation.D, qtoolkit.polarisation.A),
        (qtoolkit.polarisation.R, qtoolkit.polarisation.L),
    ]:
        assert np.vdot(first, first) == pytest.approx(1.0)
        assert np.vdot(second, second) == pytest.approx(1.0)
        assert np.vdot(first, second) == pytest.approx(0.0)


def test_phi_plus_correlations_in_three_bases() -> None:
    state = qtoolkit.polarisation.PHI_PLUS

    # Phi+ is correlated in H/V and D/A, but anti-correlated in R/L.
    expected = {
        'z': (qtoolkit.polarisation.H, qtoolkit.polarisation.V, True),
        'x': (qtoolkit.polarisation.D, qtoolkit.polarisation.A, True),
        'y': (qtoolkit.polarisation.R, qtoolkit.polarisation.L, False),
    }

    for first, second, correlated in expected.values():
        p_00 = qtoolkit.polarisation.joint_projection_probability(
            state, first, first
        )
        p_01 = qtoolkit.polarisation.joint_projection_probability(
            state, first, second
        )
        p_10 = qtoolkit.polarisation.joint_projection_probability(
            state, second, first
        )
        p_11 = qtoolkit.polarisation.joint_projection_probability(
            state, second, second
        )

        if correlated:
            assert p_00 + p_11 == pytest.approx(1.0)
            assert p_01 + p_10 == pytest.approx(0.0)
        else:
            assert p_00 + p_11 == pytest.approx(0.0)
            assert p_01 + p_10 == pytest.approx(1.0)


def test_half_waveplate_at_45_degrees_swaps_h_and_v() -> None:
    plate = qtoolkit.polarisation.HalfWaveplate(angle_deg=45.0)

    output_h = plate.apply(qtoolkit.polarisation.H)
    output_v = plate.apply(qtoolkit.polarisation.V)

    # Global phase is physically irrelevant, so compare projection probability.
    assert qtoolkit.polarisation.projection_probability(
        output_h, qtoolkit.polarisation.V
    ) == pytest.approx(1.0)
    assert qtoolkit.polarisation.projection_probability(
        output_v, qtoolkit.polarisation.H
    ) == pytest.approx(1.0)


def test_quarter_waveplate_at_zero_maps_d_to_l() -> None:
    plate = qtoolkit.polarisation.QuarterWaveplate(angle_deg=0.0)
    output = plate.apply(qtoolkit.polarisation.D)

    assert qtoolkit.polarisation.projection_probability(
        output, qtoolkit.polarisation.L
    ) == pytest.approx(1.0)


def test_composed_waveplates_apply_in_list_order() -> None:
    plates = [
        qtoolkit.polarisation.QuarterWaveplate(angle_deg=17.0),
        qtoolkit.polarisation.HalfWaveplate(angle_deg=-23.0),
        qtoolkit.polarisation.QuarterWaveplate(angle_deg=41.0),
    ]

    sequential = qtoolkit.polarisation.H.copy()
    for plate in plates:
        sequential = plate.apply(sequential)

    composed = qtoolkit.polarisation.compose_waveplates(plates)

    assert composed @ qtoolkit.polarisation.H == pytest.approx(sequential)
