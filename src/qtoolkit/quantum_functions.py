import typing

import numpy as np

# qber functions

def qber(
        correct: float,
        incorrect: float
) -> float:
    """
    Quantum bit error rate.

    Parameters
    ----------
    correct: float
        Number of correct detections/coincidences
    incorrect: float
        Number of erroneous detections/coincidences
    
    Returns
    -------
    float
        QBER as a fraction in the range [0, 1].
    
    Example
    -------
    >>> qber(correct=950, incorrect=50)
    0.05
    """
    return incorrect / (incorrect + correct)

def qber_from_coincidences(
        c_00: float,
        c_01: float,
        c_10: float,
        c_11: float,
        correlated: bool = True
) -> float:
    """
    Calculate QBER from a 2x2 coincidence matrix.

    The matrix is::

                 Bob
                  0     1
        Alice 0  c_00   c_01
              1  c_10   c_11

    For correlated outcomes:
        correct   = c_00 + c_11
        incorrect = c_01 + c_10

    For anti-correlated outcomes:
        correct   = c_01 + c_10
        incorrect = c_00 + c_11

    Parameters
    ----------
        c_00: float
            c_00

        c_01: float
            c_01

        c_10: float
            c_10

        c_11: float
            c_11

        correleated: bool = True
            True for correlated, False for anti-correlated
    
    Returns
    -------
    float
        QBER as a fraction in the range [0, 1].
    """
    if correlated:
        correct = c_00 + c_11
        incorrect = c_01 + c_10
    else:
        correct = c_01 + c_10
        incorrect = c_00 + c_11

    return qber(correct, incorrect)

def qz(
        c_hh: float,
        c_hv: float,
        c_vh: float,
        c_vv: float,
        correlated: bool = True
) -> float:
    """
    QBER in the Z (H/V) basis.

    Parameters
    ----------
        c_hh: float
            c_hh
        c_hv: float
            c_hv
        c_vh: float
            c_vh
        c_vv: float
            c_vv
        correleated: bool = True
            True for correlated, False for anti-correlated
    
    Returns
    -------
    float
        QBER as a fraction in the range [0, 1].
    """
    return qber_from_coincidences(
        c_00=c_hh,
        c_01=c_hv,
        c_10=c_vh,
        c_11=c_vv,
        correlated=correlated
    )

def qx(
        c_dd: float,
        c_da: float,
        c_ad: float,
        c_aa: float,
        correlated: bool = True
) -> float:
    """
    QBER in the X (D/A) basis.

    Parameters
    ----------
        c_dd: float
            c_dd
        c_da: float
            c_da
        c_ad: float
            c_ad
        c_aa: float
            c_aa
        correleated: bool = True
            True for correlated, False for anti-correlated
    
    Returns
    -------
    float
        QBER as a fraction in the range [0, 1].
    """
    return qber_from_coincidences(
        c_00=c_dd,
        c_01=c_da,
        c_10=c_ad,
        c_11=c_aa,
        correlated=correlated
    )

def qy(
        c_rr: float,
        c_rl: float,
        c_lr: float,
        c_ll: float,
        correlated: bool = True
) -> float:
    """
    QBER in the Y (R/L) basis.

    Parameters
    ----------
        c_rr: float
            c_rr
        c_rl: float
            c_rl
        c_lr: float
            c_lr
        c_ll: float
            c_ll
        correleated: bool = True
            True for correlated, False for anti-correlated
    
    Returns
    -------
    float
        QBER as a fraction in the range [0, 1].
    """
    return qber_from_coincidences(
        c_00=c_rr,
        c_01=c_rl,
        c_10=c_lr,
        c_11=c_ll,
        correlated=correlated
    )

def qber_from_visibility(visibility: float) -> float:
    """
    Calculate QBER from visibility

    .. math:: 
        QBER = (1 - V)/2

    Parameters
    ----------
    visibility: float
        Visibility

    Returns
    -------
    float
        QBER
    """
    return (1 - visibility)/2

# visibility functions

def visibility(max: float, min: float) -> float:
    """
    Calculate visibility

    .. math::
        V = (C_{max} - C_{min}) / (C_{max} + C_{min})
    
    Parameters
    ----------
    max: float
        max
    min: float
        min

    Returns
    -------
    float
    """
    return (max - min) / (max + min)

def visibility_from_qber(qber: float) -> float:
    """
    Calculate visbility from QBER

    .. math::
        V = 1 - 2 * \\text{QBER}

    Parameters
    ----------
    qber: float
        QBER
    
    Returns
    -------
    float
        Visibility
    """
    return 1 - 2 * qber

# singles and coincidences functions

def symmetric_heralding_efficiency(
        coincidences: float,
        singles_a: float,
        singles_b: float
) -> float:
    return coincidences / np.sqrt(singles_a * singles_b)

# entanglement functions

def fidelity_from_visibility(
        visibility_z: float,
        visibility_x: float,
        visibility_y: typing.Optional[float] = None
) -> float:
    """
    Estimate Bell-state fidelity from measured visibilities.

    With measurements in all three mutually unbiased bases:

    .. math::
        F ~= (1 + V_x + V_y + V_z) / 4

    If only X and Z are supplied, this function returns the common
    two-basis estimate:

    .. math::
        F ~= (V_x + V_z) / 2

    Parameters
    ----------
    visibility_z: float
        Visibility in the Z basis
    visibility_x: float
        Visibility in the X basis
    visibility_y: float
        Visibility in the Y basis
    
    Returns
    -------
    float
        Fidelity
    """
    if visibility_y is None:
        return (visibility_x + visibility_z) / 2

    return (
        1
        + visibility_x
        + visibility_y
        + visibility_z
    ) / 4

def fidelity_from_qber(
    qx: float,
    qz: float,
) -> float:
    """
    Two-basis Bell-state fidelity estimate.

    .. math::
        V = 1 - 2 \\text{QBER}

        F ~= (V_x + V_z) / 2 \\\\
           = 1 - Q_x - Q_z
    """
    return 1 - qx - qz

# denisty matrix functions

def purity(
        density_matrix: typing.Sequence[typing.Sequence[complex]]
) -> float:
    """
    Calculate quantum-state purity:

    .. math::
        \\text{P} = \\text{Tr}(ρ^2)
    """
    rho = np.asarray(density_matrix, dtype=complex)

    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError("Density matrix must be square.")

    return float(np.real(np.trace(rho @ rho)))
