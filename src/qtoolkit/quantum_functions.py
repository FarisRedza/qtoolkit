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