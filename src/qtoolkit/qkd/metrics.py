import typing
import dataclasses
import warnings

import numpy as np
import numpy.typing as npt

from ..timetags.channels import ChannelPair

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
    denominator = incorrect + correct
    if denominator == 0:
        warnings.warn('Warning: Division by zero. Returing 0')
        return 0

    return incorrect / denominator

    

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

        correlated: bool = True
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
        correlated: bool = True
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
        correlated: bool = True
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
        correlated: bool = True
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

def correlation_from_coincidences(
        c_00: float,
        c_01: float,
        c_10: float,
        c_11: float,
) -> float:
    r"""
    Calculate the signed same-versus-different outcome correlation.

    .. math::
        C = \frac{(c_{00}+c_{11})-(c_{01}+c_{10})}
                 {c_{00}+c_{01}+c_{10}+c_{11}}.

    ``C=+1`` denotes perfectly correlated outcomes and ``C=-1`` denotes
    perfectly anti-correlated outcomes.  Unlike a QBER-derived visibility,
    the sign is not adjusted to make the expected outcome positive.
    """
    total = c_00 + c_01 + c_10 + c_11
    if total == 0:
        return float('nan')

    return (c_00 + c_11 - c_01 - c_10) / total


def correlation_from_qber(
        qber: float,
        correlated: bool = True,
) -> float:
    r"""Convert QBER to a signed same-versus-different correlation.

    For a target with correlated outcomes, :math:`C=1-2Q`.  For a target
    with anti-correlated outcomes, :math:`C=-(1-2Q)`.
    """
    target_sign = 1 if correlated else -1
    return target_sign * (1 - 2 * qber)


def qber_from_visibility(visibility: float) -> float:
    r"""
    Calculate QBER from target-adjusted visibility.

    Here ``visibility`` is positive for the expected outcome pattern, whether
    that pattern is correlated or anti-correlated:

    .. math::
        \text{QBER} = (1 - V)/2.

    It should not be confused with a signed same-versus-different correlation,
    which is negative for ideal anti-correlated outcomes.

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
    r"""
    Calculate visibility

    .. math::
        V = (C_\text{max} - C_\text{min}) / (C_\text{max} + C_\text{min})
    
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
    r"""
    Calculate target-adjusted visibility from QBER.

    .. math::
        V = 1 - 2 * \text{QBER}.

    Since QBER already defines which outcomes are correct, this quantity is
    positive for an ideal target regardless of whether the expected outcomes
    are correlated or anti-correlated.  Use :func:`correlation_from_qber` when
    a signed same-versus-different correlation is required.

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

def fidelity_from_correlations(
        correlation_z: float,
        correlation_x: float,
        correlation_y: typing.Optional[float] = None
) -> float:
    r"""
    Fidelity with :math:`|\Phi^+\rangle` from basis correlations.

    The inputs are signed same-versus-different correlations,

    .. math::
        C_i = \langle \sigma_i \otimes \sigma_i \rangle.

    For :math:`|\Phi^+\rangle`, the ideal correlations are
    :math:`C_x=+1`, :math:`C_y=-1`, and :math:`C_z=+1`.  If all three
    correlations are supplied, the Bell-state projector gives

    .. math::
        F_{\Phi^+} = (1 + C_x - C_y + C_z) / 4.

    If only X and Z are supplied, the function returns the two-basis
    lower bound

    .. math::
        F_{\Phi^+} \ge (C_x + C_z) / 2.

    These are correlations rather than unsigned fringe contrasts or
    target-adjusted QBER visibilities.

    Parameters
    ----------
    correlation_z : float
        Signed Z-basis correlation.
    correlation_x : float
        Signed X-basis correlation.
    correlation_y : float, optional
        Signed Y-basis correlation.

    Returns
    -------
    float
        Exact :math:`|\Phi^+\rangle` fidelity when all three correlations
        are supplied; otherwise the X/Z lower bound.
    """
    if correlation_y is None:
        return (correlation_x + correlation_z) / 2

    return (
        1
        + correlation_x
        - correlation_y
        + correlation_z
    ) / 4


def fidelity_from_visibility(
        visibility_z: float,
        visibility_x: float,
        visibility_y: typing.Optional[float] = None,
) -> float:
    r"""
    Backwards-compatible alias for :func:`fidelity_from_correlations`.

    Despite the parameter names, the inputs are **signed basis
    correlations**, not target-adjusted QBER visibilities or unsigned fringe
    contrasts.  New code should use :func:`fidelity_from_correlations`.
    
    This will eventually be cleaned up.
    """
    return fidelity_from_correlations(
        correlation_z=visibility_z,
        correlation_x=visibility_x,
        correlation_y=visibility_y,
    )


def fidelity_from_qber(
    qx: float,
    qz: float,
) -> float:
    r"""
    Two-basis lower bound on :math:`|\Phi^+\rangle` fidelity.

    For QBERs defined relative to the correlated outcomes expected from
    :math:`|\Phi^+\rangle` in the X and Z bases,

    .. math::
        V_i = 1 - 2Q_i,

    which gives the lower bound

    .. math::
        F_{\Phi^+} \ge (V_x + V_z) / 2
        = 1 - Q_x - Q_z.
    """
    return 1 - qx - qz

# denisty matrix functions

def purity(
        density_matrix: npt.ArrayLike
) -> float:
    r"""
    Calculate quantum-state purity:

    .. math::
        \text{P} = \text{Tr}(\rho^2)
    """
    rho = np.asarray(density_matrix, dtype=complex)

    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError('Density matrix must be square.')

    return float(np.real(np.trace(rho @ rho)))


@dataclasses.dataclass(frozen=True)
class BasisMetrics:
    """
    Metrics calculated from four two-outcome coincidence counts.

    The coincidence counts correspond to the possible outcomes 00, 01,
    10, and 11 for a pair of two-outcome measurements.

    Parameters
    ----------
    c_00 : int
        Coincidences between outcome 0 and outcome 0.
    c_01 : int
        Coincidences between outcome 0 and outcome 1.
    c_10 : int
        Coincidences between outcome 1 and outcome 0.
    c_11 : int
        Coincidences between outcome 1 and outcome 1.
    """

    c_00: int
    c_01: int
    c_10: int
    c_11: int

    @classmethod
    def from_coincidences(
            cls,
            coincidences: dict[tuple[int, int], int],
            pairs: tuple[
                ChannelPair,
                ChannelPair,
                ChannelPair,
                ChannelPair,
            ],
    ) -> 'BasisMetrics':
        """
        Create basis metrics from coincidence data.

        The channel pairs must be supplied in the order 00, 01, 10, 11.

        Parameters
        ----------
        coincidences : dict[tuple[int, int], int]
            Coincidence counts indexed by channel pair.
        pairs : BasisPairs
            Channel pairs corresponding to the outcomes 00, 01, 10, and
            11, respectively.

        Returns
        -------
        BasisMetrics
        """
        pair_00, pair_01, pair_10, pair_11 = pairs

        return cls(
            c_00=coincidences[
                (
                    pair_00.first,
                    pair_00.second,
                )
            ],
            c_01=coincidences[
                (
                    pair_01.first,
                    pair_01.second,
                )
            ],
            c_10=coincidences[
                (
                    pair_10.first,
                    pair_10.second,
                )
            ],
            c_11=coincidences[
                (
                    pair_11.first,
                    pair_11.second,
                )
            ],
        )


    @property
    def odd(self) -> int:
        return self.c_01 + self.c_10

    @property
    def even(self) -> int:
        return self.c_00 + self.c_11

    @property
    def total(self) -> int:
        return self.odd + self.even

    @property
    def even_probability(self) -> float:
        if self.total == 0:
            return float('nan')

        return self.even / self.total

    @property
    def odd_probability(self) -> float:
        if self.total == 0:
            return float('nan')

        return self.odd / self.total

    def qber_for(self, correlated: bool = True) -> float:
        """Return QBER for the specified expected outcome parity."""
        return qber_from_coincidences(
            c_00=self.c_00,
            c_01=self.c_01,
            c_10=self.c_10,
            c_11=self.c_11,
            correlated=correlated,
        )

    @property
    def qber(self) -> float:
        """QBER assuming correlated outcomes are correct."""
        return self.qber_for(correlated=True)

    @property
    def correlation(self) -> float:
        """Signed same-versus-different outcome correlation."""
        return correlation_from_coincidences(
            c_00=self.c_00,
            c_01=self.c_01,
            c_10=self.c_10,
            c_11=self.c_11,
        )

    def visibility_for(self, correlated: bool = True) -> float:
        """Return target-adjusted visibility for an expected parity."""
        return visibility_from_qber(self.qber_for(correlated=correlated))

    @property
    def visibility(self) -> float:
        """Historical alias for the signed correlation.

        For the default correlated-target interpretation this is also equal to
        ``1 - 2*qber``.  Prefer :attr:`correlation` when the sign matters or
        :meth:`visibility_for` for target-adjusted visibility.
        """
        return self.correlation

    def as_row(self) -> list[typing.Union[int, float]]:
        return [
            self.c_00,
            self.c_01,
            self.c_10,
            self.c_11,
            self.odd,
            self.even,
            self.total,
            self.even_probability,
            self.qber,
            self.visibility,
        ]
