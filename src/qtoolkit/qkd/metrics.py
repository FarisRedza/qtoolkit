import typing
import dataclasses

import numpy as np
import numpy.typing as npt

from ..timetags import (
    ChannelPair,
    PolarisationChannelMap,
    BasisPairs,
    ProcessedTimetagData,
)

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

def qber_from_visibility(visibility: float) -> float:
    """
    Calculate QBER from visibility

    .. math:: 
        \\text{QBER} = (1 - V)/2

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
        V = (C_\\text{max} - C_\\text{min}) / (C_\\text{max} + C_\\text{min})
    
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
        V = 1 - 2 * \\text{QBER}

        F ~= (V_x + V_z) / 2 \\\\
           = 1 - Q_x - Q_z
    """
    return 1 - qx - qz

# denisty matrix functions

def purity(
        density_matrix: npt.ArrayLike
) -> float:
    """
    Calculate quantum-state purity:

    .. math::
        \\text{P} = \\text{Tr}(\\rho^2)
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

    @property
    def qber(self) -> float:
        return qber_from_coincidences(
            c_00=self.c_00,
            c_01=self.c_01,
            c_10=self.c_10,
            c_11=self.c_11,
        )

    @property
    def visibility(self) -> float:
        return visibility_from_qber(
            qber=self.qber,
        )

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


@dataclasses.dataclass(frozen=True)
class BBM92ChannelMap:
    first: PolarisationChannelMap
    second: PolarisationChannelMap

    @staticmethod
    def _basis_pairs(
        first_0: typing.Optional[int],
        first_1: typing.Optional[int],
        second_0: typing.Optional[int],
        second_1: typing.Optional[int],
        basis: str
    ) -> BasisPairs:
        channels = (
            first_0,
            first_1,
            second_0,
            second_1,
        )

        if any(channel is None for channel in channels):
            raise ValueError(
                f'Both measurement stages must define the {basis} basis.'
            )

        assert first_0 is not None
        assert first_1 is not None
        assert second_0 is not None
        assert second_1 is not None

        return (
            ChannelPair(first_0, second_0, '00'),
            ChannelPair(first_0, second_1, '01'),
            ChannelPair(first_1, second_0, '10'),
            ChannelPair(first_1, second_1, '11'),
        )

    @property
    def zz_pairs(self) -> BasisPairs:
        """
        Return the H/V coincidence pairs.
        """
        return self._basis_pairs(
            self.first.h,
            self.first.v,
            self.second.h,
            self.second.v,
            basis='Z',
        )

    @property
    def xx_pairs(self) -> BasisPairs:
        """
        Return the D/A coincidence pairs.
        """
        return self._basis_pairs(
            self.first.d,
            self.first.a,
            self.second.d,
            self.second.a,
            basis='X',
        )

    @property
    def yy_pairs(self) -> BasisPairs:
        """
        Return the R/L coincidence pairs.
        """
        return self._basis_pairs(
            self.first.r,
            self.first.l,
            self.second.r,
            self.second.l,
            basis='Y',
        )


@dataclasses.dataclass(frozen=True)
class BBM92Metrics:
    zz: BasisMetrics
    xx: BasisMetrics

    @classmethod
    def from_processed_data(
            cls,
            processed: ProcessedTimetagData,
            channel_map: BBM92ChannelMap,
    ) -> 'BBM92Metrics':
        return cls(
            zz=processed.get_basis_metrics(channel_map.zz_pairs),
            xx=processed.get_basis_metrics(channel_map.xx_pairs),
        )

    @property
    def fidelity(self) -> float:
        return fidelity_from_visibility(
            visibility_z=self.zz.visibility,
            visibility_x=self.xx.visibility
        )

    def __str__(self) -> str:
        header = (
            f'{"Basis":<6}'
            f'{"00":>8}'
            f'{"01":>8}'
            f'{"10":>8}'
            f'{"11":>8}'
            f'{"Odd":>8}'
            f'{"Even":>8}'
            f'{"Total":>8}'
            f'{"Prob":>10}'
            f'{"QBER":>10}'
            f'{"Vis":>10}'
            f'{"Fid approx":>12}'
        )

        rows = []

        for name, metrics in [
            ('ZZ', self.zz),
            ('XX', self.xx),
        ]:
            fidelity = (
                f'{self.fidelity:.6f}'
                if name == 'ZZ'
                else ''
            )

            row = (
                f'{name:<6}'
                f'{metrics.c_00:>8}'
                f'{metrics.c_01:>8}'
                f'{metrics.c_10:>8}'
                f'{metrics.c_11:>8}'
                f'{metrics.odd:>8}'
                f'{metrics.even:>8}'
                f'{metrics.total:>8}'
                f'{metrics.even_probability:>10.6f}'
                f'{metrics.qber:>10.6f}'
                f'{metrics.visibility:>10.6f}'
                f'{fidelity:>12}'
            )

            rows.append(row)

        return '\n'.join([header, *rows])