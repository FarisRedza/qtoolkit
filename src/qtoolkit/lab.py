import dataclasses

from qtoolkit.quantum_functions import (
    qber_from_coincidences,
    visibility_from_qber,
    fidelity_from_visibility
)

@dataclasses.dataclass
class BasisMetrics:
    c_00: int
    c_01: int
    c_10: int
    c_11: int

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
    def probability(self) -> float:
        return self.even / self.total

    @property
    def qber(self) -> float:
        return qber_from_coincidences(
            c_00=self.c_00,
            c_10=self.c_10,
            c_01=self.c_01,
            c_11=self.c_11,
        )

    @property
    def visibility(self) -> float:
        return visibility_from_qber(qber=self.qber)

    def as_row(self) -> list:
        return [
            self.c_00,
            self.c_01,
            self.c_10,
            self.c_11,
            self.odd,
            self.even,
            self.total,
            self.probability,
            self.qber,
            self.visibility,
        ]

def print_metrics(
        c_hh: int,
        c_hv: int,
        c_vh: int,
        c_vv: int,
        c_dd: int,
        c_da: int,
        c_ad: int,
        c_aa: int
) -> None:
    zz = BasisMetrics(
        c_00=c_hh,
        c_01=c_hv,
        c_10=c_vh,
        c_11=c_vv
    )
    xx = BasisMetrics(
        c_00=c_dd,
        c_01=c_da,
        c_10=c_ad,
        c_11=c_aa
    )
    fid = fidelity_from_visibility(
        visibility_z=zz.visibility,
        visibility_x=xx.visibility
    )

    header = (
        f'{'Basis':<6}'
        f'{'00':>8}'
        f'{'01':>8}'
        f'{'10':>8}'
        f'{'11':>8}'
        f'{'Odd':>8}'
        f'{'Even':>8}'
        f'{'Total':>8}'
        f'{'Prob':>10}'
        f'{'QBER':>10}'
        f'{'Vis':>10}'
        f'{'Fid approx':>12}'
    )

    print(header)

    for name, metrics in [('ZZ', zz), ('XX', xx)]:
        fid = f'{fid:.6f}' if name == 'ZZ' else ''

        print(
            f'{name:<6}'
            f'{metrics.c_00:>8}'
            f'{metrics.c_01:>8}'
            f'{metrics.c_10:>8}'
            f'{metrics.c_11:>8}'
            f'{metrics.odd:>8}'
            f'{metrics.even:>8}'
            f'{metrics.total:>8}'
            f'{metrics.probability:>10.6f}'
            f'{metrics.qber:>10.6f}'
            f'{metrics.visibility:>10.6f}'
            f'{fid:>12}'
        )

if __name__ == '__main__':
    zz_tt = 17003
    zz_tr = 731
    zz_rt = 346
    zz_rr = 16898
    xx_tt = 16584
    xx_tr = 538
    xx_rt = 190
    xx_rr = 17667
    print_metrics(
        c_hh=zz_tt,
        c_hv=zz_tr,
        c_vh=zz_rt,
        c_vv=zz_rr,
        c_dd=xx_tt,
        c_da=xx_tr,
        c_ad=xx_rt,
        c_aa=zz_rr
    )