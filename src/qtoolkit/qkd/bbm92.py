import dataclasses
import typing

from ..timetags.channels import (
    ChannelPair,
    BasisPairs,
    # ProcessedTimetagData,
)

from ..polarisation.channels import PolarisationChannelMap

from .metrics import (
    BasisMetrics,
    fidelity_from_visibility
)


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

    # @classmethod
    # def from_processed_data(
    #         cls,
    #         processed: ProcessedTimetagData,
    #         channel_map: BBM92ChannelMap,
    # ) -> 'BBM92Metrics':
    #     return cls(
    #         zz=processed.get_basis_metrics(channel_map.zz_pairs),
    #         xx=processed.get_basis_metrics(channel_map.xx_pairs),
    #     )

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