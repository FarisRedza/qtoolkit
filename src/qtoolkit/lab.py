import dataclasses
import typing
import pathlib

import numpy as np

from .timetagging import count_coincidences
from .quantum_functions import (
    qber_from_coincidences,
    visibility_from_qber,
    fidelity_from_visibility
)


@dataclasses.dataclass(frozen=True)
class ChannelPair:
    """
    Pair of timetagger channels.

    Parameters
    ----------
    first : int
        First channel.
    second : int
        Second channel.
    name : str | None
        Optional name describing the channel pair.
    """

    first: int
    second: int
    name: typing.Optional[str] = None

    def as_tuple(self) -> tuple[int, int]:
        return self.first, self.second


BasisPairs = tuple[ChannelPair, ChannelPair, ChannelPair, ChannelPair]


@dataclasses.dataclass(frozen=True)
class ChannelMap:
    channels: dict[str, int]

    def __getitem__(self, name: str) -> int:
        return self.channels[name]

    def get(self, name: str) -> typing.Optional[int]:
        return self.channels.get(name)

    @property
    def numbers(self) -> tuple[int, ...]:
        return tuple(self.channels.values())

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.channels.keys())


@dataclasses.dataclass(frozen=True)
class PolarisationChannelMap:
    h: typing.Optional[int] = None
    v: typing.Optional[int] = None
    d: typing.Optional[int] = None
    a: typing.Optional[int] = None
    r: typing.Optional[int] = None
    l: typing.Optional[int] = None

    @property
    def channels(self) -> tuple[int, ...]:
        return tuple(
            channel
            for channel in (
                self.h,
                self.v,
                self.d,
                self.a,
                self.r,
                self.l,
            )
            if channel is not None
        )

    def as_channel_map(self) -> ChannelMap:
        channels = {
            name.upper(): channel
            for name, channel in (
                ('h', self.h),
                ('v', self.v),
                ('d', self.d),
                ('a', self.a),
                ('r', self.r),
                ('l', self.l),
            )
            if channel is not None
        }

        return ChannelMap(channels=channels)


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
            c_00=coincidences[pair_00.as_tuple()],
            c_01=coincidences[pair_01.as_tuple()],
            c_10=coincidences[pair_10.as_tuple()],
            c_11=coincidences[pair_11.as_tuple()],
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
class TimetagData:
    timetags: np.typing.NDArray[np.int64]
    channels: np.typing.NDArray[np.int8]
    file_path: typing.Optional[pathlib.Path] = None

    @classmethod
    def from_file(
            cls,
            file_path: typing.Union[pathlib.Path, str]
    ) -> 'TimetagData':
        file_path = pathlib.Path(file_path)
        file_type = file_path.suffix.lower()
        if file_type in {'.txt', '.text'}:
                data = np.loadtxt(
                    fname=file_path,
                    dtype=np.int64,
                    ndmin=2
                )
        elif file_type == '.csv':
            data = np.loadtxt(
                fname=file_path,
                dtype=np.int64,
                delimiter=',',
                ndmin=2
            )
        else:
            raise ValueError(f'Unsupported file type: {file_type}')

        if data.shape[1] != 2:
            raise ValueError(
                'Timetag file must contain exactly two columns.'
            )

        timetags = data[:,0]
        channels = data[:,1].astype(np.int8)

        return cls(
            timetags=timetags,
            channels=channels,
            file_path=file_path
        )

    @property
    def duration_ps(self) -> int:
        if len(self) < 2:
            return 0

        return int(self.timetags[-1] - self.timetags[0])

    def to_file(
            self,
            file_path: typing.Union[pathlib.Path, str]
    ) -> None:
        file_path = pathlib.Path(file_path)

        np.savetxt(
            fname=file_path,
            X=np.c_[(self.timetags, self.channels)],
            fmt='%d'
        )

    def get_channel_timetags(
            self,
            channel: int
    ) -> np.typing.NDArray[np.int64]:
        return self.timetags[self.channels == channel]

    def count(
            self,
            channel: typing.Optional[int] = None
    ) -> int:
        """
        Return the number of timetags.

        If a channel is supplied, only events on that channel are counted.

        Parameters
        ----------
        channel: int | None
            Optional parameter to only get counts from this channel.
        
        Returns
        -------
        int
        """
        if channel is None:
            return len(self.timetags)

        return int(np.count_nonzero(self.channels == channel))

    def select_channels(self, *channels: int) -> 'TimetagData':
        """
        Create a new TimetagData object with only the specified channels.
        """
        mask = np.isin(self.channels, channels)

        return TimetagData(
            timetags=self.timetags[mask],
            channels=self.channels[mask],
            file_path=self.file_path
        )

    def __len__(self) -> int:
        return len(self.timetags)

    def __post_init__(self) -> None:
        if self.timetags.ndim != 1:
            raise ValueError('timetags must be one-dimensional.')

        if self.channels.ndim != 1:
            raise ValueError('channels must be one-dimensional.')

        if len(self.timetags) != len(self.channels):
            raise ValueError(
                'timetags and channels must have the same length.'
            )

@dataclasses.dataclass(frozen=True)
class ProcessedTimetagData:
    coincidences: dict[tuple[int, int], int]
    coincidence_window: int
    file_path: typing.Optional[pathlib.Path] = None

    @classmethod
    def from_timetag_data(
            cls,
            timetag_data: TimetagData,
            pairs: typing.Iterable[
                typing.Union[ChannelPair, tuple[int, int]]
            ],
            coincidence_window: int
    ) -> 'ProcessedTimetagData':
        pairs = tuple(
            pair.as_tuple()
            if isinstance(pair, ChannelPair)
            else pair
            for pair in pairs
        )
        coincidences = count_coincidences(
            timetags=timetag_data.timetags,
            channels=timetag_data.channels,
            pairs=pairs,
            coincidence_window=coincidence_window
        )
        return cls(
            coincidences=coincidences,
            coincidence_window=coincidence_window,
            file_path=timetag_data.file_path
        )

    @classmethod
    def from_file(
            cls,
            file_path: typing.Union[pathlib.Path, str],
            pairs: list[ChannelPair],
            coincidence_window: int
    ) -> 'ProcessedTimetagData':
        timetag_data = TimetagData.from_file(file_path=file_path)
        return cls.from_timetag_data(
            timetag_data=timetag_data,
            pairs=pairs,
            coincidence_window=coincidence_window
        )

    def get_basis_metrics(
            self,
            pairs: BasisPairs,
    ) -> 'BasisMetrics':
        """
        Calculate metrics for a set of basis channel pairs.

        Parameters
        ----------
        pairs : BasisPairs
            Channel pairs corresponding to the outcomes 00, 01, 10, and 11.

        Returns
        -------
        BasisMetrics
        """
        return BasisMetrics.from_coincidences(
            coincidences=self.coincidences,
            pairs=pairs,
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
