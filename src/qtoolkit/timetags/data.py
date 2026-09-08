import dataclasses
import typing
import pathlib

import numpy as np
import numpy.typing as npt

# from ..qkd.metrics import BasisMetrics

# from .channels import (
#     ChannelPair,
#     BasisPairs,
# )
# from .coincidences import count_coincidences


@dataclasses.dataclass(frozen=True)
class TimetagData:
    timetags: npt.NDArray[np.int64]
    channels: npt.NDArray[np.int8]
    start_ps: typing.Optional[int] = None
    stop_ps: typing.Optional[int] = None
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
    def duration_ps(self) -> typing.Optional[int]:
        if (
            self.start_ps is None
            or self.stop_ps is None
        ):
            return None

        return self.stop_ps - self.start_ps

    @property
    def duration(self) -> typing.Optional[float]:
        if self.duration_ps is None:
            return None

        return self.duration_ps * 1e-12

    @property
    def span_ps(self) -> int:
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
    ) -> npt.NDArray[np.int64]:
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

# @dataclasses.dataclass(frozen=True)
# class ProcessedTimetagData:
#     coincidences: dict[tuple[int, int], int]
#     coincidence_window: int
#     file_path: typing.Optional[pathlib.Path] = None

#     @classmethod
#     def from_timetag_data(
#             cls,
#             timetag_data: TimetagData,
#             pairs: typing.Iterable[
#                 typing.Union[ChannelPair, tuple[int, int]]
#             ],
#             coincidence_window: int
#     ) -> 'ProcessedTimetagData':
#         pairs = tuple(
#             pair.as_tuple()
#             if isinstance(pair, ChannelPair)
#             else pair
#             for pair in pairs
#         )
#         coincidences = count_coincidences(
#             data=timetag_data,
#             pairs=pairs,
#             coincidence_window=coincidence_window
#         )
#         return cls(
#             coincidences=coincidences,
#             coincidence_window=coincidence_window,
#             file_path=timetag_data.file_path
#         )

#     @classmethod
#     def from_file(
#             cls,
#             file_path: typing.Union[pathlib.Path, str],
#             pairs: list[ChannelPair],
#             coincidence_window: int
#     ) -> 'ProcessedTimetagData':
#         timetag_data = TimetagData.from_file(file_path=file_path)
#         return cls.from_timetag_data(
#             timetag_data=timetag_data,
#             pairs=pairs,
#             coincidence_window=coincidence_window
#         )

#     def get_basis_metrics(
#             self,
#             pairs: BasisPairs,
#     ) -> 'BasisMetrics':
#         """
#         Calculate metrics for a set of basis channel pairs.

#         Parameters
#         ----------
#         pairs : BasisPairs
#             Channel pairs corresponding to the outcomes 00, 01, 10, and 11.

#         Returns
#         -------
#         BasisMetrics
#         """
#         return BasisMetrics.from_coincidences(
#             coincidences=self.coincidences,
#             pairs=pairs,
#         )