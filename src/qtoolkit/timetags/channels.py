import dataclasses
import typing


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

    @property
    def channels(
        self,
    ) -> tuple[int, int]:
        return (
            self.first,
            self.second,
        )

    def as_tuple(self) -> tuple[int, int]:
        return self.channels


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