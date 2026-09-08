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
