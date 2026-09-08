import dataclasses
import typing

from ..timetags.channels import ChannelMap


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