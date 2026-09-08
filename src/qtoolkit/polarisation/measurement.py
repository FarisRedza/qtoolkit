import dataclasses

import numpy as np
import numpy.typing as npt

from ..timetags.channels import (
    BasisPairs,
    ChannelPair,
)
from ..polarisation.channels import PolarisationChannelMap

from .states import (
    H,V,D,A
)

def projection_probability(
    state: npt.ArrayLike,
    measurement_state: npt.ArrayLike,
) -> float:
    """
    Probability of projecting onto a measurement state.
    """
    state_array = np.asarray(
        state,
        dtype=complex,
    )

    measurement_array = np.asarray(
        measurement_state,
        dtype=complex,
    )

    if state_array.shape != measurement_array.shape:
        raise ValueError(
            'state and measurement_state must have the same shape.'
        )

    amplitude = np.vdot(
        measurement_array,
        state_array,
    )

    return float(
        abs(amplitude) ** 2
    )


def joint_projection_probability(
    state: npt.ArrayLike,
    first_state: npt.ArrayLike,
    second_state: npt.ArrayLike,
) -> float:
    """
    Probability of jointly measuring two polarisation states.
    """
    measurement_state = np.kron(
        np.asarray(
            first_state,
            dtype=complex,
        ),
        np.asarray(
            second_state,
            dtype=complex,
        ),
    )

    return projection_probability(
        state=state,
        measurement_state=measurement_state,
    )


@dataclasses.dataclass(frozen=True)
class BB84Measurement:
    channels: PolarisationChannelMap
    z_probability: float = 0.5
    x_probability: float = 0.5

    def probabilities(
        self,
        state: np.typing.ArrayLike,
    ) -> dict[int, float]:
        """
        Calculate detector probabilities for a BB84 measurement stage.
        """
        if (
            self.channels.h is None
            or self.channels.v is None
            or self.channels.d is None
            or self.channels.a is None
        ):
            raise ValueError(
                'BB84 measurement requires H, V, D, and A channels.'
            )

        return {
            self.channels.h: (
                self.z_probability
                * projection_probability(
                    state,
                    H,
                )
            ),
            self.channels.v: (
                self.z_probability
                * projection_probability(
                    state,
                    V,
                )
            ),
            self.channels.d: (
                self.x_probability
                * projection_probability(
                    state,
                    D,
                )
            ),
            self.channels.a: (
                self.x_probability
                * projection_probability(
                    state,
                    A,
                )
            ),
        }


@dataclasses.dataclass(frozen=True)
class BB84MeasurementPair:
    first: BB84Measurement
    second: BB84Measurement

    @property
    def z_pairs(self) -> BasisPairs:
        if (
            self.first.channels.h is None
            or self.first.channels.v is None
            or self.second.channels.h is None
            or self.second.channels.v is None
        ):
            raise ValueError(
                'Z-basis pairs require H and V channels for both measurements.'
            )

        first_h = self.first.channels.h
        first_v = self.first.channels.v
        second_h = self.second.channels.h
        second_v = self.second.channels.v
        return (
            ChannelPair(
                first_h,
                second_h,
                name='HH',
            ),
            ChannelPair(
                first_h,
                second_v,
                name='HV',
            ),
            ChannelPair(
                first_v,
                second_h,
                name='VH',
            ),
            ChannelPair(
                first_v,
                second_v,
                name='VV',
            ),
        )


    @property
    def x_pairs(self) -> BasisPairs:
        if (
            self.first.channels.d is None
            or self.first.channels.a is None
            or self.second.channels.d is None
            or self.second.channels.a is None
        ):
            raise ValueError(
                'X-basis pairs require D and A channels for both measurements.'
            )

        first_d = self.first.channels.d
        first_a = self.first.channels.a
        second_d = self.second.channels.d
        second_a = self.second.channels.a
        return (
            ChannelPair(
                first_d,
                second_d,
                name='DD',
            ),
            ChannelPair(
                first_d,
                second_a,
                name='DA',
            ),
            ChannelPair(
                first_a,
                second_d,
                name='AD',
            ),
            ChannelPair(
                first_a,
                second_a,
                name='AA',
            ),
        )


    @property
    def coincidence_pairs(
        self,
    ) -> tuple[ChannelPair, ...]:
        return (
            *self.z_pairs,
            *self.x_pairs,
        )

    def joint_probabilities(
        self,
        state: np.typing.ArrayLike,
    ) -> dict[tuple[int, int], float]:
        """
        Calculate joint detector probabilities for two BB84 stages.
        """
        state_array = np.asarray(
            state,
            dtype=complex,
        )

        first_states = {
            self.first.channels.h: (
                H,
                self.first.z_probability,
            ),
            self.first.channels.v: (
                V,
                self.first.z_probability,
            ),
            self.first.channels.d: (
                D,
                self.first.x_probability,
            ),
            self.first.channels.a: (
                A,
                self.first.x_probability,
            ),
        }

        second_states = {
            self.second.channels.h: (
                H,
                self.second.z_probability,
            ),
            self.second.channels.v: (
                V,
                self.second.z_probability,
            ),
            self.second.channels.d: (
                D,
                self.second.x_probability,
            ),
            self.second.channels.a: (
                A,
                self.second.x_probability,
            ),
        }

        probabilities = {}

        for (
            first_channel,
            (first_state, first_basis_probability),
        ) in first_states.items():

            for (
                second_channel,
                (
                    second_state,
                    second_basis_probability,
                ),
            ) in second_states.items():

                probability = (
                    first_basis_probability
                    * second_basis_probability
                    * joint_projection_probability(
                        state=state_array,
                        first_state=first_state,
                        second_state=second_state,
                    )
                )

                probabilities[
                    (
                        first_channel,
                        second_channel,
                    )
                ] = probability

        return probabilities