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

    def pairs(self, first_basis: str, second_basis: str) -> BasisPairs:
        """Return outcome pairs (00, 01, 10, 11) for two Z/X bases.

        Z uses H/V outcomes and X uses D/A outcomes. Basis labels are
        case-insensitive. Missing detector channels raise ``ValueError``.
        """
        basis_channels = {
            'Z': ('h', 'v', 'H', 'V'),
            'X': ('d', 'a', 'D', 'A'),
        }
        first_basis = first_basis.upper()
        second_basis = second_basis.upper()
        if first_basis not in basis_channels or second_basis not in basis_channels:
            raise ValueError('Supported BB84 bases are Z and X.')

        def outcomes(measurement: BB84Measurement, basis: str) -> tuple[tuple[int, str], tuple[int, str]]:
            zero, one, zero_name, one_name = basis_channels[basis]
            zero_channel = getattr(measurement.channels, zero)
            one_channel = getattr(measurement.channels, one)
            if zero_channel is None or one_channel is None:
                raise ValueError(f'{basis}-basis pairs require both {zero_name} and {one_name} channels.')
            return ((zero_channel, zero_name), (one_channel, one_name))

        first = outcomes(self.first, first_basis)
        second = outcomes(self.second, second_basis)
        return tuple(
            ChannelPair(a, b, name=f'{a_name}{b_name}')
            for a, a_name in first
            for b, b_name in second
        )  # type: ignore[return-value]

    @property
    def z_pairs(self) -> BasisPairs:
        """Backwards-compatible Z/Z coincidence pairs."""
        return self.pairs('Z', 'Z')

    @property
    def x_pairs(self) -> BasisPairs:
        """Backwards-compatible X/X coincidence pairs."""
        return self.pairs('X', 'X')

    @property
    def coincidence_pairs(self) -> tuple[ChannelPair, ...]:
        """Backwards-compatible same-basis coincidence pairs."""
        return (*self.z_pairs, *self.x_pairs)

    @property
    def all_pairs(self) -> tuple[ChannelPair, ...]:
        """All 16 Z/X detector pairs, ordered ZZ, ZX, XZ, XX."""
        return tuple(
            pair
            for first_basis in ('Z', 'X')
            for second_basis in ('Z', 'X')
            for pair in self.pairs(first_basis, second_basis)
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