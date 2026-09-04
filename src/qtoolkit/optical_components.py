import dataclasses

import numpy as np


@dataclasses.dataclass
class WavePlate:
    retardance_rad: float
    angle_deg: float = 0.0

    @property
    def angle_rad(self) -> float:
        return np.radians(
            self.angle_deg
        )

    @property
    def matrix(self) -> np.ndarray:
        """Jones matrix of the waveplate."""
        theta = np.radians(self.angle_deg)

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        rotation = np.array(
            [
                [cos_theta, -sin_theta],
                [sin_theta, cos_theta],
            ],
            dtype=complex,
        )

        inverse_rotation = rotation.T

        retardance = np.array(
            [
                [1.0, 0.0],
                [0.0, np.exp(1j * self.retardance_rad)],
            ],
            dtype=complex,
        )

        return (
            inverse_rotation
            @ retardance
            @ rotation
        )

    def apply(
        self,
        state: np.ndarray,
    ) -> np.ndarray:
        """Apply the waveplate to a Jones vector."""
        state = np.asarray(
            state,
            dtype=complex,
        )

        if state.shape != (2,):
            raise ValueError(
                'Polarisation state must have shape (2,).'
            )

        return self.matrix @ state


class QuarterWavePlate(WavePlate):
    def __init__(
        self,
        angle_deg: float = 0.0,
    ) -> None:
        super().__init__(
            retardance_rad=np.pi / 2,
            angle_deg=angle_deg,
        )


class HalfWavePlate(WavePlate):
    def __init__(
        self,
        angle_deg: float = 0.0,
    ) -> None:
        super().__init__(
            retardance_rad=np.pi,
            angle_deg=angle_deg,
        )


def compose_waveplates(
    waveplates: list[WavePlate],
) -> np.ndarray:
    """Compose a sequence of waveplates into a single Jones matrix."""
    result = np.eye(2, dtype=complex)

    for waveplate in waveplates:
        result = waveplate.matrix @ result

    return result