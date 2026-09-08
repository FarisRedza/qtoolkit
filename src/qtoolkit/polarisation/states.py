import numpy as np
import numpy.typing as npt


PolarisationState = npt.NDArray[np.complex128]
JonesMatrix = npt.NDArray[np.complex128]

H: PolarisationState = np.array(
    [1.0, 0.0],
    dtype=complex,
)

V: PolarisationState = np.array(
    [0.0, 1.0],
    dtype=complex,
)

D: PolarisationState = (
    H + V
) / np.sqrt(2)

A: PolarisationState = (
    H - V
) / np.sqrt(2)

R: PolarisationState = (
    H - 1j * V
) / np.sqrt(2)

L: PolarisationState = (
    H + 1j * V
) / np.sqrt(2)


# bell states

PHI_PLUS: PolarisationState = (
    np.kron(H, H)
    + np.kron(V, V)
) / np.sqrt(2)

PHI_MINUS: PolarisationState = (
    np.kron(H, H)
    - np.kron(V, V)
) / np.sqrt(2)

PSI_PLUS: PolarisationState = (
    np.kron(H, V)
    + np.kron(V, H)
) / np.sqrt(2)

PSI_MINUS: PolarisationState = (
    np.kron(H, V)
    - np.kron(V, H)
) / np.sqrt(2)

# state transformations

def apply_jones_matrix(
    state: npt.ArrayLike,
    matrix: npt.ArrayLike,
) -> PolarisationState:
    """
    Apply a Jones matrix to a single-photon polarisation state.
    """
    state_array = np.asarray(
        state,
        dtype=complex,
    )

    matrix_array = np.asarray(
        matrix,
        dtype=complex,
    )

    if state_array.shape != (2,):
        raise ValueError(
            'Single-photon state must have shape (2,).'
        )

    if matrix_array.shape != (2, 2):
        raise ValueError(
            'Jones matrix must have shape (2, 2).'
        )

    return matrix_array @ state_array


def apply_local_jones_matrix(
    state: npt.ArrayLike,
    matrix: npt.ArrayLike,
    subsystem: int,
) -> PolarisationState:
    """
    Apply a Jones matrix to one photon of a two-photon state.

    Parameters
    ----------
    state:
        Two-photon state in the basis
        |HH>, |HV>, |VH>, |VV>.

    matrix:
        2x2 Jones matrix.

    subsystem:
        Photon to transform: 0 or 1.
    """
    state_array = np.asarray(
        state,
        dtype=complex,
    )

    matrix_array = np.asarray(
        matrix,
        dtype=complex,
    )

    if state_array.shape != (4,):
        raise ValueError(
            'Two-photon state must have shape (4,).'
        )

    if matrix_array.shape != (2, 2):
        raise ValueError(
            'Jones matrix must have shape (2, 2).'
        )

    if subsystem == 0:
        transformation = np.kron(
            matrix_array,
            np.eye(2),
        )

    elif subsystem == 1:
        transformation = np.kron(
            np.eye(2),
            matrix_array,
        )

    else:
        raise ValueError(
            'subsystem must be 0 or 1.'
        )

    return np.asarray(
        transformation @ state_array,
        dtype=np.complex128,
    )