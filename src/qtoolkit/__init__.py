"""Tools for quantum optics, quantum communication, and experimental data analysis.

qtoolkit provides utilities for working with polarisation states and optical
components, processing and simulating timetag data, and analysing quantum key
distribution (QKD) experiments.

The main subpackages are:

- ``qtoolkit.polarisation``: Polarisation states, measurements, and optics.
- ``qtoolkit.qkd``: QKD protocols, metrics, and analytical models.
- ``qtoolkit.timetags``: Timetag processing, coincidence counting, and simulation.
"""

from .misc_functions import (
    binary_entropy,
    fraction_to_dB,
    dB_to_fraction
)

from . import (
    polarisation,
    qkd,
    timetags,
)

__all__ = [
    'binary_entropy',
    'fraction_to_dB',
    'dB_to_fraction',

    'polarisation',
    'qkd',
    'timetags',
]