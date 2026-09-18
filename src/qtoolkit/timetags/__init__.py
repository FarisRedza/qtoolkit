"""Tools for processing and simulating timetag data.

This subpackage provides data structures for timetags, coincidence-counting
functions, channel utilities, and tools for generating simulated timetag data.
"""

from .channels import (
    ChannelPair,
    BasisPairs,
    ChannelMap,
)

from .coincidences import (
    count_twofold_coincidences,
    find_twofold_coincidence_indices,
    count_threefold_coincidences,
    count_fourfold_coincidences,
    count_coincidences,
)

from .data import (
    TimetagData,
    ProcessedTimetagData,
)

from .simulation import (
    CoincidenceProcess,
    generate_timetags,
    LiveTimetagSimulator,
    coincidence_processes_from_probabilities,
)

__all__ = [
    'ChannelPair',
    'BasisPairs',
    'ChannelMap',

    'count_twofold_coincidences',
    'find_twofold_coincidence_indices',
    'count_threefold_coincidences',
    'count_fourfold_coincidences',
    'count_coincidences',

    'TimetagData',
    'ProcessedTimetagData',

    'CoincidenceProcess',
    'generate_timetags',
    'LiveTimetagSimulator',
    'coincidence_processes_from_probabilities',
]