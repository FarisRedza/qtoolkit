import dataclasses
import typing
import pathlib

import numpy as np

from .timetagging import count_coincidences
from .quantum_functions import (
    qber_from_coincidences,
    visibility_from_qber,
    fidelity_from_visibility
)
from .polarisation import (
    H,V,D,A,
    projection_probability,
    joint_projection_probability
)





