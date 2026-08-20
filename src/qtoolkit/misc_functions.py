import numpy as np

def binary_entropy(p: float) -> float:
    """
    Calculate the binary entropy, which quantifies the uncertainty associated
    with a binary outcome with probability (p).

    .. math::
        \\text{H}(X) = -p\\log_2(p) - (1-p)\\log_2(1-p)

    Parameters
    ----------
    x: float
        x
    """
    return -p * np.log2(p) - (1-p) * np.log2(1-p)

def fraction_to_dB(x: float) -> float:
    return -10 * np.log10(x)

def dB_to_fraction(x: float) -> float:
    return 10**(-x/10)