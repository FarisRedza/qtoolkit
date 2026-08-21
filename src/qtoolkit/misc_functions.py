import numpy as np

def binary_entropy(p: float) -> float:
    """
    Calculate the binary entropy, which quantifies the uncertainty associated
    with a binary outcome with probability (p).

    .. math::
        \\text{H}(X) = -p\\log_2(p) - (1-p)\\log_2(1-p)

    Parameters
    ----------
    p: float
        Probability of one of the two outcomes, in the range [0, 1]
    
    Returns
    -------
    float
    """
    if not 0 <= p <= 1:
        raise ValueError('p must be between 0 and 1.')

    if p in (0, 1):
        return 0.0

    return -p * np.log2(p) - (1-p) * np.log2(1-p)

def fraction_to_dB(x: float) -> float:
    return -10 * np.log10(x)

def dB_to_fraction(x: float) -> float:
    return 10**(-x/10)