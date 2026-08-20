import numpy as np

def binary_entropy(x: float) -> float:
    """
        Binary entropy function

        Parameters
        ----------
        x: float
            x
    """
    return -x * np.log2(x) - (1-x) * np.log2(1-x)

def fraction_to_dB(x: float) -> float:
    return -10 * np.log10(x)

def dB_to_fraction(x: float) -> float:
    return 10**(-x/10)