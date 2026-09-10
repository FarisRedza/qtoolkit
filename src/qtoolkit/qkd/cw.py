# Model for optimizing quantum key distribution with continuous-wave pumped entangled-photon sources
# DOI: 10.1103/PhysRevA.104.022406

import math

import numpy as np

from ..misc_functions import binary_entropy

# idealised cw-qkd

def true_single_rate(
        brightness: float,
        efficiency: float
) -> float:
    """
    Equation 2

    .. math::
        S_\\text{i}^\\text{t} = B \\eta_\\text{i}
    """
    return brightness * efficiency

def true_coincidence_rate(
        brightness: float,
        efficiency_a: float,
        efficiency_b: float,
) -> float:
    """
    Equation 3

    .. math::
        \\text{CC}^\\text{t} = B \\eta_\\text{A} \\eta_\\text{B}
    """
    return brightness * efficiency_a * efficiency_b

def heralding_efficiency(
        coincidence_rate: float,
        single_rate: float
) -> float:
    """
    Equation 4

    .. math::
        \\eta_\\text{A} = \\frac{\\text{CC}^\\text{t}}{S_\\text{B}^\\text{{t}}}
    """
    return coincidence_rate / single_rate

def total_heralding_efficiency(
        heralding_efficiency_a: float,
        heralding_efficiency_b: float
) -> float:
    """
    .. math::
        \\eta = \\sqrt{\\eta_\\text{A} \\eta_\\text{B}}
    """
    return np.sqrt(heralding_efficiency_a * heralding_efficiency_b)

def total_polarisation_error(
        polarisation_error_a: float,
        polarisation_error_b: float
) -> float:
    """
    Equation 5

    .. math::
        e^\\text{pol} = e_\\text{A}^\\text{pol} \\left(1 - e_\\text{B}^\\text{pol}\\right) + e_\\text{B}^\\text{pol} \\left(1 - e_\\text{A}^\\text{pol}\\right) 
    """
    return polarisation_error_a * (1 - polarisation_error_b) + polarisation_error_b * (1 - polarisation_error_a)

# noise-afflicted cw-qkd

def measured_single_rate(
        true_singles_rate: float,
        dark_count_rate: float
) -> float:
    """
    Equation 6

    .. math::
        S_\\text{i}^\\text{m} = S_\\text{i}^\\text{t} + \\text{DC}_\\text{A}
    """
    return true_singles_rate + dark_count_rate

def mean_clicks(
        measured_single_rate: float,
        coincidence_window: float
) -> float:
    """
    Equation 7

    .. math::
        \\mu_\\text{A}^S = S_\\text{A}^\\text{m} t_\\text{CC}
    """
    return measured_single_rate * coincidence_window

def accidental_probability(
        mean_clicks_a: float,
        mean_clicks_b: float
) -> float:
    """
    Equation 8

    .. math::
        P^\\text{acc} = \\left(1 - e^{-\\mu_\\text{A}^S}\\right) \\left(1 - e^{-\\mu_\\text{B}^S}\\right)
    """
    return (1 - np.exp(-mean_clicks_a)) * (1 - np.exp(-mean_clicks_b))

def accidental_probability_approx(
        mean_clicks_a: float,
        mean_clicks_b: float
) -> float:
    """
    Equation 9

    .. math::
        P^\\text{acc} \\approx \\mu_\\text{A}^S \\mu_\\text{B}^S

        \\mu_\\text{i}^S \\ll 1
    """
    return mean_clicks_a * mean_clicks_b

def accidental_coincidence_rate(
        probability: float,
        coincidence_window: float
) -> float:
    """
    Equation 10

    .. math::
        \\text{CC}^\\text{acc} = \\frac{P^\\text{acc}}{t_\\text{CC}}
    """
    return probability / coincidence_window

def accidental_coincidence_rate_approx(
        measured_singles_a: float,
        measured_singles_b: float,
        coincidence_window: float
) -> float:
    """
    Equation 10

    .. math::
        \\text{CC}^\\text{acc} \\approx S_\\text{A}^\\text{m} S_\\text{B}^\\text{m} t_\\text{CC}
    """
    return measured_singles_a * measured_singles_b * coincidence_window

def temporal_correlation_density(
        time: float,
        delay: float,
        timing_imprecision: float
) -> float:
    """
    Equation 11

    .. math::
        j\\left(t,t_\\Delta,t_\\text{D}\\right) = \\frac{2}{t_\\Delta} \\sqrt{\\frac{\\ln(2)}{\\pi}} \\exp\\left[-\\frac{4\\ln(2)}{t_\\Delta^2} \\left(t-t_\\text{D}\\right)^2\\right]
    """
    return (
        2 / timing_imprecision
        * np.sqrt(np.log(2)/np.pi)
        * np.exp(-4*np.log(2)/np.square(timing_imprecision) * np.square(time - delay))
    )

def coincidence_window_efficiency(
        coincidence_window: float,
        timing_imprecision: float
) -> float:
    """
    Equation 13

    .. math::
        \\eta^{t_\\text{CC}} = \\text{erf} \\left[\\sqrt{\\ln(2)} \\frac{t_\\text{CC}}{t_\\Delta}\\right]
    """
    return math.erf(np.sqrt(np.log(2)) * coincidence_window / timing_imprecision)

def measured_coincidence_rate(
        coincidence_window_efficiency: float,
        true_coincidence_rate: float,
        accidental_coincidence_rate: float
) -> float:
    """
    Equation 14

    .. math::
        \\text{CC}^\\text{m} = \\eta^{t_\\text{CC}} \\text{CC}^\\text{t} + \\text{CC}^\\text{acc}
    """
    return coincidence_window_efficiency * true_coincidence_rate + accidental_coincidence_rate

def erroneous_coincidence_rate(
        coincidence_window_efficiency: float,
        true_coincidence_rate: float,
        polarisation_error: float,
        accidental_coincidence_rate: float
) -> float:
    """
    Equation 15

    .. math::
        \\text{CC}^\\text{err} = \\eta^{t_\\text{CC}} \\text{CC}^\\text{t} e^\\text{pol} + \\frac{1}{2}\\text{CC}^\\text{acc}
    """
    return coincidence_window_efficiency * true_coincidence_rate * polarisation_error + (0.5*accidental_coincidence_rate)

# error rate and secure key rate

def qber_from_rate(
        erroneous_coincidence_rate: float,
        measured_coincidence_rate: float
) -> float:
    """
    Equation 16

    .. math::
        E = \\frac{\\text{CC}^\\text{err}}{\\text{CC}^\\text{m}} = \\frac{\\eta^{t_\\text{CC}} \\text{CC}^\\text{t} e^\\text{pol} + \\frac{1}{2}\\text{CC}^\\text{acc}}{\\eta^{t_\\text{CC}} \\text{CC}^\\text{t} + \\text{CC}^\\text{acc}}
    """
    return erroneous_coincidence_rate / measured_coincidence_rate

def visibility(
        qber: float
) -> float:
    """
    .. math::
        V = 1 - 2E
    """
    return 1 - 2*qber

def secure_key_rate(
        measured_coincidence_rate: float,
        bit_error_rate: float,
        phase_error_rate: float,
        *,
        sifting_probability: float = 0.5,
        error_correction_efficiency: float = 1.1
) -> float:
    """
    Equation 17

    .. math::
        R^\\text{s} = q\\text{CC}^\\text{m} \\left[1 - f\\left(E_\\text{bit}\\right)\\text{H}_2\\left(E_\\text{bit}\\right) - \\text{H}_2\\left(E_\\text{ph}\\right)\\right]
    """
    return (
        sifting_probability
        * measured_coincidence_rate
        * (
            1 - error_correction_efficiency
            * binary_entropy(bit_error_rate) - binary_entropy(phase_error_rate)
        )
    )

def secure_key_rate_symmetric(
        qber: float,
        measured_coincidence_rate: float,
) -> float:
    """
    Equation 19

    .. math::
        R^\\text{s} = \\frac{1}{2} \\text{CC}^\\text{m} \\left[1 - 2.1 \\text{H}_2(E)\\right]
    """
    return 0.5 * measured_coincidence_rate * (1 - 2.1 * binary_entropy(qber))