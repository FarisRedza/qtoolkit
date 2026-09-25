Continuous-wave entanglement-based QKD model
============================================

The :mod:`qtoolkit.qkd.cw` module implements the analytical continuous-wave
entanglement-based QKD model of
:ref:`Neumann et al. (2021) <ref-neumann-2021>`.  The functions are deliberately
small and closely follow the equations in that paper, making them useful both
individually and when constructing an end-to-end link model.

Model scope
-----------

The model describes a continuously pumped entangled-photon source using its
pair brightness, channel/detection efficiencies, detector dark counts,
coincidence-window width, timing imprecision, polarisation error, and detector
dead time.  It then estimates singles, true and accidental coincidences, QBER,
and an asymptotic secure key rate.

The model is not a finite-key security analysis.  In particular, the secure-key
rate functions do not account for finite block sizes, statistical confidence
bounds, parameter-estimation failure probabilities, or composable security
parameters.

Rates and brightness
--------------------

For source brightness :math:`B` and arm efficiency :math:`\eta_i`, the ideal
single rate is

.. math::

   S_i^t = B\eta_i,

and the true coincidence rate is

.. math::

   CC^t = B\eta_A\eta_B.

These correspond to Eqs. (2) and (3) of
:ref:`Neumann et al. (2021) <ref-neumann-2021>`.

Accidental coincidences
-----------------------

Following Neumann et al., qtoolkit provides a Poisson-based estimate of the
accidental-coincidence probability and its low-occupancy approximation.  If

.. math::

   \mu_i^S = S_i^m t_{CC},

then the model uses

.. math::

   P^{acc}
   =
   \left(1-e^{-\mu_A^S}\right)
   \left(1-e^{-\mu_B^S}\right).

For :math:`\mu_i^S\ll1`, the implementation also provides the approximation

.. math::

   CC^{acc}
   \approx
   S_A^m S_B^m t_{CC}.

Neumann et al. describe the first expression itself as an estimate based on
independent Poissonian click statistics.  The second expression introduces the
additional low-occupancy approximation :math:`\mu_i^S\ll1`, so it becomes
increasingly inaccurate as the mean number of clicks per window grows.

Timing and the coincidence window
---------------------------------

For Gaussian timing uncertainty, the fraction of true coincidences captured by
a coincidence window is modelled as

.. math::

   \eta^{t_{CC}}
   =
   \operatorname{erf}\left[
       \sqrt{\ln 2}\frac{t_{CC}}{t_\Delta}
   \right].

The measured coincidence rate combines this accepted fraction of true pairs
with accidental coincidences.  The same convention for ``coincidence_window``
and ``timing_imprecision`` must therefore be used consistently; the symbols
follow the definitions in the source paper.

QBER and secure key rate
------------------------

The model assigns the intrinsic polarisation error to true coincidences and a
one-half error probability to accidental coincidences.  The resulting QBER is
the erroneous coincidence rate divided by the measured coincidence rate.

The general asymptotic key-rate helper implements

.. math::

   R^s
   =
   q CC^m
   \left[
       1-f(E_\mathrm{bit})H_2(E_\mathrm{bit})
       -H_2(E_\mathrm{ph})
   \right].

Its default ``sifting_probability`` and ``error_correction_efficiency`` are
model parameters, not universal constants.  They should be changed when the
protocol or error-correction assumptions differ from those defaults.

Detector dead time
------------------

The module also implements the dead-time efficiency and corrected accidental
coincidence approximation given in Appendix B of
:ref:`Neumann et al. (2021) <ref-neumann-2021>`.  These helpers inherit the
assumptions of that model; they are not a general simulation of arbitrary
detector recovery behaviour.

See also
--------

* :doc:`examples/skr_brightness`
* :doc:`examples/skr_loss`
* :doc:`examples/skr_loss_timing`
* :doc:`../api/qkd/cw`
* :doc:`references`
