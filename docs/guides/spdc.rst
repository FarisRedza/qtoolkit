SPDC modelling
==============

qtoolkit provides tools for modelling spontaneous parametric
down-conversion (SPDC) in nonlinear optical materials.

Conventions
-----------

Unless stated otherwise:

* vacuum wavelengths are specified in metres
* angular frequencies are specified in radians per second
* temperatures are specified in degrees Celsius
* propagation angles are specified in radians
* propagation angles are measured relative to the crystal optic axis
* poling periods are specified at the 19 degrees Celsius reference
  temperature used by the current thermal-expansion model.

Energy conservation
-------------------

For SPDC,

.. math::

   \omega_p = \omega_s + \omega_i,

or equivalently,

.. math::

   \frac{1}{\lambda_p}
   =
   \frac{1}{\lambda_s}
   +
   \frac{1}{\lambda_i}.

The :func:`qtoolkit.spdc.conjugate_wavelength` function calculates
the remaining wavelength when the pump and one generated wavelength
are known.

Phase matching
--------------

The wavevector magnitude is

.. math::

   k = \frac{2\pi n}{\lambda}.

qtoolkit defines the quasi-phase-matching mismatch as

.. math::

   \Delta k
   =
   k_p-k_s-k_i
   -\frac{2\pi m}{\Lambda(T)},

where :math:`m` is the quasi-phase-matching order and
:math:`\Lambda(T)` is the physical poling period at the crystal
temperature.

Perfect quasi-phase matching corresponds to

.. math::

   \Delta k = 0.

Spectral model
--------------

The joint spectral amplitude is modelled as

.. math::

   f(\omega_s,\omega_i)
   =
   \alpha(\omega_s+\omega_i)
   \Phi(\omega_s,\omega_i),

where :math:`\alpha` is the pump spectral amplitude and
:math:`\Phi` is the longitudinal phase-matching amplitude.

For a uniform crystal,

.. math::

   \Phi
   =
   \operatorname{sinc}
   \left(
       \frac{\Delta kL}{2}
   \right)
   \exp\left(
       i\frac{\Delta kL}{2}
   \right).

The joint spectral intensity is

.. math::

   \mathrm{JSI}=|f|^2.

The JSA returned by qtoolkit is not normalised.

Pump bandwidth
--------------

The Gaussian pump model is defined in angular-frequency space.