SPDC modelling
==============

qtoolkit provides tools for modelling spontaneous parametric down-conversion
(SPDC) in nonlinear optical materials.  The current implementation is a
plane-wave spectral model: it describes material dispersion, birefringent and
quasi-phase matching, and the longitudinal joint spectral amplitude.  It does
not currently model focusing, collection modes, spatial walk-off, nonlinear
conversion efficiency, or absolute pair-generation rate.

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

The :func:`qtoolkit.spdc.conjugate_wavelength` function calculates the
remaining wavelength when the pump and one generated wavelength are known.

Material dispersion and propagation angle
-----------------------------------------

The current :class:`qtoolkit.spdc.MgOLithiumNiobate` model implements the
temperature-dependent Sellmeier equations for 5 mol% MgO-doped congruent
lithium niobate reported by :ref:`Gayer et al. (2008) <ref-gayer-2008>`.

For a uniaxial material the ordinary refractive index is independent of
propagation angle.  For extraordinary polarisation, qtoolkit uses the
effective index

.. math::

   \frac{1}{n_\mathrm{eff}^2(\theta)}
   =
   \frac{\cos^2\theta}{n_o^2}
   +
   \frac{\sin^2\theta}{n_e^2},

where :math:`\theta` is the angle between the propagation direction and the
optic axis.  With this convention, :math:`n_\mathrm{eff}(0)=n_o` and
:math:`n_\mathrm{eff}(\pi/2)=n_e`.

The angle arguments accepted by the phase-matching functions are therefore
propagation angles, not waveplate angles or crystal-face angles.

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
:math:`\Lambda(T)` is the physical poling period at the crystal temperature.
Perfect quasi-phase matching corresponds to :math:`\Delta k=0`.  The general
QPM convention and the role of grating period, wavelength, angle, and
temperature are discussed by :ref:`Fejer et al. (1992) <ref-fejer-1992>`.

The solver functions can search for a phase-matching temperature, wavelength,
wavelength pair, or propagation angle.  They are numerical root finders: a
returned solution satisfies the model used by qtoolkit, but it should not be
interpreted as a complete model of a real crystal without checking that the
material model, geometry, and poling-period convention match the experiment.

Poling-period temperature convention
------------------------------------

The public ``poling_period`` arguments currently represent the grating period
at 19 degrees Celsius.  qtoolkit converts this reference period to the
physical period at temperature :math:`T` using

.. math::

   \Lambda(T)
   =
   \Lambda_{19}
   \left[
       1 + \alpha(T-19) + \beta(T-19)^2
   \right],

with

.. math::

   \alpha=1.53\times10^{-5}\ \mathrm{K}^{-1},
   \qquad
   \beta=5.3\times10^{-9}\ \mathrm{K}^{-2}.

The thermal-expansion coefficients are based on lithium-niobate data from
:ref:`Kim and Smith (1969) <ref-kim-smith-1969>`, while the use of a poling
period referenced to 19 degrees Celsius follows :ref:`Paul et al. (2007)
<ref-paul-2007>`. Because poling periods may be specified at a different
reference temperature, a quoted or manufacturer-specified period should be
converted consistently before it is used in the phase-matching functions.

.. important::

   Thermal expansion is currently implemented globally in the SPDC
   phase-matching module even though it is material dependent.  Consequently,
   the current poling-temperature model should be regarded as specific to the
   lithium-niobate model rather than as a generic property of every
   :class:`qtoolkit.spdc.NonlinearMaterial`.

Spectral model
--------------

The joint spectral amplitude (JSA) is modelled as

.. math::

   f(\omega_s,\omega_i)
   =
   \alpha(\omega_s+\omega_i)
   \Phi(\omega_s,\omega_i),

where :math:`\alpha` is the pump spectral amplitude and :math:`\Phi` is the
longitudinal phase-matching amplitude.  This pump-envelope times
phase-matching structure is standard in treatments of the SPDC joint spectrum;
see, for example, :ref:`Grice and Walmsley (1997) <ref-grice-walmsley-1997>`.

For a uniform crystal, qtoolkit uses

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

The JSA returned by qtoolkit is not normalised and therefore does not by itself
provide an absolute pair-generation rate.

Pump bandwidth
--------------

The Gaussian pump model is defined in angular-frequency space.  The helper
:func:`qtoolkit.spdc.pump_wavelength_fwhm_to_angular_frequency_std` converts a
wavelength FWHM into an angular-frequency standard deviation using the local
linearisation of :math:`\omega=2\pi c/\lambda` around the pump centre.  This is
a narrow-band approximation rather than an exact transformation of a Gaussian
wavelength distribution.

Wavelength-domain plots
-----------------------

The JSA is naturally formulated as a function of angular frequency.  qtoolkit
allows wavelength arrays as convenient coordinates, but a probability density
does not transform between frequency and wavelength without a Jacobian.  In
particular,

.. math::

   \left|\frac{d\omega}{d\lambda}\right|
   = \frac{2\pi c}{\lambda^2}.

For narrow wavelength ranges this distinction may have little effect on the
visual shape of a spectrum, but it matters when wavelength-domain marginals
are to be interpreted quantitatively as transformed probability densities.

See also
--------

* :doc:`examples/phasematching`
* :doc:`examples/phasematching_spectrum`
* :doc:`examples/joint_spectral_intensity`
* :doc:`../api/spdc/index`
* :doc:`references`
