Atmospheric turbulence
======================

The :mod:`qtoolkit.loss.turbulence` module currently provides a
Hufnagel--Valley refractive-index structure profile and a Fried coherence
length for a vertical path.  These functions describe atmospheric turbulence;
they do not by themselves calculate total optical link loss, beam wander,
scintillation, or received power.

Hufnagel--Valley profile
------------------------

The implemented profile is

.. math::

   C_n^2(z)
   =
   5.94\times10^{-53}
   \left(\frac{v}{27}\right)^2
   z^{10}e^{-z/1000}
   +2.7\times10^{-16}e^{-z/1500}
   +Ae^{-z/100},

where :math:`z` is altitude above ground in metres, :math:`v` is the
high-altitude RMS wind speed in metres per second, and :math:`A` controls the
near-ground turbulence strength.  The returned :math:`C_n^2` has units of
:math:`\mathrm{m}^{-2/3}`.

The Hufnagel--Valley model describes the altitude dependence of the
refractive-index structure parameter \(C_n^2\); it is not itself a model of
beam expansion. The resulting turbulence profile can be used to calculate
quantities such as the Fried coherence length and, together with an appropriate
propagation model, the resulting beam broadening, beam wander, and other
turbulence-induced effects. See :ref:`Hufnagel and Stanley (1964)
<ref-hufnagel-1964>` for the underlying atmospheric turbulence description and
:ref:`Bourgoin et al. (2013) <ref-bourgoin-2013>` and :ref:`Bonato et al. 
(2009) <ref-bonato-2009>` for applications to satellite quantum links.

Fried coherence length
----------------------

For a vertical path qtoolkit evaluates

.. math::

   r_0 =
   \left[
       0.423 k^2
       \int_0^{h_\max} C_n^2(z)\,dz
   \right]^{-3/5},

with :math:`k = 2\pi/\lambda`.  The Fried parameter :math:`r_0` is a
transverse atmospheric coherence scale associated with turbulence-induced
phase distortion.  The coherence-length concept was introduced by
:ref:`Fried (1966) <ref-fried-1966>`.

For satellite links, the corresponding expression generally depends on the
propagation geometry.  For example, :ref:`Bourgoin et al. (2013)
<ref-bourgoin-2013>` use a slant-path expression that includes the satellite
elevation angle and an altitude-dependent path-weighting term when calculating
the transverse coherence length for an uplink.

The current qtoolkit function integrates :math:`C_n^2` directly over altitude
without these geometric or path-weighting terms and therefore represents the
vertical-path approximation.  A general slant-path satellite link should not
be modelled simply by interpreting ``maximum_altitude`` as the slant range.

Example
-------

For the commonly used Hufnagel--Valley 5/7 parameter choice,
:math:`v=21\ \mathrm{m\,s^{-1}}` and
:math:`A=1.7\times10^{-14}\ \mathrm{m}^{-2/3}`:

.. code-block:: python

   from qtoolkit.loss import fried_parameter, turbulence

   cn2_ground = turbulence(
       v=21.0,
       A=1.7e-14,
       z=0.0,
   )

   r0 = fried_parameter(
       wavelength=785e-9,
       v=21.0,
       A=1.7e-14,
   )

See also
--------

* :doc:`../api/loss/index`
* :doc:`references`
