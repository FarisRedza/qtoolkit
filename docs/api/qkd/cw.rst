CW-QKD
======

The :mod:`qtoolkit.qkd.cw` module currently provides a collection of
low-level functions implementing the continuous-wave QKD model. A higher-level
interface is planned to provide a simpler way to configure and evaluate a
complete CW QKD system without manually combining the individual functions.

Some functions in this module duplicate functionality that is available
elsewhere in qtoolkit. These are included here for completeness and to keep the
individual components of the CW model directly accessible. The planned
higher-level interface will use the corresponding general qtoolkit
functionality where appropriate and will not expose these duplicate functions
as part of its public high-level API.

The low-level functions will remain useful for inspecting individual parts of
the model, reproducing specific calculations, and constructing custom
simulations.

.. automodule:: qtoolkit.qkd.cw
   :members:
   :undoc-members:
   :show-inheritance: