qtoolkit
========

**Tools for quantum optics, quantum communication, and experimental data analysis.**

qtoolkit is a Python package providing reusable tools for quantum optics
experiments, simulations, and data analysis. It includes functionality for
working with polarisation states, processing and simulating timetag data, and
calculating quantities commonly used in quantum key distribution (QKD).

The package is intended to provide useful building blocks that can be combined
for both experimental data analysis and simulation.

Features
--------

qtoolkit currently provides tools for:

* **Timetag processing** -- correlation and coincidence counting for two-,
  three-, and four-fold coincidences.
* **Timetag simulation** -- generation of simulated timetag streams, including
  live simulations of singles and correlated detection events.
* **Polarisation** -- Jones matrices, waveplates, polarisation states, and
  simulated polarisation measurements.
* **Quantum key distribution** -- calculation of QBER, visibility, fidelity,
  purity, secure key rates, and other commonly used quantities.
* **QKD modelling** -- analytical tools for modelling entanglement-based QKD
  systems, including detector noise, accidental coincidences, timing
  imprecision, and channel loss.
* **SPDC modelling** -- nonlinear-material dispersion, periodically
  poled crystal phase matching, angular phase matching, and joint
  spectral amplitude and intensity calculations.

Getting started
---------------

qtoolkit requires Python 3.9 or later. The latest version can be installed
directly from GitHub using pip:

.. code-block:: bash

   python -m pip install "qtoolkit @ git+https://github.com/FarisRedza/qtoolkit.git"

The :doc:`guides/index` section contains examples and guides demonstrating how
the different parts of qtoolkit can be used.

For detailed information about individual classes and functions, see the
:doc:`api/index`.

Examples
--------

The examples demonstrate how qtoolkit can be used for tasks such as simulating
entanglement-based QKD systems, investigating secure key rates, modelling the
effects of channel loss and detector timing, and simulating polarisation
measurements and waveplate rotations.

See the :doc:`guides/examples/index` page for complete examples with source code.

Documentation
-------------

.. toctree::
   :maxdepth: 2

   guides/index
   api/index

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`