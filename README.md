# qtoolkit

`qtoolkit` is a Python package containing tools for quantum optics, quantum communication, and experimental data analysis.

The package is intended to provide reusable implementations of common operations used in quantum optics experiments and simulations.

Current functionality includes:

- Timetag correlation for two-, three-, and four-fold coincidences
- Simulation of timetag data
- Convenience tools for handling timetag data
- Functions for calculating common experimental quantities such as QBER, visibility, fidelity, and purity
- Tools for modelling quantum key distribution (QKD) systems
- Tools for working with polarisation states and optical components

## Installation

qtoolkit requires Python 3.9 or later and supports Linux, macOS, and Windows.

A virtual environment is recommended to keep qtoolkit and its dependencies isolated from the system Python installation.

### Create a virtual environment

First, create a virtual environment in the project directory:

#### Linux and macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Windows
Using PowerShell:

```PowerShell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

or using Command Prompt:

```cmd
py -m venv .venv
.venv\Scripts\activate.bat
```

You can install the latest version of qtoolkit directly from GitHub using pip:
```bash
python -m pip install git+https://github.com/FarisRedza/qtoolkit.git
```

Alternatively, clone the repository and install the package locally:

```bash
git clone https://github.com/FarisRedza/qtoolkit.git
cd qtoolkit
python -m pip install .
```

For development, install the package in editable mode together with the development dependencies:

```bash
python -m pip install -e ".[dev]"
```

This installs qtoolkit along with the tools required for testing and development.

## Documentation

The documentation is built using [Sphinx](https://www.sphinx-doc.org/) and is available online at:

https://qtoolkit.readthedocs.io/en/latest/

To build the documentation locally, first install the documentation dependencies:

```bash
python -m pip install -e ".[doc]"
```

The HTML documentation can then be built with:

```bash
make -C docs clean && make -C docs dirhtml
```

The generated documentation will be placed under:

```text
docs/_build/dirhtml/
```

Read the Docs also builds the documentation automatically using the configuration in `.readthedocs.yaml`.

## Testing

qtoolkit uses [pytest](https://docs.pytest.org/) for its test suite.

Install the development dependencies:

```bash
python -m pip install -e ".[dev]"
```

and run the tests from the repository root:

```bash
pytest
```

The pytest configuration is stored in `pyproject.toml`. Tests are discovered from the `tests/` directory.

Coverage measurement is enabled automatically when running pytest. The test run reports statement and branch coverage for `qtoolkit`, including lines that are not currently covered:

```text
--cov=qtoolkit
--cov-report=term-missing
--cov-branch
```

Individual test files can also be run directly, for example:

```bash
pytest tests/qkd/test_cw.py
```

or a specific test can be selected with:

```bash
pytest tests/qkd/test_cw.py::test_name
```

Nox is used to run the test suite against all Python versions supported by qtoolkit.

To run the complete Nox test matrix:

```bash
nox
```

Nox creates isolated environments for the configured Python versions, installs qtoolkit and its test dependencies into each environment, and runs the test suite independently. This provides a convenient way to check that changes remain compatible with all supported Python versions rather than only the version installed in the development environment.

The supported Python versions are declared by the project metadata in pyproject.toml.

The test suite is also run automatically using GitHub Actions.

The workflows in .github/workflows/ provide continuous integration testing for the repository. They run the project's automated tests in clean environments, allowing changes pushed to GitHub to be checked independently of the local development environment.

This gives the project three complementary levels of testing:

pytest provides fast testing in the current development environment.
nox tests locally across the supported Python versions.
GitHub Actions performs automated testing in clean CI environments when changes are pushed to GitHub.

The GitHub Actions workflows can also be tested locally using act, if installed. For example:

```bash
act push -W .github/workflows/test_main.yaml
```

A custom event file can be supplied when testing workflows for other GitHub events:

```bash
act -e .act/tag-push.json
```


## Project structure

The main package is organised into modules covering different parts of the toolkit, including:

```text
qtoolkit
├── polarisation
├── qkd
└── timetags
```

* **`polarisation`** contains tools for working with polarisation states and optical components.
* **`qkd`** contains QKD metrics, protocol-related tools, and analytical models.
* **`timetags`** contains tools for processing, correlating, and simulating timetag data.

Examples demonstrating the use of qtoolkit can be found in the `examples/` directory.

## License

qtoolkit is distributed under the MIT License. See `LICENCE` for details.
