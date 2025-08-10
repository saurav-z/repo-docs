# QuTiP: Quantum Toolbox in Python

**QuTiP is a powerful open-source software for simulating the dynamics of closed and open quantum systems, making it an essential tool for quantum mechanics research and education.**  [Visit the original repo](https://github.com/qutip/qutip)

[![Build Status](https://github.com/qutip/qutip/actions/workflows/tests.yml/badge.svg?branch=master)](https://github.com/qutip/qutip/actions/workflows/tests.yml)
[![Coverage Status](https://img.shields.io/coveralls/qutip/qutip.svg?logo=Coveralls)](https://coveralls.io/r/qutip/qutip)
[![Maintainability](https://api.codeclimate.com/v1/badges/df502674f1dfa1f1b67a/maintainability)](https://codeclimate.com/github/qutip/qutip/maintainability)
[![license](https://img.shields.io/badge/license-New%20BSD-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![PyPi Downloads](https://img.shields.io/pypi/dm/qutip?label=downloads%20%7C%20pip&logo=PyPI)](https://pypi.org/project/qutip)
[![Conda-Forge Downloads](https://img.shields.io/conda/dn/conda-forge/qutip?label=downloads%20%7C%20conda&logo=Conda-Forge)](https://anaconda.org/conda-forge/qutip)

## Key Features

*   **Simulates Quantum Systems:**  Provides tools for simulating the time-evolution of quantum systems, both closed and open.
*   **Versatile:** Handles a wide range of quantum mechanical problems, including those with time-dependent Hamiltonians and collapse operators.
*   **Python-Based:** Built upon the popular Python scientific computing stack (NumPy, SciPy, Cython, and Matplotlib).
*   **Open Source & Free:**  Available under a permissive open-source license (New BSD), allowing for free use, modification, and redistribution.
*   **User-Friendly:** Designed to be accessible for both researchers and students.

## Core Functionality

QuTiP is a comprehensive library for quantum simulations, featuring:

*   Time-dependent Hamiltonians and collapse operators.
*   Support for various quantum systems, including those with time-dependence.
*   Efficient numerical simulations.
*   User-friendly interface.

## Installation

QuTiP is easily installable via `pip` or `conda`:

```bash
pip install qutip  # for a minimal installation
pip install qutip[full] # for installation with optional dependencies.
```
or
```bash
conda install -c conda-forge qutip
```

Refer to the [detailed installation guide in the documentation](https://qutip.readthedocs.io/en/stable/installation.html) for further instructions, including building from source.

## Documentation

*   [Latest Stable Release Documentation](https://qutip.readthedocs.io/en/latest/)
*   [Development Documentation (Master Branch)](https://qutip.readthedocs.io/en/master/)
*   [QuTiP Website Documentation](https://qutip.org/documentation.html)
*   [Tutorials and Examples](https://qutip.org/tutorials.html)

## Contribute

Contributions to QuTiP are welcome!  Please see the [contributing guide](https://qutip.readthedocs.io/en/stable/development/contributing.html) for details.

*   Fork the repository and submit pull requests.
*   Report bugs via the [issues page](https://github.com/qutip/qutip/issues).
*   Engage in discussions on the [QuTiP discussion group](https://groups.google.com/g/qutip).

## Support

QuTiP is supported by the [Unitary Fund](https://unitary.fund) and [NumFOCUS](https://numfocus.org).

We are grateful for [Nori's lab](https://dml.riken.jp/) at RIKEN and [Blais' lab](https://www.physique.usherbrooke.ca/blais/) at the Institut Quantique
for providing developer positions to work on QuTiP.

We also thank Google for supporting us by financing GSoC students to work on the QuTiP as well as [other supporting organizations](https://qutip.org/#supporting-organizations) that have been supporting QuTiP over the years.

## Citing QuTiP

If you use QuTiP in your research, please cite the original QuTiP papers, which are available [here](https://dml.riken.jp/?s=QuTiP).