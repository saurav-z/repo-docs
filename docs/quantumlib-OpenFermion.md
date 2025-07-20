<!-- OpenFermion Logo -->
<p align="center">
  <img src="https://raw.githubusercontent.com/quantumlib/OpenFermion/refs/heads/master/docs/images/logo_horizontal.svg" alt="OpenFermion Logo" width="75%">
</p>

<!-- Badges -->
<p align="center">
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/Python-3.10+-fcbc2c.svg?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+">
  </a>
  <a href="https://github.com/quantumlib/OpenFermion/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-Apache%202.0-3c60b1.svg?logo=opensourceinitiative&logoColor=white&style=flat-square" alt="License: Apache 2.0">
  </a>
  <a href="https://pypi.org/project/OpenFermion">
    <img src="https://img.shields.io/pypi/v/OpenFermion.svg?logo=semantic-release&logoColor=white&label=Release&style=flat-square&color=fcbc2c" alt="PyPI Version">
  </a>
  <a href="https://img.shields.io/pypi/dm/openfermion?logo=PyPI&color=d56420&logoColor=white&style=flat-square&label=Downloads">
    <img src="https://img.shields.io/pypi/dm/openfermion?logo=PyPI&color=d56420&logoColor=white&style=flat-square&label=Downloads" alt="PyPI Downloads">
  </a>
</p>

## OpenFermion: Accelerate Quantum Algorithm Development for Fermionic Systems

OpenFermion is a powerful, open-source library for simulating and analyzing quantum algorithms, specifically designed for fermionic systems, including quantum chemistry, offering tools for manipulating and understanding quantum computations.  Explore the source code on [GitHub](https://github.com/quantumlib/OpenFermion).

**Key Features:**

*   **Fermionic and Qubit Hamiltonian Manipulation:** Efficiently work with representations of fermionic and qubit Hamiltonians.
*   **Quantum Algorithm Compilation:** Compile quantum algorithms for simulating fermionic systems.
*   **Quantum Chemistry Focus:** Designed with quantum chemistry applications in mind.
*   **Modular Plugin Architecture:** Extensible with plugins for circuit compilation, high-performance simulation, and electronic structure calculations.
*   **Integration with Popular Tools:** Seamlessly integrates with tools like Psi4, PySCF, and more.
*   **Comprehensive Documentation & Tutorials:**  Learn how to get started with the  [official documentation](https://quantumai.google/openfermion).
*   **Active Development and Community Support:** Benefit from ongoing development and a supportive community.

**Quick Start**

Run interactive Jupyter Notebooks in [Google Colab](https://colab.research.google.com/github/quantumlib/OpenFermion) or [MyBinder](https://mybinder.org/v2/gh/quantumlib/OpenFermion/master?filepath=examples).

## Installation

**Prerequisites**: Python 3.10+ and `pip`

### Installing OpenFermion

Install the latest stable release using pip:

```bash
python -m pip install --user openfermion
```

### Developer Installation

To install the latest development version (in editable mode):

```bash
git clone https://github.com/quantumlib/OpenFermion
cd OpenFermion
python -m pip install -e .
```

## Plugins

OpenFermion's functionality is extended through modular plugins.

### High-Performance Simulators

*   [OpenFermion-FQE](https://github.com/quantumlib/OpenFermion-FQE): High-performance emulator for fermionic quantum evolutions, leveraging symmetries.

### Circuit Compilation Plugins

*   [Forest-OpenFermion](https://github.com/rigetticomputing/forestopenfermion): Integration with Rigetti's Forest platform.
*   [SFOpenBoson](https://github.com/XanaduAI/SFOpenBoson): Integration with Xanadu's Strawberry Fields.

### Electronic Structure Package Plugins

*   [OpenFermion-Psi4](http://github.com/quantumlib/OpenFermion-Psi4): Integration with Psi4.
*   [OpenFermion-PySCF](http://github.com/quantumlib/OpenFermion-PySCF): Integration with PySCF.
*   [OpenFermion-Dirac](https://github.com/bsenjean/Openfermion-Dirac): Integration with DIRAC.
*   [OpenFermion-QChem](https://github.com/qchemsoftware/OpenFermion-QChem): Integration with Q-Chem.

## How to Contribute

We welcome contributions! Please review the [contributing guidelines](https://github.com/quantumlib/OpenFermion/blob/master/CONTRIBUTING.md) for details on submitting code, testing, and style guidelines.

## Authors

A list of contributors can be found in the original README.

## How to Cite

Please cite the following paper when using OpenFermion in your research:

```
Jarrod R McClean, et al. *OpenFermion: The Electronic Structure Package for Quantum Computers*. Quantum Science and Technology 5.3 (2020): 034014.
```

## Disclaimer

Copyright 2017 The OpenFermion Developers.
This is not an official Google product.