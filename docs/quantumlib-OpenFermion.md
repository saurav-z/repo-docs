<div align="center">
  <img src="https://raw.githubusercontent.com/quantumlib/OpenFermion/refs/heads/master/docs/images/logo_horizontal.svg" alt="OpenFermion Logo" width="75%">
</div>

<div align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.10+-fcbc2c.svg?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+"></a>
  <a href="https://github.com/quantumlib/OpenFermion/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-3c60b1.svg?logo=opensourceinitiative&logoColor=white&style=flat-square" alt="License: Apache 2.0"></a>
  <a href="https://pypi.org/project/OpenFermion"><img src="https://img.shields.io/pypi/v/OpenFermion.svg?logo=semantic-release&logoColor=white&label=Release&style=flat-square&color=fcbc2c" alt="PyPI Release"></a>
  <a href="https://img.shields.io/pypi/dm/openfermion?logo=PyPI&color=d56420&logoColor=white&style=flat-square&label=Downloads" alt="PyPI Downloads"></a>
</div>

OpenFermion is an open-source library for simulating fermionic systems on quantum computers, providing tools for quantum chemistry and beyond.

**[View the original repository on GitHub](https://github.com/quantumlib/OpenFermion)**

## Key Features

*   **Fermionic Hamiltonian Manipulation:** Construct, manipulate, and analyze fermionic and qubit Hamiltonians.
*   **Quantum Algorithm Compilation:** Compile and optimize quantum algorithms for fermionic simulations.
*   **Integration with Quantum Chemistry Packages:** Seamlessly integrate with popular electronic structure packages like Psi4 and PySCF.
*   **High-Performance Simulators:** Leverage high-performance emulators for fermionic quantum evolutions, like OpenFermion-FQE.
*   **Modular Plugin Architecture:** Extensible through plugins for circuit compilation and integration with other quantum software (e.g., Forest, Strawberry Fields).

## Installation and Documentation

### Installation

Install the latest stable release using pip:

```bash
python -m pip install --user openfermion
```

For developers, install the latest development version:

```bash
git clone https://github.com/quantumlib/OpenFermion
cd OpenFermion
python -m pip install -e .
```

### Documentation

Comprehensive documentation is available at [https://quantumai.google/openfermion](https://quantumai.google/openfermion).

*   [Installation](https://quantumai.google/openfermion/install)
*   [API Docs](https://quantumai.google/reference/python/openfermion/all_symbols)
*   [Tutorials](https://quantumai.google/openfermion/tutorials/intro_to_openfermion)

### Platform Compatibility

OpenFermion is tested on Mac, Windows, and Linux. While most plugins are compatible with Mac and Linux, a Docker image is provided for broader compatibility and easier installation of the core library and plugins, which can be found in the [docker folder](https://github.com/quantumlib/OpenFermion/tree/master/docker).

## Plugins

OpenFermion leverages plugins for extending functionality. Key plugins include:

### High-Performance Simulators

*   **OpenFermion-FQE:** High-performance emulator for fermionic quantum evolutions. ([https://github.com/quantumlib/OpenFermion-FQE](https://github.com/quantumlib/OpenFermion-FQE))

### Circuit Compilation Plugins

*   **Forest-OpenFermion:** Integration with Forest.
*   **SFOpenBoson:** Integration with Strawberry Fields.

### Electronic Structure Package Plugins

*   **OpenFermion-Psi4:** Integration with Psi4.
*   **OpenFermion-PySCF:** Integration with PySCF.
*   **OpenFermion-Dirac:** Integration with DIRAC.
*   **OpenFermion-QChem:** Integration with Q-Chem.

## How to Contribute

Contributions are welcome! Please review the [contributing guidelines](https://github.com/quantumlib/OpenFermion/blob/main/CONTRIBUTING.md) for details on contributing code, including the need for a Contributor License Agreement (CLA), code review processes, and testing requirements.

*   Use GitHub pull requests for contributions.
*   Ensure new code has extensive tests.
*   Follow PEP 8 style guidelines.
*   Provide documentation for any new code.

## Authors

See the original README for a full list of authors.

## How to Cite

When using OpenFermion in your research, please cite the following paper:

> Jarrod R McClean, et al. *OpenFermion: The Electronic Structure Package for Quantum Computers*. Quantum Science and Technology 5.3 (2020): 034014. ([https://iopscience.iop.org/article/10.1088/2058-9565/ab8ebc/meta](https://iopscience.iop.org/article/10.1088/2058-9565/ab8ebc/meta))

## Disclaimer

Copyright 2017 The OpenFermion Developers.
This is not an official Google product.