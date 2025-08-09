<div align="center">
  <img src="https://raw.githubusercontent.com/quantumlib/OpenFermion/refs/heads/master/docs/images/logo_horizontal.svg" alt="OpenFermion Logo" width="75%">
</div>

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-fcbc2c.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-3c60b1.svg?logo=opensourceinitiative&logoColor=white&style=flat-square)](https://github.com/quantumlib/OpenFermion/blob/main/LICENSE)
[![PyPI Release](https://img.shields.io/pypi/v/OpenFermion.svg?logo=semantic-release&logoColor=white&label=Release&style=flat-square&color=fcbc2c)](https://pypi.org/project/OpenFermion)
[![Downloads](https://img.shields.io/pypi/dm/openfermion?logo=PyPI&color=d56420&logoColor=white&style=flat-square&label=Downloads)](https://img.shields.io/pypi/dm/OpenFermion)

# OpenFermion: Simulate and Analyze Fermionic Systems for Quantum Computing

**OpenFermion is an open-source library designed for simulating and analyzing fermionic systems, particularly in quantum chemistry, offering tools for compiling and manipulating Hamiltonians.** Explore the power of OpenFermion to advance your quantum computing research!

**[Explore the OpenFermion Repository](https://github.com/quantumlib/OpenFermion)**

## Key Features:

*   **Fermionic Hamiltonian Manipulation:** Provides data structures and tools to obtain and manipulate representations of fermionic and qubit Hamiltonians.
*   **Integration with Quantum Simulators:** Offers plugins for high-performance simulators and circuit compilation, supporting integration with platforms like Forest and Strawberry Fields.
*   **Electronic Structure Package Plugins:** Integrates with popular electronic structure packages such as Psi4, PySCF, DIRAC, and Q-Chem.
*   **Comprehensive Documentation:** Extensive documentation, including installation guides, API references, and tutorials.
*   **Cross-Platform Compatibility:** Tested on Mac, Windows, and Linux, with Docker support for broader compatibility.

## Installation and Documentation

### Installation

Install the latest stable version using pip:

```bash
python -m pip install --user openfermion
```

For developers, install the latest version in development mode:

```bash
git clone https://github.com/quantumlib/OpenFermion
cd OpenFermion
python -m pip install -e .
```

### Documentation

*   **Official Documentation:** [quantumai.google/openfermion](https://quantumai.google/openfermion)
*   **Installation:** [quantumai.google/openfermion/install](https://quantumai.google/openfermion/install)
*   **API Docs:** [quantumai.google/reference/python/openfermion/all_symbols](https://quantumai.google/reference/python/openfermion/all_symbols)
*   **Tutorials:** [quantumai.google/openfermion/tutorials/intro_to_openfermion](https://quantumai.google/openfermion/tutorials/intro_to_openfermion)

## Plugins

OpenFermion's modular design leverages plugins for advanced functionality.

### High-Performance Simulators

*   **OpenFermion-FQE:**  A high-performance emulator for fermionic quantum evolutions, designed to exploit symmetries. ([OpenFermion-FQE](https://github.com/quantumlib/OpenFermion-FQE))

### Circuit Compilation Plugins

*   **Forest-OpenFermion:**  Supports integration with Rigetti's Forest. ([Forest-OpenFermion](https://github.com/rigetticomputing/forestopenfermion))
*   **SFOpenBoson:** Supports integration with Xanadu's Strawberry Fields. ([SFOpenBoson](https://github.com/XanaduAI/SFOpenBoson))

### Electronic Structure Package Plugins

*   **OpenFermion-Psi4:** Integration with Psi4. ([OpenFermion-Psi4](http://github.com/quantumlib/OpenFermion-Psi4))
*   **OpenFermion-PySCF:** Integration with PySCF. ([OpenFermion-PySCF](http://github.com/quantumlib/OpenFermion-PySCF))
*   **OpenFermion-Dirac:** Integration with DIRAC. ([Openfermion-Dirac](https://github.com/bsenjean/Openfermion-Dirac))
*   **OpenFermion-QChem:** Integration with Q-Chem. ([OpenFermion-QChem](https://github.com/qchemsoftware/OpenFermion-QChem))

## How to Contribute

We welcome contributions!  Please review our [contribution guidelines](https://github.com/quantumlib/OpenFermion/blob/main/CONTRIBUTING.md) and ensure your contributions are accompanied by a Contributor License Agreement (CLA).

*   **CLA:**  Sign the CLA at [https://cla.developers.google.com/](https://cla.developers.google.com/).
*   **Pull Requests:** Submit contributions via GitHub pull requests.
*   **Testing:**  Ensure your code includes comprehensive tests.
*   **Style Guide:**  Follow PEP 8 style guidelines.
*   **Documentation:**  Include documentation for all new code.
*   **Issues:** Report bugs and feature requests using [Github issues](https://github.com/quantumlib/OpenFermion/issues)

## Authors

(See original README for author list)

## How to Cite

When using OpenFermion in your research, please cite the following publication:

Jarrod R McClean, et al.  *OpenFermion: The Electronic Structure Package for Quantum Computers*. Quantum Science and Technology 5.3 (2020): 034014. [https://iopscience.iop.org/article/10.1088/2058-9565/ab8ebc/meta](https://iopscience.iop.org/article/10.1088/2058-9565/ab8ebc/meta)

## Disclaimer

Copyright 2017 The OpenFermion Developers.  This is not an official Google product.