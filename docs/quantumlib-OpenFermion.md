<div align="center">
  <img src="https://raw.githubusercontent.com/quantumlib/OpenFermion/refs/heads/master/docs/images/logo_horizontal.svg" alt="OpenFermion Logo" width="75%">
</div>

<div align="center">
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/Python-3.10+-fcbc2c.svg?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+">
  </a>
  <a href="https://github.com/quantumlib/OpenFermion/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-Apache%202.0-3c60b1.svg?logo=opensourceinitiative&logoColor=white&style=flat-square" alt="License: Apache 2.0">
  </a>
  <a href="https://pypi.org/project/OpenFermion">
    <img src="https://img.shields.io/pypi/v/OpenFermion.svg?logo=semantic-release&logoColor=white&label=Release&style=flat-square&color=fcbc2c" alt="PyPI">
  </a>
  <a href="https://img.shields.io/pypi/dm/openfermion?logo=PyPI&color=d56420&logoColor=white&style=flat-square&label=Downloads" >
    <img src="https://img.shields.io/pypi/dm/openfermion?logo=PyPI&color=d56420&logoColor=white&style=flat-square&label=Downloads" alt="PyPI downloads">
  </a>
</div>

# OpenFermion: Simulate and Analyze Fermionic Systems for Quantum Algorithms

OpenFermion is a powerful open-source library designed for quantum computing researchers to simulate and analyze fermionic systems, including quantum chemistry applications, providing essential tools and data structures for quantum algorithm development.  [**Explore the OpenFermion Repository**](https://github.com/quantumlib/OpenFermion).

**Key Features:**

*   **Fermionic Hamiltonian Manipulation:** Create, manipulate, and analyze representations of fermionic and qubit Hamiltonians.
*   **Quantum Algorithm Compilation:** Tools to compile and analyze quantum algorithms, particularly for simulating fermionic systems.
*   **Integration with Quantum Computing Platforms:** Seamless integration with platforms like Forest, Strawberry Fields, and others via plugins.
*   **Electronic Structure Package Integration:** Plugins to interface with popular electronic structure packages such as Psi4, PySCF, and Q-Chem.
*   **Open Source and Community-Driven:** Benefit from a collaborative environment with contributions and support from the quantum computing community.

## Installation and Documentation

### Installation

Install the latest stable release using pip:

```bash
python -m pip install --user openfermion
```

For developers, install the latest version:

```bash
git clone https://github.com/quantumlib/OpenFermion
cd OpenFermion
python -m pip install -e .
```

### Documentation

Comprehensive documentation is available at:

*   [Documentation](https://quantumai.google/openfermion)
*   [Installation](https://quantumai.google/openfermion/install)
*   [API Reference](https://quantumai.google/reference/python/openfermion/all_symbols)
*   [Tutorials](https://quantumai.google/openfermion/tutorials/intro_to_openfermion)

### Docker

For cross-platform compatibility, utilize the provided Docker image:

*   [Docker Instructions](https://github.com/quantumlib/OpenFermion/tree/master/docker)

## Plugins

Enhance OpenFermion's capabilities with these plugins:

*   **High-Performance Simulators:**
    *   [OpenFermion-FQE](https://github.com/quantumlib/OpenFermion-FQE)
*   **Circuit Compilation:**
    *   [Forest-OpenFermion](https://github.com/rigetticomputing/forestopenfermion)
    *   [SFOpenBoson](https://github.com/XanaduAI/SFOpenBoson)
*   **Electronic Structure Packages:**
    *   [OpenFermion-Psi4](http://github.com/quantumlib/OpenFermion-Psi4)
    *   [OpenFermion-PySCF](http://github.com/quantumlib/OpenFermion-PySCF)
    *   [OpenFermion-Dirac](https://github.com/bsenjean/Openfermion-Dirac)
    *   [OpenFermion-QChem](https://github.com/qchemsoftware/OpenFermion-QChem)

## Contributing

We welcome contributions!  Please review the [Contribution Guidelines](https://github.com/quantumlib/OpenFermion/blob/main/CONTRIBUTING.md) before submitting pull requests.

*   **CLA:** Contributions require a Contributor License Agreement (CLA).
*   **Pull Requests:** All submissions undergo review via GitHub pull requests.
*   **Testing:** Ensure new code includes extensive tests and adheres to the style guide (PEP 8).
*   **Documentation:** Always include documentation for new code.
*   **Issues:** Use [GitHub issues](https://github.com/quantumlib/OpenFermion/issues) to report bugs or request features.
*   **Questions:** For questions, use the [Quantum Computing Stack Exchange](https://quantumcomputing.stackexchange.com/) with the 'openfermion' tag.

## Authors

(List of Authors)

## How to Cite

If you use OpenFermion in your research, please cite the following paper:

*   Jarrod R McClean, et al. *OpenFermion: The Electronic Structure Package for Quantum Computers*. Quantum Science and Technology 5.3 (2020): 034014.  [Link to Publication](https://iopscience.iop.org/article/10.1088/2058-9565/ab8ebc/meta)

## Disclaimer

Copyright 2017 The OpenFermion Developers. This is not an official Google product.