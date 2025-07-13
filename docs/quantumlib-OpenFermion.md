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
    <img src="https://img.shields.io/pypi/v/OpenFermion.svg?logo=semantic-release&logoColor=white&label=Release&style=flat-square&color=fcbc2c" alt="PyPI Version">
  </a>
  <a href="https://img.shields.io/pypi/dm/openfermion?logo=PyPI&color=d56420&logoColor=white&style=flat-square&label=Downloads" >
    <img src="https://img.shields.io/pypi/dm/openfermion?logo=PyPI&color=d56420&logoColor=white&style=flat-square&label=Downloads" alt="PyPI Downloads">
  </a>
</div>

# OpenFermion: Simulate and Analyze Fermionic Systems for Quantum Computing

OpenFermion is an open-source library designed to help researchers compile and analyze quantum algorithms for simulating fermionic systems, including quantum chemistry, providing tools and data structures for working with fermionic and qubit Hamiltonians. Explore the original repository [here](https://github.com/quantumlib/OpenFermion).

## Key Features

*   **Fermionic Hamiltonian Manipulation:** Create, manipulate, and analyze representations of fermionic Hamiltonians.
*   **Qubitization:** Convert fermionic Hamiltonians to qubit representations suitable for quantum computers.
*   **Electronic Structure Plugins:** Integrate with popular electronic structure packages (Psi4, PySCF, Q-Chem, etc.) for calculations.
*   **Circuit Compilation Plugins:** Support integration with quantum circuit compilation tools like Forest and Strawberry Fields.
*   **High-Performance Simulators:** Utilize plugins like OpenFermion-FQE for efficient simulation of fermionic quantum evolutions.

## Installation and Documentation

### Installation

Install the latest stable release using pip:

```bash
python -m pip install --user openfermion
```

For developers, install the latest version from source:

```bash
git clone https://github.com/quantumlib/OpenFermion
cd OpenFermion
python -m pip install -e .
```

### Documentation

Comprehensive documentation is available at:

*   [Official Documentation](https://quantumai.google/openfermion)
*   [Installation Guide](https://quantumai.google/openfermion/install)
*   [API Reference](https://quantumai.google/reference/python/openfermion/all_symbols)
*   [Tutorials](https://quantumai.google/openfermion/tutorials/intro_to_openfermion)

### Docker

For those encountering installation difficulties, a Docker image is available with OpenFermion and select plugins pre-installed:

*   [Docker Instructions](https://github.com/quantumlib/OpenFermion/tree/master/docker)

## Plugins

OpenFermion's modular architecture relies on plugins to extend functionality.

### High-Performance Simulators

*   **OpenFermion-FQE:** High-performance emulator of fermionic quantum evolutions: [OpenFermion-FQE](https://github.com/quantumlib/OpenFermion-FQE)

### Circuit Compilation Plugins

*   **Forest-OpenFermion:** Support for Rigetti's Forest: [Forest-OpenFermion](https://github.com/rigetticomputing/forestopenfermion)
*   **SFOpenBoson:** Support for Xanadu's Strawberry Fields: [SFOpenBoson](https://github.com/XanaduAI/SFOpenBoson)

### Electronic Structure Package Plugins

*   **OpenFermion-Psi4:** Integration with Psi4: [OpenFermion-Psi4](http://github.com/quantumlib/OpenFermion-Psi4)
*   **OpenFermion-PySCF:** Integration with PySCF: [OpenFermion-PySCF](http://github.com/quantumlib/OpenFermion-PySCF)
*   **OpenFermion-Dirac:** Integration with DIRAC: [OpenFermion-Dirac](https://github.com/bsenjean/Openfermion-Dirac)
*   **OpenFermion-QChem:** Integration with Q-Chem: [OpenFermion-QChem](https://github.com/qchemsoftware/OpenFermion-QChem)

## How to Contribute

Contributions are welcome!  Please review the [contributing guidelines](https://github.com/quantumlib/OpenFermion/blob/main/CONTRIBUTING.md) and adhere to the following:

*   **Contributor License Agreement (CLA):**  All contributions require a CLA.
*   **Pull Requests:**  Submit contributions via GitHub pull requests.
*   **Testing:**  Ensure your code includes comprehensive tests.
*   **Style Guide:** Follow PEP 8 guidelines.
*   **Documentation:**  Include thorough documentation for your code.
*   **Issue Tracker:** Use [GitHub issues](https://github.com/quantumlib/OpenFermion/issues) for bugs and feature requests.
*   **Stack Exchange:** Ask questions on the [Quantum Computing Stack Exchange](https://quantumcomputing.stackexchange.com/) with the 'openfermion' tag.

## Authors

(A list of authors is included in the original README)

## How to Cite

When citing OpenFermion, please use the following:

```bibtex
@article{mcclean2020openfermion,
  title={OpenFermion: The Electronic Structure Package for Quantum Computers},
  author={McClean, Jarrod R and Rubin, Nicholas C and Sung, Kevin J and Kivlichan, Ian D and Bonet-Monroig, Xavier and Cao, Yudong and Dai, Chengyu and Fried, E Schuyler and Gidney, Craig and Gimby, Brendan and others},
  journal={Quantum Science and Technology},
  volume={5},
  number={3},
  pages={034014},
  year={2020},
  publisher={IOP Publishing}
}
```

## Disclaimer

Copyright 2017 The OpenFermion Developers. This is not an official Google product.