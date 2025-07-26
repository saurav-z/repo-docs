<div align="center">
  <img src="https://raw.githubusercontent.com/quantumlib/OpenFermion/refs/heads/master/docs/images/logo_horizontal.svg" alt="OpenFermion Logo" width="75%">
</div>

[![Python](https://img.shields.io/badge/Python-3.10+-fcbc2c.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-3c60b1.svg?logo=opensourceinitiative&logoColor=white&style=flat-square)](https://github.com/quantumlib/OpenFermion/blob/main/LICENSE)
[![PyPI Version](https://img.shields.io/pypi/v/OpenFermion.svg?logo=semantic-release&logoColor=white&label=Release&style=flat-square&color=fcbc2c)](https://pypi.org/project/OpenFermion)
[![Downloads](https://img.shields.io/pypi/dm/openfermion?logo=PyPI&color=d56420&logoColor=white&style=flat-square&label=Downloads)](https://img.shields.io/pypi/dm/OpenFermion)

<br>

**OpenFermion is an open-source library designed for simulating and analyzing fermionic systems, including quantum chemistry, making it a powerful tool for quantum algorithm research.**

<br>

## Key Features

*   **Fermionic Hamiltonian Manipulation:** Provides data structures and tools for obtaining and manipulating representations of fermionic and qubit Hamiltonians.
*   **Quantum Algorithm Compilation:** Compiles and analyzes quantum algorithms to simulate fermionic systems.
*   **Extensive Plugin Support:** Integrates with various plugins for high-performance simulations, circuit compilation, and electronic structure calculations.
*   **Cross-Platform Compatibility:** Tested on Mac, Windows, and Linux, with Docker support for easy setup across all operating systems.
*   **Comprehensive Documentation:** Offers detailed documentation, tutorials, and API references to facilitate usage and development.

## Getting Started

### Installation

You can install OpenFermion using `pip`:

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

Explore the documentation for detailed information:

*   **Website:** [https://quantumai.google/openfermion](https://quantumai.google/openfermion)
*   **Installation:** [https://quantumai.google/openfermion/install](https://quantumai.google/openfermion/install)
*   **API Docs:** [https://quantumai.google/reference/python/openfermion/all_symbols](https://quantumai.google/reference/python/openfermion/all_symbols)
*   **Tutorials:** [https://quantumai.google/openfermion/tutorials/intro_to_openfermion](https://quantumai.google/openfermion/tutorials/intro_to_openfermion)

## Plugins

OpenFermion's modular design allows for extending functionality through plugins:

### High-Performance Simulators

*   [OpenFermion-FQE](https://github.com/quantumlib/OpenFermion-FQE): Emulator of fermionic quantum evolutions.

### Circuit Compilation Plugins

*   [Forest-OpenFermion](https://github.com/rigetticomputing/forestopenfermion): Integration with Forest.
*   [SFOpenBoson](https://github.com/XanaduAI/SFOpenBoson): Integration with Strawberry Fields.

### Electronic Structure Package Plugins

*   [OpenFermion-Psi4](http://github.com/quantumlib/OpenFermion-Psi4): Integration with Psi4.
*   [OpenFermion-PySCF](http://github.com/quantumlib/OpenFermion-PySCF): Integration with PySCF.
*   [OpenFermion-Dirac](https://github.com/bsenjean/Openfermion-Dirac): Integration with DIRAC.
*   [OpenFermion-QChem](https://github.com/qchemsoftware/OpenFermion-QChem): Integration with Q-Chem.

## Contributing

We welcome contributions!  Please see the [GitHub repository](https://github.com/quantumlib/OpenFermion) for instructions on how to contribute. All contributions require a Contributor License Agreement (CLA).

## Authors

See the original [Authors](https://github.com/quantumlib/OpenFermion#authors) section of the README for a full list.

## How to Cite

When using OpenFermion in your research, please cite:

> Jarrod R McClean, Nicholas C Rubin, Kevin J Sung, Ian D Kivlichan, Xavier Bonet-Monroig,
> Yudong Cao, Chengyu Dai, E Schuyler Fried, Craig Gidney, Brendan Gimby, Pranav Gokhale,
> Thomas Häner, Tarini Hardikar, Vojtĕch Havlíček, Oscar Higgott, Cupjin Huang, Josh Izaac,
> Zhang Jiang, Xinle Liu, Sam McArdle, Matthew Neeley, Thomas O'Brien, Bryan O'Gorman,
> Isil Ozfidan, Maxwell D Radin, Jhonathan Romero, Nicolas P D Sawaya, Bruno Senjean,
> Kanav Setia, Sukin Sim, Damian S Steiger, Mark Steudtner, Qiming Sun, Wei Sun, Daochen Wang,
> Fang Zhang, and Ryan Babbush
> *OpenFermion: The Electronic Structure Package for Quantum Computers*.
> `Quantum Science and Technology 5.3 (2020): 034014 <https://iopscience.iop.org/article/10.1088/2058-9565/ab8ebc/meta>`__.

##  Disclaimer

Copyright 2017 The OpenFermion Developers.  This is not an official Google product.